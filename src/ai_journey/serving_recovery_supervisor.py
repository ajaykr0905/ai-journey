"""Supervise loopback server lifetimes: launch, await readiness, probe liveness.

Each lifetime starts from an argument vector without a shell, in a new
session, and the supervisor keeps its ``Popen`` handle. It only signals the
process group of a child it spawned and has not reaped, under the same lock
that polls exit status, so a recycled pid is never signaled. Liveness is the
child's exit status, polled every probe interval, plus ``GET /health`` probes;
a live child failing probes for the detection timeout is killed as unresponsive.
After an observed exit the supervisor waits a doubling backoff and launches a
replacement lifetime, at most ``restart_limit`` times per run; a replacement
that fails to spawn or to become ready consumes one restart. Exhaustion, or
any unexpected error that stops the monitor, is recorded as the supervisor's
failure, not hidden; ``endpoint`` then raises it so clients stop retrying a
server that will not come back.
"""

from __future__ import annotations

import http.client
import os
import selectors
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .serving_recovery_config import (
    ServingRecoveryConfig,
    ServingRecoveryError,
    require_digest,
)
from .serving_recovery_identity import require_run_id, server_lifetime_id
from .serving_recovery_server import (
    LOOPBACK,
    MAX_HANDSHAKE_BYTES,
    health_body,
    parse_handshake,
)

LIFECYCLE_KINDS = (
    "spawn",
    "ready",
    "unresponsive",
    "exit_observed",
    "restart",
    "exhausted",
    "stopped",
)
CommandFactory = Callable[[int, str], Sequence[str]]


@dataclass(frozen=True)
class LifecycleObservation:
    """One supervisor-observed lifecycle fact on the monotonic clock."""

    kind: str
    lifetime_number: int
    lifetime_id: str
    monotonic_ns: int
    pid: int
    port: int | None = None
    exit_code: int | None = None
    backoff_ms: int | None = None


@dataclass(frozen=True)
class LifetimeView:
    """Read-only snapshot of one supervised server lifetime."""

    number: int
    lifetime_id: str
    pid: int
    port: int | None
    ready: bool
    exit_code: int | None

    @property
    def endpoint(self) -> str:
        if self.port is None:
            raise ServingRecoveryError("lifetime has not published a port")
        return f"http://{LOOPBACK}:{self.port}"


class _Lifetime:
    def __init__(self, number: int, lifetime_id: str, process: subprocess.Popen):
        self.number = number
        self.lifetime_id = lifetime_id
        self.process = process
        self.port: int | None = None
        self.ready = False
        self.exit_code: int | None = None

    def view(self) -> LifetimeView:
        return LifetimeView(
            self.number,
            self.lifetime_id,
            self.process.pid,
            self.port,
            self.ready,
            self.exit_code,
        )


def probe_health(
    port: int, *, lifetime_id: str, pid: int, model_sha256: str, timeout_ms: int
) -> bool:
    """Return True only for an exact healthy identity from one lifetime."""

    expected = health_body(lifetime_id=lifetime_id, pid=pid, model_sha256=model_sha256)
    connection = http.client.HTTPConnection(LOOPBACK, port, timeout=timeout_ms / 1000)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        body = response.read(len(expected) + 1)
        return response.status == 200 and body == expected
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


class ServingSupervisor:
    """Own the server processes of one run and observe their lifecycle."""

    def __init__(
        self,
        *,
        run_id: str,
        config: ServingRecoveryConfig,
        model_sha256: str,
        command_for: CommandFactory,
        log_dir: Path | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(config, ServingRecoveryConfig):
            raise TypeError("config must be ServingRecoveryConfig")
        if not callable(command_for) or not callable(clock_ns):
            raise TypeError("command_for and clock_ns must be callable")
        if log_dir is not None and not isinstance(log_dir, Path):
            raise TypeError("log_dir must be pathlib.Path")
        self.run_id = require_run_id(run_id)
        self.config = config
        self.model_sha256 = require_digest(model_sha256, "model_sha256")
        self._command_for = command_for
        self._log_dir = log_dir
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._stopping = threading.Event()
        self._lifetimes: list[_Lifetime] = []
        self._observations: list[LifecycleObservation] = []
        self._monitor: threading.Thread | None = None
        self._endpoint: str | None = None
        self._restarts = 0
        self._failure: ServingRecoveryError | None = None

    # Observation and snapshots -------------------------------------------------

    def _observe(
        self, kind: str, lifetime: _Lifetime, *, backoff_ms: int | None = None
    ) -> LifecycleObservation:
        with self._changed:
            observation = LifecycleObservation(
                kind,
                lifetime.number,
                lifetime.lifetime_id,
                self._clock_ns(),
                lifetime.process.pid,
                lifetime.port,
                lifetime.exit_code,
                backoff_ms,
            )
            self._observations.append(observation)
            self._changed.notify_all()
            return observation

    def observations(self) -> tuple[LifecycleObservation, ...]:
        with self._lock:
            return tuple(self._observations)

    @property
    def restarts_used(self) -> int:
        with self._lock:
            return self._restarts

    @property
    def failure(self) -> ServingRecoveryError | None:
        """Return the error that ended supervision: exhaustion or a monitor fault."""

        with self._lock:
            return self._failure

    def current(self) -> LifetimeView:
        with self._lock:
            if not self._lifetimes:
                raise ServingRecoveryError("supervisor has not started a lifetime")
            return self._lifetimes[-1].view()

    def endpoint(self) -> str:
        """Return the most recently published endpoint, which may be stale.

        Once supervision has failed no lifetime will be published again, so the
        recorded failure is raised instead.
        """

        with self._lock:
            if self._failure is not None:
                raise ServingRecoveryError(str(self._failure)) from self._failure
            if self._endpoint is None:
                raise ServingRecoveryError("no server lifetime has become ready")
            return self._endpoint

    def wait_until(self, predicate: Callable[[], bool], *, timeout_ms: int) -> bool:
        """Wait for a lifecycle predicate without exceeding a bounded timeout."""

        deadline = time.monotonic() + timeout_ms / 1000
        with self._changed:
            while not predicate():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._changed.wait(remaining)
            return True

    # Process ownership -------------------------------------------------------

    def _spawn(self, number: int) -> _Lifetime:
        lifetime_id = server_lifetime_id(self.run_id, number)
        argv = tuple(self._command_for(number, lifetime_id))
        if not argv or any(type(argument) is not str for argument in argv):
            raise TypeError("command_for must return a non-empty string sequence")
        log_path = None
        stderr = subprocess.DEVNULL
        if self._log_dir is not None:
            log_path = self._log_dir / f"lifetime-{number}.stderr"
            stderr = log_path.open("xb")
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr,
                shell=False,
                start_new_session=True,
                env={},
            )
        except BaseException:
            # No child started, so the next restart reuses this lifetime number.
            if log_path is not None:
                log_path.unlink(missing_ok=True)
            raise
        finally:
            if stderr is not subprocess.DEVNULL:
                stderr.close()
        lifetime = _Lifetime(number, lifetime_id, process)
        with self._lock:
            self._lifetimes.append(lifetime)
        return lifetime

    def _poll(self, lifetime: _Lifetime) -> int | None:
        with self._lock:
            code = lifetime.process.poll()
            if code is not None and lifetime.exit_code is None:
                lifetime.exit_code = code
                lifetime.ready = False
                self._observe("exit_observed", lifetime)
            return code

    def _signal_group(self, lifetime: _Lifetime, signum: int) -> int:
        """Signal an owned, unreaped child group and return the signal time."""

        with self._lock:
            if lifetime not in self._lifetimes:
                raise ServingRecoveryError("refusing to signal an unsupervised process")
            if lifetime.exit_code is not None or lifetime.process.poll() is not None:
                raise ServingRecoveryError("refusing to signal an exited lifetime")
            signaled_ns = self._clock_ns()
            os.killpg(lifetime.process.pid, signum)
            return signaled_ns

    def signal_current(self, lifetime_id: str, signum: int) -> tuple[LifetimeView, int]:
        """Signal the live, ready, current lifetime named by ``lifetime_id``."""

        if type(signum) is not int and not isinstance(signum, signal.Signals):
            raise TypeError("signum must be a signal number")
        with self._lock:
            if not self._lifetimes or self._lifetimes[-1].lifetime_id != lifetime_id:
                raise ServingRecoveryError("target is not the current lifetime")
            lifetime = self._lifetimes[-1]
            if not lifetime.ready:
                raise ServingRecoveryError("target lifetime is not ready")
            view = lifetime.view()
            return view, self._signal_group(lifetime, signum)

    # Readiness ----------------------------------------------------------------

    def _read_handshake(self, lifetime: _Lifetime, deadline_ns: int) -> bytes:
        stream = lifetime.process.stdout
        data = b""
        with selectors.DefaultSelector() as selector:
            selector.register(stream, selectors.EVENT_READ)
            while not data.endswith(b"\n"):
                remaining = (deadline_ns - self._clock_ns()) / 1e9
                if remaining <= 0 or self._stopping.is_set():
                    raise ServingRecoveryError("server did not handshake in time")
                if not selector.select(min(remaining, 0.05)):
                    continue
                chunk = os.read(stream.fileno(), MAX_HANDSHAKE_BYTES + 1)
                if not chunk:
                    try:
                        code = lifetime.process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        raise ServingRecoveryError(
                            "server closed its handshake stream before readiness"
                        ) from None
                    raise ServingRecoveryError(
                        f"server exited with code {code} before readiness"
                    )
                data += chunk
                if len(data) > MAX_HANDSHAKE_BYTES:
                    raise ServingRecoveryError("server handshake exceeds its bound")
        return data

    def _await_ready(self, lifetime: _Lifetime, spawned_ns: int) -> None:
        deadline_ns = spawned_ns + self.config.readiness_timeout_ms * 1_000_000
        try:
            handshake = parse_handshake(
                self._read_handshake(lifetime, deadline_ns),
                lifetime_id=lifetime.lifetime_id,
                pid=lifetime.process.pid,
                model_sha256=self.model_sha256,
            )
        finally:
            lifetime.process.stdout.close()
        with self._lock:
            lifetime.port = handshake.port
        while not self._probe(lifetime):
            if self._poll(lifetime) is not None:
                raise ServingRecoveryError("server exited before readiness")
            remaining = (deadline_ns - self._clock_ns()) / 1e9
            if remaining <= 0 or self._stopping.is_set():
                raise ServingRecoveryError("server was not healthy before timeout")
            self._stopping.wait(min(remaining, self.config.probe_interval_ms / 1000))
        with self._lock:
            lifetime.ready = True
            self._endpoint = lifetime.view().endpoint
            self._observe("ready", lifetime)

    def _probe(self, lifetime: _Lifetime) -> bool:
        return lifetime.port is not None and probe_health(
            lifetime.port,
            lifetime_id=lifetime.lifetime_id,
            pid=lifetime.process.pid,
            model_sha256=self.model_sha256,
            timeout_ms=self.config.probe_timeout_ms,
        )

    def _launch(
        self, number: int, kind: str, *, backoff_ms: int | None = None
    ) -> _Lifetime:
        lifetime = self._spawn(number)
        spawned = self._observe(kind, lifetime, backoff_ms=backoff_ms)
        try:
            self._await_ready(lifetime, spawned.monotonic_ns)
        except BaseException:
            self._terminate(lifetime)
            raise
        return lifetime

    def _terminate(self, lifetime: _Lifetime) -> None:
        with self._lock:
            if lifetime.exit_code is None and lifetime.process.poll() is None:
                os.killpg(lifetime.process.pid, signal.SIGKILL)
        lifetime.process.wait(timeout=10)
        self._poll(lifetime)

    # Liveness -----------------------------------------------------------------

    def start(self) -> LifetimeView:
        """Launch lifetime 1, wait for readiness, then start liveness probes."""

        with self._lock:
            if self._lifetimes:
                raise ServingRecoveryError("supervisor already started")
        lifetime = self._launch(1, "spawn")
        self._monitor = threading.Thread(target=self._watch, daemon=True)
        self._monitor.start()
        return lifetime.view()

    def _watch(self) -> None:
        try:
            lifetime: _Lifetime | None = self._lifetimes[-1]
            while lifetime is not None and not self._stopping.is_set():
                self._watch_lifetime(lifetime)
                lifetime = None if self._stopping.is_set() else self._restart()
        except BaseException as exc:  # recorded: the run must fail, not hang
            failure = ServingRecoveryError(
                f"supervisor monitor failed: {type(exc).__name__}: {exc}"
            )
            failure.__cause__ = exc
            with self._changed:
                self._failure = failure
                self._changed.notify_all()

    def _restart(self) -> _Lifetime | None:
        """Launch a replacement after backoff until one is ready or none remain."""

        while not self._stopping.is_set():
            with self._lock:
                if self._restarts >= self.config.restart_limit:
                    self._failure = ServingRecoveryError(
                        f"restart limit of {self.config.restart_limit} exhausted"
                    )
                    self._observe("exhausted", self._lifetimes[-1])
                    return None
                self._restarts += 1
                backoff_ms = self.config.restart_delay_ms(self._restarts)
                number = len(self._lifetimes) + 1
            if self._stopping.wait(backoff_ms / 1000):
                return None
            try:
                return self._launch(number, "restart", backoff_ms=backoff_ms)
            except (ServingRecoveryError, OSError, TypeError):
                continue
        return None

    def _watch_lifetime(self, lifetime: _Lifetime) -> None:
        interval = self.config.probe_interval_ms / 1000
        detection_ns = self.config.detection_timeout_ms * 1_000_000
        last_healthy_ns = self._clock_ns()
        while not self._stopping.is_set():
            if self._poll(lifetime) is not None:
                return
            if self._probe(lifetime):
                last_healthy_ns = self._clock_ns()
            elif self._clock_ns() - last_healthy_ns >= detection_ns:
                with self._lock:
                    if lifetime.exit_code is None and lifetime.process.poll() is None:
                        self._observe("unresponsive", lifetime)
                        os.killpg(lifetime.process.pid, signal.SIGKILL)
            self._stopping.wait(interval)

    def stop(self) -> int:
        """Stop probing, SIGTERM the current lifetime and return its exit code."""

        self._stopping.set()
        if self._monitor is not None:
            self._monitor.join(timeout=self.config.readiness_timeout_ms / 1000 + 5)
        lifetime = self._lifetimes[-1]
        with self._lock:
            if lifetime.exit_code is None and lifetime.process.poll() is None:
                os.killpg(lifetime.process.pid, signal.SIGTERM)
        try:
            code = lifetime.process.wait(timeout=self.config.shutdown_timeout_ms / 1000)
        except subprocess.TimeoutExpired:
            with self._lock:
                if lifetime.process.poll() is None:
                    os.killpg(lifetime.process.pid, signal.SIGKILL)
            code = lifetime.process.wait(timeout=10)
        with self._lock:
            if lifetime.exit_code is None:
                lifetime.exit_code = code
                lifetime.ready = False
                self._observe("stopped", lifetime)
            return lifetime.exit_code

    def close(self) -> None:
        """Kill and reap every child this supervisor still owns."""

        self._stopping.set()
        if (
            self._monitor is not None
            and self._monitor is not threading.current_thread()
        ):
            self._monitor.join(timeout=self.config.readiness_timeout_ms / 1000 + 5)
        for lifetime in list(self._lifetimes):
            with self._lock:
                if lifetime.process.poll() is None:
                    os.killpg(lifetime.process.pid, signal.SIGKILL)
            lifetime.process.wait(timeout=10)
            if lifetime.process.stdout is not None:
                lifetime.process.stdout.close()

    def __enter__(self) -> ServingSupervisor:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
