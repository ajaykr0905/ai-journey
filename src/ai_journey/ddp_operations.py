"""Operating commands for comparison, evaluation and loopback serving."""

import json
import os
from pathlib import Path
import tempfile
import threading

import torch

from .ddp_benchmark import benchmark_service
from .ddp_comparison import compare_decompositions, write_comparison, verify_comparison
from .ddp_generation import SamplingConfig
from .ddp_http import make_server
from .ddp_limits import preflight
from .ddp_pipeline import run_pipeline, load_serving_model
from .ddp_service import GenerationService, GenerationRequest


def write_report(path: Path, report):
    body = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    if path.is_symlink():
        raise ValueError("report output must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def add_commands(commands):
    for name in ("matrix", "pipeline"):
        command = commands.add_parser(name)
        for field in ("source", "spec", "output", "workdir"):
            command.add_argument(f"--{field}", type=Path, required=True)
        command.add_argument("--steps", type=int, default=4)
    verify = commands.add_parser("verify-comparison")
    verify.add_argument("--report", type=Path, required=True)
    for name in ("serve", "benchmark"):
        command = commands.add_parser(name)
        for field in ("source", "spec", "checkpoint"):
            command.add_argument(f"--{field}", type=Path, required=True)
        if name == "serve":
            command.add_argument("--port", type=int, default=8080)
            command.add_argument("--max-seconds", type=int, default=60)
            command.add_argument("--max-connections", type=int, default=8)
            command.add_argument("--drain-seconds", type=int, default=10)
        else:
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--repeats", type=int, default=5)


def run_operation(arguments):
    if arguments.command in ("benchmark", "matrix", "pipeline"):
        _validate_operation_paths(arguments)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        command = arguments.command
        if command == "verify-comparison":
            if (
                arguments.report.is_symlink()
                or arguments.report.stat().st_size > 4 * 1024 * 1024
            ):
                raise ValueError("comparison report exceeds file constraints")
            return verify_comparison(json.loads(arguments.report.read_text()))
        if command == "matrix":
            report = compare_decompositions(
                source=arguments.source,
                spec_path=arguments.spec,
                directory=arguments.workdir,
                steps=arguments.steps,
            )
            write_comparison(arguments.output, report)
            return report
        if command == "pipeline":
            report = run_pipeline(
                source=arguments.source,
                spec_path=arguments.spec,
                directory=arguments.workdir,
                steps=arguments.steps,
            )
            write_report(arguments.output, report)
            return report
        spec = preflight(arguments.source, arguments.spec, world_size=1, steps=1)
        service = GenerationService(
            load_serving_model(
                arguments.checkpoint, spec, arguments.source.read_bytes()
            )
        )
        if command == "benchmark":
            request = GenerationRequest(
                "cli-benchmark",
                tuple(b"a"),
                SamplingConfig(max_new_tokens=4, temperature=0),
            )
            report = benchmark_service(service, request, repeats=arguments.repeats)
            write_report(arguments.output, report)
            return report
        if not 0 <= arguments.port <= 65535 or not 1 <= arguments.max_seconds <= 3600:
            raise ValueError("server port or lifetime exceeds lab limits")
        if not 0 <= arguments.drain_seconds <= 60:
            raise ValueError("drain timeout exceeds lab limits")
        server = make_server(
            service, port=arguments.port, max_connections=arguments.max_connections
        )

        def stop():
            service.drain(timeout_seconds=arguments.drain_seconds)
            server.shutdown()

        timer = threading.Timer(arguments.max_seconds, stop)
        timer.start()
        print(
            json.dumps(
                {
                    "listening": f"http://127.0.0.1:{server.server_port}",
                    **service.health(),
                }
            ),
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.05)
        finally:
            timer.cancel()
            service.drain(timeout_seconds=arguments.drain_seconds)
            timer.join(timeout=arguments.drain_seconds + 2)
            server.server_close()
        return service.health()
    finally:
        torch.set_num_threads(threads)


def _validate_operation_paths(arguments):
    """Reject destructive output aliases before model, worker or file creation."""
    output = arguments.output
    inputs = [arguments.source, arguments.spec]
    if arguments.command in ("benchmark", "train"):
        inputs.append(arguments.checkpoint)
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise ValueError("operation output must be a regular non-symlink path")
    for source in inputs:
        if output.resolve() == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise ValueError("operation output aliases a protected input")
    if arguments.command == "train" and output.resolve().is_relative_to(
        arguments.workdir.resolve()
    ):
        raise ValueError("training report must be outside its worker workdir")
    if arguments.command in ("matrix", "pipeline", "recover", "compare"):
        directory = arguments.workdir
        if directory.is_symlink() or (
            directory.exists() and (not directory.is_dir() or any(directory.iterdir()))
        ):
            raise ValueError("operation workdir must be new or empty")
        resolved = directory.resolve()
        if any(path.resolve().is_relative_to(resolved) for path in [*inputs, output]):
            raise ValueError("operation inputs and report must be outside its workdir")
