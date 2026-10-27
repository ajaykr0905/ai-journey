"""Run the installed serving operator against an owned tiny CPU control."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import http.client
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from .ddp_generation import SamplingConfig, generate_tokens
from .gpt2_training import (
    GPT2TrainingConfig,
    GPT2TrainingSource,
    GPT2TrainingSpec,
    build_gpt2_optimizer,
)
from .http_measurements import loopback_address, measure_http_generation
from .hellaswag_io import read_evaluation_bytes, parse_evaluation_json
from .prefix_cache import prefix_identity
from .prefix_report import (
    BUDGET,
    SCOPE,
    SNAPSHOT_FIELDS,
    control_config,
    schedule,
    seal_prefix_report,
    verify_prefix_report,
)
from .training_models import (
    build_training_model,
    model_config_mapping,
    training_model_fingerprint,
)


def _get(endpoint, route):
    connection = http.client.HTTPConnection(*loopback_address(endpoint), timeout=2)
    try:
        connection.request("GET", route)
        response = connection.getresponse()
        body = response.read(64 * 1024 + 1)
        if response.status != 200 or len(body) > 64 * 1024:
            raise RuntimeError(
                "owned prefix readiness server returned invalid telemetry"
            )
        return body.decode()
    finally:
        connection.close()


def _metrics(endpoint):
    values = {}
    for line in _get(endpoint, "/metrics").splitlines():
        if line and not line.startswith("#"):
            name, value = line.split()
            if name in SNAPSHOT_FIELDS:
                if name in values:
                    raise RuntimeError("duplicate prefix readiness metric")
                values[name] = int(value)
    if set(values) != set(SNAPSHOT_FIELDS):
        raise RuntimeError("installed serving operator lacks prefix telemetry")
    return values


def _wait_idle(endpoint, *, attempts):
    until = time.monotonic() + 3
    while time.monotonic() < until:
        if json.loads(_get(endpoint, "/health"))["active_requests"] == 0:
            values = _metrics(endpoint)
            observed = sum(
                values[f'ai_journey_prefix_attempts_total{{outcome="{outcome}"}}']
                for outcome in ("hit", "miss", "bypass")
            )
            if observed == attempts:
                return values
            if observed > attempts:
                raise RuntimeError("owned readiness service observed unexpected work")
        time.sleep(0.01)
    raise RuntimeError("owned prefix readiness inference did not settle")


def _assert_installed(root):
    root = Path(root).resolve()
    if not (root / "ai_journey").is_dir():
        raise ValueError("installed-root must contain the installed ai_journey package")
    for name, module in tuple(sys.modules.items()):
        if (
            (name == "ai_journey" or name.startswith("ai_journey."))
            and getattr(module, "__file__", None)
            and not Path(module.__file__).resolve().is_relative_to(root)
        ):
            raise ValueError(
                "prefix readiness must import only the requested installed package"
            )
    return root


def run_prefix_readiness(*, installed_root, output_dir):
    """Use a new directory and only this function's explicitly owned child.

    The saved checkpoint is step zero, not evidence of training. Timings are
    actual client receipt observations, never GPU or speedup assertions.
    """
    installed_root = _assert_installed(installed_root)
    directory = Path(output_dir).absolute()
    if directory.is_symlink() or (
        directory.exists() and (not directory.is_dir() or any(directory.iterdir()))
    ):
        raise ValueError("prefix readiness output-dir must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    original_threads = torch.get_num_threads()
    child = None
    try:
        torch.set_num_threads(1)
        config = control_config()
        source = b"Independently authored public prefix-cache CPU control.\n" * 32
        training = GPT2TrainingConfig(2, 1, 1, 4, 0.001, 0.0001, 0, 0.1, 1.0, 53)
        spec = GPT2TrainingSpec(
            config,
            training,
            GPT2TrainingSource(
                "Untrained prefix readiness control",
                "https://github.com/ajaykr0905/ai-journey",
                "independently-authored-v1",
                "CC0-1.0",
                sha256(source).hexdigest(),
            ),
        )
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.manual_seed(53)
            model = build_training_model(config).double().eval()
        (directory / "source.txt").write_bytes(source)
        (directory / "spec.json").write_text(
            json.dumps(spec.to_mapping(), sort_keys=True)
        )
        torch.save(
            {
                "schema_version": 1,
                "world_size": 1,
                "step": 0,
                "model_config": model_config_mapping(config),
                "training_config": asdict(training),
                "source_sha256": sha256(source).hexdigest(),
                "model_sha256": training_model_fingerprint(model),
                "model": model.state_dict(),
                "optimizer": build_gpt2_optimizer(model, training).state_dict(),
                "rank_rngs": [torch.get_rng_state()],
            },
            directory / "checkpoint.pt",
        )
        environment = os.environ.copy()
        environment.update(
            PYTHONPATH=str(installed_root),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        command = [
            sys.executable,
            "-m",
            "ai_journey.ddp_cli",
            "serve",
            "--source",
            str(directory / "source.txt"),
            "--spec",
            str(directory / "spec.json"),
            "--checkpoint",
            str(directory / "checkpoint.pt"),
            "--port",
            "0",
            "--max-seconds",
            "12",
            "--drain-seconds",
            "2",
            "--prefix-cache-bytes",
            str(BUDGET),
            "--prefix-cache-entries",
            "8",
            "--prefix-namespace",
            "readiness",
        ]
        with (directory / "server.stdout.log").open("w") as stdout, (
            directory / "server.stderr.log"
        ).open("w") as stderr:
            child = subprocess.Popen(
                command,
                cwd=directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
        until = time.monotonic() + 15
        endpoint = None
        while time.monotonic() < until:
            lines = (directory / "server.stdout.log").read_text().splitlines()
            if lines:
                announcement = json.loads(lines[0])
                endpoint = announcement["listening"]
                if announcement["model_sha256"] != training_model_fingerprint(model):
                    raise RuntimeError("installed operator loaded a different model")
                break
            if child.poll() is not None:
                raise RuntimeError(
                    "installed prefix serving operator failed before readiness"
                )
            time.sleep(0.02)
        if endpoint is None:
            raise RuntimeError(
                "owned prefix serving operator did not announce readiness"
            )
        rows = []
        for index, planned in enumerate(schedule()):
            request = planned["request"]
            before = _wait_idle(endpoint, attempts=index)
            control = generate_tokens(
                model,
                request["prompt"],
                SamplingConfig(**request["sampling"]),
                use_cache=False,
            )
            observation = measure_http_generation(
                endpoint, request, streaming=index != 0, timeout_seconds=5
            )
            after = _wait_idle(endpoint, attempts=index + 1)
            rows.append(
                {
                    "label": planned["label"],
                    "request": request,
                    "observation": observation,
                    "control_tokens": control["tokens"],
                    "before": before,
                    "after": after,
                }
            )
        cancellation = measure_http_generation(
            endpoint,
            {
                "request_id": "prefix-cancel",
                "prompt": [1, 2, 3, 4],
                "sampling": {"max_new_tokens": 8, "temperature": 0, "seed": 53},
            },
            timeout_seconds=5,
            cancel_after_tokens=1,
        )
        final = _wait_idle(endpoint, attempts=7)
        child.wait(timeout=15)
        identity = prefix_identity(model, namespace="readiness")
        report = seal_prefix_report(
            {
                "schema": "ai-journey-prefix-readiness-v1",
                "scope": SCOPE,
                "model_config": model_config_mapping(config),
                "model_sha256": identity.model_sha256,
                "architecture_sha256": identity.architecture_sha256,
                **{
                    field: sha256((directory / filename).read_bytes()).hexdigest()
                    for field, filename in (
                        ("source_sha256", "source.txt"),
                        ("spec_sha256", "spec.json"),
                        ("checkpoint_sha256", "checkpoint.pt"),
                    )
                },
                "runtime": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "device": "cpu",
                    "dtype": "float64",
                    "checkpoint_step": 0,
                    "trained": False,
                },
                "rows": rows,
                "cancellation": cancellation,
                "final_metrics": final,
                "child_exit_code": child.returncode,
            }
        )
        (directory / "raw-observations.json").write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        verify_prefix_report(report, directory=directory)
        (directory / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        return report
    finally:
        if child is not None and child.poll() is None:
            # Only this invocation's explicitly spawned process is terminated.
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
        torch.set_num_threads(original_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-root", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--verify-report", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.verify_report is not None:
            path = arguments.verify_report
            report = verify_prefix_report(
                parse_evaluation_json(
                    read_evaluation_bytes(
                        path, maximum=1024 * 1024, name="prefix readiness report"
                    ),
                    name="prefix readiness report",
                ),
                directory=arguments.artifact_dir,
            )
        else:
            if arguments.installed_root is None or arguments.artifact_dir is not None:
                raise ValueError("readiness run requires installed-root and output-dir")
            report = run_prefix_readiness(
                installed_root=arguments.installed_root, output_dir=arguments.output_dir
            )
        print(
            json.dumps(
                {
                    "report_sha256": report["report_sha256"],
                    "completed_requests": len(report["rows"]),
                    "completed_output_tokens": sum(
                        len(row["observation"]["tokens"]) for row in report["rows"]
                    ),
                    "child_exit_code": report["child_exit_code"],
                    "scope": report["scope"],
                },
                sort_keys=True,
            )
        )
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
