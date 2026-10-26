"""Exercise the installed measurement command against an owned tiny CPU service."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import torch

from .ddp_http import make_server
from .ddp_service import GenerationService
from .gpt2_config import GPT2Config
from .gpt2_model import GPT2LanguageModel
from .http_load_report import verify_http_load_report


def run_installed_http_readiness(output, *, command="ai-journey-http-load"):
    """Retain actual socket evidence and verify it through the installed CLI.

    The exclusively owned server uses an explicitly seeded, untrained, tiny
    float32 CPU control. No existing service is contacted, stopped or replaced.
    """

    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("readiness output already exists; choose a new path")
    executable = shutil.which(str(command))
    if executable is None:
        raise ValueError("the installed ai-journey-http-load command is required")
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"
    )
    if environment.get("PYTHONPATH"):
        environment["PYTHONPATH"] = os.pathsep.join(
            str(Path(path).absolute())
            for path in environment["PYTHONPATH"].split(os.pathsep)
        )

    def invoke(arguments):
        completed = subprocess.run(
            [executable, *arguments],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"installed HTTP readiness command failed: {completed.stderr.strip()}"
            )

    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    server = thread = None
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(2026)
            model = GPT2LanguageModel(
                GPT2Config(
                    8,
                    4,
                    8,
                    2,
                    1,
                    embedding_dropout=0.0,
                    attention_dropout=0.0,
                    residual_dropout=0.0,
                )
            )
        service = GenerationService(model)
        server = make_server(service)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()
        with tempfile.TemporaryDirectory(
            prefix="ai-journey-http-readiness-"
        ) as temporary:
            profile_path = Path(temporary) / "profile.json"
            invoke(["profile", "--output", str(profile_path)])
            invoke(
                [
                    "measure",
                    "--endpoint",
                    f"http://127.0.0.1:{server.server_port}",
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output),
                ]
            )
            invoke(["verify", "--report", str(output)])
        report = verify_http_load_report(json.loads(output.read_text()))
        totals = report["measurement"]["totals"]
        if (
            totals["attempted_requests"] != 8
            or totals["outcomes"]["completed"] != 8
            or totals["completed_output_tokens"] != 32
            or service.completed_requests != 9
            or service.generated_tokens != 36
        ):
            raise RuntimeError(
                "installed HTTP readiness did not complete the declared workload"
            )
        if {
            row["observation"]["model_sha256"]
            for row in report["measurement"]["measured"]
        } != {service.model_sha256}:
            raise RuntimeError("HTTP readiness measured a different model identity")
        return {
            "report": str(output),
            "model_sha256": service.model_sha256,
            "measured_requests": 8,
            "measured_output_tokens": 32,
            "server_requests_including_warmup": service.completed_requests,
        }
    finally:
        try:
            if server is not None:
                if thread is not None and thread.ident is not None:
                    server.shutdown()
                    thread.join(timeout=5)
                    if thread.is_alive():
                        raise RuntimeError("HTTP readiness server did not shut down")
                server.server_close()
        finally:
            torch.set_default_dtype(dtype)
            torch.set_num_threads(threads)
