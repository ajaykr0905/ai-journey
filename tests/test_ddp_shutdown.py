from argparse import Namespace
import http.client
import json
from pathlib import Path
import threading

from ai_journey import ddp_operations
from test_ddp_metrics_integration import service, server_for


def test_http_liveness_and_readiness_have_distinct_drain_behavior():
    instance = service()
    with server_for(instance) as server:
        client = http.client.HTTPConnection(*server.server_address, timeout=2)
        for route in ("/health", "/ready"):
            client.request("GET", route)
            response = client.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["ready"]
        assert instance.drain(timeout_seconds=0)
        for route, status in (("/health", 200), ("/ready", 503)):
            client.request("GET", route)
            response = client.getresponse()
            assert response.status == status
            assert not json.loads(response.read())["ready"]
        client.close()


def test_serving_operation_drains_before_shutdown_and_closes_socket(
    monkeypatch, tmp_path
):
    instance = service()
    seen = {}
    monkeypatch.setattr(ddp_operations, "preflight", lambda *args, **kwargs: object())
    monkeypatch.setattr(ddp_operations, "load_serving_model", lambda *args: object())
    monkeypatch.setattr(ddp_operations, "GenerationService", lambda model: instance)
    original = ddp_operations.make_server

    def make(*args, **kwargs):
        seen["max_connections"] = kwargs["max_connections"]
        server = original(*args, **kwargs)
        seen["server"] = server
        shutdown = server.shutdown

        def checked_shutdown():
            assert not instance.health()["ready"]
            seen["drained_before_stop"] = True
            shutdown()

        server.shutdown = checked_shutdown
        return server

    monkeypatch.setattr(ddp_operations, "make_server", make)
    source = tmp_path / "source.txt"
    source.write_bytes(b"public fixture")
    args = Namespace(
        command="serve",
        source=source,
        spec=Path("unused"),
        checkpoint=Path("unused"),
        port=0,
        max_seconds=1,
        max_connections=3,
        drain_seconds=1,
    )
    before = {thread.ident for thread in threading.enumerate()}
    result = ddp_operations.run_operation(args)
    assert not result["ready"]
    assert seen["drained_before_stop"]
    assert seen["max_connections"] == 3
    assert seen["server"].socket.fileno() == -1
    assert not {thread.ident for thread in threading.enumerate()} - before
