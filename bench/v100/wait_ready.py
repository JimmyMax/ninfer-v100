#!/usr/bin/env python3
"""Readiness helper for the v100 A/B scripts.

The server answers 503 with {"code": "model_loading"} until the engine can serve, and it logs
"listening on" at exactly that point. The log line is only reliable for a freshly created
container: `docker start`/`restart` and `docker compose up` on an existing container keep the
history, so grepping for it matches a *previous* launch and the first request races the load.
That is what produced the 503s chased on 2026-10-16 -- the server was behaving correctly.

/health cannot go stale: it returns 503 while loading and 200 once ready, with no authentication
needed (verified: 503 at t<=18s, 200 at t=21s, and a request sent right after it returns 200).
"""
import subprocess
import time


def container_ip(name: str) -> str:
    return subprocess.run(
        ["docker", "inspect", "-f",
         "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
        capture_output=True, text=True).stdout.strip()


def wait_ready_ip(ip: str, port: int = 8000, timeout_s: int = 900) -> None:
    """Block until /health answers 200, so the next request cannot hit 503 model_loading."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        code = subprocess.run(
            ["curl", "-s", "--noproxy", "*", "-m", "10", "-o", "/dev/null", "-w", "%{http_code}",
             f"http://{ip}:{port}/health"], capture_output=True, text=True).stdout.strip()
        if code == "200":
            return
        time.sleep(5)
    raise RuntimeError(f"engine at {ip}:{port} never reported /health 200 within {timeout_s}s")


def wait_ready(name: str, port: int = 8000, timeout_s: int = 900) -> str:
    """Resolve the container IP, wait for readiness, and return the IP."""
    ip = container_ip(name)
    if not ip:
        raise RuntimeError(f"no IP for container {name}")
    wait_ready_ip(ip, port, timeout_s)
    return ip
