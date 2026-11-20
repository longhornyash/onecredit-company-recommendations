"""Demonstrate a real HTTP request, then clean up the temporary API process."""

import json
import subprocess
import sys
import time

import httpx


def main():
    # Fixed demo port is separate from the documented long-running API port.
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "onecredit.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "18080",
            "--no-access-log",
        ]
    )
    try:
        with httpx.Client(base_url="http://127.0.0.1:18080", timeout=10) as client:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("demo API exited before readiness")
                try:
                    if client.get("/ready").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("demo API did not become ready")
            response = client.get("/companies/SYN-001550/similar?limit=4")
            response.raise_for_status()
            assert len(response.json()["results"]) == 4
            print(json.dumps(response.json(), indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
