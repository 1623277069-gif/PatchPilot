"""Submit and optionally approve the scripted public example through the running API."""

import argparse
import time

import httpx

from patchpilot.model import DEMO_COMMIT, DEMO_ISSUE, DEMO_REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--mode", choices=["mock", "base", "lora"], default="mock")
    parser.add_argument("--approve", action="store_true", help="Approve this specific public example")
    parser.add_argument("--timeout", type=int, default=450)
    args = parser.parse_args()
    # With Compose only the API token is used on this host.
    from pydantic import SecretStr
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Operator(BaseSettings):
        model_config = SettingsConfigDict(env_prefix="PATCHPILOT_", env_file=".env", extra="ignore")
        api_token: SecretStr

    headers = {"Authorization": f"Bearer {Operator().api_token.get_secret_value()}"}
    with httpx.Client(base_url=args.api, headers=headers, trust_env=False, timeout=10) as client:
        repo = client.post("/repositories", json={"url": DEMO_REPO}).raise_for_status().json()
        job = (
            client.post(
                "/jobs",
                json={
                    "repository_id": repo["id"],
                    "commit": DEMO_COMMIT,
                    "issue": DEMO_ISSUE,
                    "model_mode": args.mode,
                    "profile": "demo",
                },
            )
            .raise_for_status()
            .json()
        )
        path = f"/jobs/{job['id']}"
        print(f"{args.mode} job {job['id']}", flush=True)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            current = client.get(path).raise_for_status().json()
            status = current["status"]
            if status == "awaiting_approval":
                patch = client.get(path + "/patch").raise_for_status()
                print(patch.text, flush=True)
                sha = patch.headers["X-Patch-SHA256"]
                if not args.approve:
                    print(f"Review SHA {sha}. Submit the approval through /docs.")
                    return
                client.post(
                    path + "/approval",
                    json={
                        "approve": True,
                        "patch_sha": sha,
                    },
                ).raise_for_status()
            if status in {"succeeded", "unresolved", "failed", "rejected"}:
                print(f"{status}: {current['result']}; error={current['error']}", flush=True)
                print(client.get(path + "/events").raise_for_status().text)
                if status != "succeeded":
                    raise SystemExit(1)
                return
            time.sleep(1)
        raise SystemExit("Demo timed out while waiting for the API job")


if __name__ == "__main__":
    main()
