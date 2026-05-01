"""Simple smoke test for local auth flow.

Checks:
1) GET /v1/health
2) POST /v1/auth/register (or reuse existing user)
3) POST /v1/auth/login (OAuth form fields)
4) GET /v1/users/me with bearer token
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request


def request_json(
    url: str,
    *,
    method: str = "GET",
    json_body: dict | None = None,
    form_body: dict | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    req_headers = {"accept": "application/json"}
    if headers:
        req_headers.update(headers)

    data: bytes | None = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        req_headers["content-type"] = "application/json"
    elif form_body is not None:
        data = urllib.parse.urlencode(form_body).encode("utf-8")
        req_headers["content-type"] = "application/x-www-form-urlencoded"

    req = urllib.request.Request(url=url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        payload = {}
        if raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
        return exc.code, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backend auth smoke test.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--email", default="test@example.com")
    parser.add_argument("--password", default="test1234")
    parser.add_argument("--display-name", default="Test User")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    status, health = request_json(f"{base}/v1/health")
    if status != 200:
        print(f"Health failed: status={status}, body={health}")
        return 1
    print("Health OK")

    status, register_body = request_json(
        f"{base}/v1/auth/register",
        method="POST",
        json_body={
            "email": args.email,
            "password": args.password,
            "display_name": args.display_name,
        },
    )
    if status == 200:
        print("Register OK")
    elif status == 409:
        print("Register skipped (user already exists)")
    else:
        print(f"Register failed: status={status}, body={register_body}")
        return 1

    status, login_body = request_json(
        f"{base}/v1/auth/login",
        method="POST",
        form_body={"username": args.email, "password": args.password},
    )
    if status != 200 or "access_token" not in login_body:
        print(f"Login failed: status={status}, body={login_body}")
        return 1
    token = login_body["access_token"]
    print("Login OK")

    status, me_body = request_json(
        f"{base}/v1/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    if status != 200:
        print(f"users/me failed: status={status}, body={me_body}")
        return 1
    print(f"users/me OK ({me_body.get('email', 'unknown')})")
    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
