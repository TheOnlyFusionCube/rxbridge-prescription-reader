#!/usr/bin/env python3
"""RxBridge LLM endpoint smoke test.

For each configured provider, issue ONE minimal chat request (no retries, no
concurrency) and print one line per provider:

    PROVIDER=<name> MODEL=<model> OK|FAIL latency=<sec> chars=<n>

Featherless with no key prints UNAVAILABLE_NEEDS_REGISTRATION instead of OK/FAIL
(it is not an endpoint failure).

Exit code: 0 if at least one provider returned a non-empty completion, else 1.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import PROVIDERS, probe_provider  # noqa: E402


def main() -> int:
    ok_count = 0
    for provider in PROVIDERS:
        result = probe_provider(provider)
        print(
            f"PROVIDER={result['name']} "
            f"MODEL={result['model']} "
            f"{result['status']} "
            f"latency={result['latency']:.3f} "
            f"chars={result['chars']}"
        )
        if result["error"]:
            # Diagnostics go to stderr so stdout stays one-line-per-provider.
            print(f"  # {result['name']}: {result['error']}", file=sys.stderr)
        if result["status"] == "OK":
            ok_count += 1

    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
