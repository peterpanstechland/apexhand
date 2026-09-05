#!/usr/bin/env python3
"""Empty-load (air) smoke test for Rysen Apex Hand SDK.

Default: ping + connect + read joint states only (NO motion).
Optional tiny motion requires ``--wiggle --i-know-what-im-doing``.

Usage:
  source env_real.sh
  python scripts/real_sdk_smoke.py --ip 192.168.0.103          # right hand default
  python scripts/real_sdk_smoke.py --ip 192.168.0.102          # left hand
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time


def ping_ok(ip: str, timeout_s: float = 2.0) -> bool:
    try:
        r = subprocess.run(
            ["ping", "-c", "1", "-W", str(max(1, int(timeout_s))), ip],
            capture_output=True,
            text=True,
            timeout=timeout_s + 2,
        )
        return r.returncode == 0
    except Exception:
        return False


def tcp_probe(ip: str, port: int, timeout_s: float = 2.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Apex Hand SDK air smoke test")
    parser.add_argument("--ip", default="192.168.0.103", help="Hand IP (right=103, left=102)")
    parser.add_argument(
        "--wiggle",
        action="store_true",
        help="After readback, do a tiny index-j1 motion (requires confirm flag)",
    )
    parser.add_argument(
        "--i-know-what-im-doing",
        action="store_true",
        help="Required together with --wiggle",
    )
    args = parser.parse_args()

    print("=== network ===")
    print(f"ping {args.ip}: {'OK' if ping_ok(args.ip) else 'FAIL'}")
    for port in (5856, 5857):
        print(f"tcp {args.ip}:{port}: {'OK' if tcp_probe(args.ip, port) else 'FAIL'}")

    from rysen_apexhand_sdk import ConnectionType, ErrorCode, Rysen

    print("=== connect ===")
    sdk = Rysen()
    ret = sdk.connect(args.ip, ConnectionType.CONNECTION_TYPE_ETHERNET)
    if ret != ErrorCode.ERROR_CODE_OK:
        print(f"CONNECT_FAIL code={ret}")
        print(
            "Hint: host NIC must be on 192.168.0.x (hand defaults).\n"
            "This laptop's enp109s0 is currently 192.168.88.102 — see docs/REAL_SDK.zh.md"
        )
        return 1
    print("CONNECT_OK")

    try:
        print("=== read joints ===")
        # API name may be get_joint_states (snake) in wrapper
        getter = getattr(sdk, "get_joint_states", None) or getattr(sdk, "GetJointStates", None)
        if getter is None:
            print("WARN: no get_joint_states API found; connect-only smoke passed")
        else:
            for i in range(5):
                st = getter()
                print(f"[{i}] {st}")
                time.sleep(0.2)

        if args.wiggle:
            if not args.i_know_what_im_doing:
                print("Refusing --wiggle without --i-know-what-im-doing")
                return 2
            print("=== tiny wiggle (air) skipped in smoke v1 — use vendor example.py ===")
            print("  cd third_party/Rysen_SDK/python && python example.py --ip", args.ip)
    finally:
        disc = getattr(sdk, "disconnect", None) or getattr(sdk, "Disconnect", None)
        if disc:
            disc()
            print("DISCONNECTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
