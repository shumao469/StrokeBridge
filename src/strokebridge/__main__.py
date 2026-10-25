"""Safe CLI: no fit, predict, download, upload or training subcommands."""
import argparse
import json
from pathlib import Path
from . import brain_heart_contract as contract
from .evidence import summary, verify_sources


def main():
    parser = argparse.ArgumentParser(description="StrokeBridge research closeout evidence tools")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summary", help="print audited aggregate status")
    sub.add_parser("synthetic-contract", help="run original 30 synthetic data-contract tests")
    sub.add_parser("example", help="print a fictional data-schema example, no risk estimate")
    verify = sub.add_parser("verify-sources", help="hash originals without executing them")
    verify.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summary":
        result = summary()
    elif args.command == "synthetic-contract":
        result = contract.mock_tests()
    elif args.command == "example":
        result = contract.sample_row()
    else:
        result = verify_sources(args.workspace)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if result.get("state") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

