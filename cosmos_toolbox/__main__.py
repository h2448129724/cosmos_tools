from __future__ import annotations

import argparse

from .app import main as run_app
from .preflight import main as run_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmos personal image and training toolbox")
    parser.add_argument(
        "--workspace",
        choices=("hub", "image", "labeling", "training"),
        default="hub",
        help="Workspace to open.",
    )
    parser.add_argument("--preflight", action="store_true", help="Print environment diagnostics and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight:
        return run_preflight()
    return run_app(args.workspace)


if __name__ == "__main__":
    raise SystemExit(main())
