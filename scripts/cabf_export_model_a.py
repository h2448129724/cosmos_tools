from __future__ import annotations

import argparse

from _cabf_cli import REPO_ROOT  # noqa: F401
from cabf import export_master_to_model_a


def main() -> None:
    parser = argparse.ArgumentParser(description="将 CAB-F 母标签导出为 model_a / LabelMe 点标注格式。")
    parser.add_argument("--image_dir", required=True, help="母图目录")
    parser.add_argument("--annotation_dir", required=True, help="母标签 JSON 目录")
    parser.add_argument("--output_dir", required=True, help="导出目录")
    parser.add_argument("--exclude_empty", action="store_true", help="不导出空标注样本")
    args = parser.parse_args()

    result = export_master_to_model_a(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        include_empty=not args.exclude_empty,
    )
    print(result)


if __name__ == "__main__":
    main()
