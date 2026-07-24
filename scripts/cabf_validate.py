from __future__ import annotations

import argparse
import json
from pathlib import Path

from _cabf_cli import REPO_ROOT  # noqa: F401
from cabf import summarize_validation, summarize_validation_findings, validate_master_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 CAB-F 母标签数据集。")
    parser.add_argument("--image_dir", required=True, help="母图目录")
    parser.add_argument("--annotation_dir", required=True, help="母标签 JSON 目录")
    parser.add_argument("--report_json", default="", help="可选，保存完整报告 JSON")
    parser.add_argument("--details", action="store_true", help="输出样本级详细问题")
    args = parser.parse_args()

    report = validate_master_dataset(args.image_dir, args.annotation_dir)
    print(summarize_validation(report))

    findings = summarize_validation_findings(report, include_details=args.details).strip()
    if findings:
        print()
        print(findings)

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"[REPORT] {report_path}")


if __name__ == "__main__":
    main()
