"""Pure decisions for one complete-inspection case.

This module receives facts produced by the runner shell.  It intentionally
does not know about images, model/config globals, files, clocks, or output
writers; those concerns stay in ``runner.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class CheckFact:
    method: str
    passed: bool
    message: str = ""


@dataclass(frozen=True, slots=True)
class FailureFact:
    path: str
    error: str

    @property
    def message(self) -> str:
        return f"{self.path} 执行失败：{self.error}"


@dataclass(frozen=True, slots=True)
class SlotDecision:
    slot: str
    inputs: tuple[str, ...]
    passed: bool
    messages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(str(item) for item in self.inputs))
        object.__setattr__(self, "messages", tuple(str(item) for item in self.messages if str(item)))


@dataclass(frozen=True, slots=True)
class CaseDecision:
    case_id: str
    slots: tuple[SlotDecision, ...]
    passed: bool
    error_messages: tuple[str, ...] = ()
    error_file_map: tuple[tuple[str, str], ...] = ()
    product_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(self.slots))
        object.__setattr__(self, "error_messages", tuple(self.error_messages))
        object.__setattr__(self, "error_file_map", tuple(self.error_file_map))


def collect_failed_results(node: Any, path: str = "result") -> tuple[FailureFact, ...]:
    """Collect nested ``failed`` result facts in deterministic traversal order."""

    if isinstance(node, Mapping):
        failures: list[FailureFact] = []
        if node.get("failed") is True:
            failures.append(FailureFact(path, str(node.get("error") or "未知错误")))
        for key, value in node.items():
            if key not in {"failed", "error"}:
                failures.extend(collect_failed_results(value, f"{path}.{key}"))
        return tuple(failures)
    if isinstance(node, (list, tuple)):
        failures: list[FailureFact] = []
        for index, value in enumerate(node):
            failures.extend(collect_failed_results(value, f"{path}[{index}]"))
        return tuple(failures)
    return ()


def decide_slot(
    slot: str,
    inputs: Iterable[str] = (),
    *,
    checks: Iterable[CheckFact] = (),
    failures: Iterable[FailureFact] = (),
    calibration_ok: bool = True,
    calibration_message: str = "图像校正失败",
) -> SlotDecision:
    """Combine calibration, checker, and nested-result facts for one slot."""

    check_facts = tuple(checks)
    failure_facts = tuple(failures)
    messages: list[str] = []
    if not calibration_ok:
        messages.append(calibration_message)
    messages.extend(fact.message for fact in check_facts if not fact.passed and fact.message)
    messages.extend(fact.message for fact in failure_facts)
    return SlotDecision(
        slot=str(slot),
        inputs=tuple(str(item) for item in inputs),
        passed=bool(calibration_ok) and all(fact.passed for fact in check_facts) and not failure_facts,
        messages=tuple(messages),
    )


def select_product_id(
    current: str | None,
    extracted: object,
    update_required: bool,
) -> str | None:
    """Apply the runner's ordered product-id replacement rule."""

    if extracted:
        value = str(extracted)
        if current is None or str(current).startswith("unknown_") or update_required:
            return value
    return current


def aggregate_case(
    case_id: str,
    slots: Iterable[SlotDecision],
    *,
    product_id: str | None = None,
) -> CaseDecision:
    """Aggregate ordered slots, preserving legacy last-input error mapping."""

    ordered_slots = tuple(slots)
    messages: list[str] = []
    file_by_message: dict[str, str] = {}
    for slot in ordered_slots:
        last_input = slot.inputs[-1] if slot.inputs else ""
        for message in slot.messages:
            if message not in file_by_message:
                messages.append(message)
                file_by_message[message] = last_input
    return CaseDecision(
        case_id=str(case_id),
        slots=ordered_slots,
        passed=all(slot.passed for slot in ordered_slots),
        error_messages=tuple(messages),
        error_file_map=tuple(file_by_message.items()),
        product_id=product_id,
    )


__all__ = [
    "CaseDecision",
    "CheckFact",
    "FailureFact",
    "SlotDecision",
    "aggregate_case",
    "collect_failed_results",
    "decide_slot",
    "select_product_id",
]
