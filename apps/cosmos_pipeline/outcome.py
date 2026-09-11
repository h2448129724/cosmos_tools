from __future__ import annotations

from enum import StrEnum


class PipelineOutcome(StrEnum):
    """Terminal meaning of one complete Cosmos pipeline run."""

    BUSINESS_OK = "success"
    BUSINESS_NG = "business_ng"
    TECHNICAL_FAILURE = "failed"
    STOPPED = "stopped"

    @property
    def business_label(self) -> str:
        if self is PipelineOutcome.BUSINESS_OK:
            return "OK"
        if self is PipelineOutcome.BUSINESS_NG:
            return "NG"
        raise ValueError(f"Business label is undefined for {self.value}")


def business_outcome(passed: bool) -> PipelineOutcome:
    return PipelineOutcome.BUSINESS_OK if passed else PipelineOutcome.BUSINESS_NG


def cli_exit_code(outcome: PipelineOutcome, *, fail_on_ng: bool) -> int:
    if outcome is PipelineOutcome.BUSINESS_OK:
        return 0
    if outcome is PipelineOutcome.BUSINESS_NG:
        return 2 if fail_on_ng else 0
    raise ValueError(f"CLI business exit code is undefined for {outcome.value}")


def outcome_from_process_exit(
    exit_code: int,
    *,
    cancelled: bool = False,
    crashed: bool = False,
) -> PipelineOutcome:
    if cancelled:
        return PipelineOutcome.STOPPED
    if crashed:
        return PipelineOutcome.TECHNICAL_FAILURE
    if exit_code == 0:
        return PipelineOutcome.BUSINESS_OK
    if exit_code == 2:
        return PipelineOutcome.BUSINESS_NG
    return PipelineOutcome.TECHNICAL_FAILURE


def should_register_artifact(
    outcome: PipelineOutcome,
    *,
    has_output: bool,
    output_exists: bool,
    already_registered: bool,
) -> bool:
    return (
        outcome in {PipelineOutcome.BUSINESS_OK, PipelineOutcome.BUSINESS_NG}
        and has_output
        and output_exists
        and not already_registered
    )
