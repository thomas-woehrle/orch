"""Reconcile desired jobs with W&B and the scheduler. See docs/orchestrator.md §6.2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from orch.cache import CacheEntry

# ``None`` means that cluster was not queried (a dry run). Absence of an id in
# the map means the scheduler has forgotten the job.
SchedulerStates = Callable[[str], dict[str, str] | None]


@dataclass(frozen=True)
class AttemptView:
    attempt: int
    wb_state: str | None
    scheduler_state: str | None


@dataclass(frozen=True)
class Decision:
    action: str  # submit | skip
    attempt: int
    reason: str


def decide(attempts: list[AttemptView], *, force: bool = False) -> Decision:
    """Apply the §6.2 table to the highest attempt.

    ``force`` turns every skip of an existing attempt into a submit of the
    next one. States that already submit the next attempt keep their reason.
    """
    decision = _decide(attempts)
    if force and decision.action == "skip":
        return Decision("submit", decision.attempt + 1, "force")
    return decision


def _decide(attempts: list[AttemptView]) -> Decision:
    if not attempts:
        return Decision("submit", 1, "new")
    latest = max(attempts, key=lambda item: item.attempt)
    number = latest.attempt
    wb = latest.wb_state
    sched = latest.scheduler_state
    # Eqw is visible and not auto-resubmitted while the attempt is still open.
    if sched == "error" and wb in (None, "running"):
        return Decision("skip", number, "error")
    if wb is None:
        if sched in ("queued", "running", "unknown", None):
            return Decision("skip", number, "pending")
        return Decision("submit", number + 1, "burned")
    if wb == "running":
        if sched == "gone":
            return Decision("submit", number + 1, "crashed")
        return Decision("skip", number, "running")
    if wb == "finished":
        return Decision("skip", number, "finished")
    if wb in ("failed", "crashed", "killed"):
        return Decision("submit", number + 1, wb)
    return Decision("skip", number, f"wandb:{wb}")


def attempt_views(
    wb_attempts: dict[int, str],
    cache_entries: list[CacheEntry],
    states: SchedulerStates,
) -> list[AttemptView]:
    """Combine a W&B probe with cache entries.

    ``states(cluster) is None`` means that cluster was not queried (dry-run):
    its cached jobs are ``unknown``, which the table treats as still pending.
    Each cache entry is looked up on the cluster it was submitted to.
    """
    sched_by_attempt: dict[int, str] = {}
    for entry in cache_entries:
        mapping = states(entry.cluster)
        if mapping is None:
            sched_by_attempt[entry.attempt] = "unknown"
        else:
            sched_by_attempt[entry.attempt] = mapping.get(entry.scheduler_id, "gone")
    numbers = set(wb_attempts) | set(sched_by_attempt)
    return [
        AttemptView(number, wb_attempts.get(number), sched_by_attempt.get(number))
        for number in sorted(numbers)
    ]


def upstream_allows(decision: Decision) -> bool:
    """An eval may proceed when its train run finished, is in flight, or is being submitted."""
    if decision.action == "submit":
        return True
    return decision.reason in {"finished", "running", "pending"}
