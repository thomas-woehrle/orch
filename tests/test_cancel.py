"""Cancel reports a no-op instead of exiting 0."""

import pytest

from orch.cache import Cache, CacheEntry
from orch.errors import UsageError
from orch.ops import cancel_summaries


def _entry(attempt: int, scheduler_id: str) -> CacheEntry:
    return CacheEntry(
        cluster="eddie",
        scheduler_id=scheduler_id,
        fingerprint="ce2e5b65f769",
        attempt=attempt,
        kind="eval",
        name="prior_tasks",
        submitted_at="2026-10-05T11:44:46Z",
    )


def test_fingerprint_is_reported_when_nothing_is_cancelled():
    cache = Cache(None, {"ce2e5b65f769-a1": _entry(1, "59290675")})
    with pytest.raises(UsageError) as caught:
        cancel_summaries(
            ["ce2e5b65f769"],
            cache,
            deleted={},
            tagged=set(),
            wandb_missing={"ce2e5b65f769"},
        )
    message = str(caught.value)
    assert "fingerprint, not a run id" in message
    assert "ce2e5b65f769-a1 (scheduler 59290675)" in message
    assert "no W&B run" in message


def test_unknown_id_is_reported_when_nothing_is_cancelled():
    with pytest.raises(UsageError, match="no scheduler job in the local cache"):
        cancel_summaries(
            ["missing-a1"],
            Cache(None, {}),
            deleted={},
            tagged=set(),
            wandb_missing={"missing-a1"},
        )


def test_successful_cancel_names_qdel_and_tag():
    cache = Cache(None, {"ce2e5b65f769-a1": _entry(1, "59290675")})
    lines = cancel_summaries(
        ["ce2e5b65f769-a1"],
        cache,
        deleted={"ce2e5b65f769-a1": "59290675"},
        tagged={"ce2e5b65f769-a1"},
        wandb_missing=set(),
    )
    assert lines == ["ce2e5b65f769-a1: qdel 59290675; tagged cancelled"]
