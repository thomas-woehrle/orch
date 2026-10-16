"""Every row of the reconcile table in docs/orchestrator.md §6.2."""

from orch.cache import CacheEntry
from orch.reconcile import AttemptView, Decision, attempt_views, decide, upstream_allows


def test_decision_table():
    assert decide([]).action == "submit" and decide([]).attempt == 1

    pending = decide([AttemptView(1, None, "queued")])
    assert (pending.action, pending.reason) == ("skip", "pending")
    assert decide([AttemptView(1, None, "running")]).reason == "pending"
    assert decide([AttemptView(1, None, "unknown")]).reason == "pending"

    burned = decide([AttemptView(1, None, "gone")])
    assert (burned.action, burned.attempt, burned.reason) == ("submit", 2, "burned")

    running = decide([AttemptView(2, "running", "queued")])
    assert (running.action, running.reason) == ("skip", "running")
    assert decide([AttemptView(2, "running", None)]).reason == "running"
    crashed = decide([AttemptView(2, "running", "gone")])
    assert (crashed.action, crashed.attempt, crashed.reason) == ("submit", 3, "crashed")

    finished = decide([AttemptView(1, "finished", "gone")])
    assert (finished.action, finished.reason) == ("skip", "finished")
    forced = decide([AttemptView(1, "finished", "gone")], force=True)
    assert (forced.action, forced.attempt, forced.reason) == ("submit", 2, "force")

    for view in (
        AttemptView(1, None, "queued"),
        AttemptView(1, None, "running"),
        AttemptView(2, "running", "queued"),
        AttemptView(2, "running", None),
        AttemptView(1, None, "error"),
        AttemptView(1, "running", "error"),
        AttemptView(1, "killed-by-user", "gone"),
    ):
        overridden = decide([view], force=True)
        assert (overridden.action, overridden.attempt, overridden.reason) == (
            "submit",
            view.attempt + 1,
            "force",
        )

    # States that already submit the next attempt keep their own reason.
    still_failed = decide([AttemptView(1, "failed", "gone")], force=True)
    assert (still_failed.action, still_failed.attempt, still_failed.reason) == ("submit", 2, "failed")
    assert decide([], force=True).reason == "new"

    for state in ("failed", "crashed", "killed"):
        nxt = decide([AttemptView(1, state, "gone")])
        assert (nxt.action, nxt.attempt, nxt.reason) == ("submit", 2, state)

    error = decide([AttemptView(1, None, "error")])
    assert (error.action, error.reason) == ("skip", "error")


def test_highest_attempt_wins_over_older_finished():
    decision = decide(
        [AttemptView(1, "finished", "gone"), AttemptView(2, "failed", "gone")]
    )
    assert (decision.action, decision.attempt) == ("submit", 3)


def test_attempt_views_dry_run_treats_cache_as_unknown():
    cache = [
        CacheEntry("eddie", "10", "abc", 1, "train", "name", "2026-01-01T00:00:00Z")
    ]
    views = attempt_views({}, cache, lambda _name: None)
    assert views == [AttemptView(1, None, "unknown")]
    assert decide(views).reason == "pending"

    views = attempt_views({}, cache, lambda _name: {})
    assert views[0].scheduler_state == "gone"
    assert decide(views).reason == "burned"


def test_upstream_rule():
    assert upstream_allows(decide([]))
    assert upstream_allows(decide([AttemptView(1, "finished", None)]))
    assert upstream_allows(decide([AttemptView(1, "running", "running")]))
    assert not upstream_allows(Decision("skip", 1, "failed"))
    assert not upstream_allows(decide([AttemptView(1, None, "error")]))
