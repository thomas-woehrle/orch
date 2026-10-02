"""Exit-code-bearing errors. See docs/orchestrator.md §8."""


class OrchError(Exception):
    """Base error. ``exit_code`` is what the CLI returns."""

    exit_code = 1


class UsageError(OrchError):
    """Bad invocation or a validation error. Nothing was submitted."""

    exit_code = 1


class PreconditionError(OrchError):
    """Dirty tree, unpushed HEAD, cluster check failed, or ``--resync`` required."""

    exit_code = 2


class RemoteError(OrchError):
    """Staging or submission failed. A partial submission may have happened."""

    exit_code = 3
