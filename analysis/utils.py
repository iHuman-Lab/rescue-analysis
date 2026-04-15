from contextlib import contextmanager


class _Check:
    """Returned by skip_run; calling it produces the inner context manager."""

    def __init__(self, should_run: bool, label: str):
        self._should_run = should_run
        self._label = label

    def __call__(self):
        return self._inner()

    @contextmanager
    def _inner(self):
        if self._should_run:
            print(f"[skip_run] running: {self._label}")
            yield True
        else:
            print(f"[skip_run] skipping: {self._label}")
            yield False


@contextmanager
def skip_run(mode: str, label: str):
    """Context manager that conditionally executes a block.

    Usage::

        with skip_run("run", "my_step") as check, check():
            do_work()

    Parameters
    ----------
    mode:
        ``"run"``  — execute the inner block.
        ``"skip"`` — skip the inner block (yields ``False``).
    label:
        Human-readable name printed to stdout.
    """
    should_run = mode.strip().lower() == "run"
    yield _Check(should_run, label)
