"""Shared pytest fixtures (WP6 item 3)."""
import pytest

from tests.utils.iteration_limit_as_converged import iteration_limit_counts_as_converged


@pytest.fixture
def fast_dual_counts_as_converged():
    """Function-scoped use of tests.utils.iteration_limit_as_converged (see that module for the reason)."""
    with iteration_limit_counts_as_converged():
        yield
