"""WP6 item 3 (P0-07): a fit that is not CONVERGED is withheld. Older tests of the NUMERICAL-health hold-back (the -1e6 Cholesky fallback) use
deliberately short dual fits (2 to 4 iterations) as their 'healthy' control. Inside this context a dual fit that stops at its iteration limit is
treated as converged, so the rule under test is the only one that can withhold. The P0-07 rule itself is tested in
tests/test_wp6_item3_baseline_and_withholding.py."""
from contextlib import contextmanager
from unittest.mock import patch

from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.model import FitState


@contextmanager
def iteration_limit_counts_as_converged():
    real_fit = DualTimescaleHSSM.fit

    def fit(self, *args, **kwargs):
        out = real_fit(self, *args, **kwargs)
        if self.fit_state_ == FitState.MAX_ITER_REACHED:
            self.fit_state_ = FitState.CONVERGED
        return out

    with patch.object(DualTimescaleHSSM, "fit", fit):
        yield
