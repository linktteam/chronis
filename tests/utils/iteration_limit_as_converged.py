"""WP6 item 3 (P0-07): a fit that is not CONVERGED is withheld. Older tests of the NUMERICAL-health hold-back (the -1e6 Cholesky fallback) use
deliberately short dual fits (2 to 4 iterations) as their 'healthy' control. Inside this context a dual fit that stops at its iteration limit is
treated as converged, so the rule under test is the only one that can withhold. The P0-07 rule itself is tested in
tests/test_wp6_item3_baseline_and_withholding.py.

WP7-R item 5 (found by the full-suite run): the same short fits can also end with a collapsed regime (R1) or a degenerate duration law (R2), which are
new withholding rules of the dual model. Those rules are tested in tests/test_wp7r_item5_dual_honesty.py; here, for the same reason as above, they are
switched off (the report is still computed, only its verdict flag is cleared), so that the rule under test is the only one that can withhold."""
from contextlib import contextmanager
from unittest.mock import patch

from backbone.hssm import fitting as _fitting
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

    real_collapse, real_degeneracy = _fitting.dual_collapse_report, _fitting.dual_duration_degeneracy

    def collapse(p_t_posterior):
        return {**real_collapse(p_t_posterior), "collapsed": False}

    def degeneracy(model):
        return {**real_degeneracy(model), "degenerate": False, "degenerate_regimes": []}

    with patch.object(DualTimescaleHSSM, "fit", fit), patch.object(_fitting, "dual_collapse_report", collapse), patch.object(_fitting, "dual_duration_degeneracy", degeneracy):
        yield
