"""WP9 A4: the additive `feature_scale_change` option of data/synthetic_identity.py. Default streams are bit-identical to WP8's (hashes of X_full computed with the WP8 file at f3b71a5, RUN)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

from data.synthetic_identity import generate_identity_stream

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration_report" / "wp10b" / "scripts"))
import f4_lib as F4  # noqa: E402

REF = Path(__file__).resolve().parent / "data" / "wp8_identity_streams.npz"

HASHES = {
    (9996, "default"): "52541cb96246c798", (9996, "split"): "2bdbb5fe9b8ca5fc", (9996, "merge"): "393f717a500a76eb", (9996, "drift"): "41355ac8122f9a17",
    (9997, "default"): "5a7818bd2e550967", (9997, "split"): "14c1e75a75ed9bca", (9997, "merge"): "7dce8054e54ff1bb", (9997, "drift"): "a5d4b4dfe96d956e",
}
KW = {"default": {}, "split": dict(k_change="split"), "merge": dict(k_change="merge"), "drift": dict(drift_sd=1.0)}


def _sha16(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


@pytest.mark.parametrize("seed,kind", sorted(HASHES))
def test_default_streams_are_bit_identical_to_wp8(seed, kind, record_property):
    """WP10B F4 (criteria section 13): bit-identical to WP8, EXCEPT on a machine whose OpenBLAS kernel rounds `np.linalg.norm` differently from the machine that produced the WP8 hashes (an AMD EPYC 7763 does: the
    drift stream of seed 9996 has 13 of 1050 values one unit in the last place away). On such a machine only the kinds that use a norm-made vector (drift, split) may deviate, only if this machine's BLAS norm of the probe
    vector really differs from the WP8 one, and only to within a relative 1e-15 of the stored WP8 array. A machine with the WP8 kernel must reproduce the hashes exactly."""
    x = generate_identity_stream(seed, **KW[kind]).X_full
    if _sha16(x) == HASHES[(seed, kind)]:
        record_property("bit_identical_to_wp8", True)
        return
    record_property("bit_identical_to_wp8", False)
    assert kind in F4.NORM_DEPENDENT_KINDS, f"the {kind} stream of seed {seed} must be bit-identical to WP8 on every machine"
    assert not F4.same_blas_norm_as_wp8(), "this machine's BLAS norm is the WP8 one, so the stream must be bit-identical to WP8"
    ref = np.load(REF)[f"{seed}-{kind}"]
    assert _sha16(ref) == HASHES[(seed, kind)], "the stored reference is not the WP8 stream"
    assert F4.within_last_bit(x, ref), "the stream differs from WP8 by more than the last bit"


def test_the_option_none_or_empty_changes_nothing():
    a = generate_identity_stream(9996)
    for opt in (None, {}):
        b = generate_identity_stream(9996, feature_scale_change=opt)
        assert np.array_equal(a.X_full, b.X_full) and np.array_equal(a.X_obs, b.X_obs, equal_nan=True)


def test_only_the_listed_features_from_the_start_day_change_and_their_noise_is_scaled():
    base = generate_identity_stream(9996)
    ch = generate_identity_stream(9996, feature_scale_change=dict(features=(5, 6), factor=0.15, start=60))
    assert np.array_equal(base.labels, ch.labels) and np.array_equal(np.isnan(base.X_obs), np.isnan(ch.X_obs))            # nothing else drawn
    assert np.array_equal(base.X_full[:60], ch.X_full[:60])                                                              # before the start day: identical
    assert np.array_equal(base.X_full[:, :5], ch.X_full[:, :5])                                                          # other features: identical everywhere
    means = base.truth["means_t"]
    for f in (5, 6):
        assert np.allclose(ch.X_full[60:, f] - means[60:, f], 0.15 * (base.X_full[60:, f] - means[60:, f]))
    assert np.array_equal(base.truth["means_t"], ch.truth["means_t"]) and ch.truth["feature_scale_change"]["factor"] == 0.15 and base.truth["feature_scale_change"] is None
