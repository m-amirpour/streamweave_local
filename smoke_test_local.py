"""Fast, download-free software regression tests; not scientific validation."""

import importlib.util
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).with_name("streamweave_local.py")
spec = importlib.util.spec_from_file_location("streamweave_local", SCRIPT)
sw = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sw)

rng = np.random.default_rng(2026)
X = rng.normal(size=(800, 8)).astype(np.float32)
y = (X[:, 0] + 0.35 * X[:, 1] > 0).astype(np.int64)
split = sw.split_and_preprocess(X, y, seed=2026)

# Verify calibration partitions are distinct, positive-sized, and appropriately sized.
assert len(split.X_cal_fit) > 0 and len(split.X_cal) > 0
assert len(split.X_cal_fit) + len(split.X_cal) == round(len(y) * sw.CAL_FRAC)
assert len(split.X_cal_fit) == len(split.y_cal_fit)
assert len(split.X_cal) == len(split.y_cal)

# Test exact identity completeness gate, including failure and duplicate rejection.
expected = [("D", 0, "A"), ("D", 1, "A")]
sw.require_complete_phase(
    [
        {"dataset": "D", "seed": 0, "transfer": "A", "success": True},
        {"dataset": "D", "seed": 1, "transfer": "A", "success": True},
    ],
    "test",
    expected,
    ["dataset", "seed", "transfer"],
)
for bad_rows in [
    [{"dataset": "D", "seed": 0, "transfer": "A", "success": True}],
    [
        {"dataset": "D", "seed": 0, "transfer": "A", "success": True},
        {"dataset": "D", "seed": 0, "transfer": "A", "success": True},
    ],
    [
        {"dataset": "D", "seed": 0, "transfer": "A", "success": True},
        {
            "dataset": "D",
            "seed": 1,
            "transfer": "A",
            "success": False,
            "error": "synthetic failure",
        },
    ],
]:
    try:
        sw.require_complete_phase(
            bad_rows, "test", expected, ["dataset", "seed", "transfer"]
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "Completeness gate accepted an incomplete/duplicated phase"
        )

# Verify append_csv preserves different ablation identities across appends.
with tempfile.TemporaryDirectory() as td:
    out = str(Path(td) / "ablations.csv")
    base = {
        "dataset": "D",
        "seed": 0,
        "bpu": "MLP",
        "spu": "MLP",
        "transfer": "StandardKD",
        "cu": "SimpleAvg",
        "drift": False,
        "test_f1": 0.7,
    }
    sw.append_csv(out, [{**base, "ablation": "full"}, {**base, "ablation": "spu_only"}])
    sw.append_csv(out, [{**base, "ablation": "no_kd", "test_f1": 0.6}])
    got = pd.read_csv(out)
    assert set(got["ablation"]) == {"full", "spu_only", "no_kd"}, got

# Keep runtime small; test each ablation and CU through predict-then-update.
sw.EPOCHS_BPU = 1
sw.EPOCHS_KD = 1
sw.BPU_REFRESH_EPOCHS = 1
sw.STREAM_BS = 64
sw.BATCH_INTERVAL = 64
for ablation in ["full", "no_kd", "no_online_spu", "no_refresh", "spu_only"]:
    row = sw.run_one(
        "SYNTHETIC_TEST",
        split,
        "MLP",
        "MLP",
        "StandardKD",
        "SimpleAvg",
        seed=3,
        drift=False,
        ablation=ablation,
    )
    assert row["success"]
    for metric in ["test_f1", "test_acc", "preq_f1", "preq_acc"]:
        assert np.isfinite(row[metric]), (ablation, metric, row.get(metric))

for cu in [
    "SimpleAvg",
    "ConfWeight",
    "Bayesian",
    "Stacking",
    "AttnGate",
    "TempScale",
    "HDT",
]:
    row = sw.run_one(
        "SYNTHETIC_TEST",
        split,
        "MLP",
        "MLP",
        "StandardKD",
        cu,
        seed=4,
        drift=False,
        ablation="full",
    )
    assert row["success"] and row["bpu_refresh_count"] > 0, (cu, row)
    assert np.isfinite(row["test_f1"]), (cu, row)

print(
    "StreamWeave v15 software regression tests passed. This is not scientific validation."
)
