"""Full-dataset end-to-end: normalize -> build training matrix -> train.

Emits per-stage timing (flushed) so progress is visible in the background log.
Run after a reparse to validate the corrected betsp.parquet trains cleanly.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils import normalizer
from features import builder
from models import train


def _t(t0):
    return f"{(time.monotonic() - t0) / 60:.1f} min"


t0 = time.monotonic()
print("=== full build start ===", flush=True)

uni = normalizer.normalize(write=True)
print(f"[{_t(t0)}] UNIFIED: {len(uni)} rows, position non-null: "
      f"{int(uni['position'].notna().sum())} ({uni['position'].notna().mean():.1%})",
      flush=True)

t1 = time.monotonic()
mat = builder.build_training_matrix(unified=uni, write=True)
print(f"[{_t(t0)}] MATRIX: {len(mat)} rows, {mat.shape[1]} cols "
      f"(build took {_t(t1)})", flush=True)
from models.targets import add_targets
labelled = add_targets(mat, 3)
for lbl in ("won", "placed_2", "showed"):
    if lbl in labelled.columns:
        print(f"    label {lbl}: {dict(labelled[lbl].value_counts())}", flush=True)

t2 = time.monotonic()
try:
    train.train(df=mat, no_tune=True)
    print(f"[{_t(t0)}] TRAIN done (took {_t(t2)})", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[{_t(t0)}] TRAIN skipped/failed: {exc!r}", flush=True)

print(f"=== full build done in {_t(t0)} ===", flush=True)
print("FULL_BUILD_COMPLETE", flush=True)
