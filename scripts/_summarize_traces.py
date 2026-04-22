import glob
import json

for p in sorted(glob.glob("scripts/traces/*_trace.json")):
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    s = d["summary"]
    print(
        f"{d['run_name']}: mse={s['test_mse']:.4f} "
        f"mae={s['test_mae']:.4f} "
        f"best_ep={s['best_epoch']} "
        f"alpha_eff={s['final_alpha_effective']} "
        f"epochs={len(d['per_epoch'])} "
        f"time={s['total_training_time']:.0f}s"
    )
