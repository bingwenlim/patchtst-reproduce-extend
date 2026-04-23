import json
import subprocess
from collections import defaultdict
from datetime import datetime

models_configs = [
    { "model": "PatchTST", "args": ["--d_model", "16", "--n_heads", "4", "--d_ff", "128", "--dropout", "0.3"], "name": "PatchTST" },
    { "model": "PatchTST", "args": ["--d_model", "16", "--n_heads", "4", "--d_ff", "128", "--dropout", "0.3", "--use_acca"], "name": "PatchTST (ACCA)" },
    { "model": "DLinear", "args": [], "name": "DLinear" },
    { "model": "Autoformer", "args": ["--d_model", "16", "--n_heads", "4", "--d_ff", "128", "--dropout", "0.3", "--seq_len", "96"], "name": "Autoformer" }
]

datasets = ["traffic", "air", "fx"]

results = []

for dataset in datasets:
    for config in models_configs:
        model_name = config.get("name", config["model"])
        print(f"\n--- Running {model_name} on {dataset} ---")
        
        cmd = ["uv", "run", "python", "train.py", "--model", config["model"], "--dataset", dataset] + config["args"]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        test_mse = None
        test_mae = None
        best_epoch = None
        train_time = None
        inference_time = None

        for line in process.stdout:
            print(line, end="")
            if line.startswith("best_epoch:"):
                best_epoch = line.split(":")[1].strip()
            elif line.startswith("test_mse:"):
                test_mse = line.split(":")[1].strip()
            elif line.startswith("test_mae:"):
                test_mae = line.split(":")[1].strip()
            elif line.startswith("total_training_time:"):
                train_time = float(line.split(":")[1].strip())
            elif line.startswith("test_inference_time:"):
                inference_time = float(line.split(":")[1].strip())

        process.wait()

        if process.returncode != 0:
            print("Failed to run", cmd)

        results.append({
            "dataset": dataset,
            "model": model_name,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "best_epoch": best_epoch,
            "time": f"{train_time:.1f}s" if train_time else "N/A",
            "inference_time": f"{inference_time:.3f}s" if inference_time is not None else "N/A",
        })

print("\n\n=== RESULTS ===\n")
for r in results:
    print(r)

grouped = defaultdict(list)
for r in results:
    grouped[r["dataset"]].append(r)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
json_path = "scripts/benchmark_results.json"
md_path = "scripts/benchmark_results.md"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"timestamp": timestamp, "results": results}, f, indent=2)

with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# Benchmark Results\n\nGenerated: {timestamp}\n\n")
    for dataset, rows in grouped.items():
        f.write(f"## {dataset}\n\n")
        f.write("| Model | Test MSE | Test MAE | Best Epoch | Train Time | Inference Time |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                f"| {row['model']} | {row['test_mse']} | {row['test_mae']} | "
                f"{row['best_epoch']} | {row['time']} | {row['inference_time']} |\n"
            )
        f.write("\n")

print(f"\nSaved JSON results to {json_path}")
print(f"Saved Markdown summary to {md_path}")

# Add step to automatically update RESULTS.md
import subprocess
try:
    subprocess.run(["python", "scripts/update_results_md.py"])
except Exception as e:
    print(f"Could not automatically update RESULTS.md: {e}")
