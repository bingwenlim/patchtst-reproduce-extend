import json
import re

def update_results():
    with open("scripts/benchmark_results.json", "r") as f:
        data = json.load(f)
        
    results = data["results"]
    
    grouped = {}
    for r in results:
        grouped.setdefault(r["dataset"], []).append(r)
        
    dataset_name_map = {
        "traffic": "Traffic",
        "air": "Air Quality",
        "fx": "FX"
    }

    with open("RESULTS.md", "r") as f:
        content = f.read()

    for dataset_id, rows in grouped.items():
        if dataset_id not in dataset_name_map:
            continue
            
        dataset_title = dataset_name_map[dataset_id]
        
        table_str = "| Model           | Config           | MSE (Ours) | MAE (Ours) | Best Epoch | Train Time  | Inference Time |\n"
        table_str += "| --------------- | ---------------- | ---------- | ---------- | ---------- | ----------- | -------------- |\n"

        for row in rows:

            model_name = row["model"]
            if "ACCA" in model_name:
                model_name = "PatchTST (ACCA)"

            config_name = "paper config" if model_name != "DLinear" else "default"
            mse = f"{float(row['test_mse']):.3f}" if row['test_mse'] else "N/A"
            mae = f"{float(row['test_mae']):.3f}" if row['test_mae'] else "N/A"
            time_val = f"{row.get('time', 'N/A')}"
            inf_val = f"{row.get('inference_time', 'N/A')}"

            table_str += f"| {model_name:<15} | {config_name:<16} | {mse:<10} | {mae:<10} | {row.get('best_epoch', 'N/A'):<10} | {time_val:<11} | {inf_val:<14} |\n"
        
        pattern = re.compile(
            rf"(?P<head>## {dataset_title} \(pred_len=96\)\n\n### Summary\n\n)"
            rf"(?P<table>\| Model [^\n]+\n\| -[^\n]+\n(?:\| [^\n]+\n)*)",
            re.MULTILINE
        )
        
        content = pattern.sub(rf"\g<head>{table_str}", content)

    with open("RESULTS.md", "w") as f:
        f.write(content)

    print("Updated RESULTS.md successfully.")

if __name__ == "__main__":
    update_results()
