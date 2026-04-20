with open('scripts/update_results_md.py', 'r') as f:
    text = f.read()

text = text.replace('if model_name == "PatchTST_ACCA":\n                model_name = "PatchTST (ACCA)"', 'if "ACCA" in model_name:\n                model_name = "PatchTST (ACCA)"')

with open('scripts/update_results_md.py', 'w') as f:
    f.write(text)
