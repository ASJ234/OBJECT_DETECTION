import json

with open("train.ipynb") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        
        # Fix #1: evaluate(model -> evaluate(ema.model
        for i, line in enumerate(source):
            if "coco_eval = evaluate(model, val_loader, device)" in line:
                source[i] = line.replace("evaluate(model,", "evaluate(ema.model,")
                print("Patched EMA evaluation in train.ipynb")
                
        # Fix #5: clamp(0,1)
        for i, line in enumerate(source):
            if "if self.normalize:" in line:
                # check if clamp is already there
                if i > 0 and "clamp" not in source[i-1]:
                    # Need to match the indentation of the "if self.normalize:"
                    indent = len(line) - len(line.lstrip())
                    source.insert(i, " " * indent + "image = image.clamp(0, 1)\n")
                    print("Patched clamp(0,1) in train.ipynb")
                    break # Only do it once per cell (or rather, we only need it in the dataset cell)

with open("train.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched.")
