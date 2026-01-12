import os
import sys
import torch
sys.path.append('.')
sys.path.append('megatron-lm')
sys.path.append('megatron-lm/megatron')

def inspect_one(path: str):
    print(f"\n=== {path} ===")
    obj = torch.load(path, weights_only=False, map_location="cpu")
    if isinstance(obj, dict):
        keys = sorted(list(obj.keys()))
        print("top-level keys:", keys)
        for k in ["model", "optimizer", "opt_state", "rng_state", "args", "iteration", "checkpoint_version"]:
            print(f"has {k!r}:", k in obj)
        if "optimizer" in obj and isinstance(obj["optimizer"], dict):
            print("optimizer keys:", sorted(obj["optimizer"].keys()))
    else:
        print("Loaded object type:", type(obj))
    return obj

def main(root):
    pt_files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".pt"):
                pt_files.append(os.path.join(dirpath, fn))
    pt_files.sort()
    if not pt_files:
        print("No .pt files found under:", root)
        return 1

    for f in pt_files:
        inspect_one(f)
    return 0

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "megatron-lm/ckpts/llama-150m-mcore"
    raise SystemExit(main(root))
