import subprocess
import sys
import time


PIPELINES = [
    ("FCOS", "train_fcos.py", ["--batch-size", "8"]),
    ("RetinaNet", "train_retinanet.py", ["--batch-size", "8"]),
    ("EfficientDet", "train_efficientdet.py", ["--batch-size", "4"]),
    ("DETR", "train_detr.py", ["--batch-size", "4"]),
]


def _free_gpu_memory():
    """Release GPU memory held by previous subprocess."""
    subprocess.run(
        [sys.executable, "-c", "import torch; torch.cuda.empty_cache()"],
        capture_output=True,
    )


def _check_deps():
    """Check DETR dependency."""
    result = subprocess.run(
        [sys.executable, "-c", "import transformers"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("[WARN] DETR requires 'transformers' — install with: pip install transformers")


def main():
    python = sys.executable
    total = len(PIPELINES)
    failed = []

    _check_deps()

    for i, (name, script, extra_args) in enumerate(PIPELINES, 1):
        _free_gpu_memory()
        time.sleep(2)

        print(f"\n{'='*60}")
        print(f"  [{i}/{total}] Starting {name} training...")
        print(f"{'='*60}\n")

        start = time.time()
        result = subprocess.run(
            [python, script] + extra_args,
            cwd=".",
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"\n[ERROR] {name} failed (exit code {result.returncode})")
            failed.append(name)
        else:
            print(f"\n[DONE] {name} completed in {elapsed/60:.1f} minutes.")

    print(f"\n{'='*60}")
    if failed:
        print(f"  Completed: {total - len(failed)}/{total}")
        print(f"  Failed:    {', '.join(failed)}")
    else:
        print("  All 4 pipelines complete!")
    print(f"{'='*60}")
    print("\nResults saved to results/{fcos,retinanet,efficientdet,detr}/")


if __name__ == "__main__":
    main()
