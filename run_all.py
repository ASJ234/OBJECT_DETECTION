import subprocess
import sys
import time


PIPELINES = [
    ("FCOS", "train_fcos.py"),
    ("RetinaNet", "train_retinanet.py"),
    ("EfficientDet", "train_efficientdet.py"),
    ("DETR", "train_detr.py"),
]


def main():
    python = sys.executable
    total = len(PIPELINES)
    failed = []

    for i, (name, script) in enumerate(PIPELINES, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{total}] Starting {name} training...")
        print(f"{'='*60}\n")

        start = time.time()
        result = subprocess.run(
            [python, script],
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
