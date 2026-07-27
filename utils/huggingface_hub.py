"""Hugging Face Hub integration for pushing trained models and artifacts."""

import os
import json
from pathlib import Path

from huggingface_hub import HfApi


MODEL_CARDS = {
    "fcos": {
        "arch": "FCOS (Fully Convolutional One-Stage Object Detection)",
        "backbone": "ResNet-50 + FPN",
        "framework": "torchvision",
    },
    "retinanet": {
        "arch": "RetinaNet with FPN",
        "backbone": "ResNet-50 + FPN v2",
        "framework": "torchvision",
    },
    "efficientdet": {
        "arch": "EfficientDet-D2",
        "backbone": "EfficientNet-D2 + BiFPN",
        "framework": "effdet / timm",
    },
    "detr": {
        "arch": "DETR (DEtection TRansformer)",
        "backbone": "ResNet-50 + Transformer Encoder-Decoder",
        "framework": "HuggingFace Transformers",
    },
}


def _resolve_token(token=None):
    if token:
        return token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token
    return None


def _generate_model_card(model_type, cfg, best_map, best_epoch, metrics):
    info = MODEL_CARDS.get(model_type, {
        "arch": model_type.upper(),
        "backbone": "Unknown",
        "framework": "PyTorch",
    })

    model_name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]
    training = cfg.get("training", {})
    aug = cfg.get("augmentation", {})

    lines = [
        f"# {model_name} — TB Lesion Detection",
        "",
        "## Model Description",
        "",
        f"This model is a **{info['arch']}** for detecting Tuberculosis (TB) lesions "
        "on chest X-ray images from the TBX11K dataset.",
        "",
        f"- **Architecture**: {info['arch']}",
        f"- **Backbone**: {info['backbone']}",
        f"- **Framework**: {info['framework']}",
        f"- **Number of classes**: {num_classes}",
        "",
        "### Classes",
        "",
        "| ID | Class |",
        "|----|-------|",
        "| 1 | ActiveTuberculosis |",
        "| 2 | ObsoletePulmonaryTuberculosis |",
        "",
        "## Training Details",
        "",
        f"- **Dataset**: TBX11K (chest X-rays)",
        f"- **Epochs**: {training.get('epochs', 'N/A')}",
        f"- **Batch size**: {training.get('batch_size', 'N/A')}",
        f"- **Optimizer**: {training.get('optimizer', 'AdamW')}",
        f"- **Learning rate**: {training.get('lr', training.get('lr_backbone', 'N/A'))}",
        f"- **Weight decay**: {training.get('weight_decay', 'N/A')}",
        f"- **Warmup epochs**: {training.get('warmup_epochs', 'N/A')}",
        f"- **EMA decay**: {training.get('ema_decay', 'N/A')}",
        f"- **Gradient clipping**: {training.get('clip_norm', 'N/A')}",
        "",
        "### Augmentation",
        "",
        f"- Horizontal flip: {aug.get('hflip_prob', 0.5)}",
        f"- Brightness: +/-{aug.get('brightness', 0.1)}",
        f"- Contrast: +/-{aug.get('contrast', 0.1)}",
        f"- Gamma: {aug.get('gamma', 0.0)}",
        f"- Noise std: {aug.get('noise_std', 0.0)}",
        "",
        "## Performance",
        "",
    ]

    if best_map is not None:
        lines.append(f"**Best mAP@0.5:0.95**: {best_map:.4f} (epoch {best_epoch})")
        lines.append("")

    if metrics:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
        lines.append("")

    lines.extend([
        "## Usage",
        "",
        "### Loading Weights",
        "",
        "```python",
        "import torch",
        "",
        "# Load the EMA weights (recommended) or best_model weights",
        "state_dict = torch.load('weights/ema_model.pth', weights_only=True)",
        "model.load_state_dict(state_dict)",
        "```",
        "",
        "### Files",
        "",
        "```",
        "weights/",
        "  ema_model.pth        # EMA weights (recommended for inference)",
        "  best_model.pth       # Best weights by validation mAP",
        "  last_checkpoint.pth  # Full checkpoint (includes optimizer state)",
        "config.json           # Training configuration",
        "metrics.json          # Validation metrics (COCO evaluation)",
        "metrics_tta.json      # Test-time augmentation metrics",
        "confusion_matrix.png  # Confusion matrix visualization",
        "curves/               # Training curves",
        "explain/              # Grad-CAM / attention visualizations",
        "```",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{tbx11k_detection,",
        "  title={TB Lesion Detection on Chest X-rays},",
        "  year={2024},",
        "  note={TBX11K Dataset},",
        "}",
        "```",
    ])

    return "\n".join(lines)


def push_to_hub(results_dir, cfg, best_map=None, best_epoch=None,
                model_type="fcos", repo_id=None, token=None,
                private=False):
    """Push trained model and all artifacts to Hugging Face Hub.

    Args:
        results_dir: Path to results directory (e.g. "results/fcos").
        cfg: Training configuration dict.
        best_map: Best validation mAP value.
        best_epoch: Epoch number of best model.
        model_type: One of "fcos", "retinanet", "efficientdet", "detr".
        repo_id: HF repo ID (e.g. "username/model-tbx11k"). If None, auto-generated.
        token: HF API token. Falls back to HF_TOKEN env var.
        private: Whether to create a private repo.

    Returns:
        The URL of the pushed model repo.
    """
    api = HfApi()
    auth_token = _resolve_token(token)
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"[HF] Results directory not found: {results_dir}")
        return None

    if repo_id is None:
        repo_id = model_type

    print(f"[HF] Pushing to hub: {repo_id}")

    repo_url = api.create_repo(
        repo_id, token=auth_token, repo_type="model",
        exist_ok=True, private=private,
    )
    print(f"[HF] Repo: {repo_url}")

    pushed_files = []

    weight_files = ["weights/ema_model.pth", "weights/best_model.pth"]
    for wf in weight_files:
        fpath = results_path / wf
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=wf,
                repo_id=repo_id,
                token=auth_token,
                commit_message=f"Upload {wf}",
            )
            pushed_files.append(wf)
            print(f"  Uploaded {wf}")

    metadata_files = ["config.json", "metrics.json", "metrics_tta.json"]
    for mf in metadata_files:
        fpath = results_path / mf
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=mf,
                repo_id=repo_id,
                token=auth_token,
                commit_message=f"Upload {mf}",
            )
            pushed_files.append(mf)
            print(f"  Uploaded {mf}")

    metrics = {}
    metrics_path = results_path / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    model_card = _generate_model_card(model_type, cfg, best_map, best_epoch, metrics)
    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=auth_token,
        commit_message="Upload model card",
    )
    pushed_files.append("README.md")
    print("  Uploaded README.md (model card)")

    asset_dirs = [
        ("curves", ["training_curves.png"]),
        (None, ["confusion_matrix.png"]),
    ]
    for subdir, filenames in asset_dirs:
        for fname in filenames:
            if subdir:
                fpath = results_path / subdir / fname
                repo_path = f"{subdir}/{fname}"
            else:
                fpath = results_path / fname
                repo_path = fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    token=auth_token,
                    commit_message=f"Upload {repo_path}",
                )
                pushed_files.append(repo_path)
                print(f"  Uploaded {repo_path}")

    explain_dir = results_path / "explain"
    if explain_dir.exists():
        explain_files = sorted(explain_dir.glob("*.png"))[:10]
        for ef in explain_files:
            api.upload_file(
                path_or_fileobj=str(ef),
                path_in_repo=f"explain/{ef.name}",
                repo_id=repo_id,
                token=auth_token,
                commit_message=f"Upload explain/{ef.name}",
            )
            pushed_files.append(f"explain/{ef.name}")
        if explain_files:
            print(f"  Uploaded {len(explain_files)} XAI visualizations")

    hub_url = f"https://huggingface.co/{repo_id}"
    print(f"\n[HF] Push complete! {len(pushed_files)} files uploaded.")
    print(f"[HF] View at: {hub_url}")
    return hub_url
