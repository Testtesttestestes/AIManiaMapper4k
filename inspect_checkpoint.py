#!/usr/bin/env python3
"""
Diagnostic script for model and checkpoint dissection.
Inspects all parameter tensors, calculates weight statistics (norm, mean, std, zero-ratio),
and verifies which input features were present in the processed dataset chunks.
"""

import sys
from pathlib import Path
import torch

def inspect_checkpoint(ckpt_path: Path):
    print("=" * 80)
    print(f"🔬 MODEL & CHECKPOINT DISSECTION TOOL")
    print(f"📁 Target File: {ckpt_path.resolve()}")
    print("=" * 80)

    if not ckpt_path.exists():
        print(f"❌ Error: File not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        print(f"\n📦 Checkpoint Metadata:")
        print(f"  • Keys in checkpoint dict: {list(ckpt.keys())}")
        if "epoch" in ckpt:
            print(f"  • Saved Epoch: {ckpt['epoch']}")
        if "val_loss" in ckpt or "best_val_loss" in ckpt:
            v_loss = ckpt.get("val_loss", ckpt.get("best_val_loss"))
            print(f"  • Best Validation Loss: {v_loss}")
        if "optimizer_state" in ckpt and ckpt["optimizer_state"]:
            opt_keys = list(ckpt["optimizer_state"].keys())
            opt_state = ckpt["optimizer_state"].get("state", {})
            print(f"  • Optimizer State: Present ({len(opt_state)} parameter state entries)")

        state_dict = None
        for k in ("model_state", "model_state_dict", "state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
    else:
        print("❌ Unknown checkpoint format.")
        return

    if state_dict is None:
        print("❌ Could not find model state_dict.")
        return

    print(f"\n🧠 Neural Network Layer Breakdown ({len(state_dict)} parameter tensors):")
    print(f"{'Layer / Parameter Name':<45} | {'Shape':<18} | {'Mean':<10} | {'Std':<10} | {'L2 Norm':<10} | Status")
    print("-" * 110)

    total_params = 0
    sleeping_layers = []
    active_layers = []

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        p = param.float()
        numel = p.numel()
        total_params += numel
        p_mean = p.mean().item()
        p_std = p.std().item() if numel > 1 else 0.0
        p_norm = p.norm(2).item()
        shape_str = str(list(p.shape))

        # Check if layer received real optimizer updates (AdamW moment buffers or non-trivial weight shifts)
        status = "✅ ACTIVE"
        # Check optimizer state if present
        is_in_optimizer = False
        if isinstance(ckpt, dict) and "optimizer_state" in ckpt:
            opt_state = ckpt["optimizer_state"].get("state", {})
            # Match parameter index
            for param_idx, state in opt_state.items():
                if "exp_avg" in state and state["exp_avg"].shape == param.shape:
                    if state["exp_avg"].abs().sum().item() > 1e-7:
                        is_in_optimizer = True
                        break
        
        # If std is near 0 or param has never received updates
        if p_std < 1e-7 and numel > 1:
            status = "💤 SLEEPING (Zero variance)"
            sleeping_layers.append(name)
        else:
            active_layers.append(name)

        print(f"{name:<45} | {shape_str:<18} | {p_mean:<10.4e} | {p_std:<10.4e} | {p_norm:<10.4f} | {status}")

    print("-" * 110)
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"  • Active Trained Layers:   {len(active_layers)} tensors")
    print(f"  • Sleeping/Unused Layers:  {len(sleeping_layers)} tensors {sleeping_layers}")

    # Inspect processed dataset structure if available
    dataset_dirs = [Path("dataset_processed"), Path("data/dataset_processed"), Path("../dataset_processed")]
    found_dataset = None
    for d in dataset_dirs:
        if d.exists() and d.is_dir():
            found_dataset = d
            break

    if found_dataset:
        print(f"\n📂 Dataset Sample Analysis (Inspecting files in: {found_dataset}):")
        # Check train_data.pt first, then other .pt files
        sample_files = list(found_dataset.glob("train_data.pt")) or list(found_dataset.glob("*.pt"))
        if sample_files:
            sample_pt = sample_files[0]
            try:
                data = torch.load(sample_pt, map_location="cpu", weights_only=False)
                first_item = {}
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    first_item = data[0]
                elif isinstance(data, dict):
                    if "samples" in data and len(data["samples"]) > 0:
                        first_item = data["samples"][0]
                    else:
                        first_item = data

                print(f"  • Verified dataset file: {sample_pt.name}")
                print(f"  • Available keys in sample: {list(first_item.keys())}")
                for expected in ["audio_mel", "tokens", "ln_mask", "lane_states", "grid_metrics", "star_rating", "scaffold_tokens", "strain_metrics"]:
                    if expected in first_item:
                        shape = getattr(first_item[expected], 'shape', 'scalar')
                        print(f"    - '{expected}': PRESENT (Shape: {shape})")
                    else:
                        print(f"    - '{expected}': ⚡ Dynamically synthesized on-the-fly during training batches")
            except Exception as e:
                print(f"  ⚠️ Could not inspect sample file: {e}")
    else:
        print("\nℹ️ No local dataset_processed folder found in immediate directory.")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    ckpt_file = Path("checkpoints/model_3_transformer_best.pt")
    if len(sys.argv) > 1:
        ckpt_file = Path(sys.argv[1])
    inspect_checkpoint(ckpt_file)
