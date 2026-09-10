"""
Phase 3 Training Entrypoint.
Trains SpeedNet on IO-VNBD data, evaluates on test split, and exports ONNX.

Usage:
    python train_speed_model.py [--epochs 60] [--batch 256] [--window 50]
"""

import argparse
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Train SpeedNet on IO-VNBD dataset")
    parser.add_argument("--data", default="data/iovnbd/S-S1.csv", help="Path to IO-VNBD CSV")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Phase 3: AI/ML Speed Estimation — SpeedNet Training")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Build dataset
    # ---------------------------------------------------------------
    print(f"\n[1/4] Loading dataset: {args.data}")
    from src.speed_estimation.dataset import SpeedDataset

    ds = SpeedDataset(
        csv_path=args.data,
        window_size=args.window,
        stride=5,
    )
    train_loader, val_loader, test_loader = ds.get_torch_loaders(batch_size=args.batch)

    print(f"       Train windows : {len(ds.X_train)}")
    print(f"       Val windows   : {len(ds.X_val)}")
    print(f"       Test windows  : {len(ds.X_test)}")

    # Save normalization stats for inference engine
    stats_path = os.path.join(args.models_dir, "norm_stats.json")
    os.makedirs(args.models_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(ds.norm_stats, f, indent=2)
    print(f"       Norm stats saved → {stats_path}")

    # ---------------------------------------------------------------
    # 2. Train
    # ---------------------------------------------------------------
    from src.speed_estimation.model import SpeedNet
    from src.speed_estimation.trainer import SpeedModelTrainer

    model = SpeedNet(window_size=args.window)
    print(f"\n[2/4] Training SpeedNet ({model.count_parameters():,} parameters)")

    trainer = SpeedModelTrainer(
        model=model,
        window_size=args.window,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=args.models_dir,
    )
    history = trainer.fit(train_loader, val_loader, verbose=True)
    trainer.save_history(os.path.join(args.models_dir, "training_history.json"))

    # ---------------------------------------------------------------
    # 3. Evaluate on test split
    # ---------------------------------------------------------------
    print("\n[3/4] Evaluating on test split ...")
    metrics = trainer.evaluate(test_loader, load_best=True)

    print(f"\n  ✓ Test RMSE : {metrics['rmse_ms']:.4f} m/s  ({metrics['rmse_kmh']:.2f} km/h)")
    print(f"  ✓ Test MAE  : {metrics['mae_ms']:.4f} m/s")
    print(f"  ✓ Best val  : {metrics['best_val_rmse']:.4f} m/s  (epoch {metrics['best_epoch']})")

    target_rmse = 1.5   # SPEED-03 requirement
    if metrics["rmse_ms"] <= target_rmse:
        print(f"\n  ✅ SPEED-03 PASSED: RMSE {metrics['rmse_ms']:.3f} ≤ {target_rmse} m/s")
    else:
        print(f"\n  ⚠  SPEED-03 NOTE:  RMSE {metrics['rmse_ms']:.3f} > {target_rmse} m/s")
        print("     (Single-session IO-VNBD may be insufficient; consider multi-run data)")

    # ---------------------------------------------------------------
    # 4. Export to ONNX
    # ---------------------------------------------------------------
    print("\n[4/4] Exporting to ONNX ...")
    from src.speed_estimation.inference import OnnxExporter

    onnx_path = os.path.join(args.models_dir, "speednet.onnx")
    try:
        exporter = OnnxExporter(checkpoint_dir=args.models_dir)
        exporter.export(output_path=onnx_path, window_size=args.window)
    except Exception as e:
        print(f"  ⚠  ONNX export skipped: {e}")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Checkpoint : {args.models_dir}/speednet_best.pt")
    print(f"  ONNX model : {onnx_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
