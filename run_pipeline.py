"""
RouteIQ — Main Pipeline Runner
Orchestrates: data generation → ETL → model training
Run this once to set up the platform.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_pipeline(sample: bool = True, train_models: bool = True):
    start = time.time()
    print("\n" + "=" * 60)
    print("  RouteIQ Delivery Operations Intelligence Platform")
    print("  Pipeline Runner v2.4")
    print("=" * 60)

    # Step 1: Data Generation
    print("\n[STEP 1/3] Data Generation")
    print("-" * 40)
    from src.pipelines.data_generator import RouteIQDataGenerator
    gen = RouteIQDataGenerator(seed=42)
    gen.run(save=True, sample_only=sample)

    # Step 2: ETL
    print("\n[STEP 2/3] ETL & Feature Engineering")
    print("-" * 40)
    from src.pipelines.etl import RouteIQETL
    etl = RouteIQETL(use_sample=sample)
    etl.run()

    # Step 3: Model Training
    if train_models:
        print("\n[STEP 3/3] Predictive Model Training")
        print("-" * 40)
        from src.models.trainer import RouteIQModelTrainer
        trainer = RouteIQModelTrainer()
        results = trainer.run()

        print("\n📊 Model Results:")
        for name, res in results.items():
            acc = res.get("accuracy") or res.get("r2", 0)
            print(f"  {name:<30} → {acc:.1%}")

    elapsed = time.time() - start
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s")
    print("\nTo launch dashboard:")
    print("  streamlit run dashboard/app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RouteIQ Pipeline Runner")
    parser.add_argument("--full", action="store_true",
                        help="Generate full 1M record dataset (slow)")
    parser.add_argument("--no-models", action="store_true",
                        help="Skip model training")
    args = parser.parse_args()

    run_pipeline(
        sample=not args.full,
        train_models=not args.no_models,
    )
