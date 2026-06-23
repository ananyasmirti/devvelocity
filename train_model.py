"""
train_model.py
Trains a LightGBM classifier to predict developer efficiency class
(Low / Medium / High) based on SPACE framework features extracted
from the Stack Overflow Developer Survey 2024.

Usage:
    python train_model.py --csv data/survey_results_public.csv
    python train_model.py --csv data/survey_results_public.csv --output models/devvelocity.pkl
"""

import argparse
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

# Adjust import path when running directly
sys.path.insert(0, str(Path(__file__).parent))
from data_prep import create_space_target, engineer_features, load_survey

import warnings
warnings.filterwarnings("ignore")


# ── Model config ──────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",   # handles class imbalance automatically
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ── Training pipeline ─────────────────────────────────────────────────────────

def train(csv_path: str, output_path: str = "devvelocity_model.pkl") -> lgb.LGBMClassifier:
    """
    Full training pipeline:
      1. Load SO Survey CSV
      2. Engineer SPACE features
      3. Create composite efficiency target
      4. Train LightGBM with early stopping
      5. Evaluate and save model
    """
    # 1. Load data
    df = load_survey(csv_path)

    # 2. Feature engineering
    print("\n⚙️  Engineering SPACE features...")
    X = engineer_features(df)
    y = create_space_target(X)

    # 3. Drop rows with NaN targets
    valid = y.notna()
    X = X[valid].copy()
    y = y[valid].astype(int)

    print(f"   Feature columns ({len(X.columns)}): {list(X.columns)}")
    print(f"   Valid samples: {len(X):,}")
    print(f"\n   Class distribution:")
    dist = y.value_counts().sort_index().rename({0: "Low", 1: "Medium", 2: "High"})
    for label, count in dist.items():
        bar = "█" * (count * 30 // dist.max())
        pct = count / len(y) * 100
        print(f"     {label:8s} {bar} {count:,}  ({pct:.1f}%)")

    # 4. Fill remaining NaNs with column medians
    X = X.fillna(X.median(numeric_only=True))

    # 5. Train / test split (stratified to preserve class ratios)
    print("\n🔀 Splitting 80% train / 20% test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 6. Train LightGBM
    print("\n🚀 Training LightGBM classifier...")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"   Best iteration: {model.best_iteration_}")

    # 7. Evaluate
    y_pred = model.predict(X_test)
    print("\n📊 Test-set evaluation:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Low efficiency", "Medium efficiency", "High efficiency"],
        )
    )

    print("Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Low", "Medium", "High"]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    # 8. Feature importances
    print("\n🔍 Feature importances (gain-based):")
    fi = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )
    for feat, imp in fi.items():
        bar = "█" * int(imp * 30 / fi.max())
        print(f"  {feat:35s} {bar} {imp:.0f}")

    # 9. Save model bundle (model + feature column list)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_cols": list(X.columns),
        "label_map": {0: "Low", 1: "Medium", 2: "High"},
    }
    joblib.dump(bundle, output_path)
    print(f"\n✅ Model saved to: {output_path}")

    return model


# ── Inference helper (called by MCP server at runtime) ───────────────────────

def predict(features: dict, model_path: str = "devvelocity_model.pkl") -> dict:
    """
    Predict efficiency class for a single developer.
    
    Args:
        features: dict with keys matching feature_cols (missing keys → NaN → median)
        model_path: path to saved model bundle
    
    Returns:
        {
          "efficiency_class": "Low" | "Medium" | "High",
          "confidence": 0.0-1.0,
          "probabilities": {"Low": ..., "Medium": ..., "High": ...}
        }
    """
    bundle = joblib.load(model_path)
    model: lgb.LGBMClassifier = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]
    label_map: dict = bundle["label_map"]

    row = pd.DataFrame([{col: features.get(col, np.nan) for col in feature_cols}])
    row = row.fillna(row.median())

    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]

    return {
        "efficiency_class": label_map[pred],
        "confidence": round(float(proba[pred]), 3),
        "probabilities": {
            label_map[i]: round(float(p), 3) for i, p in enumerate(proba)
        },
    }


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DevVelocity LightGBM model")
    parser.add_argument(
        "--csv",
        default="data/2024 - survey_results_public.csv",
        help="Path to survey_results_public.csv",
    )
    parser.add_argument(
        "--output",
        default="models/devvelocity_model.pkl",
        help="Where to save the trained model bundle",
    )
    args = parser.parse_args()
    train(args.csv, args.output)
