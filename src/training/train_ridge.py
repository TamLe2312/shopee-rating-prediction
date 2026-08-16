"""
Production pipeline: TF-IDF + Ridge Regression dự đoán Rating Shopee.

Quy trình end-to-end:
  1. Load dữ liệu đã preprocess (~20,486 reviews) từ data/raw/shopee_reviews_for_regression.csv
  2. Split 90/10 (Train 18,437 / Test 2,049) — không có dev set
  3. Fit TF-IDF trên train (1-2 gram, max 30k features, min_df=3)
  4. Grid search α qua 5-fold cross-validation trên tập train
  5. So sánh Ridge vs LinearRegression (OLS) vs Lasso (CV)
  6. Train Ridge với α tối ưu trên TOÀN BỘ train
  7. Đánh giá trên Test set (chưa từng thấy)
  8. Lưu model.pkl + vectorizer.pkl + metadata.json vào models/

Cách dùng:
    python -m src.training.train_ridge
    # hoặc
    python src/training/train_ridge.py
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "shopee_reviews_for_regression.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
RANDOM_STATE = 42

TFIDF_CONFIG = {
    "max_features": 30000,
    "ngram_range": (1, 2),
    "min_df": 3,
    "sublinear_tf": True,
}

ALPHA_GRID = [0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def evaluate(y_true, y_pred, label=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pred_clipped = np.clip(y_pred, 1, 5)
    diff = np.abs(pred_clipped - y_true)
    acc05 = float((diff <= 0.5).mean())
    acc10 = float((diff <= 1.0).mean())

    print(f"  {label}")
    print(f"    MSE      = {mse:.4f}")
    print(f"    RMSE     = {rmse:.4f}")
    print(f"    MAE      = {mae:.4f}")
    print(f"    R²       = {r2:.4f}")
    print(f"    Acc ±0.5 = {acc05*100:.2f}%")
    print(f"    Acc ±1.0 = {acc10*100:.2f}%")
    return {
        "mse": float(mse), "rmse": rmse, "mae": float(mae), "r2": float(r2),
        "acc_within_0_5": acc05, "acc_within_1": acc10,
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print("Hãy chạy trước: python -m src.preprocessing.prepare_regression")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df):,} rows")

    X = df["Comment"].fillna("").astype(str)
    y = df["Target"].astype(float)

    # ---- 1. Split 90/10 ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # ---- 2. TF-IDF ----
    print("\n[1/4] Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v = vectorizer.transform(X_test)
    vocab_size = X_train_v.shape[1]
    print(f"  Vocab size: {vocab_size:,} features")

    # ---- 3. Grid search α qua 5-fold CV trên train ----
    print(f"\n[2/4] Grid search α ∈ {ALPHA_GRID} (5-fold CV trên train)...")
    print(f"  {'alpha':<8} {'CV R² mean':<12} {'CV R² std':<10}")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    best_alpha = None
    best_cv_r2 = -np.inf
    best_cv_std = 0.0
    for alpha in ALPHA_GRID:
        scores = cross_val_score(
            Ridge(alpha=alpha, random_state=RANDOM_STATE),
            X_train_v, y_train,
            cv=kf, scoring="r2", n_jobs=-1,
        )
        cv_mean = float(scores.mean())
        cv_std = float(scores.std())
        cv_results[alpha] = {"r2_mean": cv_mean, "r2_std": cv_std}
        marker = " ★" if cv_mean > best_cv_r2 else ""
        print(f"  {alpha:<8.2f} {cv_mean:<12.4f} {cv_std:<10.4f}{marker}")
        if cv_mean > best_cv_r2:
            best_alpha = alpha
            best_cv_r2 = cv_mean
            best_cv_std = cv_std

    print(f"\n  → Best α = {best_alpha} (CV R² = {best_cv_r2:.4f} ± {best_cv_std:.4f})")

    # ---- 4. So sánh Ridge vs LinearReg vs Lasso (CV) ----
    print("\n[3/4] So sánh 3 mô hình Linear (5-fold CV trên train)...")
    comparison = {
        f"Ridge α={best_alpha}": Ridge(alpha=best_alpha, random_state=RANDOM_STATE),
        "LinearReg (OLS)": LinearRegression(),
        "Lasso α=0.001": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000),
    }
    comp_results = {}
    for name, model in comparison.items():
        scores = cross_val_score(model, X_train_v, y_train, cv=kf, scoring="r2", n_jobs=-1)
        comp_results[name] = {"r2_mean": float(scores.mean()), "r2_std": float(scores.std())}
        print(f"  {name:<20} CV R² = {scores.mean():>7.4f} ± {scores.std():.4f}")

    # ---- 5. Train final Ridge trên toàn bộ train ----
    print(f"\n[4/4] Train Ridge(α={best_alpha}) trên toàn bộ {len(X_train):,} mẫu train...")
    final_model = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    final_model.fit(X_train_v, y_train)

    # ---- 6. Đánh giá trên Test ----
    print("\n=== Final evaluation trên Test set ===")
    pred_test = final_model.predict(X_test_v)
    metrics = evaluate(y_test, pred_test, "Test")

    # ---- 7. Lưu artifacts ----
    joblib.dump(final_model, os.path.join(MODEL_DIR, "rating_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "rating_vectorizer.pkl"))

    metadata = {
        "model_type": "Ridge",
        "task": "regression",
        "target": "Rating (1-5)",
        # Test metrics
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "acc_within_0_5": metrics["acc_within_0_5"],
        "acc_within_1": metrics["acc_within_1"],
        # CV metrics
        "cv_r2_mean": best_cv_r2,
        "cv_r2_std": best_cv_std,
        # Data
        "n_train": len(X_train),
        "n_test": len(X_test),
        "vocab_size": int(vocab_size),
        # Config
        "config": {
            "max_features": TFIDF_CONFIG["max_features"],
            "ngram_range": list(TFIDF_CONFIG["ngram_range"]),
            "min_df": TFIDF_CONFIG["min_df"],
            "sublinear_tf": TFIDF_CONFIG["sublinear_tf"],
            "alpha": best_alpha,
            "solver": "auto",
        },
        # Audit trail
        "alpha_cv_results": {str(k): v for k, v in cv_results.items()},
        "model_comparison_cv": comp_results,
    }
    with open(os.path.join(MODEL_DIR, "rating_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved to {MODEL_DIR}/")
    print("  - rating_model.pkl")
    print("  - rating_vectorizer.pkl")
    print("  - rating_metadata.json")


if __name__ == "__main__":
    main()
