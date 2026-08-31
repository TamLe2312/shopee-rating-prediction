import os
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "shopee_reviews_for_regression.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "max_features_linear": 5000,
    "min_df_linear": 5,
    "max_features_lightgbm": 30000,
    "min_df_lightgbm": 3,
    "sublinear_tf": True
}


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
    print(f"    Acc ±0.5 = {acc05 * 100:.2f}%")
    print(f"    Acc ±1.0 = {acc10 * 100:.2f}%")
    print("-" * 30)
    return {
        "mse": float(mse), "rmse": rmse, "mae": float(mae), "r2": float(r2),
        "acc_within_0_5": acc05, "acc_within_1": acc10,
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print(f"Không tìm thấy file data tại: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df):,} rows")

    X = df["Comment"].fillna("").astype(str)
    y = df["Target"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=RANDOM_STATE)

    print("\nĐang Vector hóa dữ liệu cho Linear...")
    vec_linear = TfidfVectorizer(
        ngram_range=TFIDF_CONFIG["ngram_range"],
        max_features=TFIDF_CONFIG["max_features_linear"],
        min_df=TFIDF_CONFIG["min_df_linear"],
        sublinear_tf=TFIDF_CONFIG["sublinear_tf"]
    )
    X_train_linear = vec_linear.fit_transform(X_train)
    X_test_linear = vec_linear.transform(X_test)
    vocab_size = len(vec_linear.vocabulary_)
    print(f"Kích thước từ điển (Vocab size): {vocab_size}")

    print("\n--- ĐÁNH GIÁ & HUẤN LUYỆN LINEAR REGRESSION ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    linear_model = LinearRegression(n_jobs=-1)

    scores = cross_val_score(
        linear_model,
        X_train_linear, y_train,
        cv=kf, scoring="r2", n_jobs=-1,
    )
    cv_mean = float(scores.mean())
    cv_std = float(scores.std())

    print(f"  Cross-Validation R²: {cv_mean:<8.4f} ± {cv_std:<8.4f}")

    linear_model.fit(X_train_linear, y_train)
    y_pred_linear = linear_model.predict(X_test_linear)
    metrics_linear = evaluate(y_test, y_pred_linear, "LINEAR REGRESSION (TEST SET)")

    print("\n--- HUẤN LUYỆN MÔ HÌNH LIGHTGBM ---")
    print("Vector hóa cho LightGBM...")
    vec_lgbm = TfidfVectorizer(ngram_range=(1, 2), max_features=TFIDF_CONFIG["max_features_lightgbm"],
                               min_df=TFIDF_CONFIG["min_df_lightgbm"])
    X_train_lgbm = vec_lgbm.fit_transform(X_train)
    X_test_lgbm = vec_lgbm.transform(X_test)

    lgbm_model = LGBMRegressor(
        n_estimators=588,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    lgbm_model.fit(X_train_lgbm, y_train)
    y_pred_lgbm = lgbm_model.predict(X_test_lgbm)
    metrics_lgbm = evaluate(y_test, y_pred_lgbm, "LIGHTGBM (TEST SET)")

    print("\nĐang lưu các file mô hình và cấu hình...")
    joblib.dump(vec_linear, os.path.join(MODEL_DIR, "rating_vectorizer_linear.pkl"))
    joblib.dump(vec_lgbm, os.path.join(MODEL_DIR, "rating_vectorizer_lightgbm.pkl"))

    # Lưu Linear Regression
    joblib.dump(linear_model, os.path.join(MODEL_DIR, "rating_model_linear.pkl"))
    metadata_linear = {
        "model_type": "LinearRegression",
        "task": "regression",
        "target": "Rating (1-5)",
        "mse": metrics_linear["mse"],
        "rmse": metrics_linear["rmse"],
        "mae": metrics_linear["mae"],
        "r2": metrics_linear["r2"],
        "acc_within_0_5": metrics_linear["acc_within_0_5"],
        "acc_within_1": metrics_linear["acc_within_1"],
        "cv_r2_mean": cv_mean,
        "cv_r2_std": cv_std,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "vocab_size": int(vocab_size),
        "config": {
            "max_features": TFIDF_CONFIG["max_features_linear"],
            "ngram_range": list(TFIDF_CONFIG["ngram_range"]),
            "min_df": TFIDF_CONFIG["min_df_linear"],
            "sublinear_tf": TFIDF_CONFIG["sublinear_tf"]
        }
    }
    with open(os.path.join(MODEL_DIR, "rating_metadata_linear.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_linear, f, indent=4, ensure_ascii=False)

    # Lưu LightGBM
    joblib.dump(lgbm_model, os.path.join(MODEL_DIR, "rating_model_lightgbm.pkl"))
    metadata_lgbm = {
        "model_type": "LGBMRegressor",
        "task": "regression",
        "target": "Rating (1-5)",
        "mse": metrics_lgbm["mse"],
        "rmse": metrics_lgbm["rmse"],
        "mae": metrics_lgbm["mae"],
        "r2": metrics_lgbm["r2"],
        "acc_within_0_5": metrics_lgbm["acc_within_0_5"],
        "acc_within_1": metrics_lgbm["acc_within_1"],
        "cv_r2_mean": None,
        "cv_r2_std": None,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "vocab_size": int(vocab_size),
        "config": {
            "max_features": TFIDF_CONFIG["max_features_lightgbm"],
            "ngram_range": list(TFIDF_CONFIG["ngram_range"]),
            "min_df": TFIDF_CONFIG["min_df_lightgbm"],
            "n_estimators": lgbm_model.n_estimators,
            "learning_rate": lgbm_model.learning_rate,
            "num_leaves": lgbm_model.num_leaves,
        }
    }
    with open(os.path.join(MODEL_DIR, "rating_metadata_lightgbm.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_lgbm, f, indent=4, ensure_ascii=False)

    print(f"✅ Đã lưu xong thành công 5 file tại thư mục: {MODEL_DIR}")


if __name__ == "__main__":
    main()
