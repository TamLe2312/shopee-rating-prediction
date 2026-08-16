"""
Sinh báo cáo trực quan cho bài toán Linear/Ridge Regression.

Output PNG (lưu vào reports/):
  r01_rating_distribution.png   - Phân bố target rating 1-5
  r02_alpha_tuning.png          - Đường cong R²/MSE theo alpha
  r03_predicted_vs_actual.png   - Scatter plot
  r04_residual_plot.png         - Residual vs predicted
  r05_residual_hist.png         - Histogram lỗi
  r06_top_features.png          - Top từ ảnh hưởng (+/-)
  r07_learning_curve.png        - Learning curve (R² theo size)
  r08_model_comparison.png      - So sánh LR/Ridge/Lasso

Cách dùng:
    python -m src.reporting.plots
    # hoặc
    python src/reporting/plots.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Color palette (Shopee orange theme)
C_PRIMARY = "#EE4D2D"
C_SECONDARY = "#0F5257"
C_ACCENT = "#FFC72C"
C_POS = "#2CA02C"
C_NEG = "#D62728"
C_GRAY = "#64748B"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "shopee_reviews_for_regression.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def load_data():
    df = pd.read_csv(DATA_FILE)
    X = df["Comment"].fillna("").astype(str)
    y = df["Target"].astype(float)
    return X, y


def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def fit_tfidf(X_train, X_test):
    v = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    return v, v.fit_transform(X_train), v.transform(X_test)


# --------------------- 1. Rating distribution ---------------------
def plot_rating_distribution(y, out):
    counts = pd.Series(y).astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=[C_NEG, C_NEG, C_GRAY, C_POS, C_POS], edgecolor="black")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, v + 50, f"{v:,}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Rating (sao)", fontsize=12)
    ax.set_ylabel("Số lượng review", fontsize=12)
    ax.set_title(f"Phân bố Rating (n = {len(y):,})", fontsize=13, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- 2. Alpha tuning curve ---------------------
def plot_alpha_tuning(X_train_v, y_train, X_test_v, y_test, out):
    alphas = [0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]  # khớp ALPHA_GRID (8 cấu hình)
    r2_dev_list, rmse_dev_list = [], []
    r2_train_list, cv_list = [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for alpha in alphas:
        m = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        m.fit(X_train_v, y_train)
        pred_test = m.predict(X_test_v)
        pred_train = m.predict(X_train_v)
        r2_dev_list.append(r2_score(y_test, pred_test))
        r2_train_list.append(r2_score(y_train, pred_train))
        rmse_dev_list.append(np.sqrt(mean_squared_error(y_test, pred_test)))
        # CV R² trên train — tiêu chí chọn α (giống train_ridge.py)
        cv = cross_val_score(Ridge(alpha=alpha, random_state=RANDOM_STATE),
                             X_train_v, y_train, cv=kf, scoring="r2", n_jobs=-1)
        cv_list.append(float(cv.mean()))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_xlabel("Alpha (regularization)", fontsize=12)
    ax1.set_ylabel("R²", color=C_PRIMARY, fontsize=12)
    ax1.plot(alphas, r2_train_list, "s--", color=C_SECONDARY, label="R² Train", linewidth=2, alpha=0.7)
    ax1.plot(alphas, cv_list, "o-", color=C_PRIMARY, label="CV R² (5-fold, train)", linewidth=2, markersize=8)
    ax1.plot(alphas, r2_dev_list, "D-.", color=C_ACCENT, label="R² Test", linewidth=1.5, alpha=0.9)
    ax1.set_xscale("log")
    ax1.tick_params(axis="y", labelcolor=C_PRIMARY)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower left", fontsize=9)

    ax2 = ax1.twinx()
    color2 = C_GRAY
    ax2.set_ylabel("RMSE", color=color2, fontsize=12)
    ax2.plot(alphas, rmse_dev_list, "^:", color=color2, label="RMSE Test", linewidth=1.5)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.legend(loc="upper right", fontsize=9)

    # Chọn α tối ưu bằng CV (đúng phương pháp — không peek test set)
    best_idx = int(np.argmax(cv_list))
    ax1.axvline(alphas[best_idx], color="red", linestyle=":", alpha=0.5)
    ax1.annotate(f"Best α = {alphas[best_idx]} (chọn bằng 5-fold CV)\nCV R² = {cv_list[best_idx]:.3f}",
                 xy=(alphas[best_idx], cv_list[best_idx]),
                 xytext=(15, -28), textcoords="offset points",
                 fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor=C_ACCENT, alpha=0.8))

    plt.title("Ridge Regression — Tuning Alpha (chọn α bằng 5-fold CV)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- 3. Predicted vs Actual scatter ---------------------
def plot_predicted_vs_actual(y_true, y_pred, out):
    fig, ax = plt.subplots(figsize=(7, 7))
    hb = ax.hexbin(y_true, y_pred, gridsize=30, cmap="YlOrRd", mincnt=1)
    ax.plot([1, 5], [1, 5], "k--", linewidth=1.5, alpha=0.5, label="Perfect (y=x)")
    ax.set_xlabel("Rating thực tế", fontsize=12)
    ax.set_ylabel("Rating dự đoán", fontsize=12)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_aspect("equal")
    plt.colorbar(hb, ax=ax, label="Số mẫu")

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    ax.text(0.05, 0.95, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor=C_PRIMARY))

    ax.set_title("Predicted vs Actual Rating", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- 4. Residual plot ---------------------
def plot_residual(y_true, y_pred, out):
    residual = y_true - y_pred
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_pred, residual, alpha=0.2, color=C_PRIMARY, edgecolor=None, s=15)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Rating dự đoán", fontsize=12)
    ax.set_ylabel("Residual (thực tế - dự đoán)", fontsize=12)
    ax.set_title("Residual Plot", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- 5. Residual histogram ---------------------
def plot_residual_hist(y_true, y_pred, out):
    residual = y_true - y_pred
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residual, bins=50, color=C_SECONDARY, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {residual.mean():.3f}")
    ax.set_xlabel("Residual (thực tế - dự đoán)", fontsize=12)
    ax.set_ylabel("Tần số", fontsize=12)
    ax.set_title(f"Phân bố Residual  (std = {residual.std():.3f})", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- 6. Top features ---------------------
def plot_top_features(model, vectorizer, out, n=20):
    feat_names = vectorizer.get_feature_names_out()
    coef = model.coef_
    top_pos_idx = np.argsort(coef)[-n:][::-1]
    top_neg_idx = np.argsort(coef)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].barh([feat_names[i] for i in top_pos_idx[::-1]],
                 [coef[i] for i in top_pos_idx[::-1]],
                 color=C_POS)
    axes[0].set_title(f"Top {n} từ tăng rating", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Hệ số (coef)")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh([feat_names[i] for i in top_neg_idx[::-1]],
                 [coef[i] for i in top_neg_idx[::-1]],
                 color=C_NEG)
    axes[1].set_title(f"Top {n} từ giảm rating", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Hệ số (coef)")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- 7. Learning curve ---------------------
def plot_learning_curve(X_v, y, out):
    sizes, train_scores, val_scores = learning_curve(
        Ridge(alpha=1.0, random_state=RANDOM_STATE),
        X_v, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5, scoring="r2", n_jobs=-1, random_state=RANDOM_STATE,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, train_mean, "o-", color=C_SECONDARY, label="Train R²", linewidth=2)
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color=C_SECONDARY)
    ax.plot(sizes, val_mean, "s-", color=C_PRIMARY, label="Validation R² (5-fold)", linewidth=2)
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color=C_PRIMARY)
    ax.set_xlabel("Số mẫu huấn luyện", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Learning Curve — Ridge(α=1.0)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- 8. Model comparison ---------------------
def plot_model_comparison(X_train_v, y_train, X_test_v, y_test, out):
    models = {
        "LinearReg (OLS)": LinearRegression(),
        "Ridge α=1.0": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso α=0.001": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train_v, y_train)
        pred = m.predict(X_test_v)
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        results[name] = {"R²": r2, "RMSE": rmse, "MAE": mae}

    metrics = ["R²", "RMSE", "MAE"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, metrics):
        values = [results[m][metric] for m in models]
        colors = [C_GRAY, C_PRIMARY, C_SECONDARY]
        bars = ax.bar(list(models.keys()), values, color=colors, edgecolor="black")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=11, fontweight="bold")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        if metric == "R²":
            ax.set_ylim(min(min(values) - 0.1, -0.2), 1.0)
            ax.axhline(0, color="black", linewidth=0.5)

    plt.suptitle("So sánh các mô hình Linear Regression", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- MAIN ---------------------
def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading data...")
    X, y = load_data()
    print(f"  {len(X):,} samples")

    print("\n[1/8] Plotting rating distribution...")
    plot_rating_distribution(y, os.path.join(REPORTS_DIR, "r01_rating_distribution.png"))

    print("\nSplitting + TF-IDF...")
    X_train, X_test, y_train, y_test = split(X, y)
    v, X_train_v, X_test_v = fit_tfidf(X_train, X_test)

    print("\n[2/8] Plotting alpha tuning curve...")
    plot_alpha_tuning(X_train_v, y_train, X_test_v, y_test,
                      os.path.join(REPORTS_DIR, "r02_alpha_tuning.png"))

    print("\n[3-5/8] Training Ridge + plotting predictions...")
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train_v, y_train)
    y_pred = model.predict(X_test_v)

    plot_predicted_vs_actual(y_test.values, y_pred,
                             os.path.join(REPORTS_DIR, "r03_predicted_vs_actual.png"))
    plot_residual(y_test.values, y_pred,
                  os.path.join(REPORTS_DIR, "r04_residual_plot.png"))
    plot_residual_hist(y_test.values, y_pred,
                       os.path.join(REPORTS_DIR, "r05_residual_hist.png"))

    print("\n[6/8] Plotting top features...")
    plot_top_features(model, v,
                      os.path.join(REPORTS_DIR, "r06_top_features.png"))

    print("\n[7/8] Plotting learning curve (cv=5, mất ~1 phút)...")
    plot_learning_curve(X_train_v, y_train,
                        os.path.join(REPORTS_DIR, "r07_learning_curve.png"))

    print("\n[8/8] Plotting model comparison...")
    plot_model_comparison(X_train_v, y_train, X_test_v, y_test,
                          os.path.join(REPORTS_DIR, "r08_model_comparison.png"))

    print("\n=== Done ===")
    print(f"All plots saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
