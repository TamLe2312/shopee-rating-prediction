"""
Sinh báo cáo trực quan cho bài toán Regression với LightGBM và Linear Regression.

Output PNG (lưu vào reports/):
  r01_rating_distribution.png   - Phân bố target rating 1-5
  r02_lgbm_tuning.png           - Đường cong R²/MSE theo n_estimators của LightGBM
  r03_predicted_vs_actual.png   - Scatter plot (LightGBM)
  r04_residual_plot.png         - Residual vs predicted (LightGBM)
  r05_residual_hist.png         - Histogram lỗi (LightGBM)
  r06_top_features_linear.png   - Top từ ảnh hưởng (+/-) của Linear Regression
  r07_learning_curve.png        - Learning curve (R² theo size của LightGBM)
  r08_model_comparison.png      - So sánh Linear Regression vs LightGBM
  r09_stats_linear.png          - Min, Max, Mean, Mode, Std của Linear vs Thực tế
  r10_stats_lightgbm.png        - Min, Max, Mean, Mode, Std của LightGBM vs Thực tế
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mode
from lightgbm import LGBMRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# Color palette
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
        ax.text(b.get_x() + b.get_width() / 2, v + 50, f"{v:,}",
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


# --------------------- 2. LightGBM Tuning curve ---------------------
def plot_lgbm_tuning(X_train_v, y_train, X_test_v, y_test, out):
    # Thay vì Alpha, ta tune n_estimators của LightGBM
    n_estimators_list = [50, 100, 200, 300, 500]
    r2_test_list, rmse_test_list = [], []
    r2_train_list, cv_list = [], []
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  # Dùng 3-fold để tiết kiệm thời gian

    for n_est in n_estimators_list:
        m = LGBMRegressor(n_estimators=n_est, random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X_train_v, y_train)
        pred_test = m.predict(X_test_v)
        pred_train = m.predict(X_train_v)

        r2_test_list.append(r2_score(y_test, pred_test))
        r2_train_list.append(r2_score(y_train, pred_train))
        rmse_test_list.append(np.sqrt(mean_squared_error(y_test, pred_test)))

        cv = cross_val_score(LGBMRegressor(n_estimators=n_est, random_state=RANDOM_STATE, n_jobs=-1),
                             X_train_v, y_train, cv=kf, scoring="r2", n_jobs=-1)
        cv_list.append(float(cv.mean()))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_xlabel("Số lượng cây (n_estimators)", fontsize=12)
    ax1.set_ylabel("R²", color=C_PRIMARY, fontsize=12)
    ax1.plot(n_estimators_list, r2_train_list, "s--", color=C_SECONDARY, label="R² Train", linewidth=2, alpha=0.7)
    ax1.plot(n_estimators_list, cv_list, "o-", color=C_PRIMARY, label="CV R² (3-fold, train)", linewidth=2,
             markersize=8)
    ax1.plot(n_estimators_list, r2_test_list, "D-.", color=C_ACCENT, label="R² Test", linewidth=1.5, alpha=0.9)
    ax1.tick_params(axis="y", labelcolor=C_PRIMARY)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="center left", fontsize=9)

    ax2 = ax1.twinx()
    color2 = C_GRAY
    ax2.set_ylabel("RMSE", color=color2, fontsize=12)
    ax2.plot(n_estimators_list, rmse_test_list, "^:", color=color2, label="RMSE Test", linewidth=1.5)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.legend(loc="center right", fontsize=9)

    best_idx = int(np.argmax(cv_list))
    ax1.axvline(n_estimators_list[best_idx], color="red", linestyle=":", alpha=0.5)
    ax1.annotate(f"Best n_est = {n_estimators_list[best_idx]}\nCV R² = {cv_list[best_idx]:.3f}",
                 xy=(n_estimators_list[best_idx], cv_list[best_idx]),
                 xytext=(15, -28), textcoords="offset points",
                 fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor=C_ACCENT, alpha=0.8))

    plt.title("LightGBM — Tuning n_estimators", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- 3, 4, 5. Scatter, Residuals (dùng cho mô hình chính) ---------------------
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

    ax.set_title("Predicted vs Actual Rating (LightGBM)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


def plot_residual(y_true, y_pred, out):
    residual = y_true - y_pred
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_pred, residual, alpha=0.2, color=C_PRIMARY, edgecolor=None, s=15)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Rating dự đoán", fontsize=12)
    ax.set_ylabel("Residual (thực tế - dự đoán)", fontsize=12)
    ax.set_title("Residual Plot (LightGBM)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


def plot_residual_hist(y_true, y_pred, out):
    residual = y_true - y_pred
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residual, bins=50, color=C_SECONDARY, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {residual.mean():.3f}")
    ax.set_xlabel("Residual (thực tế - dự đoán)", fontsize=12)
    ax.set_ylabel("Tần số", fontsize=12)
    ax.set_title(f"Phân bố Residual LightGBM (std = {residual.std():.3f})", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- 6. Top features (Dùng Linear Regression để thấy rõ +/-) ---------------------
def plot_top_features_linear(model, vectorizer, out, n=20):
    feat_names = vectorizer.get_feature_names_out()
    coef = model.coef_
    top_pos_idx = np.argsort(coef)[-n:][::-1]
    top_neg_idx = np.argsort(coef)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].barh([feat_names[i] for i in top_pos_idx[::-1]],
                 [coef[i] for i in top_pos_idx[::-1]],
                 color=C_POS)
    axes[0].set_title(f"Top {n} từ làm TĂNG rating (Linear)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Hệ số (coef)")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh([feat_names[i] for i in top_neg_idx[::-1]],
                 [coef[i] for i in top_neg_idx[::-1]],
                 color=C_NEG)
    axes[1].set_title(f"Top {n} từ làm GIẢM rating (Linear)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Hệ số (coef)")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- 7. Learning curve (LightGBM) ---------------------
def plot_learning_curve(X_v, y, out):
    sizes, train_scores, val_scores = learning_curve(
        LGBMRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        X_v, y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=3, scoring="r2", n_jobs=-1, random_state=RANDOM_STATE,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, train_mean, "o-", color=C_SECONDARY, label="Train R²", linewidth=2)
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color=C_SECONDARY)
    ax.plot(sizes, val_mean, "s-", color=C_PRIMARY, label="Validation R² (3-fold)", linewidth=2)
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color=C_PRIMARY)
    ax.set_xlabel("Số mẫu huấn luyện", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Learning Curve — LightGBM", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- 8. Model comparison ---------------------
def plot_model_comparison(X_train_v, y_train, X_test_v, y_test, out):
    models = {
        "LinearReg (OLS)": LinearRegression(n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
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
        colors = [C_GRAY, C_PRIMARY]
        bars = ax.bar(list(models.keys()), values, color=colors, edgecolor="black", width=0.5)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=11, fontweight="bold")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if metric == "R²":
            ax.set_ylim(min(min(values) - 0.1, -0.2), 1.0)
            ax.axhline(0, color="black", linewidth=0.5)

    plt.suptitle("So sánh mô hình Linear Regression và LightGBM", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


# --------------------- 9 & 10. Prediction Statistics (Min, Max, Mean, Mode, Std) ---------------------
def calculate_stats(arr):
    arr = np.array(arr)
    _min = np.min(arr)
    _max = np.max(arr)
    _mean = np.mean(arr)
    _std = np.std(arr)
    # Mode tính trên số làm tròn 1 chữ số thập phân
    _mode = float(mode(np.round(arr, 1), keepdims=True)[0][0])
    return [_min, _max, _mean, _mode, _std]


def plot_prediction_statistics(y_true, y_pred, model_name, out):
    stats_true = calculate_stats(y_true)
    stats_pred = calculate_stats(y_pred)

    labels = ["Min", "Max", "Mean", "Mode (~1 decimal)", "Std"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, stats_true, width, label='Thực tế', color=C_GRAY, edgecolor="black")
    rects2 = ax.bar(x + width / 2, stats_pred, width, label=f'Dự đoán ({model_name})', color=C_PRIMARY,
                    edgecolor="black")

    ax.set_ylabel('Giá trị', fontsize=12)
    ax.set_title(f'Thống kê mô tả: Thực tế vs Dự đoán ({model_name})', fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    ax.grid(axis='y', alpha=0.3)
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

    print("\n[1/10] Plotting rating distribution...")
    plot_rating_distribution(y, os.path.join(REPORTS_DIR, "r01_rating_distribution.png"))

    print("\nSplitting + TF-IDF...")
    X_train, X_test, y_train, y_test = split(X, y)
    v, X_train_v, X_test_v = fit_tfidf(X_train, X_test)

    print("\n[2/10] Plotting LightGBM tuning curve (cv=3, có thể mất chút thời gian)...")
    plot_lgbm_tuning(X_train_v, y_train, X_test_v, y_test,
                     os.path.join(REPORTS_DIR, "r02_lgbm_tuning.png"))

    print("\n[3-5/10] Training LightGBM + plotting predictions...")
    lgbm = LGBMRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    lgbm.fit(X_train_v, y_train)
    y_pred_lgbm = lgbm.predict(X_test_v)

    plot_predicted_vs_actual(y_test.values, y_pred_lgbm,
                             os.path.join(REPORTS_DIR, "r03_predicted_vs_actual.png"))
    plot_residual(y_test.values, y_pred_lgbm,
                  os.path.join(REPORTS_DIR, "r04_residual_plot.png"))
    plot_residual_hist(y_test.values, y_pred_lgbm,
                       os.path.join(REPORTS_DIR, "r05_residual_hist.png"))

    print("\n[6/10] Training Linear Reg & Plotting top features...")
    linear = LinearRegression(n_jobs=-1)
    linear.fit(X_train_v, y_train)
    y_pred_linear = linear.predict(X_test_v)
    plot_top_features_linear(linear, v,
                             os.path.join(REPORTS_DIR, "r06_top_features_linear.png"))

    print("\n[7/10] Plotting learning curve (cv=3, có thể mất 1-2 phút)...")
    plot_learning_curve(X_train_v, y_train,
                        os.path.join(REPORTS_DIR, "r07_learning_curve.png"))

    print("\n[8/10] Plotting model comparison...")
    plot_model_comparison(X_train_v, y_train, X_test_v, y_test,
                          os.path.join(REPORTS_DIR, "r08_model_comparison.png"))

    print("\n[9-10/10] Plotting Prediction Statistics (Linear & LightGBM)...")
    plot_prediction_statistics(y_test.values, y_pred_linear, "Linear Regression",
                               os.path.join(REPORTS_DIR, "r09_stats_linear.png"))
    plot_prediction_statistics(y_test.values, y_pred_lgbm, "LightGBM",
                               os.path.join(REPORTS_DIR, "r10_stats_lightgbm.png"))

    print("\n=== Done ===")
    print(f"All 10 plots saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
