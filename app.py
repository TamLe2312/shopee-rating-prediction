"""
Streamlit Demo - Dự đoán rating Shopee bằng Ridge Regression / LightGBM
Chạy: streamlit run app.py
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.preprocessing.text import preprocess_text

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Shopee Rating Predictor",
    page_icon="⭐",
    layout="centered",
    initial_sidebar_state="expanded",  # Mở sidebar mặc định
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_models():
    """Load Vectorizer và cả 2 mô hình cùng lúc vào bộ nhớ tạm"""
    # Load chung Vectorizer
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "rating_vectorizer.pkl"))

    # Load Ridge
    model_ridge = joblib.load(os.path.join(MODEL_DIR, "rating_model_ridge.pkl"))
    with open(os.path.join(MODEL_DIR, "rating_metadata_ridge.json"), "r", encoding="utf-8") as f:
        meta_ridge = json.load(f)

    # Load LightGBM
    model_lgbm = joblib.load(os.path.join(MODEL_DIR, "rating_model_lightgbm.pkl"))
    with open(os.path.join(MODEL_DIR, "rating_metadata_lightgbm.json"), "r", encoding="utf-8") as f:
        meta_lgbm = json.load(f)

    return vectorizer, {"Ridge": model_ridge, "LightGBM": model_lgbm}, {"Ridge": meta_ridge, "LightGBM": meta_lgbm}


def predict_rating(text, model, vectorizer, model_name):
    processed = preprocess_text(text)
    if not processed:
        return None
    X = vectorizer.transform([processed])

    # Raw prediction (continuous)
    raw_pred = float(model.predict(X)[0])
    # Clip vào [1, 5] cho hợp lý
    clipped = float(np.clip(raw_pred, 1, 5))

    # Tính Top features dựa trên loại mô hình
    feat_names = vectorizer.get_feature_names_out()
    nonzero = X.toarray()[0].nonzero()[0]
    contribs = []

    if len(nonzero) > 0:
        if model_name == "Ridge":
            coef = model.coef_
            contribs = sorted(
                [(feat_names[i], coef[i] * X[0, i]) for i in nonzero],
                key=lambda x: abs(x[1]), reverse=True,
            )[:6]
        elif model_name == "LightGBM":
            # LightGBM không có coef_ (âm dương), chỉ có feature_importances_ (độ quan trọng >= 0)
            importance = model.feature_importances_
            contribs = sorted(
                [(feat_names[i], importance[i] * X[0, i]) for i in nonzero],
                key=lambda x: x[1], reverse=True,
            )[:6]

    return {
        "raw_pred": raw_pred,
        "clipped_pred": clipped,
        "rounded": int(round(clipped)),
        "processed": processed,
        "top_features": contribs,
    }


def render_stars(rating: float) -> str:
    full = int(rating)
    half = (rating - full) >= 0.5
    s = "★" * full
    if half and full < 5:
        s += "½"
        full += 1
    s += "☆" * (5 - full)
    return s


def rating_color(rating: float) -> str:
    if rating >= 4: return "#2ca02c"  # green
    if rating >= 3: return "#FFC72C"  # gold
    if rating >= 2: return "#FF8C00"  # orange
    return "#d62728"  # red


def rating_label(rating: float) -> str:
    if rating >= 4.5: return "Rất tốt"
    if rating >= 3.5: return "Tốt"
    if rating >= 2.5: return "Trung bình"
    if rating >= 1.5: return "Kém"
    return "Rất tệ"


# ==========================
# UI & APP LOGIC
# ==========================
try:
    vectorizer, models, metas = load_models()
except FileNotFoundError:
    st.error("Chưa có model. Hãy chạy code train cả Ridge và LightGBM trước.")
    st.stop()

# ----- SIDEBAR (Chọn mô hình) -----
with st.sidebar:
    st.title("⚙️ Cài đặt")
    selected_model_name = st.radio(
        "Chọn Mô Hình Dự Đoán:",
        options=["Ridge", "LightGBM"],
        index=0,
        help="Chuyển đổi qua lại để xem sự khác biệt giữa thuật toán Tuyến tính và thuật toán dạng Cây."
    )

    st.markdown("---")
    st.caption(
        "🔍 **Mẹo:** Ridge giải thích từ khóa tốt hơn (Âm/Dương), trong khi LightGBM đôi khi dự đoán điểm số nhạy hơn.")

# Lấy mô hình đang được active
active_model = models[selected_model_name]
active_meta = metas[selected_model_name]

# ----- HEADER -----
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title(f"⭐ Dự đoán bằng {selected_model_name}")
with col2:
    st.metric("R² (Test)", f"{active_meta['r2']:.3f}")
with col3:
    st.metric("RMSE", f"{active_meta['rmse']:.3f}")

st.caption(f"Đang sử dụng **{selected_model_name}** để dự đoán điểm đánh giá **1.0 → 5.0**.")

tab1, tab2, tab3 = st.tabs(["🔍 Dự đoán", "📂 Hàng loạt", "ℹ️ Giới thiệu mô hình"])

# ============ TAB 1 - SINGLE PREDICTION ============
with tab1:
    examples = {
        "(Tự nhập)": "",
        "VD 5★ — Rất tốt": "Sản phẩm cực tốt, đóng gói cẩn thận, ship nhanh, sẽ mua lại 💖",
        "VD 4★ — Tốt": "Hàng ok, giao đúng mô tả, ship hơi chậm chút",
        "VD 3★ — Trung bình": "Sản phẩm bình thường, không có gì nổi bật",
        "VD 2★ — Kém": "Hàng dùng được, nhưng giao chậm và đóng gói cẩu thả",
        "VD 1★ — Rất tệ": "Hàng giả, lừa đảo, shop bùng đơn, ko đáng tiền 😡",
        "🔄 Phủ định (positive)": "Không có gì để chê, sản phẩm rất tốt",
        "🔄 Phủ định (negative)": "Không đáng tiền, chất lượng kém",
        "😍 Emoji nhiều": "Tốt 👍💯❤️🌟😍🔥",
        "🗣 Slang tiếng Việt": "Shop ok, sp đẹp, ship nhanh, mn nên mua nha",
        "⚖️ Mixed sentiment": "Chất lượng tốt nhưng giao hàng quá chậm và đóng gói xấu",
    }
    chosen = st.selectbox("Ví dụ:", list(examples.keys()), label_visibility="collapsed")
    text = st.text_area(
        "Bình luận",
        value=examples[chosen],
        height=100,
        placeholder="Nhập bình luận đánh giá sản phẩm...",
        label_visibility="collapsed",
    )

    if st.button("🔍 Dự đoán Rating", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Vui lòng nhập bình luận.")
        else:
            result = predict_rating(text, active_model, vectorizer, selected_model_name)
            if result is None:
                st.warning("Bình luận trống sau khi xử lý.")
            else:
                rating = result["clipped_pred"]
                color = rating_color(rating)
                label = rating_label(rating)
                stars = render_stars(rating)

                # Hiển thị Điểm
                st.markdown(
                    f"""<div style='padding:30px;border-radius:12px;
                    background-color:{color}15;border-left:6px solid {color};
                    text-align:center;margin:15px 0;'>
                    <div style='font-size:56px;color:{color};line-height:1;'>{rating:.2f}</div>
                    <div style='font-size:13px;color:#666;margin-top:5px;'>trên 5.0</div>
                    <div style='font-size:38px;color:{color};margin-top:12px;letter-spacing:4px;'>{stars}</div>
                    <div style='font-size:18px;font-weight:bold;color:{color};margin-top:8px;'>{label}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Detail metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Raw prediction", f"{result['raw_pred']:.3f}")
                c2.metric("Sau clipping [1,5]", f"{result['clipped_pred']:.3f}")
                c3.metric("Làm tròn", f"{result['rounded']} ★")

                # Top features
                if result["top_features"]:
                    if selected_model_name == "Ridge":
                        st.markdown("**Top từ khóa tác động đến điểm (Tích cực / Tiêu cực):**")
                    else:
                        st.markdown("**Top từ khóa được LightGBM chú ý nhất (Độ quan trọng):**")
                        st.caption(
                            "*(Lưu ý: LightGBM chỉ đánh giá mức độ quan trọng, không phân biệt từ đó mang ý nghĩa âm hay dương)*")

                    fig, ax = plt.subplots(figsize=(8, 2.8))
                    words = [w for w, _ in result["top_features"]]
                    vals = [v for _, v in result["top_features"]]

                    # Set màu dựa theo mô hình
                    if selected_model_name == "Ridge":
                        colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
                    else:
                        colors = ["#1f77b4" for _ in vals]  # LightGBM chỉ dùng 1 màu xanh dương

                    ax.barh(words[::-1], vals[::-1], color=colors[::-1])
                    if selected_model_name == "Ridge":
                        ax.axvline(0, color="black", linewidth=0.5)
                        ax.set_xlabel("Đóng góp vào rating (+/-)", fontsize=9)
                    else:
                        ax.set_xlabel("Mức độ quan trọng (Feature Importance)", fontsize=9)

                    ax.tick_params(labelsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)

                with st.expander("Chi tiết kỹ thuật"):
                    st.caption(f"**Văn bản sau xử lý:** `{result['processed']}`")

# ============ TAB 2 - BATCH PREDICTION ============
with tab2:
    st.caption(f"Tải file CSV có cột **`Comment`** để dự đoán hàng loạt bằng **{selected_model_name}**.")
    uploaded = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        if "Comment" not in df.columns:
            st.error(f"Thiếu cột 'Comment'. Cột hiện có: {list(df.columns)}")
        else:
            st.success(f"Đã tải {len(df):,} dòng.")
            if st.button(f"🚀 Dự đoán bằng {selected_model_name}", type="primary", use_container_width=True):
                with st.spinner(f"Đang dự đoán {len(df):,} bình luận..."):
                    progress = st.progress(0)
                    ratings = []
                    for i, c in enumerate(df["Comment"].fillna("").astype(str)):
                        r = predict_rating(c, active_model, vectorizer, selected_model_name)
                        ratings.append(r["clipped_pred"] if r else None)
                        if i % 100 == 0:
                            progress.progress(i / len(df))
                    progress.progress(1.0)

                df["Predicted_Rating"] = ratings
                df["Predicted_Stars"] = [int(round(r)) if r else None for r in ratings]

                valid = df["Predicted_Rating"].dropna()
                if len(valid) > 0:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Trung bình", f"{valid.mean():.2f} ★")
                    c2.metric("Trung vị", f"{valid.median():.2f} ★")
                    c3.metric("Min", f"{valid.min():.2f}")
                    c4.metric("Max", f"{valid.max():.2f}")

                    # Distribution Chart
                    fig, ax = plt.subplots(figsize=(8, 3))
                    star_counts = df["Predicted_Stars"].value_counts().sort_index()
                    colors = {1: "#d62728", 2: "#FF8C00", 3: "#FFC72C", 4: "#7CB342", 5: "#2ca02c"}
                    ax.bar(
                        star_counts.index, star_counts.values,
                        color=[colors.get(s, "#999") for s in star_counts.index],
                        edgecolor="black",
                    )
                    ax.set_xlabel("Sao (làm tròn)")
                    ax.set_ylabel("Số lượng")
                    ax.set_xticks([1, 2, 3, 4, 5])
                    ax.set_title(f"Phân bố rating dự đoán ({selected_model_name})")
                    plt.tight_layout()
                    st.pyplot(fig)

                csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    "💾 Tải kết quả CSV",
                    csv,
                    file_name=f"predictions_{selected_model_name}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ============ TAB 3 - INTRO ============
with tab3:
    st.markdown(f"### Đang xem thông tin của: **{selected_model_name}**")
    st.markdown(
        f"""
| Thuộc tính | Giá trị |
|---|---|
| **Loại bài toán** | Regression (hồi quy) |
| **Mô hình** | {active_meta.get('model_type', selected_model_name)} |
| **Đặc trưng** | TF-IDF (1-2 gram, max {active_meta['config']['max_features']:,}) |
| **Vocabulary size** | {active_meta.get('vocab_size', 'N/A')} từ |
| **Dataset** | {active_meta.get('n_train', 0) + active_meta.get('n_test', 0):,} reviews |
| **MSE** | {active_meta.get('mse', 0):.4f} |
| **RMSE** | {active_meta.get('rmse', 0):.4f} sao |
| **MAE** | {active_meta.get('mae', 0):.4f} sao |
| **R²** | {active_meta.get('r2', 0):.4f} |
| **Accuracy ±0.5 sao** | {active_meta.get('acc_within_0_5', 0) * 100:.2f}% |
| **Accuracy ±1.0 sao** | {active_meta.get('acc_within_1', 0) * 100:.2f}% |
"""
    )

# ----- FOOTER -----
st.markdown("---")
st.caption(
    f"Mô hình đang chạy: {selected_model_name} · "
    f"Test R²: **{active_meta.get('r2', 0):.3f}** · "
    f"RMSE: **{active_meta.get('rmse', 0):.3f}** sao"
)
