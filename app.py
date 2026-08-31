"""
Streamlit Demo - Dự đoán rating Shopee bằng Linear Regression / LightGBM
Chạy: streamlit run app.py
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure Matplotlib backend for Streamlit (Headless)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.preprocessing.text import preprocess_text

# ==========================
# CONFIG & THEME
# ==========================
# Shopee Theme Colors
SHOPEE_ORANGE = "#EE4D2D"
SHOPEE_ORANGE_LIGHT = "#FF7337"
SHOPEE_BG_LIGHT = "#FFF0EE"
SHOPEE_TEXT_DARK = "#000000"
SHOPEE_TEXT_MUTED = "#757575"

st.set_page_config(
    page_title="Shopee Rating Predictor",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for Shopee Theme
st.markdown(f"""
    <style>
    /* Primary Color Overrides */
    :root {{
        --primary-color: {SHOPEE_ORANGE};
    }}
    
    /* Headers and Text */
    h1, h2, h3 {{
        color: {SHOPEE_ORANGE} !important;
    }}
    
    /* Buttons */
    .stButton>button[data-baseweb="button"] {{
        background-color: {SHOPEE_ORANGE};
        color: white;
        border: none;
        border-radius: 4px;
        transition: background-color 0.3s;
    }}
    .stButton>button[data-baseweb="button"]:hover {{
        background-color: {SHOPEE_ORANGE_LIGHT};
        color: white;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }}
    .stTabs [aria-selected="true"] {{
        color: {SHOPEE_ORANGE} !important;
        border-bottom-color: {SHOPEE_ORANGE} !important;
    }}
    
    /* Radio Buttons */
    div[role="radiogroup"] label[data-baseweb="radio"] div:first-child {{
        background-color: {SHOPEE_ORANGE} !important;
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background-color: {SHOPEE_ORANGE} !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        color: {SHOPEE_ORANGE};
    }}
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_models():
    """Load Vectorizers và cả 2 mô hình cùng lúc vào bộ nhớ tạm"""

    # 1. Load Vectorizers riêng biệt
    vec_linear_path = os.path.join(MODEL_DIR, "rating_vectorizer_linear.pkl")
    vec_lgbm_path = os.path.join(MODEL_DIR, "rating_vectorizer_lightgbm.pkl")

    if not os.path.exists(vec_linear_path) or not os.path.exists(vec_lgbm_path):
        raise FileNotFoundError("Thiếu file vectorizer riêng biệt cho Linear hoặc LightGBM.")

    # 2. Load Linear Regression
    linear_path = os.path.join(MODEL_DIR, "rating_model_linear.pkl")
    linear_meta_path = os.path.join(MODEL_DIR, "rating_metadata_linear.json")
    if os.path.exists(linear_path) and os.path.exists(linear_meta_path):
        model_linear = joblib.load(linear_path)
        with open(linear_meta_path, "r", encoding="utf-8") as f:
            meta_linear = json.load(f)
    else:
        model_linear = None
        meta_linear = {"r2": 0.0, "rmse": 0.0, "model_type": "Linear Regression (Missing)"}

    # 3. Load LightGBM
    lgbm_path = os.path.join(MODEL_DIR, "rating_model_lightgbm.pkl")
    lgbm_meta_path = os.path.join(MODEL_DIR, "rating_metadata_lightgbm.json")
    if os.path.exists(lgbm_path) and os.path.exists(lgbm_meta_path):
        model_lgbm = joblib.load(lgbm_path)
        with open(lgbm_meta_path, "r", encoding="utf-8") as f:
            meta_lgbm = json.load(f)
    else:
        model_lgbm = None
        meta_lgbm = {"r2": 0.0, "rmse": 0.0, "model_type": "LightGBM (Missing)"}

    return (
        {
            "Linear": joblib.load(vec_linear_path),
            "LightGBM": joblib.load(vec_lgbm_path)
        },
        {"Linear": model_linear, "LightGBM": model_lgbm},
        {"Linear": meta_linear, "LightGBM": meta_lgbm}
    )


def predict_rating(text, model, vectorizer, model_name):
    if model is None:
        return None

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
        if model_name == "Linear":
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
    if rating >= 4: return "#00bfa5"  # Shopee Mint Green for good
    if rating >= 3: return "#ffc107"  # Yellow for average
    if rating >= 2: return "#ff9800"  # Orange
    return "#ff5722"  # Deep orange/red for bad


def rating_label(rating: float) -> str:
    if rating >= 4.5: return "Tuyệt vời"
    if rating >= 3.5: return "Hài lòng"
    if rating >= 2.5: return "Bình thường"
    if rating >= 1.5: return "Không hài lòng"
    return "Rất tệ"


# ==========================
# UI & APP LOGIC
# ==========================
try:
    vectorizer, models, metas = load_models()
except FileNotFoundError as e:
    st.error(f"Lỗi tải mô hình: {e}. Hãy chạy code train Linear và LightGBM trước.")
    st.stop()
except Exception as e:
    st.error(f"Đã xảy ra lỗi: {e}")
    st.stop()

# ----- SIDEBAR (Chọn mô hình) -----
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Shopee.svg/3840px-Shopee.svg.png?utm_source=vi.wikipedia.org&utm_campaign=index&utm_content=thumbnail",
        width=150)
    st.title("⚙️ Cấu hình")
    selected_model_name = st.radio(
        "Thuật toán:",
        options=["Linear", "LightGBM"],
        index=0,
        help="Chuyển đổi qua lại để xem sự khác biệt giữa thuật toán Tuyến tính và dạng Cây."
    )
# Lấy mô hình đang được active
active_model = models[selected_model_name]
active_meta = metas[selected_model_name]

if active_model is None:
    st.warning(
        f"Mô hình **{selected_model_name}** chưa được huấn luyện hoặc không tìm thấy file. Vui lòng train mô hình trước.")
    st.stop()

# ----- HEADER -----
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.title("Phân Tích Đánh Giá Bình Luận")
with col2:
    st.metric("Độ chính xác (R²)", f"{active_meta.get('r2', 0):.3f}")
with col3:
    st.metric("Sai số (RMSE)", f"{active_meta.get('rmse', 0):.3f}")

st.caption(f"Sử dụng thuật toán **{selected_model_name}** để dự đoán rating từ 1 đến 5 sao.")

tab1, tab2, tab3 = st.tabs(["💬 Phân Tích Đơn", "📁 Phân Tích Hàng Loạt", "ℹ️ Chi Tiết Mô Hình"])

# ============ TAB 1 - SINGLE PREDICTION ============
with tab1:
    examples = {
        "Viết đánh giá của riêng bạn...": "",
        "Đánh giá 5 sao": "Sản phẩm xịn xò, giao hàng hỏa tốc, shop tư vấn nhiệt tình 💖",
        "Đánh giá 4 sao": "Chất lượng ổn áp so với tầm giá, nhưng hộp hơi móp",
        "Đánh giá 3 sao": "Tạm được, không giống ảnh lắm nhưng vẫn dùng được",
        "Đánh giá 2 sao": "Giao nhầm size, nhắn tin shop không rep nhanh",
        "Đánh giá 1 sao": "Hàng pha ke, chất vải nóng, khuyên mọi người né gấp 😡",
        "Nhiều biểu tượng cảm xúc": "Quá tuyệt vời luôn shop ơi 👍💯❤️🌟😍🔥",
    }
    chosen = st.selectbox("Chọn mẫu đánh giá:", list(examples.keys()), label_visibility="collapsed")
    text = st.text_area(
        "Nội dung đánh giá",
        value=examples[chosen],
        height=120,
        placeholder="Nhập trải nghiệm mua hàng của bạn tại đây...",
        label_visibility="collapsed",
    )

    if st.button("🚀 Dự Đoán Rating", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Vui lòng nhập nội dung đánh giá để phân tích.")
        else:
            result = predict_rating(text, active_model, vectorizer, selected_model_name)
            if result is None:
                st.warning("Nội dung sau khi lọc rác quá ngắn để phân tích.")
            else:
                rating = result["clipped_pred"]
                color = rating_color(rating)
                label = rating_label(rating)
                stars = render_stars(rating)

                # Hiển thị Điểm với phong cách Shopee Card
                st.markdown(
                    f"""<div style='padding:20px; border-radius:8px; border: 1px solid #f9ecea;
                    background-color:{SHOPEE_BG_LIGHT}; box-shadow: 0 1px 2px 0 rgba(0,0,0,.05);
                    text-align:center; margin:15px 0;'>
                    <div style='font-size:16px; color:{SHOPEE_TEXT_MUTED}; font-weight: 500; text-transform: uppercase;'>Hệ thống đánh giá</div>
                    <div style='font-size:64px; color:{SHOPEE_ORANGE}; line-height:1; font-weight: 700; margin-top: 10px;'>{rating:.1f}<span style='font-size: 24px; color: {SHOPEE_TEXT_MUTED}'> / 5</span></div>
                    <div style='font-size:42px; color:{SHOPEE_ORANGE}; margin-top:5px; letter-spacing:2px;'>{stars}</div>
                    <div style='font-size:20px; font-weight:600; color:{SHOPEE_TEXT_DARK}; margin-top:10px;'>{label}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Detail metrics
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Dự đoán thô", f"{result['raw_pred']:.2f}")
                c2.metric("Sau chuẩn hóa", f"{result['clipped_pred']:.2f}")
                c3.metric("Sao hiển thị", f"{result['rounded']} ⭐")

                # Top features
                if result["top_features"]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if selected_model_name == "Linear":
                        st.markdown(f"<h4 style='color:{SHOPEE_TEXT_DARK}'>Từ khóa tác động mạnh nhất</h4>",
                                    unsafe_allow_html=True)
                        st.caption("Màu cam/xanh thể hiện chiều hướng tác động (Tích cực/Tiêu cực) lên số sao.")
                    else:
                        st.markdown(f"<h4 style='color:{SHOPEE_TEXT_DARK}'>Từ khóa được chú ý nhất</h4>",
                                    unsafe_allow_html=True)
                        st.caption("LightGBM tính toán độ quan trọng của từ khóa, không phân biệt âm dương.")

                    fig, ax = plt.subplots(figsize=(8, 3))
                    words = [w for w, _ in result["top_features"]]
                    vals = [v for _, v in result["top_features"]]

                    # Set màu dựa theo mô hình (Sử dụng màu Shopee)
                    if selected_model_name == "Linear":
                        colors = [SHOPEE_ORANGE if v > 0 else "#424242" for v in vals]
                    else:
                        colors = [SHOPEE_ORANGE for _ in vals]

                    bars = ax.barh(words[::-1], vals[::-1], color=colors[::-1], height=0.6)

                    # Remove borders for cleaner look
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)

                    if selected_model_name == "Linear":
                        ax.axvline(0, color=SHOPEE_TEXT_MUTED, linewidth=1, linestyle="--")
                        ax.set_xlabel("Mức độ đóng góp (+/-)", fontsize=10, color=SHOPEE_TEXT_MUTED)
                    else:
                        ax.set_xlabel("Độ quan trọng", fontsize=10, color=SHOPEE_TEXT_MUTED)

                    ax.tick_params(labelsize=10, colors=SHOPEE_TEXT_DARK, bottom=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)  # Prevent memory leaks

                with st.expander("🛠️ Xem văn bản sau khi tiền xử lý"):
                    st.info(f"`{result['processed']}`")

# ============ TAB 2 - BATCH PREDICTION ============
with tab2:
    st.markdown("### Phân tích File CSV")
    st.write(f"Tải lên tệp danh sách đánh giá để hệ thống dự đoán tự động bằng **{selected_model_name}**.")
    st.info("💡 Lưu ý: Tệp CSV của bạn bắt buộc phải có cột tên là **`Comment`** chứa nội dung bình luận.")

    uploaded = st.file_uploader("Kéo thả file CSV vào đây", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        if "Comment" not in df.columns:
            st.error(f"❌ Không tìm thấy cột 'Comment'. Các cột hiện có: {', '.join(list(df.columns))}")
        else:
            st.success(f"✅ Đã tải thành công **{len(df):,}** đánh giá.")
            if st.button(f"🚀 Bắt đầu phân tích ({selected_model_name})", type="primary", use_container_width=True):
                with st.spinner("Hệ thống đang xử lý..."):
                    progress = st.progress(0)
                    ratings = []
                    for i, c in enumerate(df["Comment"].fillna("").astype(str)):
                        r = predict_rating(c, active_model, vectorizer, selected_model_name)
                        ratings.append(r["clipped_pred"] if r else None)
                        if i % max(1, len(df) // 100) == 0:
                            progress.progress(min(1.0, i / len(df)))
                    progress.progress(1.0)

                df["Predicted_Rating"] = ratings
                df["Predicted_Stars"] = [int(round(r)) if r else None for r in ratings]

                valid = df["Predicted_Rating"].dropna()
                if len(valid) > 0:
                    st.markdown("### Thống kê kết quả")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Trung bình", f"{valid.mean():.2f} ⭐")
                    c2.metric("Trung vị", f"{valid.median():.2f} ⭐")
                    c3.metric("Thấp nhất", f"{valid.min():.2f}")
                    c4.metric("Cao nhất", f"{valid.max():.2f}")

                    # Distribution Chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    star_counts = df["Predicted_Stars"].value_counts().sort_index()

                    # Ensure all 1-5 exist for chart
                    for i in range(1, 6):
                        if i not in star_counts:
                            star_counts[i] = 0
                    star_counts = star_counts.sort_index()

                    colors_map = {1: "#424242", 2: "#757575", 3: "#ffc107", 4: SHOPEE_ORANGE_LIGHT, 5: SHOPEE_ORANGE}
                    bar_colors = [colors_map.get(s, "#999") for s in star_counts.index]

                    ax.bar(
                        star_counts.index, star_counts.values,
                        color=bar_colors,
                        edgecolor="white",
                        width=0.6
                    )

                    # Clean up chart appearance
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xlabel("Số sao dự đoán", color=SHOPEE_TEXT_MUTED)
                    ax.set_ylabel("Số lượng đánh giá", color=SHOPEE_TEXT_MUTED)
                    ax.set_xticks([1, 2, 3, 4, 5])
                    ax.tick_params(colors=SHOPEE_TEXT_DARK)
                    ax.set_title("Phân bố đánh giá", pad=20, color=SHOPEE_TEXT_DARK, weight='bold')

                    # Add value labels on top of bars
                    for i, v in enumerate(star_counts.values):
                        ax.text(i + 1, v + (max(star_counts.values) * 0.02), str(v), ha='center',
                                color=SHOPEE_TEXT_MUTED)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    "📥 Tải Về Kết Quả (CSV)",
                    csv,
                    file_name=f"shopee_predictions_{selected_model_name.lower()}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ============ TAB 3 - INTRO ============
with tab3:
    st.markdown(f"### Thông số mô hình: <span style='color:{SHOPEE_ORANGE}'>{selected_model_name}</span>",
                unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #eee;">
        <table style="width:100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; color: #757575; font-weight: 500;">Thuật toán cốt lõi</td>
                <td style="padding: 10px 0; text-align: right; font-weight: 600;">{active_meta.get('model_type', selected_model_name)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; color: #757575; font-weight: 500;">Phương pháp xử lý từ</td>
                <td style="padding: 10px 0; text-align: right; font-weight: 600;">TF-IDF</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; color: #757575; font-weight: 500;">Số lượng từ vựng (Vocabulary)</td>
                <td style="padding: 10px 0; text-align: right; font-weight: 600;">{active_meta.get('vocab_size', 'N/A')} từ</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; color: #757575; font-weight: 500;">Kích thước dữ liệu huấn luyện</td>
                <td style="padding: 10px 0; text-align: right; font-weight: 600;">{active_meta.get('n_train', 0) + active_meta.get('n_test', 0):,} đánh giá</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; color: #757575; font-weight: 500;">Sai số bình phương (MSE)</td>
                <td style="padding: 10px 0; text-align: right; font-weight: 600;">{active_meta.get('mse', 0):.4f}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; color: #757575; font-weight: 500;">Sai số trung bình (MAE)</td>
                <td style="padding: 10px 0; text-align: right; font-weight: 600;">{active_meta.get('mae', 0):.4f} sao</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; color: {SHOPEE_ORANGE}; font-weight: bold;">Tỷ lệ dự đoán đúng (±0.5 sao)</td>
                <td style="padding: 10px 0; text-align: right; font-weight: bold; color: {SHOPEE_ORANGE};">{active_meta.get('acc_within_0_5', 0) * 100:.2f}%</td>
            </tr>
            <tr>
                <td style="padding: 10px 0; color: {SHOPEE_ORANGE}; font-weight: bold;">Tỷ lệ dự đoán đúng (±1.0 sao)</td>
                <td style="padding: 10px 0; text-align: right; font-weight: bold; color: {SHOPEE_ORANGE};">{active_meta.get('acc_within_1', 0) * 100:.2f}%</td>
            </tr>
        </table>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----- FOOTER -----
st.markdown("<br><hr style='margin-top: 40px; margin-bottom: 20px; border-color: #eee;'>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align: center; color: {SHOPEE_TEXT_MUTED}; font-size: 12px;'>"
    f"Đang sử dụng: <strong>{selected_model_name}</strong> | "
    f"Độ chính xác R²: <strong>{active_meta.get('r2', 0):.3f}</strong> | "
    f"Sai số RMSE: <strong>{active_meta.get('rmse', 0):.3f}</strong> sao"
    f"</div>",
    unsafe_allow_html=True
)
