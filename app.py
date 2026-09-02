"""
Streamlit Frontend - Giao diện dự đoán rating Shopee (Gọi qua FastAPI)
Chạy: streamlit run app.py
"""
import os
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# ==========================
# CONFIG & THEME
# ==========================
# Địa chỉ của FastAPI Backend
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

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
    :root {{ --primary-color: {SHOPEE_ORANGE}; }}
    h1, h2, h3 {{ color: {SHOPEE_ORANGE} !important; }}
    .stButton>button[data-baseweb="button"] {{
        background-color: {SHOPEE_ORANGE}; color: white; border: none; border-radius: 4px; transition: 0.3s;
    }}
    .stButton>button[data-baseweb="button"]:hover {{
        background-color: {SHOPEE_ORANGE_LIGHT}; color: white;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; padding: 10px 0;
    }}
    .stTabs [aria-selected="true"] {{ color: {SHOPEE_ORANGE} !important; border-bottom-color: {SHOPEE_ORANGE} !important; }}
    div[role="radiogroup"] label[data-baseweb="radio"] div:first-child {{ background-color: {SHOPEE_ORANGE} !important; }}
    .stProgress > div > div > div > div {{ background-color: {SHOPEE_ORANGE} !important; }}
    .streamlit-expanderHeader {{ color: {SHOPEE_ORANGE}; }}
    </style>
    """, unsafe_allow_html=True)


# ==========================
# HELPER FUNCTIONS
# ==========================
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
    if rating >= 4: return "#00bfa5"
    if rating >= 3: return "#ffc107"
    if rating >= 2: return "#ff9800"
    return "#ff5722"


def rating_label(rating: float) -> str:
    if rating >= 4.5: return "Tuyệt vời"
    if rating >= 3.5: return "Hài lòng"
    if rating >= 2.5: return "Bình thường"
    if rating >= 1.5: return "Không hài lòng"
    return "Rất tệ"


# ==========================
# UI & APP LOGIC
# ==========================

# ----- SIDEBAR -----
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Shopee.svg/3840px-Shopee.svg.png", width=150)
    st.title("Cấu hình API")
    selected_model_name = st.radio(
        "Gửi request yêu cầu mô hình:",
        options=["Linear", "LightGBM"],
        index=0,
        help="Chọn mô hình mà Backend FastAPI sẽ sử dụng để dự đoán."
    )

# ----- HEADER -----
st.title("Phân Tích Đánh Giá Bình Luận")
st.caption(f"Yêu cầu dự đoán qua API bằng thuật toán **{selected_model_name}**.")

tab1, tab2, tab3 = st.tabs(["💬 Phân Tích Đơn", "📁 Phân Tích Hàng Loạt", "ℹ️ Cấu Trúc Hệ Thống"])

# ============ TAB 1 - SINGLE PREDICTION ============
with tab1:
    examples = {
        "Viết đánh giá của riêng bạn...": "",
        "Đánh giá 5 sao": "Sản phẩm xịn xò, giao hàng hỏa tốc, shop tư vấn nhiệt tình 💖",
        "Đánh giá 4 sao": "Chất lượng ổn áp so với tầm giá, nhưng hộp hơi móp",
        "Đánh giá 3 sao": "Tạm được, không giống ảnh lắm nhưng vẫn dùng được",
        "Đánh giá 2 sao": "Giao nhầm size, nhắn tin shop không rep nhanh",
        "Đánh giá 1 sao": "Hàng pha ke, chất vải nóng, khuyên mọi người né gấp 😡",
    }
    chosen = st.selectbox("Chọn mẫu đánh giá:", list(examples.keys()), label_visibility="collapsed")
    text = st.text_area(
        "Nội dung đánh giá", value=examples[chosen], height=120,
        placeholder="Nhập trải nghiệm mua hàng của bạn tại đây...", label_visibility="collapsed",
    )

    if st.button("🚀 Gửi Yêu Cầu Dự Đoán", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Vui lòng nhập nội dung đánh giá.")
        else:
            with st.spinner("Đang chờ FastAPI phản hồi..."):
                try:
                    # GỌI API THAY VÌ CHẠY HÀM LOCAL
                    payload = {"text": text, "model_type": selected_model_name}
                    response = requests.post(API_URL, json=payload, timeout=5)
                    response.raise_for_status()

                    result = response.json()

                    rating = result["clipped_prediction"]
                    color = rating_color(rating)
                    label = rating_label(rating)
                    stars = render_stars(rating)

                    st.markdown(
                        f"""<div style='padding:20px; border-radius:8px; border: 1px solid #f9ecea;
                        background-color:{SHOPEE_BG_LIGHT}; box-shadow: 0 1px 2px 0 rgba(0,0,0,.05);
                        text-align:center; margin:15px 0;'>
                        <div style='font-size:16px; color:{SHOPEE_TEXT_MUTED}; font-weight: 500; text-transform: uppercase;'>API Trả Về</div>
                        <div style='font-size:64px; color:{SHOPEE_ORANGE}; line-height:1; font-weight: 700; margin-top: 10px;'>{rating:.1f}<span style='font-size: 24px; color: {SHOPEE_TEXT_MUTED}'> / 5</span></div>
                        <div style='font-size:42px; color:{SHOPEE_ORANGE}; margin-top:5px; letter-spacing:2px;'>{stars}</div>
                        <div style='font-size:20px; font-weight:600; color:{SHOPEE_TEXT_DARK}; margin-top:10px;'>{label}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Dự đoán thô", f"{result['raw_prediction']:.2f}")
                    c2.metric("Sau chuẩn hóa", f"{result['clipped_prediction']:.2f}")
                    c3.metric("Sao hiển thị", f"{result['stars']} ⭐")

                    with st.expander("🛠️ Văn bản đã lọc rác (Tiền xử lý tại Backend)"):
                        st.info(f"`{result['processed_text']}`")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Không thể kết nối tới Backend. Hãy chắc chắn bạn đã chạy lệnh `uvicorn api:app --port 8000` ở một terminal khác.")
                except Exception as e:
                    st.error(f"❌ Lỗi API: {e}")

# ============ TAB 2 - BATCH PREDICTION ============
with tab2:
    st.markdown("### Phân tích File CSV (Gửi API hàng loạt)")
    st.write("Tải lên tệp danh sách đánh giá. Frontend sẽ gửi từng dòng lên FastAPI để xử lý.")
    uploaded = st.file_uploader("Kéo thả file CSV vào đây", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        if "Comment" not in df.columns:
            st.error("❌ Không tìm thấy cột 'Comment'.")
        else:
            st.success(f"✅ Đã tải thành công **{len(df):,}** đánh giá.")
            if st.button(f"🚀 Bắt đầu gọi API ({selected_model_name})", type="primary", use_container_width=True):
                with st.spinner("Đang giao tiếp với FastAPI..."):
                    progress = st.progress(0)
                    ratings = []

                    for i, c in enumerate(df["Comment"].fillna("").astype(str)):
                        try:
                            # Gọi API cho từng dòng
                            payload = {"text": c, "model_type": selected_model_name}
                            res = requests.post(API_URL, json=payload, timeout=5)
                            if res.status_code == 200:
                                ratings.append(res.json()["clipped_prediction"])
                            else:
                                ratings.append(None)
                        except:
                            ratings.append(None)

                        if i % max(1, len(df) // 100) == 0:
                            progress.progress(min(1.0, i / len(df)))

                    progress.progress(1.0)

                df["Predicted_Rating"] = ratings
                df["Predicted_Stars"] = [int(round(r)) if pd.notnull(r) else None for r in ratings]
                valid = df["Predicted_Rating"].dropna()

                if len(valid) > 0:
                    st.markdown("### Thống kê kết quả từ Backend")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Trung bình", f"{valid.mean():.2f} ⭐")
                    c2.metric("Trung vị", f"{valid.median():.2f} ⭐")
                    c3.metric("Thấp nhất", f"{valid.min():.2f}")
                    c4.metric("Cao nhất", f"{valid.max():.2f}")

                    # Distribution Chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    star_counts = df["Predicted_Stars"].value_counts().sort_index()
                    for i in range(1, 6):
                        if i not in star_counts: star_counts[i] = 0
                    star_counts = star_counts.sort_index()

                    colors_map = {1: "#424242", 2: "#757575", 3: "#ffc107", 4: SHOPEE_ORANGE_LIGHT, 5: SHOPEE_ORANGE}
                    ax.bar(star_counts.index, star_counts.values,
                           color=[colors_map.get(s, "#999") for s in star_counts.index], edgecolor="white", width=0.6)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xticks([1, 2, 3, 4, 5])

                    for i, v in enumerate(star_counts.values):
                        ax.text(i + 1, v + (max(star_counts.values) * 0.02), str(v), ha='center',
                                color=SHOPEE_TEXT_MUTED)

                    st.pyplot(fig)
                    plt.close(fig)

                st.download_button(
                    "📥 Tải Về Kết Quả (CSV)",
                    df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                    file_name=f"api_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
# ============ TAB 3 - CHI TIẾT MÔ HÌNH ============
with tab3:
    st.markdown(f"### Thông số kỹ thuật: <span style='color:{SHOPEE_ORANGE}'>{selected_model_name}</span>",
                unsafe_allow_html=True)

    try:
        # Đọc trực tiếp file metadata JSON lưu thông số trong quá trình huấn luyện
        import os, json

        meta_path = os.path.join("models", f"rating_metadata_{selected_model_name.lower()}.json")

        with open(meta_path, "r", encoding="utf-8") as f:
            active_meta = json.load(f)

        st.markdown(
            f"""
            <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #eee;">
            <table style="width:100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px 0; color: #757575; font-weight: 500;">Độ phù hợp (R²)</td>
                    <td style="padding: 10px 0; text-align: right; font-weight: 600;">{active_meta.get('r2', 0):.4f}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px 0; color: #757575; font-weight: 500;">Sai số bình phương (MSE)</td>
                    <td style="padding: 10px 0; text-align: right; font-weight: 600;">{active_meta.get('mse', 0):.4f}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px 0; color: #757575; font-weight: 500;">Sai số chuẩn (RMSE)</td>
                    <td style="padding: 10px 0; text-align: right; font-weight: 600;">{active_meta.get('rmse', 0):.4f} sao</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px 0; color: #757575; font-weight: 500;">Sai số tuyệt đối (MAE)</td>
                    <td style="padding: 10px 0; text-align: right; font-weight: 600;">{active_meta.get('mae', 0):.4f} sao</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px 0; color: {SHOPEE_ORANGE}; font-weight: bold;">Chính xác (±0.5 sao)</td>
                    <td style="padding: 10px 0; text-align: right; font-weight: bold; color: {SHOPEE_ORANGE};">{active_meta.get('acc_within_0_5', 0) * 100:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding: 10px 0; color: {SHOPEE_ORANGE}; font-weight: bold;">Chính xác (±1.0 sao)</td>
                    <td style="padding: 10px 0; text-align: right; font-weight: bold; color: {SHOPEE_ORANGE};">{active_meta.get('acc_within_1', 0) * 100:.2f}%</td>
                </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Tự động trích xuất và hiển thị các tham số thiết lập (nếu có trong key 'config')
        if "config" in active_meta:
            with st.expander("🔍 Xem thêm cấu hình tham số (Hyperparameters)"):
                st.json(active_meta["config"])

    except FileNotFoundError:
        st.warning(
            f"⚠️ Không tìm thấy file `{meta_path}`. Vui lòng đảm bảo bạn đã chạy file huấn luyện để sinh ra file này.")
    except Exception as e:
        st.error(f"❌ Có lỗi khi đọc thông số mô hình: {e}")
