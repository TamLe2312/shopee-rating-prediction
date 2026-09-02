# ============================================================
# LỆNH THỰC THI FILE
# ============================================================
# streamlit run shopee_rating_preprocessing/src/preprocessing/08_stopword_removing.py

# ============================================================
# THƯ VIỆN
# ============================================================
import os
import re
import pandas as pd
import streamlit as st
from shopee_rating_preprocessing.utils.present import print_header

# ============================================================
# BIẾN TOÀN CỤC
# ============================================================
try:
    from shopee_rating_preprocessing.config.paths import ROOT_DIR
except Exception as e:
    print("Lỗi khi import:", e)
    ROOT_DIR = os.getcwd()

# ============================================================
# CẤU HÌNH
# ============================================================
INPUT_DATA_PATH = f"{ROOT_DIR}/data/processed/06_word_segmented.csv"
OUTPUT_DATA_PATH = f"{ROOT_DIR}/data/processed/07_stopword_removed.csv"

# ============================================================
# REGEX
# ============================================================
SPACE_PATTERN = re.compile(r"\s+")
PUNCT_PATTERN = re.compile(r"[^\wÀ-ỹĐđ_]+", re.UNICODE)
UNDERSCORE_PATTERN = re.compile(r"_+")
TOKEN_PATTERN = re.compile(r"^[\wÀ-ỹĐđ]+(?:_[\wÀ-ỹĐđ]+)*$", re.UNICODE)

# ============================================================
# DANH SÁCH STOPWORD TIẾNG VIỆT
# ============================================================
STOPWORDS = {
    "à", "á", "ạ", "ả", "ã",
    "â", "ấ", "ậ", "ẩ", "ẫ",
    "ă", "ắ", "ặ", "ẳ", "ẵ",
    "è", "é", "ẹ", "ẻ", "ẽ",
    "ê", "ế", "ệ", "ể", "ễ",
    "ì", "í", "ị", "ỉ", "ĩ",
    "ò", "ó", "ọ", "ỏ", "õ",
    "ô", "ố", "ộ", "ổ", "ỗ",
    "ơ", "ớ", "ợ", "ở", "ỡ",
    "ù", "ú", "ụ", "ủ", "ũ",
    "ư", "ứ", "ự", "ử", "ữ",
    "ỳ", "ý", "ỵ", "ỷ", "ỹ",

    "bị", "bởi", "cả", "các", "cái", "càng", "chỉ", "cho",
    "chứ", "có", "còn", "cùng", "của", "cũng", "đã", "đang",
    "đây", "để", "đến", "đều", "đi", "đó", "do", "giữa",
    "hay", "hơn", "khi", "không", "là", "lại", "lên", "mà",
    "mỗi", "một", "này", "nên", "nếu", "như", "những", "ra",
    "rằng", "rất", "sau", "sẽ", "so", "tại", "theo", "thì",
    "trên", "trong", "từ", "và", "vào", "vẫn", "với", "về",
    "vì", "việc", "xem", "được",

    "ai", "anh", "chị", "em", "họ", "mình", "người", "ta",
    "tôi", "bạn", "ông", "bà", "cô", "chú", "bác", "nó",
    "hắn", "chúng", "chúng_ta", "chúng_tôi", "chúng_mình",

    "bao", "bao_nhiêu", "bằng", "bất", "bất_cứ", "bất_kỳ",
    "chẳng", "chưa", "đâu", "đừng", "hết", "hoặc", "mấy",
    "nào", "nấy", "nữa", "phải", "qua", "quá", "thật", "thế",
    "thì", "thôi", "từng", "vậy", "vẫn", "vốn",

    "mang", "mang_lại", "nhằm", "theo", "thuộc", "tới",
    "trước", "trong_khi", "bên", "bên_cạnh", "dưới", "ngoài",
    "giữa", "gần", "sang", "từ", "tại",

    "mới", "đã", "sắp", "sẽ", "đang", "vừa", "từng",
    "luôn", "thường", "thường_xuyên", "đôi_khi",

    "ơi", "ạ", "nhé", "nha", "nè", "ha", "hen", "hén",
    "nhỉ", "nhá", "thôi", "đấy", "đó",

    "ơi", "ừ", "ừm", "uh", "uhm", "à", "ờ", "ờm",
}

# ============================================================
# STOPWORD ĐẶC BIỆT KHÔNG XÓA
# ============================================================
KEEP_WORDS = {
    "không",
    "chẳng",
    "chả",
    "chưa",
    "đừng",
    "chớ",
    "đâu",
}

STOPWORDS -= KEEP_WORDS

# ============================================================
# NẠP DỮ LIỆU
# ============================================================
@st.cache_data
def load_input_data():
    if not os.path.exists(INPUT_DATA_PATH):
        return None
    return pd.read_csv(INPUT_DATA_PATH).copy()

# ============================================================
# LÀM SẠCH TOKEN
# ============================================================
def clean_token(token: str) -> str:
    if not token:
        return ""

    token = token.strip().lower()
    token = UNDERSCORE_PATTERN.sub("_", token)
    token = token.strip("_")

    if not token:
        return ""

    if TOKEN_PATTERN.fullmatch(token):
        return token

    return ""

# ============================================================
# XỬ LÝ STOPWORD + DẤU CÂU
# ============================================================
def remove_stopwords(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower().strip()
    text = PUNCT_PATTERN.sub(" ", text)
    text = SPACE_PATTERN.sub(" ", text).strip()

    if not text:
        return ""

    tokens = text.split()
    result = []

    for token in tokens:
        token = clean_token(token)

        if not token:
            continue

        if token in STOPWORDS:
            continue

        result.append(token)

    return " ".join(result)

# ============================================================
# XỬ LÝ DATAFRAME
# ============================================================
def process_dataframe(df: pd.DataFrame, progress_bar, status_text) -> pd.DataFrame:
    total_rows = len(df)
    comments = df["Comment"].tolist()
    processed_comments = []
    batch_size = 500

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)

        processed_comments.extend(
            remove_stopwords(comment)
            for comment in comments[start:end]
        )

        progress = int(end / total_rows * 90)
        progress_bar.progress(progress)
        status_text.text(f"Đang xử lý: {end:,}/{total_rows:,}")

    df = df.copy()
    df["Comment"] = processed_comments
    return df

# ============================================================
# MAIN - STREAMLIT
# ============================================================
def main():
    st.set_page_config(page_title="Vietnamese Stopword Removal")
    st.title("Xóa Stopword + Làm sạch dấu câu")
    st.markdown("---")

    df = load_input_data()

    if df is None or df.empty:
        st.error(
            f"Không tìm thấy file dữ liệu tại: `{INPUT_DATA_PATH}` "
            f"hoặc file bị trống!"
        )
        return

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**Tổng số dòng dữ liệu:** `{len(df):,}`")

    with col2:
        st.info(f"**Các cột hiện tại:** `{list(df.columns)}`")

    st.subheader("1. Dữ liệu trước khi xử lý")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("---")

    st.subheader("Các hạng mục xử lý")

    st.markdown(
        """
        Hệ thống xử lý cột `Comment` theo thứ tự:

        1. Chuyển văn bản về chữ thường.
        2. Thay dấu câu và ký hiệu bằng khoảng trắng.
        3. Chuẩn hóa khoảng trắng.
        4. Giữ nguyên `_` trong các token đã word segmentation.
        5. Xóa Stopword.
        6. Giữ lại các từ phủ định quan trọng.
        7. Loại token rỗng hoặc token không hợp lệ.
        8. Giữ nguyên toàn bộ các cột khác.
        """
    )

    st.markdown("---")

    with st.expander("Xem danh sách Stopword"):
        st.write(sorted(STOPWORDS))

    st.markdown("---")

    if st.button("Bắt đầu làm sạch + Xóa Stopword", type="primary"):
        print_header("Vietnamese stopword removal started via Streamlit...")

        if "Comment" not in df.columns:
            st.warning("Không tìm thấy cột 'Comment' trong tập dữ liệu.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Đang làm sạch dấu câu và xóa Stopword...")

        df_processed = process_dataframe(
            df,
            progress_bar,
            status_text
        )

        output_dir = os.path.dirname(OUTPUT_DATA_PATH)
        os.makedirs(output_dir, exist_ok=True)

        try:
            df_processed.to_csv(
                OUTPUT_DATA_PATH,
                index=False,
                encoding="utf-8-sig"
            )

            progress_bar.progress(100)
            status_text.text("Hoàn tất!")

            st.success(
                f"Đã hoàn tất làm sạch + xóa Stopword. "
                f"File kết quả: `{OUTPUT_DATA_PATH}`"
            )

        except Exception as e:
            st.error(f"Lỗi khi lưu file: {e}")
            return

        st.markdown("---")
        st.subheader("2. Dữ liệu sau khi xử lý")
        st.dataframe(
            df_processed.head(10),
            use_container_width=True
        )

# ============================================================
# CHẠY CHƯƠNG TRÌNH
# ============================================================
if __name__ == "__main__":
    main()