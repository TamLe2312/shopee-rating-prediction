# ============================================================
# LỆNH THỰC THI FILE
# ============================================================
# streamlit run shopee_rating_preprocessing/src/preprocessing/07_word_segmentation.py

# ============================================================
# THƯ VIỆN
# ============================================================
import os
import re
import pandas as pd
import streamlit as st
from underthesea import word_tokenize
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
INPUT_DATA_PATH = f"{ROOT_DIR}/data/processed/05_gemini_corrected.csv"
OUTPUT_DATA_PATH = f"{ROOT_DIR}/data/processed/06_word_segmented.csv"
SEGMENT_PATTERN = re.compile(r"[\wÀ-ỹĐđ]+(?:_[\wÀ-ỹĐđ]+)+", re.UNICODE)
WORD_PATTERN = re.compile(r"^\w", re.UNICODE)
SPACE_PATTERN = re.compile(r"\s+")

# ============================================================
# DANH SÁCH TỪ / CỤM TỪ PHỦ ĐỊNH
# ============================================================
NEGATION_SINGLE = {
    "không", "chẳng", "chả", "chưa", "đừng", "chớ", "đâu",
}

NEGATION_PHRASES = {
    "không hề", "chẳng hề", "chả hề", "chưa hề",
    "không có", "chẳng có", "chả có",
    "không hề có", "chẳng hề có", "chả hề có",
    "chưa có",
    "không phải", "chẳng phải", "chả phải",
    "không phải là", "chẳng phải là", "chả phải là",
    "không thể", "chẳng thể", "chả thể",
    "không thể nào", "chẳng thể nào", "chả thể nào",
    "không muốn", "chẳng muốn", "chả muốn",
    "không định", "chẳng định", "chả định",
    "không nên", "chẳng nên", "chả nên",
    "không được", "chẳng được", "chả được",
    "không còn", "chẳng còn", "chả còn",
    "không còn nữa", "chẳng còn nữa", "chả còn nữa",
    "không bao giờ", "chẳng bao giờ", "chả bao giờ",
    "chưa bao giờ", "chưa từng",
    "không hẳn", "chẳng hẳn", "chả hẳn",
    "không hoàn toàn", "chẳng hoàn toàn", "chả hoàn toàn",
    "không nhất thiết", "chẳng nhất thiết", "chả nhất thiết",
    "không cần", "chẳng cần", "chả cần",
    "không thích", "chẳng thích", "chả thích",
    "không chút nào", "chẳng chút nào", "chả chút nào",
    "không một chút nào", "chẳng một chút nào", "chả một chút nào",
    "không tí nào", "chẳng tí nào", "chả tí nào",
    "không tý nào", "chẳng tý nào", "chả tý nào",
    "không đời nào", "chẳng đời nào", "chả đời nào",
}

STRONG_NEGATIONS = {
    "hoàn toàn không", "hoàn toàn chẳng", "hoàn toàn chả",
    "tuyệt đối không", "tuyệt đối chẳng", "tuyệt đối chả",
    "nhất quyết không", "nhất quyết chẳng", "nhất quyết chả",
    "nhất định không", "nhất định chẳng", "nhất định chả",
    "hoàn toàn không hề", "hoàn toàn chẳng hề", "hoàn toàn chả hề",
    "tuyệt đối không hề", "tuyệt đối chẳng hề", "tuyệt đối chả hề",
    "không đời nào", "chẳng đời nào", "chả đời nào",
    "không bao giờ", "chẳng bao giờ", "chả bao giờ",
}

ALL_NEGATIONS = NEGATION_SINGLE | NEGATION_PHRASES | STRONG_NEGATIONS
NEGATION_TUPLES = {
    tuple(phrase.split())
    for phrase in ALL_NEGATIONS
}
NEGATION_LENGTHS = sorted(
    {len(phrase) for phrase in NEGATION_TUPLES},
    reverse=True
)
MAX_NEGATION_LENGTH = NEGATION_LENGTHS[0]

# ============================================================
# INDEX PHỦ ĐỊNH THEO TOKEN ĐẦU
# ============================================================
NEGATION_INDEX = {}

for phrase_tokens in NEGATION_TUPLES:
    first_token = phrase_tokens[0]
    NEGATION_INDEX.setdefault(first_token, set()).add(phrase_tokens)

for first_token in NEGATION_INDEX:
    NEGATION_INDEX[first_token] = sorted(
        NEGATION_INDEX[first_token],
        key=len,
        reverse=True
    )

# ============================================================
# NẠP DỮ LIỆU
# ============================================================
@st.cache_data
def load_input_data():
    if not os.path.exists(INPUT_DATA_PATH):
        return None
    return pd.read_csv(INPUT_DATA_PATH).copy()

# ============================================================
# KIỂM TRA TOKEN ĐÃ ĐƯỢC SEGMENT
# ============================================================
def has_existing_segment(token: str) -> bool:
    return bool(token and "_" in token and SEGMENT_PATTERN.fullmatch(token))

# ============================================================
# WORD SEGMENTATION
# ============================================================
def segment_vietnamese(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    text = SPACE_PATTERN.sub(" ", text.strip())
    raw_tokens = text.split()
    result = []
    normal_buffer = []

    def flush_buffer():
        if not normal_buffer:
            return
        chunk = " ".join(normal_buffer)
        try:
            result.extend(word_tokenize(chunk, format="text").split())
        except Exception as e:
            print(f"Lỗi word segmentation: {e}")
            result.extend(normal_buffer)
        normal_buffer.clear()

    for token in raw_tokens:
        if has_existing_segment(token):
            flush_buffer()
            result.append(token)
        else:
            normal_buffer.append(token)

    flush_buffer()
    return " ".join(result)

# ============================================================
# TÌM CỤM PHỦ ĐỊNH DÀI NHẤT
# ============================================================
def find_longest_negation(tokens, start):
    candidates = NEGATION_INDEX.get(tokens[start])
    if not candidates:
        return None, 0

    remaining = len(tokens) - start

    for phrase_tokens in candidates:
        phrase_len = len(phrase_tokens)
        if phrase_len > remaining:
            continue
        if tuple(tokens[start:start + phrase_len]) == phrase_tokens:
            return phrase_tokens, phrase_len

    return None, 0

# ============================================================
# GẮN PHỦ ĐỊNH
# ============================================================
def attach_negation(segmented_text: str) -> str:
    if not segmented_text:
        return ""

    tokens = segmented_text.split()
    result = []
    i = 0
    n = len(tokens)

    while i < n:
        negation_tokens, negation_len = find_longest_negation(tokens, i)

        if negation_tokens:
            next_pos = i + negation_len

            if next_pos < n and WORD_PATTERN.match(tokens[next_pos]):
                result.append("_".join(tokens[i:next_pos + 1]))
                i = next_pos + 1
                continue

            result.append("_".join(negation_tokens))
            i += negation_len
            continue

        result.append(tokens[i])
        i += 1

    return " ".join(result)

# ============================================================
# XỬ LÝ TEXT
# ============================================================
def process_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return attach_negation(segment_vietnamese(text))

# ============================================================
# MAIN - STREAMLIT
# ============================================================
def main():
    st.set_page_config(page_title="Vietnamese Word Segmentation", layout="wide")
    st.title("Tách từ tiếng Việt + Gắn từ phủ định")
    st.markdown("---")

    df_word_segmentation = load_input_data()

    if df_word_segmentation is None or df_word_segmentation.empty:
        st.error(
            f"Không tìm thấy file dữ liệu tại: `{INPUT_DATA_PATH}` "
            f"hoặc file bị trống!"
        )
        return

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**Tổng số dòng dữ liệu:** `{len(df_word_segmentation):,}`")

    with col2:
        st.info(f"**Các cột hiện tại:** `{list(df_word_segmentation.columns)}`")

    st.subheader("1. Xem trước dữ liệu trước khi xử lý")
    st.dataframe(df_word_segmentation.head(10), use_container_width=True)
    st.markdown("---")

    st.subheader("Các hạng mục xử lý (Pipeline Steps)")
    st.markdown(
        """
        Hệ thống xử lý cột `Comment` theo thứ tự:

        1. **Word Segmentation**: Tách các phần văn bản chưa được segment bằng `Underthesea`.
        2. **Bảo toàn segment cũ**: Token đã có `_` từ bước trước được giữ nguyên.
        3. **Gắn phủ định**: Gộp từ/cụm phủ định với token phía sau bằng `_`.
        4. **Không chuẩn hóa lại slang/typo/emoji**.
        5. **Giữ nguyên toàn bộ các cột khác**.
        """
    )
    st.markdown("---")

    with st.expander("Xem danh sách từ/cụm từ phủ định"):
        st.write(
            sorted(
                ALL_NEGATIONS,
                key=lambda x: (-len(x.split()), x)
            )
        )

    st.markdown("---")

    if st.button("Bắt đầu Word Segmentation + Gắn phủ định", type="primary"):
        print_header("Vietnamese word segmentation started via Streamlit...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        if "Comment" not in df_word_segmentation.columns:
            st.warning("Không tìm thấy cột 'Comment' trong tập dữ liệu.")
            return

        total_rows = len(df_word_segmentation)
        comments = df_word_segmentation["Comment"].tolist()
        processed_comments = []
        batch_size = 100

        status_text.text("Đang Word Segmentation và gắn phủ định...")

        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            processed_comments.extend(
                process_text(comment)
                for comment in comments[start:end]
            )

            progress = int(end / total_rows * 90)
            progress_bar.progress(progress)
            status_text.text(
                f"Đang xử lý: {end:,}/{total_rows:,}"
            )

        df_word_segmentation["Comment"] = processed_comments

        output_dir = os.path.dirname(OUTPUT_DATA_PATH)
        os.makedirs(output_dir, exist_ok=True)

        try:
            df_word_segmentation.to_csv(
                OUTPUT_DATA_PATH,
                index=False,
                encoding="utf-8-sig"
            )
            progress_bar.progress(100)
            status_text.text("Hoàn tất!")
            st.success(
                f"Đã hoàn tất Word Segmentation + gắn phủ định. "
                f"File kết quả đã được lưu tại: `{OUTPUT_DATA_PATH}`"
            )
        except Exception as e:
            st.error(f"Lỗi khi lưu file: {e}")
            return

        st.markdown("---")
        st.subheader("2. Dữ liệu sau khi Word Segmentation + Gắn phủ định")
        st.dataframe(df_word_segmentation.head(10), use_container_width=True)

if __name__ == "__main__":
    main()