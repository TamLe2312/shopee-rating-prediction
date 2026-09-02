# ============================================================
# LỆNH THỰC THI FILE
# ============================================================
# streamlit run shopee_rating_preprocessing/src/preprocessing/03_text_cleaning.py

# ============================================================
# THƯ VIỆN
# ============================================================

import pandas as pd
import re
import unicodedata
import os
import streamlit as st

# ============================================================
# HÀM TIỆN ÍCH
# ============================================================

from shopee_rating_preprocessing.utils.present import print_header

# ============================================================
# BIẾN TOÀN CỤC
# ============================================================

try:
    from shopee_rating_preprocessing.config.paths import ROOT_DIR
except Exception as e:
    st.error(f"Lỗi khi import ROOT_DIR: {e}")
    raise

INPUT_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "01_noise_removed.csv")
OUTPUT_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "02_text_cleaned.csv")

# ============================================================
# NẠP DỮ LIỆU (CÓ CACHE)
# ============================================================

@st.cache_data
def load_raw_data():
    if not os.path.exists(INPUT_DATA_PATH):
        return None
    df_raw = pd.read_csv(INPUT_DATA_PATH)
    # Loại bỏ các cột không cần thiết
    df_cleaned = df_raw.drop(columns=['Id', 'ProductUrl'], errors='ignore').copy()
    return df_cleaned

# ============================================================
# CODE XỬ LÝ
# ============================================================

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # --------------------------------------------------------
    # Chuẩn hóa Unicode
    # --------------------------------------------------------
    text = unicodedata.normalize("NFC", text)

    # --------------------------------------------------------
    # Chuyển về chữ thường
    # --------------------------------------------------------
    text = text.lower()

    # --------------------------------------------------------
    # Thay xuống dòng, tab
    # --------------------------------------------------------
    # 1. Thay thế tab và các khoảng trắng đặc biệt thành khoảng trắng thường
    text = re.sub(r"[\t]+", " ", text)
    
    # 2. Thay thế toàn bộ ký tự xuống dòng thành dấu chấm
    text = re.sub(r"\s*[\r\n]+\s*", ". ", text)
    
    # 3. Chuẩn hóa lại các dấu chấm bị thừa
    text = re.sub(r"\.{2,}", ".", text)

    # --------------------------------------------------------
    # Xoá "/" không nằm giữa các chữ số (ngày/tháng/năm)
    # --------------------------------------------------------
    text = re.sub(r"(?<!\d)\s*/\s*|\s*/\s*(?!\d)", " ", text)

    # --------------------------------------------------------
    # Thay thế dấu :
    # --------------------------------------------------------
    text = re.sub(r"[:\uff1a]+", " : ", text)

    # Rút gọn các ký tự lặp kéo dài âm tiết (> 2 lần) về 2 ký tự
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # --------------------------------------------------------
    # Xóa ký tự rác NHƯNG GIỮ LẠI EMOJI VÀ CÁC DẤU CÂU HỢP LỆ
    # --------------------------------------------------------
    # Bổ sung các dấu câu thông dụng muốn giữ (nếu muốn): 
    # Ví dụ: ngoặc kép (", '), ngoặc đơn ((), []), dấu gạch ngang (-) có thể đưa vào nhóm bị xóa thành khoảng trắng.
    # Dưới đây giữ lại: \w (chữ cái/số), \s (khoảng trắng), ? . ! , : và dải emoji.
    emoji_chars = r"\U0001F000-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF\U0001F1E6-\U0001F1FF\ufe0e\ufe0f\u200d"
    
    # Mọi ký tự KHÔNG PHẢI các ký tự dưới đây sẽ bị thay bằng khoảng trắng:
    text = re.sub(rf'[^\w\s?.!,:{emoji_chars}]+', ' ', text, flags=re.UNICODE)
    
    # Chuẩn hóa các dấu câu lặp vô nghĩa (giữ lại 1 dấu duy nhất cho gọn văn bản)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\!{2,}', '!', text)

    # --------------------------------------------------------
    # Chuẩn hóa khoảng trắng
    # --------------------------------------------------------
    text = re.sub(r"\s+", " ", text).strip()
    
    return text
# ============================================================
# HÀM THỰC THI MAIN() - GIAO DIỆN STREAMLIT
# ============================================================

def main():
    st.set_page_config(page_title="Text Cleaning")
    
    st.title("Làm sạch văn bản (Text Cleaning)")
    st.markdown("---")

    # Tải dữ liệu
    df_text_cleaning = load_raw_data()

    if df_text_cleaning is None or df_text_cleaning.empty:
        st.error(f"Không tìm thấy file dữ liệu thô tại: `{INPUT_DATA_PATH}` hoặc file bị trống!")
        return

    # Hiển thị thông tin tổng quan
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Tổng số dòng dữ liệu:** `{len(df_text_cleaning):,}`")
    with col2:
        st.info(f"**Các cột hiện tại:** `{list(df_text_cleaning.columns)}`")

    st.subheader("1. Xem trước dữ liệu thô (Raw Data)")
    st.dataframe(df_text_cleaning.head(10), use_container_width=True)
    st.markdown("---")

    # --- HIỂN THỊ CÁC HẠNG MỤC XỬ LÝ TRÊN GIAO DIỆN ---
    st.subheader("Các hạng mục xử lý (Pipeline Steps)")
    st.markdown(
        """
        Hệ thống sẽ thực thi tuần tự các bước làm sạch sau đây cho cột `Comment`:
        1. **Chuẩn hóa Unicode**: Đưa về dạng NFC thống nhất để đảm bảo toàn vẹn tiếng Việt có dấu.
        2. **Chuyển chữ thường**: Chuyển toàn bộ văn bản về chữ thường (lowercase).
        3. **Xử lý xuống dòng/tab**: Thay thế các ký tự `\\r`, `\\n`, `\\tab` thành khoảng trắng.
        4. **Xử lý ký tự gạch chéo (/)**: Xóa bỏ dấu `/` ngoại trừ trường hợp nằm giữa các chữ số (ngày/tháng/năm).
        5. **Thay thế dấu hai chấm (:)**: Chuyển đổi dấu `:` và `：` thành khoảng trắng.
        6. **Lọc ký tự rác**: Loại bỏ các ký tự đặc biệt, giữ lại trọn vẹn chữ cái tiếng Việt có dấu, chữ số, khoảng trắng và các dấu câu quan trọng như `?`, `!`, `.`.
        7. **Chuẩn hóa khoảng trắng**: Gom nhiều khoảng trắng liên tiếp thành 1 khoảng trắng duy nhất và xóa khoảng trắng thừa đầu cuối.
        """
    )
    st.markdown("---")

    # Nút kích hoạt tiến trình làm sạch
    if st.button("Bắt đầu làm sạch văn bản (Text Cleaning)", type="primary"):
        print_header("Text cleaning started via Streamlit...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Giả lập hoặc xử lý trực tiếp trên cột Comment
        if "Comment" in df_text_cleaning.columns:
            status_text.text("Đang tiến hành làm sạch cột 'Comment'...")
            
            # Áp dụng hàm clean_text
            df_text_cleaning["Comment"] = df_text_cleaning["Comment"].apply(clean_text)
            progress_bar.progress(70)
        else:
            st.warning("Không tìm thấy cột 'Comment' trong tập dữ liệu.")

        # Tự động lưu file kết quả
        output_dir = os.path.dirname(OUTPUT_DATA_PATH)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            df_text_cleaning.to_csv(OUTPUT_DATA_PATH, index=False, encoding='utf-8-sig')
            progress_bar.progress(100)
            status_text.text("Hoàn tất!")
            st.success(f"Đã tự động lưu file kết quả thành công tại: `{OUTPUT_DATA_PATH}`")
        except Exception as e:
            st.error(f"Lỗi khi lưu file: {e}")

        st.markdown("---")
        st.subheader("2. Dữ liệu sau khi làm sạch (Cleaned Data Preview)")
        st.dataframe(df_text_cleaning.head(10), use_container_width=True)
        
if __name__ == "__main__":
    main()