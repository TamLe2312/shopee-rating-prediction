# ============================================================
# LỆNH THỰC THI FILE
# ============================================================
# streamlit run shopee_rating_preprocessing/src/preprocessing/01_eda.py

# ============================================================
# THƯ VIỆN
# ============================================================

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
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
    print("Lỗi khi import:", e)

# ============================================================
# NẠP DỮ LIỆU
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(f"{ROOT_DIR}/data/raw/shopee_reviews.csv")

df_text_cleaned = load_data()

# ============================================================
# CODE XỬ LÝ
# ============================================================

# ------------------------------------------------------------
# Hàm In ra số lượng dòng và cột của một Pandas DataFrame
# ------------------------------------------------------------

def print_shape(df: pd.DataFrame) -> None:
    st.subheader(f"Kích thước dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")


# ------------------------------------------------------------
# Kiểm tra dữ liệu Missing/Null
# ------------------------------------------------------------

def print_missing_data(df: pd.DataFrame) -> None:
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().mean()) * 100
    
    missing_report = pd.DataFrame({
        'Missing_Count': missing_count,
        'Percentage (%)': missing_percent
    })
    
    st.subheader("Báo cáo dữ liệu khuyết thiếu (Missing Values)")
    st.dataframe(missing_report, use_container_width=True)


# ------------------------------------------------------------
# Kiểm tra dữ liệu trùng lặp (Duplicates) và hiển thị chi tiết
# ------------------------------------------------------------

def check_duplicates(df: pd.DataFrame, subset_col: str = 'Comment', target_cols: list = ['Rating', 'Comment', 'SubCategory'], head_num: int = 10) -> tuple[int, int]:  
    total_duplicates = df.duplicated().sum()
    
    # Lọc các dòng trùng lặp dựa trên cột Comment
    subset_mask = df.duplicated(subset=[subset_col], keep=False)
    subset_duplicates = subset_mask.sum()
    
    # Lấy danh sách các dòng bị trùng nội dung comment để hiển thị
    dup_df = df[subset_mask].sort_values(by=subset_col)
    
    st.subheader("Thống kê dữ liệu trùng lặp")
    st.write(f"- Số dòng trùng lặp hoàn toàn (tất cả các cột): **{total_duplicates}**")
    st.write(f"- Số lượng dòng có nội dung `Comment` bị trùng lặp: **{subset_duplicates}**")
    
    if subset_duplicates > 0:
        st.write(f"**Xem trước các đánh giá bị trùng nội dung (Hiển thị tối đa {head_num} dòng):**")
        st.dataframe(dup_df[target_cols].head(head_num), use_container_width=True)
    
    return total_duplicates, subset_duplicates


# ------------------------------------------------------------
# Xem phân bố nhãn (Rating)
# ------------------------------------------------------------

def plot_rating_distribution(df: pd.DataFrame, column_name: str = 'Rating') -> pd.Series:
    rating_percentage = df[column_name].value_counts(normalize=True) * 100
    
    st.subheader(f"Phân bố nhãn ({column_name})")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(x=column_name, data=df, ax=ax)
    ax.set_title(f'Phân bố {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Số lượng')
    st.pyplot(fig)
    
    return rating_percentage


# ------------------------------------------------------------
# Xem độ dài câu (Comment)
# ------------------------------------------------------------

def analyze_comment_length(df: pd.DataFrame, text_col: str = 'Comment') -> pd.Series:
    word_counts = df[text_col].fillna('').astype(str).apply(lambda x: len(x.split()))
    stats_summary = word_counts.describe().round(2)
    
    st.subheader(f"Phân bố độ dài câu theo cột: {text_col}")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.histplot(word_counts, bins=100, kde=True, ax=ax)
    ax.set_title('Phân bố độ dài câu (số từ)')
    ax.set_xlabel('Số lượng từ')
    ax.set_ylabel('Tần suất')
    st.pyplot(fig)
    
    return stats_summary


# ------------------------------------------------------------
# Lọc các comment có số từ dưới 3
# ------------------------------------------------------------

def get_short_comments_report(df: pd.DataFrame, text_col: str = 'Comment', threshold: int = 3) -> tuple[int, int, pd.DataFrame]:
    word_counts = df[text_col].fillna('').astype(str).apply(lambda x: len(x.split()))
    short_comments = df[word_counts < threshold]
    
    short_freq = short_comments[text_col].value_counts().reset_index()
    short_freq.columns = ['Comment', 'Frequency']
    
    total_count = len(short_comments)
    unique_count = len(short_freq)
    
    return total_count, unique_count, short_freq


# ------------------------------------------------------------
# Lọc Spam / Review đáng ngờ
# ------------------------------------------------------------

def check_and_report_spam(df: pd.DataFrame, text_col: str = 'Comment', target_cols: list = ['Rating', 'Comment', 'SubCategory'], head_num: int = 10) -> int:
    def is_spam_or_trash(text):
        if not isinstance(text, str) or not text.strip():
            return True 

        text_str = text.strip()

        if re.search(r'http[s]?://|www\.|bit\.ly', text_str, re.IGNORECASE):
            return True

        if re.search(r'(0[3|5|7|8|9])+([0-9]{8})\b', text_str):
            return True

        if re.search(r'(.)\1{4,}', text_str):
            return True

        if not re.search(r'[a-zA-Zà-ỹÀ-Ỹ]', text_str):
            return True

        return False

    spam_mask = df[text_col].apply(is_spam_or_trash)
    total_spam = spam_mask.sum()
    
    st.subheader("Báo cáo Spam và Review đáng ngờ")
    st.write(f"Số lượng comment spam/rác thực sự: **{total_spam}**")

    suspicious_reviews = df[spam_mask][target_cols]
    st.dataframe(suspicious_reviews.head(head_num), use_container_width=True)

    return total_spam

# ============================================================
# HÀM THỰC THI MAIN()
# ============================================================

def main():
    # Cấu hình hiển thị full layout màn hình
    st.set_page_config(page_title="Khám phá dữ liệu EDA", layout="wide")
    
    st.title("Khám phá dữ liệu EDA")
    st.markdown("---")

    # Kích thước dữ liệu
    print_shape(df_text_cleaned)
    st.markdown("---")

    # Missing values
    print_missing_data(df_text_cleaned)
    st.markdown("---")

    # Trùng lặp (Gọi gọn gàng, hiển thị tích hợp bên trong hàm)
    check_duplicates(df_text_cleaned, subset_col='Comment')
    st.markdown("---")

    # Phân bố Rating
    stats_rating = plot_rating_distribution(df_text_cleaned, column_name='Rating')
    st.write("Tỷ lệ phần trăm Rating:", stats_rating)
    st.markdown("---")

    # Độ dài câu
    stats_len = analyze_comment_length(df_text_cleaned, text_col='Comment')
    st.write("Thống kê mô tả độ dài câu:", stats_len)
    st.markdown("---")

    # Comment quá ngắn
    total_short, unique_short, report_df = get_short_comments_report(df_text_cleaned, threshold=3)
    st.subheader("Thống kê Comment quá ngắn (<3 từ)")
    st.write(f"- Tổng số lượng comment quá ngắn: **{total_short}**")
    st.write(f"- Số lượng comment ngắn duy nhất (unique): **{unique_short}**")
    st.dataframe(report_df.head(10), use_container_width=True)
    st.markdown("---")

    # Spam & Review đáng ngờ
    check_and_report_spam(df_text_cleaned, text_col='Comment')

if __name__ == "__main__":
    main()