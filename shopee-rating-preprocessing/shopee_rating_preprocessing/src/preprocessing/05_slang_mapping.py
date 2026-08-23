# ============================================================
# LỆNH THỰC THI FILE
# ============================================================

# streamlit run shopee_rating_preprocessing/src/preprocessing/05_slang_mapping.py


# ============================================================
# THƯ VIỆN
# ============================================================

import pandas as pd
import re
import streamlit as st


# ============================================================
# BIẾN TOÀN CỤC
# ============================================================

try:
    from shopee_rating_preprocessing.config.paths import ROOT_DIR
except Exception as e:
    st.error(f"Lỗi khi import ROOT_DIR: {e}")
    raise

INPUT_DATA_PATH = f"{ROOT_DIR}/data/processed/03_emoji_mapped.csv"
SLANG_DICT_PATH = f"{ROOT_DIR}/data/dict/slang_dict.csv"
OUTPUT_DATA_PATH = f"{ROOT_DIR}/data/processed/04_slang_mapped.csv"


# ============================================================
# NẠP DỮ LIỆU
# ============================================================

@st.cache_data
def load_data():
    """Nạp dữ liệu comment và từ điển slang."""
    return (
        pd.read_csv(INPUT_DATA_PATH, encoding="utf-8-sig"),
        pd.read_csv(SLANG_DICT_PATH, encoding="utf-8-sig")
    )


# ============================================================
# XÂY DỰNG SLANG RESOURCES
# ============================================================

@st.cache_data
def build_slang_resources(slang_dict_df: pd.DataFrame):
    """Chuẩn hóa dictionary và tạo regex dùng chung."""

    required = {"slang", "meaning"}
    missing = required - set(slang_dict_df.columns)

    if missing:
        raise ValueError(f"slang_dict.csv thiếu cột: {', '.join(sorted(missing))}")

    slang_df = slang_dict_df[["slang", "meaning"]].dropna().copy()
    slang_df["slang"] = slang_df["slang"].astype(str).str.strip()
    slang_df["meaning"] = slang_df["meaning"].astype(str).str.strip()
    slang_df = slang_df[(slang_df["slang"] != "") & (slang_df["meaning"] != "")]
    slang_df["slang_lower"] = slang_df["slang"].str.lower()
    slang_df = slang_df.drop_duplicates("slang_lower")

    mapping = dict(zip(slang_df["slang_lower"], slang_df["meaning"]))

    # Ưu tiên cụm slang dài trước.
    slangs = sorted(mapping, key=len, reverse=True)
    escaped = [re.escape(slang) for slang in slangs]

    if not escaped:
        return {}, None

    # Regex duy nhất để tìm và thay slang.
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)",
        re.IGNORECASE
    )

    return mapping, pattern


# ============================================================
# TRÍCH XUẤT SLANG
# ============================================================

def extract_slangs_in_text(text: str, pattern, mapping) -> str:
    """Lấy các slang xuất hiện trong comment."""

    if pd.isna(text) or not str(text).strip():
        return "Không có"

    matches = pattern.findall(str(text))

    if not matches:
        return "Không có"

    found = list(dict.fromkeys(match.lower() for match in matches))
    return ", ".join(found)


# ============================================================
# LẤY SLANG ĐÃ THAY
# ============================================================

def get_replaced_slangs(text: str, pattern, mapping) -> str:
    """Trả về các slang thực sự được thay và meaning tương ứng."""

    if pd.isna(text) or not str(text).strip():
        return "Không có"

    matches = pattern.findall(str(text))

    if not matches:
        return "Không có"

    replaced = []
    seen = set()

    for match in matches:
        slang = match.lower()
        meaning = mapping.get(slang)

        if meaning and slang not in seen:
            replaced.append(f"{match} → {meaning}")
            seen.add(slang)

    return ", ".join(replaced) if replaced else "Không có"


# ============================================================
# THAY THẾ SLANG
# ============================================================

def replace_slang_in_text(text: str, pattern, mapping):
    """Thay slang bằng meaning tương ứng."""

    if pd.isna(text):
        return text

    def replace_match(match):
        key = match.group(0).lower()
        return mapping.get(key, match.group(0))

    return pattern.sub(replace_match, str(text))


# ============================================================
# THAY THẾ SLANG TRONG DATAFRAME
# ============================================================

def replace_slang_in_dataframe(df: pd.DataFrame, slang_dict_df: pd.DataFrame, text_col="Comment"):
    """Thay slang trong toàn bộ DataFrame và hiển thị tiến trình."""

    if text_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{text_col}' trong DataFrame.")

    mapping, pattern = build_slang_resources(slang_dict_df)
    processed_df = df.copy()

    if pattern is None or processed_df.empty:
        return processed_df

    total = len(processed_df)
    chunk_size = 10_000
    progress = st.progress(0)
    status = st.empty()
    chunks = []

    # Chia chunk để Streamlit cập nhật tiến trình.
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = processed_df.iloc[start:end].copy()
        chunk[text_col] = chunk[text_col].map(lambda x: replace_slang_in_text(x, pattern, mapping))
        chunks.append(chunk)

        ratio = end / total
        progress.progress(ratio)
        status.text(f"Đang xử lý: {end:,}/{total:,} dòng ({ratio:.1%})")

    progress.progress(1.0)
    status.success(f"Đã xử lý {total:,} dòng.")

    return pd.concat(chunks, ignore_index=True)


# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(page_title="Slang Mapping", layout="wide")
    st.title("Slang Mapping")
    st.markdown("---")

    # Nạp dữ liệu đầu vào.
    try:
        df_data, df_slang = load_data()
    except Exception as e:
        st.error(f"Lỗi khi nạp dữ liệu: {e}")
        return

    if df_data.empty:
        st.warning("Dữ liệu comment đầu vào bị trống!")
        return

    if df_slang.empty:
        st.warning("Từ điển tiếng lóng bị trống!")
        return

    if "Comment" not in df_data.columns:
        st.error("Không tìm thấy cột 'Comment' trong file 02_emoji_mapped.csv.")
        return

    # Tạo dictionary và regex dùng chung.
    try:
        mapping, pattern = build_slang_resources(df_slang)
    except Exception as e:
        st.error(f"Lỗi khi xây dựng slang dictionary: {e}")
        return

    if pattern is None:
        st.warning("Không có slang hợp lệ trong dictionary.")
        return

    # ========================================================
    # THÔNG TIN DỮ LIỆU
    # ========================================================

    col1, col2, col3 = st.columns(3)
    col1.metric("Số comment", f"{len(df_data):,}")
    col2.metric("Số slang", f"{len(mapping):,}")
    col3.metric("Số cột", f"{len(df_data.columns):,}")

    st.markdown("---")

    # ========================================================
    # BẢNG 1
    # ========================================================

    st.subheader("1. Xem trước dữ liệu gốc & Trích xuất Tiếng lóng")

    demo = df_data.head(10).copy()

    preview = pd.DataFrame({
        "Comment": demo["Comment"],
        "Các từ slang": demo["Comment"].map(lambda x: extract_slangs_in_text(x, pattern, mapping))
    })

    st.dataframe(preview, use_container_width=True)
    st.markdown("---")

    # ========================================================
    # SLANG MAPPING
    # ========================================================

    if st.button("Slang Mapping", type="primary"):

        with st.spinner("Đang xử lý thay thế tiếng lóng..."):
            try:
                result = replace_slang_in_dataframe(df_data, df_slang, "Comment")
            except Exception as e:
                st.error(f"Lỗi trong quá trình Slang Mapping: {e}")
                return

        # ====================================================
        # BẢNG 2 - CHỈ HIỆN CÁC DÒNG CÓ THAY THẾ
        # ====================================================

        st.subheader("2. Dữ liệu sau khi thay thế Tiếng lóng (10 mẫu)")

        changed_mask = (
            df_data["Comment"].fillna("").astype(str)
            != result["Comment"].fillna("").astype(str)
        )

        changed_indices = df_data.index[changed_mask][:10]

        if len(changed_indices) == 0:
            st.info("Không tìm thấy comment nào có slang được thay thế.")
        else:
            preview_result = pd.DataFrame({
                "Sau thay thế": result.loc[changed_indices, "Comment"].values,
                "Các từ slang": df_data.loc[changed_indices, "Comment"].map(
                    lambda x: extract_slangs_in_text(x, pattern, mapping)
                ).values,
                "Các từ đã thay": df_data.loc[changed_indices, "Comment"].map(
                    lambda x: get_replaced_slangs(x, pattern, mapping)
                ).values
            })

            st.dataframe(preview_result, use_container_width=True)

        st.markdown("---")

        # Lưu dữ liệu sau khi mapping.
        try:
            result.to_csv(OUTPUT_DATA_PATH, index=False, encoding="utf-8-sig")
            st.success("Đã xử lý thành công toàn bộ dữ liệu!")
            st.info(f"File kết quả: `{OUTPUT_DATA_PATH}`")
        except Exception as e:
            st.error(f"Lỗi khi lưu file: {e}")


# ============================================================
# CHẠY CHƯƠNG TRÌNH
# ============================================================

if __name__ == "__main__":
    main()