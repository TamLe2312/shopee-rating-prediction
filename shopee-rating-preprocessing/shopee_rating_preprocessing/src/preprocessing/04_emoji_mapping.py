# ============================================================
# LỆNH THỰC THI FILE
# ============================================================
# streamlit run shopee_rating_preprocessing/src/preprocessing/04_emoji_mapping.py

# ============================================================
# THƯ VIỆN
# ============================================================

import os
import re
import unicodedata
import pandas as pd
import streamlit as st

from shopee_rating_preprocessing.utils.present import print_header

try:
    from shopee_rating_preprocessing.config.paths import ROOT_DIR
except Exception as e:
    st.error(f"Lỗi khi import ROOT_DIR: {e}")
    raise

INPUT_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "02_text_cleaned.csv")
DICT_DATA_PATH = os.path.join(ROOT_DIR, "data", "dict", "emoji_dict.csv")
OUTPUT_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "03_emoji_mapped.csv")


# ============================================================
# NẠP DỮ LIỆU
# ============================================================

@st.cache_data
def load_data():
    if not os.path.exists(INPUT_DATA_PATH) or not os.path.exists(DICT_DATA_PATH):
        return None, None

    df_data = pd.read_csv(
        INPUT_DATA_PATH,
        dtype=str,
        keep_default_na=False
    )

    df_emoji_dict = pd.read_csv(
        DICT_DATA_PATH,
        dtype=str,
        keep_default_na=False
    )

    return df_data, df_emoji_dict


df_text_cleaned, df_emoji_dict = load_data()


# ============================================================
# CHUẨN HÓA TEXT
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    return unicodedata.normalize(
        "NFC",
        str(text).replace("\ufeff", "")
    )


# ============================================================
# CHUẨN HÓA EMOJI
# ============================================================

def normalize_emoji(emoji):
    emoji = normalize_text(emoji).strip()

    # --------------------------------------------------------
    # Chuẩn hóa KEYCAP
    #
    # 0️    -> 0⃣
    # 0️⃣   -> 0⃣
    # 0⃣    -> 0⃣
    #
    # 1️    -> 1⃣
    # 2️    -> 2⃣
    # ...
    # 9️    -> 9⃣
    # #️    -> #⃣
    # *️    -> *⃣
    # --------------------------------------------------------

    emoji = re.sub(
        r"([0-9#*])(?:\ufe0f)?\u20e3",
        lambda m: m.group(1) + "\u20e3",
        emoji
    )

    # --------------------------------------------------------
    # Digit / symbol + Variation Selector
    #
    # 0️ -> 0⃣
    # 1️ -> 1⃣
    # ...
    # 9️ -> 9⃣
    # #️ -> #⃣
    # *️ -> *⃣
    #
    # Chỉ xử lý khi TOÀN BỘ chuỗi là dạng này.
    # --------------------------------------------------------

    emoji = re.sub(
        r"^([0-9#*])\ufe0f$",
        lambda m: m.group(1) + "\u20e3",
        emoji
    )

    # --------------------------------------------------------
    # Xóa Variation Selector còn lại
    # --------------------------------------------------------

    emoji = (
        emoji
        .replace("\ufe0f", "")
        .replace("\ufe0e", "")
    )

    return emoji


# ============================================================
# CHUẨN HÓA KEYCAP
# ============================================================

def normalize_keycap(emoji):
    return normalize_emoji(emoji)


# ============================================================
# LẤY UNICODE
# ============================================================

def unicode_codes(text):
    return " ".join(
        f"U+{ord(char):04X}"
        for char in normalize_text(text)
    )


# ============================================================
# LẤY EMOJI GỐC
# ============================================================

def get_base_emoji(emoji):
    emoji = normalize_emoji(emoji)

    # --------------------------------------------------------
    # Skin tone
    #
    # 👍🏻 👍🏼 👍🏽 👍🏾 👍🏿
    #      ↓
    #      👍
    # --------------------------------------------------------

    emoji = re.sub(
        r"[\U0001F3FB-\U0001F3FF]",
        "",
        emoji
    )

    # --------------------------------------------------------
    # Gender trong ZWJ
    #
    # 🙆‍♂️ -> 🙆
    # 🙆‍♀️ -> 🙆
    #
    # Không xóa ZWJ thông thường
    # --------------------------------------------------------

    emoji = re.sub(
        r"\u200d[\u2640\u2642]",
        "",
        emoji
    )

    return emoji


# ============================================================
# REGEX NHẬN DIỆN EMOJI
# ============================================================

EMOJI_BASE = (
    r"[\U0001F000-\U0001FAFF"
    r"\u2300-\u23FF"
    r"\u2600-\u27BF"
    r"\u2B00-\u2BFF]"
)

EMOJI_MODIFIER = (
    r"[\U0001F3FB-\U0001F3FF]"
)

VARIATION = (
    r"[\ufe0e\ufe0f]"
)

ZWJ = r"\u200d"

# Emoji thông thường
EMOJI_SEQUENCE = (
    rf"{EMOJI_BASE}"
    rf"(?:{VARIATION})?"
    rf"(?:{EMOJI_MODIFIER})?"
    rf"(?:"
    rf"{ZWJ}"
    rf"{EMOJI_BASE}"
    rf"(?:{VARIATION})?"
    rf"(?:{EMOJI_MODIFIER})?"
    rf")*"
)

# Cờ quốc gia
# 🇻🇳 🇺🇸 🇯🇵 ...
FLAG_SEQUENCE = (
    r"[\U0001F1E6-\U0001F1FF]{2}"
)

# ------------------------------------------------------------
# Keycap chuẩn
#
# 0️⃣ 1️⃣ 2️⃣ 3️⃣ ... 9️⃣ #️⃣ *️⃣
# ------------------------------------------------------------

KEYCAP_SEQUENCE = (
    r"(?:[0-9#*])"
    r"(?:\ufe0f)?"
    r"\u20e3"
)

# ------------------------------------------------------------
# Digit / symbol + Variation Selector
#
# 0️ 1️ 2️ 3️ ... 9️ #️ *️
#
# Đây là các ký tự có FE0F nhưng không có U+20E3.
# ------------------------------------------------------------

DIGIT_VARIATION_SEQUENCE = (
    r"(?:[0-9#*])"
    r"\ufe0f"
)

EMOJI_PATTERN = re.compile(
    rf"{FLAG_SEQUENCE}"
    rf"|{KEYCAP_SEQUENCE}"
    rf"|{DIGIT_VARIATION_SEQUENCE}"
    rf"|{EMOJI_SEQUENCE}"
)


# ============================================================
# TÁCH EMOJI
# ============================================================

def extract_emoji_tokens(text):
    if pd.isna(text):
        return []

    return [
        normalize_emoji(match.group(0))
        for match in EMOJI_PATTERN.finditer(
            normalize_text(text)
        )
    ]


# ============================================================
# BUILD MAPPING
# ============================================================

@st.cache_data
def build_emoji_mapping(emoji_dict_df):
    df = emoji_dict_df.copy()

    # --------------------------------------------------------
    # Tự động tìm cột Emoji
    # --------------------------------------------------------

    emoji_col = next(
        (
            col
            for col in [
                "emoji",
                "raw_sample",
                "raw_emoji"
            ]
            if col in df.columns
        ),
        None
    )

    if emoji_col is None:
        raise KeyError(
            "Không tìm thấy cột emoji trong "
            "emoji_dict.csv.\n"
            f"Các cột hiện có: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Kiểm tra vi_emoji
    # --------------------------------------------------------

    if "vi_emoji" not in df.columns:
        raise KeyError(
            "Không tìm thấy cột 'vi_emoji' "
            "trong emoji_dict.csv.\n"
            f"Các cột hiện có: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Chuẩn hóa emoji
    # --------------------------------------------------------

    df["raw_emoji"] = (
        df[emoji_col]
        .fillna("")
        .astype(str)
        .map(normalize_emoji)
        .str.strip()
    )

    # --------------------------------------------------------
    # Chuẩn hóa nghĩa tiếng Việt
    # --------------------------------------------------------

    df["vi_emoji"] = (
        df["vi_emoji"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Loại dữ liệu rỗng
    # --------------------------------------------------------

    df = df[
        (df["raw_emoji"] != "")
        & (df["vi_emoji"] != "")
    ]

    # --------------------------------------------------------
    # Nếu duplicate emoji: giữ dòng đầu tiên
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["raw_emoji"],
        keep="first"
    )

    # --------------------------------------------------------
    # EXACT MAPPING
    # --------------------------------------------------------

    exact_map = dict(
        zip(
            df["raw_emoji"],
            df["vi_emoji"]
        )
    )

    # --------------------------------------------------------
    # BASE MAPPING
    # --------------------------------------------------------

    base_map = {}

    for emoji, meaning in exact_map.items():
        base_emoji = get_base_emoji(emoji)

        if (
            base_emoji
            and base_emoji not in base_map
        ):
            base_map[base_emoji] = meaning

    return exact_map, base_map


# ============================================================
# TÌM NGHĨA EMOJI
# ============================================================

def resolve_emoji(
    emoji,
    exact_map,
    base_map
):
    normalized = normalize_emoji(emoji)

    # --------------------------------------------------------
    # 1. Exact mapping
    # --------------------------------------------------------

    if normalized in exact_map:
        return exact_map[normalized]

    # --------------------------------------------------------
    # 2. Base mapping
    # --------------------------------------------------------

    base_emoji = get_base_emoji(
        normalized
    )

    return base_map.get(
        base_emoji
    )


# ============================================================
# TRÍCH XUẤT EMOJI TRONG COMMENT
# ============================================================

def extract_emojis_in_text(
    text,
    emoji_dict
):
    if pd.isna(text):
        return "Không có"

    exact_map, base_map = emoji_dict

    result = []

    for emoji in extract_emoji_tokens(text):
        meaning = resolve_emoji(
            emoji,
            exact_map,
            base_map
        )

        if meaning:
            result.append(
                normalize_emoji(emoji)
            )

    result = list(
        dict.fromkeys(result)
    )

    return (
        ", ".join(result)
        if result
        else "Không có"
    )


# ============================================================
# THAY THẾ EMOJI
# ============================================================

def replace_emoji_in_dataframe(
    df: pd.DataFrame,
    emoji_dict_df: pd.DataFrame,
    text_col: str = "Comment"
) -> pd.DataFrame:
    emoji_dict = build_emoji_mapping(
        emoji_dict_df
    )

    exact_map, base_map = emoji_dict

    def replace_emoji(text):
        if pd.isna(text):
            return text

        text = normalize_text(text)

        def replace_match(match):
            emoji = normalize_emoji(
                match.group(0)
            )

            meaning = resolve_emoji(
                emoji,
                exact_map,
                base_map
            )

            if meaning:
                return f" {meaning} "

            return match.group(0)

        return EMOJI_PATTERN.sub(
            replace_match,
            text
        )

    result = df.copy()

    result[text_col] = (
        result[text_col]
        .map(replace_emoji)
    )

    return result


# ============================================================
# TÌM EMOJI CHƯA MAPPING
# ============================================================

def find_unmapped_emojis(
    df,
    emoji_dict,
    text_col="Comment"
):
    exact_map, base_map = emoji_dict

    found = {}

    for text in df[text_col]:
        if pd.isna(text):
            continue

        for emoji in extract_emoji_tokens(text):
            meaning = resolve_emoji(
                emoji,
                exact_map,
                base_map
            )

            if meaning:
                continue

            emoji = normalize_emoji(
                emoji
            )

            found[emoji] = (
                found.get(emoji, 0) + 1
            )

    return found


# ============================================================
# DEBUG EMOJI
# ============================================================

def debug_emoji(
    target_emoji,
    emoji_dict_df
):
    target = normalize_emoji(
        target_emoji
    )

    target_base = get_base_emoji(
        target
    )

    result = []

    # --------------------------------------------------------
    # Tự xác định cột emoji
    # --------------------------------------------------------

    emoji_col = next(
        (
            col
            for col in [
                "emoji",
                "raw_sample",
                "raw_emoji"
            ]
            if col in emoji_dict_df.columns
        ),
        None
    )

    if emoji_col is None:
        return result

    # --------------------------------------------------------
    # Duyệt dictionary
    # --------------------------------------------------------

    for _, row in emoji_dict_df.iterrows():
        raw = normalize_emoji(
            row[emoji_col]
        )

        if not raw:
            continue

        base = get_base_emoji(
            raw
        )

        if (
            raw == target
            or base == target_base
        ):
            result.append({
                "raw_emoji": repr(raw),
                "base_emoji": repr(base),
                "unicode": unicode_codes(raw),
                "base_unicode": unicode_codes(base),
                "vi_emoji": row["vi_emoji"]
            })

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(
        page_title="Emoji Mapping"
    )

    st.title("Emoji Mapping")
    st.markdown("---")

    # --------------------------------------------------------
    # Kiểm tra dữ liệu
    # --------------------------------------------------------

    if df_text_cleaned is None or df_emoji_dict is None:
        st.error(
            f"Không tìm thấy file dữ liệu tại: `{INPUT_DATA_PATH}` hoặc từ điển tại: `{DICT_DATA_PATH}`!"
        )
        return

    if (
        df_text_cleaned.empty
        or df_emoji_dict.empty
    ):
        st.warning(
            "Dữ liệu đầu vào hoặc từ điển emoji bị trống!"
        )
        return

    # --------------------------------------------------------
    # Build mapping
    # --------------------------------------------------------

    try:
        emoji_dict = build_emoji_mapping(
            df_emoji_dict
        )
    except Exception as e:
        st.error(
            f"Lỗi khi tạo Emoji Mapping: {e}"
        )
        st.write(
            "Các cột hiện có trong dictionary:"
        )
        st.write(
            list(df_emoji_dict.columns)
        )
        return

    # --------------------------------------------------------
    # Tìm comment có emoji
    # --------------------------------------------------------

    df_sample = df_text_cleaned.copy()

    df_sample["Has_Emoji"] = (
        df_sample["Comment"]
        .map(
            lambda x:
            extract_emojis_in_text(
                x,
                emoji_dict
            ) != "Không có"
        )
    )

    df_demo = (
        df_sample[
            df_sample["Has_Emoji"]
        ]
        .head(5)
        .copy()
    )

    # --------------------------------------------------------
    # Không có emoji
    # --------------------------------------------------------

    if df_demo.empty:
        st.error(
            "Không tìm thấy bình luận nào trong "
            "tập dữ liệu khớp với emoji trong dictionary!"
        )
        st.info(
            f"Vui lòng kiểm tra lại `{DICT_DATA_PATH}` "
            "hoặc dữ liệu Comment."
        )
        return

    # ========================================================
    # BẢNG 1
    # ========================================================

    st.subheader(
        "1. Dữ liệu gốc"
    )

    df_demo_1 = pd.DataFrame({
        "Comment":
            df_demo["Comment"],

        "Detected_Emojis":
            df_demo["Comment"].map(
                lambda x:
                extract_emojis_in_text(
                    x,
                    emoji_dict
                )
            )
    })

    st.dataframe(
        df_demo_1,
        use_container_width=True
    )

    st.markdown("---")

    # ========================================================
    # THAY THẾ EMOJI
    # ========================================================

    df_emoji_replaced = (
        replace_emoji_in_dataframe(
            df_text_cleaned,
            df_emoji_dict,
            text_col="Comment"
        )
    )

    # ========================================================
    # BẢNG 2
    # ========================================================

    st.subheader(
        "2. Dữ liệu sau khi thay thế Emoji"
    )

    df_demo_replaced = (
        df_emoji_replaced
        .loc[df_demo.index]
        .copy()
    )

    df_demo_2 = pd.DataFrame({
        "Detected_Emojis":
            df_demo["Comment"].map(
                lambda x:
                extract_emojis_in_text(
                    x,
                    emoji_dict
                )
            ),

        "Comment_Replaced":
            df_demo_replaced["Comment"]
    })

    st.dataframe(
        df_demo_2,
        use_container_width=True
    )

    # ========================================================
    # DEBUG
    # ========================================================

    st.markdown("---")

    st.subheader(
        "3. Kiểm tra Emoji"
    )

    target_emoji = st.text_input(
        "Nhập emoji cần kiểm tra:",
        value="🙆"
    )

    if target_emoji:
        debug_result = debug_emoji(
            target_emoji,
            df_emoji_dict
        )

        if debug_result:
            st.write(
                f"Các biến thể của `{target_emoji}` "
                "trong dictionary:"
            )

            st.dataframe(
                pd.DataFrame(debug_result),
                use_container_width=True
            )
        else:
            st.warning(
                f"Không tìm thấy `{target_emoji}` "
                "hoặc biến thể của nó trong dictionary."
            )

    # ========================================================
    # EMOJI CHƯA MAPPING
    # ========================================================

    st.markdown("---")

    st.subheader(
        "4. Emoji chưa được mapping"
    )

    unmapped_emojis = find_unmapped_emojis(
        df_text_cleaned,
        emoji_dict,
        text_col="Comment"
    )

    if unmapped_emojis:
        df_unmapped = pd.DataFrame([
            {
                "Emoji":
                    emoji,

                "Frequency":
                    count,

                "Unicode":
                    unicode_codes(emoji),

                "Base_Emoji":
                    get_base_emoji(emoji)
            }
            for emoji, count
            in sorted(
                unmapped_emojis.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ])

        st.warning(
            f"Có {len(df_unmapped)} "
            "emoji chưa được mapping."
        )

        st.dataframe(
            df_unmapped,
            use_container_width=True
        )
    else:
        st.success(
            "Tất cả emoji phát hiện được "
            "đều đã được mapping."
        )

    # ========================================================
    # LƯU FILE
    # ========================================================

    try:
        os.makedirs(
            os.path.dirname(
                OUTPUT_DATA_PATH
            ),
            exist_ok=True
        )

        df_emoji_replaced.to_csv(
            OUTPUT_DATA_PATH,
            index=False,
            encoding="utf-8-sig"
        )

        st.success(
            "Đã tự động lưu file kết quả tại: "
            f"`{OUTPUT_DATA_PATH}`"
        )

    except Exception as e:
        st.error(
            f"Lỗi khi tự động lưu file: {e}"
        )


# ============================================================
# CHẠY
# ============================================================

if __name__ == "__main__":
    main()