# ============================================================
# LỆNH THỰC THI FILE
# ============================================================

# streamlit run shopee_rating_preprocessing/src/preprocessing/02_noise_removing.py


# ============================================================
# THƯ VIỆN
# ============================================================

import os
import re
import pandas as pd
import streamlit as st


# ============================================================
# BIẾN TOÀN CỤC
# ============================================================

try:
    from shopee_rating_preprocessing.config.paths import ROOT_DIR
except Exception as e:
    st.error(f"Lỗi khi import ROOT_DIR: {e}")
    raise

INPUT_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "shopee_reviews.csv")
OUTPUT_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "01_noise_removed.csv")


# ============================================================
# NẠP DỮ LIỆU
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(INPUT_DATA_PATH, encoding="utf-8-sig")


# ============================================================
# CHUẨN HÓA COMMENT
# ============================================================

def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1", text)
    return text


# ============================================================
# PHÁT HIỆN COMMENT RÁC / SPAM / QUẢNG CÁO
# ============================================================

ADVERTISEMENT_PATTERNS = [
    # Kiếm xu / coin
    r"\bkiếm\s+xu\b",
    r"\bsăn\s+xu\b",
    r"\bnhận\s+xu\b",
    r"\blấy\s+xu\b",
    r"\btặng\s+xu\b",
    r"\bcho\s+xu\b",
    r"\bxu\s+miễn\s+phí\b",
    r"\bfree\s+xu\b",
    r"\bcày\s+xu\b",
    r"\bkiếm\s+coin\b",
    r"\bsăn\s+coin\b",
    r"\bnhận\s+coin\b",
    r"\bfree\s+coin\b",

    # Quà / thưởng
    r"\bnhận\s+thưởng\b",
    r"\btrúng\s+thưởng\b",
    r"\bnhận\s+quà\b",
    r"\bquà\s+miễn\s+phí\b",
    r"\bquà\s+free\b",
    r"\bquà\s+tặng\b",

    # Voucher / coupon
    r"\bnhận\s+voucher\b",
    r"\blấy\s+voucher\b",
    r"\bvoucher\s+miễn\s+phí\b",
    r"\bvoucher\s+free\b",
    r"\bvoucher\s+cho\s+không\b",
    r"\bnhận\s+coupon\b",
    r"\blấy\s+coupon\b",
    r"\bcoupon\s+miễn\s+phí\b",

    # Code / giảm giá
    r"\bcode\s+free\b",
    r"\bcode\s+miễn\s+phí\b",
    r"\bnhận\s+code\b",
    r"\blấy\s+code\b",
    r"\bcode\s+giảm\s+giá\b",
    r"\bmã\s+giảm\s+giá\s+miễn\s+phí\b",

    # Quảng cáo / bán hàng
    r"\bquảng\s+cáo\b",
    r"\bmua\s+ngay\b",
    r"\bđặt\s+hàng\s+ngay\b",
    r"\bgiá\s+sốc\b",
    r"\bgiá\s+hủy\s+diệt\b",
    r"\bgiá\s+siêu\s+rẻ\b",
    r"\bsiêu\s+rẻ\b",
    r"\bdeal\s+sốc\b",
    r"\bdeal\s+hot\b",
    r"\bflash\s+sale\b",
    r"\bsale\s+sốc\b",
    r"\bsale\s+đậm\b",

    # Inbox / liên hệ
    r"\binbox\s+mình\b",
    r"\binbox\s+em\b",
    r"\binbox\s+shop\b",
    r"\bib\s+mình\b",
    r"\bib\s+em\b",
    r"\bib\s+shop\b",
    r"\bcheck\s+ib\b",
    r"\bcheck\s+inbox\b",
    r"\bliên\s+hệ\s+mình\b",
    r"\bliên\s+hệ\s+em\b",
    r"\bliên\s+hệ\s+shop\b",
    r"\bquan\s+tâm\s+ib\b",
    r"\bquan\s+tâm\s+inbox\b",
    r"\bai\s+cần\s+ib\b",
    r"\bai\s+cần\s+inbox\b",
    r"\bcần\s+thì\s+ib\b",
    r"\bcần\s+thì\s+inbox\b",

    # Kênh liên hệ
    r"\bzalo\b",
    r"\btelegram\b",
    r"\bwhatsapp\b",
    r"\bhotline\b",
    r"\bsđt\b",
    r"\bsdt\b",
    r"\bsố\s+điện\s+thoại\b",

    # Follow / like / sub
    r"\bfollow\s+mình\b",
    r"\bfollow\s+em\b",
    r"\bfollow\s+shop\b",
    r"\bfollow\s+nhé\b",
    r"\bfllow\s+mình\b",
    r"\bfllow\s+em\b",
    r"\btheo\s+dõi\s+mình\b",
    r"\btheo\s+dõi\s+em\b",
    r"\btheo\s+dõi\s+shop\b",
    r"\blike\s+page\b",
    r"\blike\s+fanpage\b",
    r"\bthả\s+tim\s+giúp\b",
    r"\bcomment\s+giúp\b",
    r"\bđăng\s+ký\s+kênh\b",
    r"\bsub\s+kênh\b",

    # Affiliate / CTV
    r"\baffiliate\b",
    r"\btiếp\s+thị\s+liên\s+kết\b",
    r"\bhoa\s+hồng\b",
    r"\bcộng\s+tác\s+viên\b",
    r"\bctv\b",
    r"\btuyển\s+ctv\b",
    r"\btuyển\s+cộng\s+tác\s+viên\b",
    r"\bviệc\s+làm\s+online\b",
    r"\bkiếm\s+tiền\s+online\b",
    r"\bkiếm\s+tiền\s+tại\s+nhà\b",

    # Link / kéo traffic
    r"\blink\s+bio\b",
    r"\blink\s+bên\s+dưới\b",
    r"\blink\s+phía\s+dưới\b",
    r"\bclick\s+link\b",
    r"\bbấm\s+link\b",
    r"\btruy\s+cập\s+link\b",
    r"\btruy\s+cập\s+website\b",

    # Kêu gọi tham gia
    r"\btham\s+gia\s+ngay\b",
    r"\bđăng\s+ký\s+ngay\b",
    r"\btham\s+gia\s+nhận\b",
    r"\btham\s+gia\s+để\s+nhận\b",

    # Mẫu spam phổ biến
    r"\bai\s+quan\s+tâm\s+ib\b",
    r"\bai\s+quan\s+tâm\s+inbox\b",
    r"\bquan\s+tâm\s+thì\s+ib\b",
    r"\bquan\s+tâm\s+thì\s+inbox\b",
    r"\bmuốn\s+biết\s+thêm\s+ib\b",
    r"\bmuốn\s+biết\s+thêm\s+inbox\b",
    r"\bcó\s+nhu\s+cầu\s+ib\b",
    r"\bcó\s+nhu\s+cầu\s+inbox\b",
]

ADVERTISEMENT_KEYWORDS = [
    "voucher", "coupon", "xu", "coin", "shop", "sale", "deal",
    "khuyến mãi", "giảm giá", "ưu đãi", "quà", "thưởng", "code"
]

ADVERTISEMENT_ACTIONS = [
    "nhận", "lấy", "kiếm", "săn", "mua", "đặt", "ib", "inbox",
    "liên hệ", "follow", "theo dõi", "click", "bấm", "đăng ký",
    "tham gia", "tặng", "cho"
]


def detect_spam_reason(text: str):
    if pd.isna(text) or not str(text).strip():
        return "EMPTY"

    text = normalize_text(text)

    # Link
    if re.search(r"https?://|www\.|bit\.ly|tinyurl|t\.co/|goo\.gl", text, re.IGNORECASE):
        return "URL / LINK"

    # Số điện thoại
    if re.search(r"(?:0|\+84)(?:3|5|7|8|9)\d{8}\b", text):
        return "PHONE"

    # Ký tự lặp
    if re.search(r"(.)\1{4,}", text):
        return "REPEATED CHARACTER"

    # Không có chữ
    if not re.search(r"[a-zA-ZÀ-ỹ]", text):
        return "NO LETTER"

    # Quảng cáo / spam rõ ràng
    for pattern in ADVERTISEMENT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "ADVERTISEMENT / SPAM"

    # Keyword quảng cáo + hành động quảng cáo
    has_ad_keyword = any(keyword in text for keyword in ADVERTISEMENT_KEYWORDS)
    has_ad_action = any(action in text for action in ADVERTISEMENT_ACTIONS)

    if has_ad_keyword and has_ad_action:
        return "ADVERTISEMENT / PROMOTION"

    # Spam tương tác ngắn
    interaction_patterns = [
        r"^\s*follow\s*$",
        r"^\s*fllow\s*$",
        r"^\s*follow\s+nhé\s*$",
        r"^\s*fllow\s+nhé\s*$",
        r"^\s*ib\s*$",
        r"^\s*inbox\s*$",
        r"^\s*check\s*ib\s*$",
        r"^\s*check\s*inbox\s*$",
        r"^\s*quan\s+tâm\s+ib\s*$",
        r"^\s*quan\s+tâm\s+inbox\s*$",
        r"^\s*sub\s+kênh\s*$",
    ]

    for pattern in interaction_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "INTERACTION SPAM"

    # Mật độ ký tự đặc biệt quá cao
    if len(text) >= 10:
        special_chars = re.findall(r"[^a-zA-ZÀ-ỹ0-9\s]", text)
        if len(special_chars) / len(text) > 0.6:
            return "SPECIAL CHARACTER SPAM"

    # Quá ít nội dung chữ
    alnum_count = len(re.findall(r"[a-zA-ZÀ-ỹ0-9]", text))
    if len(text) >= 15 and alnum_count <= 2:
        return "LOW TEXT CONTENT"

    return None


def is_spam_or_trash(text: str) -> bool:
    return detect_spam_reason(text) is not None


# ============================================================
# XỬ LÝ DỮ LIỆU
# ============================================================

def remove_noise(df: pd.DataFrame, text_col="Comment"):
    if text_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{text_col}' trong DataFrame.")

    result = df.copy()
    total_before = len(result)

    result[text_col] = (
        result[text_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Xóa spam / rác / quảng cáo
    spam_reasons = result[text_col].map(detect_spam_reason)
    spam_mask = spam_reasons.notna()
    spam_count = int(spam_mask.sum())

    reason_stats = spam_reasons[spam_mask].value_counts().to_dict()
    result = result[~spam_mask].copy()

    # Xóa duplicate, giữ bản ghi đầu tiên
    duplicate_mask = result.duplicated(subset=[text_col], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    result = result[~duplicate_mask].copy()

    result = result.reset_index(drop=True)

    stats = {
        "total_before": total_before,
        "spam_count": spam_count,
        "duplicate_count": duplicate_count,
        "total_after": len(result),
        "removed_count": total_before - len(result),
        "reason_stats": reason_stats
    }

    return result, stats


# ============================================================
# MAIN - GIAO DIỆN STREAMLIT
# ============================================================

def main():
    st.set_page_config(page_title="Remove Noise")
    st.title("Remove Noise")
    st.markdown("---")

    # Nạp dữ liệu
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Lỗi khi nạp dữ liệu: {e}")
        return

    if df.empty:
        st.warning("Dữ liệu đầu vào bị trống!")
        return

    if "Comment" not in df.columns:
        st.error("Không tìm thấy cột 'Comment'.")
        return

    # Thông tin dữ liệu
    col1, col2 = st.columns(2)
    col1.metric("Số dòng ban đầu", f"{len(df):,}")
    col2.metric("Số cột", f"{len(df.columns):,}")

    st.markdown("---")

    # Xem trước dữ liệu
    st.subheader("1. Xem trước dữ liệu")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # Thực hiện xử lý
    if st.button("Remove Noise", type="primary"):
        with st.spinner("Đang loại bỏ dữ liệu rác, spam, quảng cáo và trùng lặp..."):
            try:
                result, stats = remove_noise(df, text_col="Comment")
            except Exception as e:
                st.error(f"Lỗi trong quá trình xử lý: {e}")
                return

        # Thống kê
        st.subheader("2. Thống kê xử lý")

        col1, col2, col3 = st.columns(3)
        col1.metric("Spam / quảng cáo đã xóa", f"{stats['spam_count']:,}")
        col2.metric("Trùng lặp đã xóa", f"{stats['duplicate_count']:,}")
        col3.metric("Dữ liệu còn lại", f"{stats['total_after']:,}")

        st.write(
            f"Đã loại bỏ tổng cộng **{stats['removed_count']:,}** / "
            f"**{stats['total_before']:,}** dòng."
        )

        st.info(
            "Không xóa comment chỉ vì có từ 'voucher', 'xu', 'shop' hoặc 'sale'. "
            "Các từ này chỉ bị loại khi nằm trong ngữ cảnh quảng cáo/spam."
        )

        # Chi tiết lý do xóa
        if stats["reason_stats"]:
            st.markdown("---")
            st.subheader("3. Chi tiết lý do xóa")

            reason_df = pd.DataFrame(
                list(stats["reason_stats"].items()),
                columns=["Lý do", "Số lượng"]
            ).sort_values("Số lượng", ascending=False)

            st.dataframe(reason_df, use_container_width=True, hide_index=True)

        # Dữ liệu sau xử lý
        st.markdown("---")
        st.subheader("4. Dữ liệu sau khi làm sạch")
        st.dataframe(result.head(10), use_container_width=True)

        # Lưu dữ liệu
        st.markdown("---")

        try:
            os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)

            result.to_csv(OUTPUT_DATA_PATH, index=False, encoding="utf-8-sig")

            st.success("Đã xử lý và lưu dữ liệu thành công!")
            st.info(f"File kết quả: `{OUTPUT_DATA_PATH}`")

        except Exception as e:
            st.error(f"Lỗi khi lưu file: {e}")


# ============================================================
# CHẠY CHƯƠNG TRÌNH
# ============================================================

if __name__ == "__main__":
    main()