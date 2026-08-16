"""
Core text-preprocessing cho tiếng Việt: emoji, slang, segmentation, negation.

Module này được import bởi:
  - src.preprocessing.prepare_regression  (pipeline production regression)
  - app.py                                  (predict-time preprocessing)
  - scripts/smoke_test.py                   (CI test)

Chạy trực tiếp file này sẽ chạy main() — preprocess legacy cho binary classification
(drop 3★, gắn nhãn Sentiment). Pipeline regression dùng prepare_regression.py.
"""
import os
import re
import sys
import pandas as pd

# Word segmentation tiếng Việt (optional - chỉ cần khi USE_SEGMENT=True)
USE_SEGMENT = True
try:
    from underthesea import word_tokenize
except Exception:
    USE_SEGMENT = False
    print("⚠️ Chưa cài underthesea, bỏ qua word segmentation.")


# ==========================
# DICTIONARIES
# ==========================
EMOJI_DICT = {
    # Positive
    "❤️": " tuyệt_vời ", "❤": " tuyệt_vời ",
    "👍": " tốt ", "🤩": " hài_lòng ",
    "😊": " vui ", "😄": " vui ", "😃": " vui ", "🙂": " vui ",
    "😍": " yêu_thích ", "🥰": " yêu_thích ", "😘": " yêu_thích ",
    "👌": " ok ", "💯": " hoàn_hảo ",
    "⭐": " 5_sao ", "🌟": " 5_sao ", "✨": " tốt ",
    "🔥": " cực_hot ",
    "💖": " yêu_thích ", "💕": " yêu_thích ", "💗": " yêu_thích ", "💓": " yêu_thích ",
    "😁": " vui ", "😆": " vui ",
    "🥳": " vui ", "🎉": " vui ",
    "✅": " tốt ", "☑": " tốt ",
    # Negative
    "😡": " tệ_quá ", "😠": " tệ_quá ", "🤬": " tệ_quá ",
    "👎": " kém ",
    "🙄": " thất_vọng ", "😒": " thất_vọng ",
    "😅": " hơi_tệ ", "😬": " hơi_tệ ",
    "😭": " quá_buồn ", "😢": " quá_buồn ", "😞": " buồn ", "😔": " buồn ",
    "🤮": " ghê ", "🤢": " ghê ",
    "❌": " không_nên_mua ", "⛔": " không_nên_mua ",
    "💩": " tệ_quá ",
    "😤": " bực ", "😣": " khó_chịu ",
}

SLANG_DICT = {
    # Phủ định
    "ko": "không", "k": "không", "kh": "không", "khong": "không", "kg": "không",
    "khg": "không", "hông": "không", "hong": "không",
    # Phản hồi/shop/sản phẩm
    "rep": "phản hồi", "fb": "phản hồi", "feedback": "phản hồi",
    "ship": "giao hàng", "shipper": "người giao hàng", "shiper": "người giao hàng",
    "shop": "cửa hàng", "st": "cửa hàng",
    "sp": "sản phẩm", "sphm": "sản phẩm",
    "sd": "sử dụng", "sdung": "sử dụng",
    # OK
    "ok": "tốt", "oke": "tốt", "okay": "tốt", "okey": "tốt", "oki": "tốt",
    "okela": "tốt", "okie": "tốt",
    # Cảm ơn
    "tks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn",
    "thx": "cảm ơn", "cam on": "cảm ơn", "camon": "cảm ơn",
    # Được
    "dc": "được", "đc": "được", "duoc": "được",
    "bt": "bình thường", "btthuong": "bình thường",
    # Nhưng/vẫn
    "nma": "nhưng mà", "nhưg": "nhưng", "nhg": "nhưng",
    # Mình/bạn/mọi người
    "mik": "mình", "mk": "mình", "mn": "mọi người",
    # Khác
    "h": "giờ", "j": "gì", "đt": "điện thoại", "dt": "điện thoại",
    "vs": "với", "v": "vậy", "z": "vậy", "vay": "vậy",
    "r": "rồi", "rui": "rồi", "rồiii": "rồi",
    "ms": "mới", "mơi": "mới",
    "đẹppp": "đẹp", "xinhh": "xinh",
    "qa": "quá", "wa": "quá",
    "lém": "lắm", "lém ạ": "lắm",
    # English sentiment
    "good": "tốt", "nice": "đẹp", "perfect": "hoàn_hảo",
    "bad": "tệ", "very": "rất", "so": "rất",
    "great": "tuyệt_vời", "excellent": "xuất_sắc",
    "awesome": "tuyệt_vời", "amazing": "tuyệt_vời",
    "hot": "nóng", "cool": "mát",
    "best": "tốt_nhất", "cheap": "rẻ",
    "high": "cao", "quality": "chất_lượng",
    "fake": "giả", "real": "thật",
    "auth": "chính_hãng", "authentic": "chính_hãng",
    "love": "yêu_thích", "like": "thích",
    "fast": "nhanh", "slow": "chậm",
    "yes": "có", "no": "không",
}

# Các từ phủ định cần ghép với từ kế tiếp
NEGATION_WORDS = {"không", "chẳng", "chả", "kg", "đâu", "đéo", "ko", "k"}

VIETNAMESE_CHARS = (
    r"a-z0-9A-Zàáảãạâầấẩẫậăằắẳẵặ"
    r"èéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
    r"ùúủũụưừứửữựỳýỷỹỵđ"
)


def normalize_text(text):
    """Bước 1-7: lowercase, emoji, dedup chars, URL, slang, special chars, whitespace."""
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Emoji → từ
    for emo, rep in EMOJI_DICT.items():
        text = text.replace(emo, rep)

    # 3. Khử ký tự lặp (≥3 lần → 1)
    text = re.sub(
        r"([" + VIETNAMESE_CHARS.lower() + r"])\1{2,}",
        r"\1",
        text,
    )

    # 4. Xóa URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # 5. Slang/English mapping (token-level)
    words = text.split()
    words = [SLANG_DICT.get(w, w) for w in words]
    text = " ".join(words)

    # 6. Xóa special chars (giữ chữ VN + số + underscore)
    text = re.sub(r"[^" + VIETNAMESE_CHARS + r"_\s]", " ", text)

    # 7. Trim whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def handle_negation(tokens):
    """Ghép 'không' (và các từ phủ định) với token kế tiếp: 'không tốt' → 'không_tốt'."""
    out = []
    i = 0
    while i < len(tokens):
        w = tokens[i]
        if w in NEGATION_WORDS and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if "_" not in nxt and nxt not in NEGATION_WORDS:
                out.append(f"{w}_{nxt}")
                i += 2
                continue
        out.append(w)
        i += 1
    return out


def segment(text):
    """Word segmentation tiếng Việt: 'sản phẩm' → 'sản_phẩm'."""
    if not USE_SEGMENT or not text:
        return text
    try:
        toks = word_tokenize(text)  # list of multi-word strings
        # Ghép multi-word tokens bằng "_"
        merged = [t.replace(" ", "_") for t in toks]
        return " ".join(merged)
    except Exception:
        return text


def preprocess_text(text):
    text = normalize_text(text)
    if not text:
        return ""
    # Segmentation tiếng Việt (sản phẩm → sản_phẩm)
    text = segment(text)
    # Negation: làm sau segmentation để 'không tốt' vẫn merge được nếu underthesea không tự gộp
    tokens = text.split()
    tokens = handle_negation(tokens)
    return " ".join(tokens)


# ==========================
# Legacy main: preprocess cho binary classification (drop 3★)
# Pipeline production regression dùng src/preprocessing/prepare_regression.py
# ==========================
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    raw_dir = os.path.join(project_root, "data", "raw")
    input_file = os.path.join(raw_dir, "shopee_reviews.csv")
    output_file = os.path.join(raw_dir, "shopee_reviews_preprocessed.csv")

    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file: {input_file}")
        return

    print(f"📂 Đang đọc dữ liệu từ: {input_file}...")
    df = pd.read_csv(input_file)

    print(f"🧹 Đang thực hiện tiền xử lý văn bản (USE_SEGMENT={USE_SEGMENT})...")
    df["Raw_Comment"] = df["Comment"]
    df["Comment"] = df["Comment"].apply(preprocess_text)

    print("🏷️ Đang gắn nhãn cảm xúc (Sentiment) - 2 lớp, drop Neutral...")
    def label_sentiment(rating):
        if rating <= 2:
            return 0  # Negative
        elif rating == 3:
            return None  # Drop
        else:
            return 1  # Positive

    df["Sentiment"] = df["Rating"].apply(label_sentiment)
    before = len(df)
    df = df.dropna(subset=["Sentiment"]).copy()
    df["Sentiment"] = df["Sentiment"].astype(int)
    print(f"🗑️ Đã drop {before - len(df)} dòng Neutral (3★).")

    initial_len = len(df)
    df = df[df["Comment"] != ""]
    print(f"🗑️ Đã loại bỏ {initial_len - len(df)} bình luận rỗng.")

    print(f"💾 Đang lưu vào: {output_file}...")
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print("✅ Hoàn thành!")


if __name__ == "__main__":
    main()
