# ============================================================
# LỆNH THỰC THI FILE
# ============================================================
# streamlit run shopee_rating_preprocessing/src/preprocessing/06_gemini_correction.py

# ============================================================
# THƯ VIỆN
# ============================================================
import os
import time
import logging
import pandas as pd
import streamlit as st

from google import genai
from shopee_rating_preprocessing.config.api_config import GEMINI_API_KEY
from shopee_rating_preprocessing.config.paths import ROOT_DIR


# ============================================================
# CẤU HÌNH
# ============================================================
INPUT_DATA_PATH = f"{ROOT_DIR}/data/processed/04_slang_mapped.csv"
OUTPUT_DATA_PATH = f"{ROOT_DIR}/data/processed/05_gemini_corrected.csv"

BATCH_SIZE = 50
DELAY_SECONDS = 15.0
MAX_RETRIES = 3
MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# LOG
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


# ============================================================
# GEMINI CLIENT
# ============================================================
def get_gemini_client():
    if not GEMINI_API_KEY:
        raise ValueError(
            "Không tìm thấy GEMINI_API_KEY trong file .env."
        )

    return genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# NẠP DỮ LIỆU INPUT
# ============================================================
@st.cache_data
def load_input_data():
    return pd.read_csv(
        INPUT_DATA_PATH,
        encoding="utf-8-sig"
    )


# ============================================================
# ĐỌC OUTPUT CHECKPOINT
#
# QUAN TRỌNG:
# Checkpoint = số bản ghi trong file output
#
# Output chỉ chứa các dòng đã xử lý.
# Không có dòng chưa xử lý.
# ============================================================
def load_output_data():
    if not os.path.exists(OUTPUT_DATA_PATH):
        return pd.DataFrame()

    try:
        output_df = pd.read_csv(
            OUTPUT_DATA_PATH,
            encoding="utf-8-sig"
        )

        return output_df

    except pd.errors.EmptyDataError:
        return pd.DataFrame()


# ============================================================
# PROMPT GEMINI
# ============================================================
def build_prompt(comments):

    numbered_comments = "\n".join(
        [
            f"{i + 1}. {c}"
            for i, c in enumerate(comments)
        ]
    )

    return f"""Bạn là chuyên gia xử lý và chuẩn hóa văn bản tiếng Việt chuyên dụng cho dữ liệu đánh giá (reviews) thương mại điện tử Shopee.

    Nhiệm vụ: Kiểm tra từng comment và thực hiện chuẩn hóa văn bản theo đúng các quy tắc bắt buộc dưới đây.

    QUY TẮC BẮT BUỘC:
    1. CHUẨN HÓA TỪ LÓNG & TEENCODE:
    - Quy đổi các teencode phổ biến, viết tắt tiếng Việt quen thuộc trên mạng (vd: k, ko, hok -> không; đc -> được; j -> gì; trc -> trước; mk -> mình;...) về dạng tiếng Việt chuẩn.
    - Giữ lại sắc thái tự nhiên của đánh giá, không biến đổi thành văn phong quá cứng nhắc, sách vở.

    2. XỬ LÝ TIẾNG ANH, LAI TẠP & BẢO TOÀN TỪ GỐC:
    - Dịch các cụm từ/câu tiếng Anh thuần túy sang tiếng Việt tự nhiên (vd: good -> tốt, fast delivery -> giao hàng nhanh).
    - GIỮ NGUYÊN TUYỆT ĐỐI các từ mượn phổ biến (vd: shop, sale, voucher, live, flash sale, order, feedback).
    - **QUY TẮC CHỐNG TỰ Ý ĐỔI TỪ (Crucial)**: KHÔNG ĐƯỢC tự ý sửa các từ tiếng Việt hợp lệ thành từ tiếng Anh chỉ vì có âm tiết gần giống (ví dụ: từ "sóc", "bụt", "sốt",... giữ nguyên, tuyệt đối KHÔNG đổi thành "shop"). Chỉ chuẩn hóa teencode và lỗi chính tả rõ ràng.

    3. GIỮ NGUYÊN Ý NGHĨA & CẤU TRÚC:
    - Tuyệt đối không làm sai lệch ý nghĩa khen, chê hoặc cảm xúc gốc của khách hàng.
    - Không tự ý viết hoa, không thêm bớt dấu câu vô tội vạ nếu văn bản gốc đã rõ nghĩa.

    4. ĐỊNH DẠNG ĐẦU RA (RẤT QUAN TRỌNG):
    - Giữ nguyên chính xác số lượng comment và đúng thứ tự ban đầu.
    - Chỉ trả về nội dung các comment đã được xử lý.
    - Mỗi comment nằm trên một dòng riêng biệt.
    - TUYỆT ĐỐI KHÔNG thêm số thứ tự, dấu gạch đầu dòng, lời dẫn, giải thích hoặc bất kỳ nhận xét nào ngoài danh sách comment kết quả.

    DANH SÁCH COMMENT CẦN XỬ LÝ:
    {numbered_comments}"""

# ============================================================
# GEMINI RETRY
# ============================================================
def fix_batch_comments_with_retry(client, comments):

    prompt = build_prompt(comments)

    backoff_time = 5.0

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            if not response.text:
                raise ValueError(
                    "Gemini trả về kết quả rỗng."
                )

            results = [
                line.strip()
                for line in response.text.strip().splitlines()
                if line.strip()
            ]

            cleaned_results = []

            for line in results:

                line = line.strip()

                # Gemini có thể tự thêm số thứ tự
                if (
                    ". " in line[:5]
                    and line[0].isdigit()
                ):
                    line = line.split(". ", 1)[1]

                elif line.startswith("- "):
                    line = line[2:]

                elif line.startswith("* "):
                    line = line[2:]

                cleaned_results.append(
                    line.strip()
                )

            # Kiểm tra số lượng kết quả
            if len(cleaned_results) != len(comments):

                raise ValueError(
                    f"Số câu Gemini trả về "
                    f"({len(cleaned_results)}) "
                    f"không khớp số câu đầu vào "
                    f"({len(comments)})."
                )

            return cleaned_results

        except Exception as e:

            error_str = str(e)

            is_retryable = any(
                x in error_str
                for x in [
                    "503",
                    "429",
                    "ResourceExhausted",
                    "Unavailable",
                    "overloaded",
                    "timeout",
                    "Timeout"
                ]
            )

            if (
                attempt < MAX_RETRIES
                and is_retryable
            ):

                logging.warning(
                    f"Lỗi Gemini lần "
                    f"{attempt}/{MAX_RETRIES}. "
                    f"Chờ {backoff_time:.1f}s."
                )

                time.sleep(backoff_time)

                backoff_time *= 2

            else:

                logging.error(
                    f"Gemini thất bại sau "
                    f"{attempt} lần thử: {e}"
                )

                raise


# ============================================================
# KIỂM TRA OUTPUT CÓ HỢP LỆ KHÔNG
# ============================================================
def validate_output_structure(input_df, output_df):

    if output_df.empty:
        return True

    input_columns = list(input_df.columns)
    output_columns = list(output_df.columns)

    if input_columns != output_columns:

        st.warning(
            "Cấu trúc cột của file output không khớp "
            "file input. Output sẽ được xử lý lại."
        )

        return False

    # Output không được lớn hơn input
    if len(output_df) > len(input_df):

        st.warning(
            "File output có số dòng lớn hơn input. "
            "Checkpoint không hợp lệ."
        )

        return False

    return True


# ============================================================
# KHỞI TẠO CHECKPOINT
#
# Nếu chưa có output:
#     tạo file rỗng nhưng có đầy đủ header.
#
# Nếu đã có output:
#     giữ nguyên.
#
# KHÔNG tạo toàn bộ input vào output.
# ============================================================
def initialize_checkpoint(input_df):

    os.makedirs(
        os.path.dirname(OUTPUT_DATA_PATH),
        exist_ok=True
    )

    output_df = load_output_data()

    # --------------------------------------------------------
    # Chưa có output
    # --------------------------------------------------------
    if not os.path.exists(OUTPUT_DATA_PATH):

        empty_output = pd.DataFrame(
            columns=input_df.columns
        )

        empty_output.to_csv(
            OUTPUT_DATA_PATH,
            index=False,
            encoding="utf-8-sig"
        )

        return empty_output

    # --------------------------------------------------------
    # File tồn tại nhưng rỗng
    # --------------------------------------------------------
    if output_df.empty:

        # Đảm bảo header đúng input
        empty_output = pd.DataFrame(
            columns=input_df.columns
        )

        empty_output.to_csv(
            OUTPUT_DATA_PATH,
            index=False,
            encoding="utf-8-sig"
        )

        return empty_output

    # --------------------------------------------------------
    # Kiểm tra cấu trúc
    # --------------------------------------------------------
    if not validate_output_structure(
        input_df,
        output_df
    ):

        st.error(
            "File output hiện tại không hợp lệ "
            "so với file input."
        )

        return None

    return output_df


# ============================================================
# CHECKPOINT
#
# CỰC KỲ QUAN TRỌNG:
#
# checkpoint = số dòng thực tế trong output
#
# Không dùng:
#     _gemini_processed
#     index
#     max processed index
#
# Ví dụ:
#
# input = 10.000 dòng
# output = 350 dòng
#
# checkpoint = 350
#
# Lô tiếp theo:
# input[350:400]
# ============================================================
def get_checkpoint(output_df):

    if output_df is None:
        return 0

    return len(output_df)


# ============================================================
# APPEND MỘT LÔ VÀO OUTPUT
#
# KHÔNG GHI ĐÈ FILE CŨ.
#
# Cách làm:
#
# output cũ:
#   0 -> 349
#
# batch mới:
#   350 -> 399
#
# sau append:
#   0 -> 399
# ============================================================
def append_batch_to_output(batch_df):

    os.makedirs(
        os.path.dirname(OUTPUT_DATA_PATH),
        exist_ok=True
    )

    file_exists = os.path.exists(
        OUTPUT_DATA_PATH
    )

    file_has_data = False

    if file_exists:
        try:
            file_has_data = (
                os.path.getsize(
                    OUTPUT_DATA_PATH
                ) > 0
            )
        except Exception:
            file_has_data = False

    # --------------------------------------------------------
    # Nếu chưa có file hoặc file rỗng:
    # ghi header + dữ liệu
    # --------------------------------------------------------
    if not file_exists or not file_has_data:

        batch_df.to_csv(
            OUTPUT_DATA_PATH,
            index=False,
            encoding="utf-8-sig",
            mode="w",
            header=True
        )

    # --------------------------------------------------------
    # Nếu đã có dữ liệu:
    # APPEND, không ghi header
    # --------------------------------------------------------
    else:

        batch_df.to_csv(
            OUTPUT_DATA_PATH,
            index=False,
            encoding="utf-8-sig",
            mode="a",
            header=False
        )


# ============================================================
# ĐỌC LẠI TOÀN BỘ OUTPUT SAU KHI APPEND
# ============================================================
def reload_output_data():

    return pd.read_csv(
        OUTPUT_DATA_PATH,
        encoding="utf-8-sig"
    )


# ============================================================
# LỊCH SỬ CÁC LÔ ĐÃ XỬ LÝ
#
# Dựa hoàn toàn vào output.
#
# Ví dụ output có 150 dòng:
#
# Lô 1: 0-49
# Lô 2: 50-99
# Lô 3: 100-149
#
# Tất cả đều được hiển thị.
# ============================================================
def build_processed_history(
    input_df,
    output_df
):

    if output_df.empty:
        return pd.DataFrame(
            columns=[
                "Lô",
                "Dòng",
                "Comment trước",
                "Comment sau",
                "Thay đổi"
            ]
        )

    history = output_df.copy()

    processed_count = min(
        len(output_df),
        len(input_df)
    )

    original_comments = (
        input_df.iloc[:processed_count]["Comment"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    corrected_comments = (
        output_df.iloc[:processed_count]["Comment"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    history_df = pd.DataFrame({
        "Dòng": range(
            0,
            processed_count
        ),
        "Comment trước": original_comments,
        "Comment sau": corrected_comments
    })

    history_df["Thay đổi"] = (
        history_df["Comment trước"]
        != history_df["Comment sau"]
    )

    history_df["Lô"] = (
        history_df["Dòng"] // BATCH_SIZE
    ) + 1

    return history_df[
        [
            "Lô",
            "Dòng",
            "Comment trước",
            "Comment sau",
            "Thay đổi"
        ]
    ]


# ============================================================
# THỐNG KÊ
# ============================================================
def get_statistics(
    input_df,
    output_df
):

    total = len(input_df)

    processed = min(
        len(output_df),
        total
    )

    if processed == 0:
        changed = 0
    else:

        input_comments = (
            input_df.iloc[:processed]["Comment"]
            .fillna("")
            .astype(str)
            .reset_index(drop=True)
        )

        output_comments = (
            output_df.iloc[:processed]["Comment"]
            .fillna("")
            .astype(str)
            .reset_index(drop=True)
        )

        changed = int(
            (
                input_comments
                != output_comments
            ).sum()
        )

    return (
        total,
        processed,
        changed
    )


# ============================================================
# MAIN
# ============================================================
def main():

    st.set_page_config(
        page_title="Gemini Text Correction"
    )

    st.title(
        "Gemini Text Correction"
    )

    st.caption(
        "Sửa lỗi chính tả, lỗi đánh máy "
        "và dịch tiếng Anh sang tiếng Việt."
    )

    st.markdown("---")

    # ========================================================
    # SESSION STATE
    # ========================================================
    if "running" not in st.session_state:
        st.session_state.running = False

    if "current_index" not in st.session_state:
        st.session_state.current_index = None

    if "last_batch_message" not in st.session_state:
        st.session_state.last_batch_message = ""

    # ========================================================
    # NẠP INPUT
    # ========================================================
    try:

        input_df = load_input_data()

    except Exception as e:

        st.error(
            f"Lỗi khi đọc dữ liệu đầu vào: {e}"
        )

        return

    # ========================================================
    # KIỂM TRA INPUT
    # ========================================================
    if input_df.empty:

        st.warning(
            "Dữ liệu đầu vào đang trống."
        )

        return

    if "Comment" not in input_df.columns:

        st.error(
            "Không tìm thấy cột 'Comment'."
        )

        return

    # ========================================================
    # KHỞI TẠO / ĐỌC OUTPUT
    # ========================================================
    try:

        output_df = initialize_checkpoint(
            input_df
        )

        if output_df is None:
            return

    except Exception as e:

        st.error(
            f"Lỗi khi tạo/đọc checkpoint: {e}"
        )

        return

    # ========================================================
    # CHECKPOINT = SỐ DÒNG OUTPUT
    # ========================================================
    checkpoint = get_checkpoint(
        output_df
    )

    total = len(input_df)

    # --------------------------------------------------------
    # Nếu session chưa có vị trí:
    # lấy trực tiếp từ số dòng output
    # --------------------------------------------------------
    if st.session_state.current_index is None:

        st.session_state.current_index = checkpoint

    # --------------------------------------------------------
    # Nếu output đã tăng lên:
    # luôn đồng bộ checkpoint từ output
    # --------------------------------------------------------
    else:

        st.session_state.current_index = checkpoint

    current_index = (
        st.session_state.current_index
    )

    # ========================================================
    # BẢO VỆ CHECKPOINT
    # ========================================================
    if current_index > total:

        st.error(
            "Checkpoint lớn hơn số dòng input. "
            "File output không hợp lệ."
        )

        st.session_state.running = False

        return

    # ========================================================
    # THỐNG KÊ
    # ========================================================
    (
        total_rows,
        processed_rows,
        changed_rows
    ) = get_statistics(
        input_df,
        output_df
    )

    # ========================================================
    # PROGRESS
    # ========================================================
    progress_value = (
        processed_rows / total_rows
        if total_rows
        else 1.0
    )

    st.progress(
        progress_value
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Tổng số dòng",
        f"{total_rows:,}"
    )

    col2.metric(
        "Đã xử lý",
        f"{processed_rows:,}"
    )

    col3.metric(
        "Còn lại",
        f"{max(total_rows - processed_rows, 0):,}"
    )

    st.caption(
        f"Tiến trình: "
        f"{processed_rows:,}/{total_rows:,} "
        f"({progress_value:.1%})"
    )

    if st.session_state.last_batch_message:

        st.success(
            st.session_state.last_batch_message
        )

    st.markdown("---")

    # ========================================================
    # LỊCH SỬ CÁC LÔ ĐÃ XỬ LÝ
    #
    # QUAN TRỌNG:
    # Không chỉ hiển thị lô hiện tại.
    #
    # Toàn bộ các dòng đã xử lý được đọc từ output
    # và hiển thị lại sau mỗi rerun.
    # ========================================================
    if processed_rows > 0:

        st.subheader(
            "Các lô đã xử lý"
        )

        history_df = build_processed_history(
            input_df,
            output_df
        )

        # ----------------------------------------------------
        # Hiển thị toàn bộ lịch sử
        # ----------------------------------------------------
        st.dataframe(
            history_df,
            use_container_width=True,
            height=500,
            hide_index=True
        )

        changed_history = int(
            history_df["Thay đổi"].sum()
        )

        st.caption(
            f"Đã lưu {processed_rows:,} dòng. "
            f"Có {changed_history:,} comment được thay đổi."
        )

        st.markdown("---")

    # ========================================================
    # NÚT ĐIỀU KHIỂN
    # ========================================================
    col_start, col_stop = st.columns(2)

    with col_start:

        start_button = st.button(
            "▶ Bắt đầu / Tiếp tục",
            type="primary",
            use_container_width=True,
            disabled=(
                st.session_state.running
                or current_index >= total
            )
        )

    with col_stop:

        stop_button = st.button(
            "⏹ Dừng",
            use_container_width=True,
            disabled=(
                not st.session_state.running
            )
        )

    # ========================================================
    # BẮT ĐẦU / TIẾP TỤC
    # ========================================================
    if start_button:

        st.session_state.running = True

        st.session_state.last_batch_message = ""

        st.rerun()

    # ========================================================
    # DỪNG
    # ========================================================
    if stop_button:

        st.session_state.running = False

        st.session_state.last_batch_message = (
            f"Đã dừng tại checkpoint "
            f"{current_index:,}/{total:,}. "
            f"Dữ liệu các lô trước vẫn được giữ nguyên."
        )

        st.rerun()

    # ========================================================
    # HOÀN TẤT
    # ========================================================
    if current_index >= total:

        st.session_state.running = False

        st.success(
            "Đã xử lý hoàn tất toàn bộ dữ liệu."
        )

        st.info(
            f"File kết quả: `{OUTPUT_DATA_PATH}`"
        )

        return

    # ========================================================
    # KHÔNG CHẠY
    # ========================================================
    if not st.session_state.running:

        st.info(
            f"Đang chờ xử lý. "
            f"Lần chạy tiếp theo sẽ bắt đầu từ dòng "
            f"{current_index:,}."
        )

        return

    # ========================================================
    # XÁC ĐỊNH LÔ HIỆN TẠI
    # ========================================================
    batch_start = current_index

    batch_end = min(
        batch_start + BATCH_SIZE,
        total
    )

    batch_number = (
        batch_start // BATCH_SIZE
    ) + 1

    total_batches = (
        total + BATCH_SIZE - 1
    ) // BATCH_SIZE

    st.markdown("---")

    st.subheader(
        f"Đang xử lý lô {batch_number}/{total_batches}"
    )

    st.write(
        f"Dòng **{batch_start:,} → "
        f"{batch_end - 1:,}** "
        f"({batch_end - batch_start:,} comment)"
    )

    # ========================================================
    # GEMINI CLIENT
    # ========================================================
    try:

        client = get_gemini_client()

    except Exception as e:

        st.session_state.running = False

        st.error(
            f"Lỗi khởi tạo Gemini: {e}"
        )

        return

    # ========================================================
    # COMMENT INPUT CỦA LÔ
    # ========================================================
    batch_original = (
        input_df
        .iloc[batch_start:batch_end]["Comment"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # ========================================================
    # GỌI GEMINI
    # ========================================================
    try:

        with st.spinner(
            f"Gemini đang xử lý "
            f"lô {batch_number}/{total_batches}..."
        ):

            fixed_batch = (
                fix_batch_comments_with_retry(
                    client,
                    batch_original
                )
            )

    except Exception as e:

        st.session_state.running = False

        st.error(
            f"Lỗi khi xử lý lô "
            f"{batch_number}: {e}"
        )

        st.warning(
            f"Checkpoint vẫn ở "
            f"{current_index:,}/{total:,}. "
            f"Bạn có thể bấm "
            f"'Bắt đầu / Tiếp tục' "
            f"để thử lại lô này."
        )

        return

    # ========================================================
    # PREVIEW LÔ HIỆN TẠI
    # ========================================================
    preview_batch = pd.DataFrame({
        "Comment trước": batch_original,
        "Comment sau": fixed_batch
    })

    preview_batch["Thay đổi"] = (
        preview_batch["Comment trước"]
        != preview_batch["Comment sau"]
    )

    st.subheader(
        f"Kết quả lô {batch_number}"
    )

    st.dataframe(
        preview_batch,
        use_container_width=True,
        height=450,
        hide_index=True
    )

    changed_count = int(
        preview_batch["Thay đổi"].sum()
    )

    st.caption(
        f"Lô này có **{changed_count:,}/"
        f"{len(preview_batch):,}** "
        f"comment được thay đổi."
    )

    # ========================================================
    # TẠO DATAFRAME OUTPUT CHO LÔ
    #
    # Lấy NGUYÊN CÁC CỘT từ input
    # để output luôn có đầy đủ cột.
    #
    # Chỉ thay Comment.
    # ========================================================
    batch_output = (
        input_df
        .iloc[batch_start:batch_end]
        .copy()
        .reset_index(drop=True)
    )

    batch_output["Comment"] = fixed_batch

    # ========================================================
    # KIỂM TRA SỐ DÒNG TRƯỚC KHI APPEND
    # ========================================================
    if len(batch_output) != (
        batch_end - batch_start
    ):

        st.error(
            "Số dòng batch không hợp lệ. "
            "Không ghi dữ liệu."
        )

        st.session_state.running = False

        return

    # ========================================================
    # APPEND VÀO FILE OUTPUT
    #
    # KHÔNG GHI ĐÈ.
    #
    # Ví dụ:
    #
    # output hiện tại = 100 dòng
    #
    # batch mới = 50 dòng
    #
    # append -> output = 150 dòng
    # ========================================================
    try:

        append_batch_to_output(
            batch_output
        )

        # ----------------------------------------------------
        # ĐỌC LẠI FILE SAU KHI LƯU
        # ----------------------------------------------------
        output_df = reload_output_data()

        # ----------------------------------------------------
        # Checkpoint mới = số dòng output
        # ----------------------------------------------------
        new_checkpoint = get_checkpoint(
            output_df
        )

        # ----------------------------------------------------
        # Kiểm tra checkpoint
        # ----------------------------------------------------
        if new_checkpoint != batch_end:

            raise ValueError(
                "Checkpoint sau khi lưu không khớp. "
                f"Expected={batch_end}, "
                f"Actual={new_checkpoint}"
            )

    except Exception as e:

        st.session_state.running = False

        st.error(
            f"Gemini đã xử lý xong nhưng "
            f"không thể append/lưu lô "
            f"{batch_number}: {e}"
        )

        st.warning(
            "Lô này CHƯA được xem là hoàn tất. "
            "Không tăng checkpoint."
        )

        return

    # ========================================================
    # CẬP NHẬT CHECKPOINT
    # ========================================================
    st.session_state.current_index = (
        new_checkpoint
    )

    # ========================================================
    # THÔNG BÁO LƯU
    # ========================================================
    st.success(
        f"Đã append và lưu lô "
        f"{batch_number}/{total_batches} thành công."
    )

    st.info(
        f"Đã lưu thêm dòng "
        f"**{batch_start:,} → "
        f"{batch_end - 1:,}**. "
        f"Checkpoint hiện tại: "
        f"**{new_checkpoint:,}/{total:,}**."
    )

    # ========================================================
    # HOÀN TẤT TOÀN BỘ
    # ========================================================
    if new_checkpoint >= total:

        st.session_state.running = False

        st.session_state.last_batch_message = (
            f"Hoàn tất toàn bộ dữ liệu: "
            f"{new_checkpoint:,}/{total:,} dòng."
        )

        st.success(
            "Đã xử lý và lưu toàn bộ dữ liệu "
            "thành công."
        )

        st.info(
            f"File kết quả: `{OUTPUT_DATA_PATH}`"
        )

        return

    # ========================================================
    # CHỜ TRƯỚC LÔ TIẾP THEO
    # ========================================================
    st.info(
        f"Lô {batch_number} đã được append vào file. "
        f"Đang chờ {DELAY_SECONDS:.0f} giây "
        f"trước khi chạy lô tiếp theo..."
    )

    time.sleep(
        DELAY_SECONDS
    )

    # ========================================================
    # THÔNG BÁO CHO LẦN RERUN TIẾP
    # ========================================================
    st.session_state.last_batch_message = (
        f"Đã lưu lô {batch_number}/{total_batches}. "
        f"Checkpoint: "
        f"{new_checkpoint:,}/{total:,} dòng."
    )

    # ========================================================
    # RERUN
    #
    # Lần rerun:
    #
    # 1. Đọc lại output
    # 2. len(output) = checkpoint
    # 3. Hiển thị toàn bộ lịch sử
    # 4. Chạy lô tiếp theo
    # ========================================================
    st.rerun()


# ============================================================
# CHẠY CHƯƠNG TRÌNH
# ============================================================
if __name__ == "__main__":
    main()