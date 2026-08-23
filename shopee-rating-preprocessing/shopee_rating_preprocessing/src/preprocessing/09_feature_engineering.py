# ============================================================
# LỆNH THỰC THI
# ============================================================
# streamlit run shopee_rating_preprocessing/src/preprocessing/09_feature_engineering.py

# ============================================================
# THƯ VIỆN
# ============================================================
import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from shopee_rating_preprocessing.utils.present import print_header

warnings.filterwarnings("ignore")

# ============================================================
# ROOT DIR & PATH
# ============================================================
try:
    from shopee_rating_preprocessing.config.paths import ROOT_DIR
except Exception as e:
    print("Lỗi import ROOT_DIR:", e)
    ROOT_DIR = os.getcwd()

INPUT_DATA_PATH = os.path.join(
    ROOT_DIR, "data", "processed", "07_stopword_removed.csv"
)
OUTPUT_DATA_PATH = os.path.join(
    ROOT_DIR, "data", "processed", "08_feature_engineered.csv"
)

TARGET_COL = "Rating"

# Ngưỡng mặc định
NUMERIC_CORR_THRESHOLD = 0.05      # |r| < 0.05: tương quan rất yếu
CATEGORICAL_ETA_THRESHOLD = 0.01    # Eta² < 0.01: liên hệ rất yếu
MULTICOL_THRESHOLD = 0.85          # |r| > 0.85: nguy cơ đa cộng tuyến
VIF_THRESHOLD = 10.0               # VIF > 10: đa cộng tuyến nghiêm trọng


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_input_data():
    if not os.path.exists(INPUT_DATA_PATH):
        return None
    return pd.read_csv(INPUT_DATA_PATH).copy()


# ============================================================
# FEATURE ENGINEERING
# Input chỉ cần:
# Comment, CommentDate, SubCategory, Rating, Helpfulness_Score
# ============================================================
def extract_features(df, progress_bar, status_text):
    df = df.copy()

    status_text.text("Đang chuẩn hóa dữ liệu...")
    progress_bar.progress(10)

    # --------------------------------------------------------
    # Chuẩn hóa cột gốc
    # --------------------------------------------------------
    df["Comment"] = df["Comment"].fillna("").astype(str)
    df["SubCategory"] = df["SubCategory"].fillna("").astype(str)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Helpfulness_Score"] = pd.to_numeric(
        df["Helpfulness_Score"], errors="coerce"
    )

    # --------------------------------------------------------
    # TEXT FEATURES
    # --------------------------------------------------------
    comment = df["Comment"]

    # Tổng số ký tự trong Comment
    df["char_count"] = comment.str.len().astype("int32")

    # Tổng số từ trong Comment
    df["word_count"] = comment.str.split().str.len().astype("int32")

    # Độ dài trung bình của một từ
    df["avg_word_length"] = np.where(
        df["word_count"] > 0,
        df["char_count"] / df["word_count"],
        0
    )

    # Số dấu ! -> mức độ biểu cảm
    df["exclamation_count"] = comment.str.count("!").astype("int16")

    # Số dấu ? -> mức độ nghi vấn/thắc mắc
    df["question_count"] = comment.str.count(r"\?").astype("int16")

    # Số chữ số xuất hiện trong Comment
    df["digit_count"] = comment.apply(
        lambda x: sum(c.isdigit() for c in x)
    ).astype("int16")

    # Số câu
    df["sentence_count"] = comment.str.count(
        r"[.!?]+"
    ).astype("int16")

    # Số chữ in hoa
    df["uppercase_count"] = comment.apply(
        lambda x: sum(c.isupper() for c in x)
    ).astype("int16")

    # Số ký tự đặc biệt
    df["special_char_count"] = comment.apply(
        lambda x: sum(
            not c.isalnum() and not c.isspace()
            for c in x
        )
    ).astype("int16")

    # Tỷ lệ chữ số trên tổng số ký tự
    df["digit_ratio"] = np.where(
        df["char_count"] > 0,
        df["digit_count"] / df["char_count"],
        0
    )

    # Tỷ lệ dấu ! trên tổng số ký tự
    df["exclamation_ratio"] = np.where(
        df["char_count"] > 0,
        df["exclamation_count"] / df["char_count"],
        0
    )

    # Tỷ lệ dấu ? trên tổng số ký tự
    df["question_ratio"] = np.where(
        df["char_count"] > 0,
        df["question_count"] / df["char_count"],
        0
    )

    status_text.text("Đã tạo các đặc trưng từ Comment...")
    progress_bar.progress(35)

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------
    date = pd.to_datetime(
        df["CommentDate"],
        errors="coerce",
        dayfirst=True
    )

    # Năm nhận xét
    df["comment_year"] = date.dt.year.astype("Int16")

    # Tháng nhận xét: 1-12
    df["comment_month"] = date.dt.month.astype("Int8")

    # Ngày trong tháng: 1-31
    df["comment_day"] = date.dt.day.astype("Int8")

    # Thứ trong tuần: 0 = Thứ 2, 6 = Chủ nhật
    df["comment_dayofweek"] = date.dt.dayofweek.astype("Int8")

    # Giờ nhận xét: 0-23
    df["comment_hour"] = date.dt.hour.astype("Int8")

    # 1 = cuối tuần, 0 = ngày thường
    df["comment_is_weekend"] = (
        date.dt.dayofweek >= 5
    ).astype("int8")

    status_text.text("Hoàn tất tạo Feature Engineering...")
    progress_bar.progress(50)

    return df


# ============================================================
# ETA SQUARED
# Đo mức độ liên hệ giữa biến Categorical và Rating.
#
# Eta²:
#   0   -> gần như không có liên hệ
#   1   -> liên hệ rất mạnh
#
# Dùng cho SubCategory thay vì mã hóa 0,1,2,... rồi Pearson.
# ============================================================
def calculate_eta_squared(categories, target):
    data = pd.DataFrame({
        "category": categories,
        "target": target
    }).dropna()

    if data.empty or data["category"].nunique() < 2:
        return 0.0

    grand_mean = data["target"].mean()

    groups = data.groupby("category")["target"].agg(
        ["mean", "count"]
    )

    between_ss = (
        groups["count"] *
        (groups["mean"] - grand_mean) ** 2
    ).sum()

    total_ss = (
        (data["target"] - grand_mean) ** 2
    ).sum()

    if total_ss == 0:
        return 0.0

    return float(between_ss / total_ss)


# ============================================================
# PHÂN TÍCH NUMERIC FEATURES
# ============================================================
def analyze_numeric_features(df, target_col):
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_cols = [
        c for c in numeric_cols
        if c != target_col
    ]

    results = []

    for col in numeric_cols:
        data = df[[col, target_col]].dropna()

        if len(data) < 3 or data[col].nunique() < 2:
            continue

        # Pearson:
        # đo tương quan tuyến tính, [-1, 1]
        pearson = data[col].corr(
            data[target_col],
            method="pearson"
        )

        # Spearman:
        # đo tương quan theo thứ hạng, phù hợp cả quan hệ đơn điệu
        spearman = data[col].corr(
            data[target_col],
            method="spearman"
        )

        # Chọn mức liên hệ mạnh hơn giữa hai phương pháp
        association = max(
            abs(pearson),
            abs(spearman)
        )

        results.append({
            "Feature": col,
            "Pearson": pearson,
            "Spearman": spearman,
            "Association": association
        })

    result = pd.DataFrame(results)

    if not result.empty:
        result["Keep"] = (
            result["Association"] >= NUMERIC_CORR_THRESHOLD
        )

        result["Reason"] = np.where(
            result["Keep"],
            "Giữ: tương quan đủ với Rating",
            "Loại: tương quan quá thấp"
        )

    return result


# ============================================================
# PHÂN TÍCH CATEGORICAL FEATURES
# ============================================================
def analyze_categorical_features(df, target_col):
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Comment là văn bản tự do, không dùng trực tiếp.
    # CommentDate giữ dạng gốc, không xem là categorical.
    excluded = {
        "Comment",
        "CommentDate",
        "ProductName",
        target_col
    }

    categorical_cols = [
        c for c in categorical_cols
        if c not in excluded
    ]

    results = []

    for col in categorical_cols:
        eta = calculate_eta_squared(
            df[col],
            df[target_col]
        )

        results.append({
            "Feature": col,
            "Unique_Values": df[col].nunique(),
            "Eta_Squared": eta
        })

    result = pd.DataFrame(results)

    if not result.empty:
        result["Keep"] = (
            result["Eta_Squared"] >= CATEGORICAL_ETA_THRESHOLD
        )

        result["Reason"] = np.where(
            result["Keep"],
            "Giữ: Eta² đủ với Rating",
            "Loại: Eta² quá thấp"
        )

    return result


# ============================================================
# XỬ LÝ ĐA CỘNG TUYẾN BẰNG CORRELATION
# ============================================================
def remove_multicollinearity(df, features, target_col):
    if len(features) < 2:
        return features, []

    corr = df[features].corr().abs()

    target_corr = (
        df[features + [target_col]]
        .corr()[target_col]
        .abs()
    )

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )

    dropped = []

    for col in upper.columns:
        high_corr_cols = upper.index[
            upper[col] > MULTICOL_THRESHOLD
        ].tolist()

        for other in high_corr_cols:
            if col in dropped or other in dropped:
                continue

            # Giữ feature có tương quan với Rating cao hơn
            if target_corr[col] >= target_corr[other]:
                drop = other
            else:
                drop = col

            if drop not in dropped:
                dropped.append(drop)

    selected = [
        c for c in features
        if c not in dropped
    ]

    return selected, dropped


# ============================================================
# TÍNH VIF
#
# VIF = 1:
#   Không có đa cộng tuyến.
#
# VIF > 5:
#   Có thể có đa cộng tuyến.
#
# VIF > 10:
#   Đa cộng tuyến nghiêm trọng.
# ============================================================
def calculate_vif(df, features):
    if len(features) < 2:
        return pd.DataFrame({
            "Feature": features,
            "VIF": [1.0] * len(features)
        })

    data = df[features].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if data.empty:
        return pd.DataFrame({
            "Feature": features,
            "VIF": [np.nan] * len(features)
        })

    results = []

    for feature in features:
        others = [
            c for c in features
            if c != feature
        ]

        if not others:
            vif = 1.0
        elif data[feature].nunique() <= 1:
            vif = 1.0
        else:
            y = data[feature].values

            X = np.column_stack([
                np.ones(len(data)),
                data[others].values
            ])

            try:
                prediction = X @ np.linalg.lstsq(
                    X,
                    y,
                    rcond=None
                )[0]

                ss_res = np.sum(
                    (y - prediction) ** 2
                )

                ss_tot = np.sum(
                    (y - y.mean()) ** 2
                )

                r2 = (
                    1 - ss_res / ss_tot
                    if ss_tot > 0
                    else 0
                )

                vif = (
                    np.inf
                    if r2 >= 0.999999
                    else 1 / (1 - r2)
                )

            except Exception:
                vif = np.nan

        results.append({
            "Feature": feature,
            "VIF": vif
        })

    return pd.DataFrame(results)


# ============================================================
# FEATURE SELECTION
# ============================================================
def select_best_features(df, target_col):
    numeric_result = analyze_numeric_features(
        df,
        target_col
    )

    categorical_result = analyze_categorical_features(
        df,
        target_col
    )

    # --------------------------------------------------------
    # Lấy Numeric đủ tương quan
    # --------------------------------------------------------
    selected_numeric = (
        numeric_result.loc[
            numeric_result["Keep"],
            "Feature"
        ].tolist()
        if not numeric_result.empty
        else []
    )

    # --------------------------------------------------------
    # Lấy Categorical đủ Eta²
    # --------------------------------------------------------
    selected_categorical = (
        categorical_result.loc[
            categorical_result["Keep"],
            "Feature"
        ].tolist()
        if not categorical_result.empty
        else []
    )

    # --------------------------------------------------------
    # Loại đa cộng tuyến bằng correlation
    # --------------------------------------------------------
    selected_numeric, corr_dropped = remove_multicollinearity(
        df,
        selected_numeric,
        target_col
    )

    if not numeric_result.empty:
        numeric_result.loc[
            numeric_result["Feature"].isin(corr_dropped),
            "Keep"
        ] = False

        numeric_result.loc[
            numeric_result["Feature"].isin(corr_dropped),
            "Reason"
        ] = (
            "Loại: đa cộng tuyến, giữ feature "
            "có tương quan với Rating cao hơn"
        )

    # --------------------------------------------------------
    # Loại tiếp bằng VIF
    # --------------------------------------------------------
    vif_dropped = []

    while len(selected_numeric) >= 2:
        vif_result = calculate_vif(
            df,
            selected_numeric
        )

        if vif_result.empty:
            break

        max_vif = vif_result["VIF"].max()

        if (
            pd.isna(max_vif)
            or max_vif <= VIF_THRESHOLD
        ):
            break

        feature_to_drop = vif_result.loc[
            vif_result["VIF"].idxmax(),
            "Feature"
        ]

        selected_numeric.remove(
            feature_to_drop
        )

        vif_dropped.append(
            feature_to_drop
        )

    if not numeric_result.empty:
        numeric_result.loc[
            numeric_result["Feature"].isin(vif_dropped),
            "Keep"
        ] = False

        numeric_result.loc[
            numeric_result["Feature"].isin(vif_dropped),
            "Reason"
        ] = "Loại: VIF > ngưỡng, đa cộng tuyến cao"

    # --------------------------------------------------------
    # Các cột gốc luôn giữ
    # --------------------------------------------------------
    original_cols = [
        "Comment",
        "CommentDate",
        "SubCategory",
        "Rating",
        "Helpfulness_Score"
    ]

    final_cols = []

    for col in (
        original_cols
        + selected_categorical
        + selected_numeric
    ):
        if col in df.columns and col not in final_cols:
            final_cols.append(col)

    # Không giữ ProductName nếu xuất hiện
    final_cols = [
        c for c in final_cols
        if c != "ProductName"
    ]

    optimized_df = df[final_cols].copy()

    # --------------------------------------------------------
    # Danh sách feature bị loại
    # --------------------------------------------------------
    dropped_rows = []

    if not numeric_result.empty:
        for _, row in numeric_result[
            ~numeric_result["Keep"]
        ].iterrows():
            dropped_rows.append({
                "Feature": row["Feature"],
                "Type": "Numeric",
                "Score": row["Association"],
                "Reason": row["Reason"]
            })

    if not categorical_result.empty:
        for _, row in categorical_result[
            ~categorical_result["Keep"]
        ].iterrows():
            dropped_rows.append({
                "Feature": row["Feature"],
                "Type": "Categorical",
                "Score": row["Eta_Squared"],
                "Reason": row["Reason"]
            })

    dropped_df = pd.DataFrame(dropped_rows)

    return (
        optimized_df,
        numeric_result,
        categorical_result,
        dropped_df
    )


# ============================================================
# MAIN - STREAMLIT
# ============================================================
def main():
    # KHAI BÁO GLOBAL Ở DÒNG ĐẦU TIÊN CỦA HÀM MAIN
    global NUMERIC_CORR_THRESHOLD
    global CATEGORICAL_ETA_THRESHOLD
    global MULTICOL_THRESHOLD
    global VIF_THRESHOLD

    st.set_page_config(
        page_title="Feature Engineering & Selection",
        layout="wide"
    )

    st.title(
        "Feature Engineering → Correlation → Multicollinearity"
    )

    st.markdown("---")

    df = load_input_data()

    if df is None or df.empty:
        st.error(
            f"Không tìm thấy file dữ liệu:\n\n"
            f"`{INPUT_DATA_PATH}`"
        )
        return

    # --------------------------------------------------------
    # Kiểm tra 5 cột dữ liệu gốc
    # --------------------------------------------------------
    required_cols = [
        "Comment",
        "CommentDate",
        "SubCategory",
        "Rating",
        "Helpfulness_Score"
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        st.error(
            "Thiếu cột dữ liệu gốc: "
            + ", ".join(missing_cols)
        )
        return

    st.info(
        f"**Input:** `{len(df):,}` dòng | "
        f"**5 cột gốc:** `Comment`, `CommentDate`, "
        f"`SubCategory`, `Rating`, `Helpfulness_Score`"
    )

    st.caption(
        "Pearson: tương quan tuyến tính [-1,1]. "
        "Spearman: tương quan theo thứ hạng [-1,1]. "
        "Association = max(|Pearson|, |Spearman|). "
        "Eta²: mức độ Categorical liên hệ với Rating [0,1]. "
        "VIF: chỉ số đa cộng tuyến; VIF > 10 thường nghiêm trọng."
    )

    st.subheader("Dữ liệu đầu vào")
    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Cấu hình ngưỡng
    # --------------------------------------------------------
    st.subheader("Ngưỡng Feature Selection")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        numeric_threshold = st.number_input(
            "Numeric |r| tối thiểu",
            min_value=0.0,
            max_value=1.0,
            value=NUMERIC_CORR_THRESHOLD,
            step=0.01,
            help=(
                "Feature Numeric có |Pearson| hoặc |Spearman| "
                "nhỏ hơn ngưỡng sẽ bị loại."
            )
        )

    with col2:
        eta_threshold = st.number_input(
            "Categorical Eta² tối thiểu",
            min_value=0.0,
            max_value=1.0,
            value=CATEGORICAL_ETA_THRESHOLD,
            step=0.01,
            help=(
                "Eta² càng lớn thì Categorical càng có "
                "liên hệ với Rating."
            )
        )

    with col3:
        multicol_threshold = st.number_input(
            "Correlation đa cộng tuyến",
            min_value=0.5,
            max_value=1.0,
            value=MULTICOL_THRESHOLD,
            step=0.05,
            help=(
                "Nếu hai Numeric feature có |r| vượt ngưỡng, "
                "giữ feature liên hệ tốt hơn với Rating."
            )
        )

    with col4:
        vif_threshold = st.number_input(
            "VIF tối đa",
            min_value=2.0,
            max_value=20.0,
            value=VIF_THRESHOLD,
            step=1.0,
            help=(
                "VIF > 10 thường cho thấy đa cộng tuyến nghiêm trọng."
            )
        )

    if st.button(
        "Bắt đầu Feature Engineering & Selection",
        type="primary"
    ):
        print_header(
            "Feature engineering, correlation and "
            "multicollinearity analysis started..."
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        # ====================================================
        # BƯỚC 1: TẠO FEATURE
        # ====================================================
        df_features = extract_features(
            df,
            progress_bar,
            status_text
        )

        # ====================================================
        # BƯỚC 2: FEATURE SELECTION
        # Dùng cấu hình từ giao diện
        # ====================================================
        old_values = (
            NUMERIC_CORR_THRESHOLD,
            CATEGORICAL_ETA_THRESHOLD,
            MULTICOL_THRESHOLD,
            VIF_THRESHOLD
        )

        NUMERIC_CORR_THRESHOLD = numeric_threshold
        CATEGORICAL_ETA_THRESHOLD = eta_threshold
        MULTICOL_THRESHOLD = multicol_threshold
        VIF_THRESHOLD = vif_threshold

        status_text.text(
            "Đang tính tương quan và xử lý đa cộng tuyến..."
        )
        progress_bar.progress(65)

        (
            df_optimized,
            numeric_result,
            categorical_result,
            dropped_df
        ) = select_best_features(
            df_features,
            TARGET_COL
        )

        (
            NUMERIC_CORR_THRESHOLD,
            CATEGORICAL_ETA_THRESHOLD,
            MULTICOL_THRESHOLD,
            VIF_THRESHOLD
        ) = old_values

        # ====================================================
        # BƯỚC 3: LƯU FILE
        # ====================================================
        os.makedirs(
            os.path.dirname(OUTPUT_DATA_PATH),
            exist_ok=True
        )

        try:
            df_optimized.to_csv(
                OUTPUT_DATA_PATH,
                index=False,
                encoding="utf-8-sig"
            )
        except Exception as e:
            st.error(
                f"Lỗi khi lưu file: {e}"
            )
            return

        progress_bar.progress(100)
        status_text.text("Hoàn tất!")

        st.success(
            f"Đã lưu file thành công:\n\n"
            f"`{OUTPUT_DATA_PATH}`"
        )

        # ====================================================
        # TỔNG QUAN
        # ====================================================
        st.markdown("---")
        st.subheader("Tổng quan kết quả")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Feature sau Feature Engineering",
                len(df_features.columns)
            )

        with col2:
            st.metric(
                "Feature sau Selection",
                len(df_optimized.columns)
            )

        with col3:
            st.metric(
                "Feature bị loại",
                len(dropped_df)
            )

        # ====================================================
        # NUMERIC ↔ RATING
        # ====================================================
        st.markdown("---")
        st.subheader(
            "1. Tương quan Numeric với Rating"
        )

        if not numeric_result.empty:
            numeric_display = numeric_result[
                [
                    "Feature",
                    "Pearson",
                    "Spearman",
                    "Association",
                    "Keep",
                    "Reason"
                ]
            ].sort_values(
                "Association",
                ascending=False
            )

            st.dataframe(
                numeric_display,
                use_container_width=True
            )

            st.caption(
                "Pearson đo quan hệ tuyến tính. "
                "Spearman đo quan hệ theo thứ hạng. "
                "Association lấy giá trị tuyệt đối lớn hơn "
                "của hai chỉ số. Càng gần 1 thì liên hệ với "
                "Rating càng mạnh."
            )

        # ====================================================
        # CATEGORICAL ↔ RATING
        # ====================================================
        st.markdown("---")
        st.subheader(
            "2. Tương quan Categorical / Label với Rating"
        )

        if not categorical_result.empty:
            categorical_display = categorical_result[
                [
                    "Feature",
                    "Unique_Values",
                    "Eta_Squared",
                    "Keep",
                    "Reason"
                ]
            ].sort_values(
                "Eta_Squared",
                ascending=False
            )

            st.dataframe(
                categorical_display,
                use_container_width=True
            )

            st.caption(
                "Eta² nằm trong [0,1]. Eta² càng lớn nghĩa là "
                "sự khác biệt giữa các nhóm Categorical càng "
                "liên quan đến sự thay đổi của Rating. "
                "SubCategory không bị mã hóa thành 0,1,2,... "
                "để tránh tạo ra thứ tự giả."
            )

            plot_df = categorical_result.sort_values(
                "Eta_Squared"
            )

            fig, ax = plt.subplots(
                figsize=(9, 4)
            )

            ax.barh(
                plot_df["Feature"],
                plot_df["Eta_Squared"]
            )

            ax.axvline(
                eta_threshold,
                linestyle="--",
                label=f"Ngưỡng = {eta_threshold:.2f}"
            )

            ax.set_xlabel("Eta²")
            ax.set_ylabel("Feature")
            ax.set_title(
                "Mức độ liên hệ Categorical với Rating"
            )
            ax.legend()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ====================================================
        # CORRELATION MATRIX
        # ====================================================
        st.markdown("---")
        st.subheader(
            "3. Ma trận tương quan giữa Numeric Features"
        )

        numeric_cols = df_features.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_cols) > 1:
            corr_matrix = df_features[
                numeric_cols
            ].corr()

            fig, ax = plt.subplots(
                figsize=(12, 9)
            )

            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                linewidths=0.3,
                ax=ax
            )

            ax.set_title(
                "Correlation Matrix"
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Ma trận dùng để phát hiện các feature Numeric "
                "trùng lặp thông tin. Nếu |r| vượt ngưỡng đa "
                "cộng tuyến, feature có liên hệ thấp hơn với "
                "Rating sẽ được ưu tiên loại."
            )

        # ====================================================
        # FEATURE BỊ LOẠI
        # ====================================================
        st.markdown("---")
        st.subheader(
            "4. Feature bị loại"
        )

        if not dropped_df.empty:
            st.dataframe(
                dropped_df.sort_values(
                    "Score",
                    ascending=True
                ),
                use_container_width=True
            )
        else:
            st.success(
                "Không có feature nào bị loại."
            )

        # ====================================================
        # FEATURE CUỐI CÙNG
        # ====================================================
        st.markdown("---")
        st.subheader(
            "5. Các Feature cuối cùng"
        )

        st.write(
            df_optimized.columns.tolist()
        )

        st.dataframe(
            df_optimized.head(10),
            use_container_width=True
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================
        st.download_button(
            "Tải 08_feature_engineered.csv",
            data=df_optimized.to_csv(
                index=False,
                encoding="utf-8-sig"
            ),
            file_name="08_feature_engineered.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()