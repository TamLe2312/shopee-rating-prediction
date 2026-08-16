"""
Tiền xử lý dữ liệu cho bài toán Linear/Ridge Regression.
KHÁC text.py:main(): KHÔNG drop 3★, giữ rating gốc 1-5 làm target liên tục.

Input:  data/raw/shopee_reviews.csv
Output: data/raw/shopee_reviews_for_regression.csv  (thêm cột Target)

Cách dùng:
    python -m src.preprocessing.prepare_regression
    # hoặc
    python src/preprocessing/prepare_regression.py
"""
import os
import sys
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Bootstrap: cho phép chạy trực tiếp `python src/preprocessing/prepare_regression.py`
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing.text import preprocess_text  # noqa: E402


def main():
    raw_dir = os.path.join(_PROJECT_ROOT, "data", "raw")
    input_file = os.path.join(raw_dir, "shopee_reviews.csv")
    output_file = os.path.join(raw_dir, "shopee_reviews_for_regression.csv")

    if not os.path.exists(input_file):
        print(f"Khong tim thay file: {input_file}")
        return

    print(f"Doc du lieu tu: {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  -> {len(df):,} dong raw")
    print(f"  -> Phan bo rating raw:\n{df['Rating'].value_counts().sort_index()}")

    print("\nTien xu ly van ban (giu tat ca rating 1-5)...")
    df["Raw_Comment"] = df["Comment"]
    df["Comment"] = df["Comment"].apply(preprocess_text)

    initial = len(df)
    df = df[df["Comment"].str.strip() != ""].copy()
    print(f"  -> Bo {initial - len(df)} dong rong sau preprocess")

    # Target = Rating làm float (regression)
    df["Target"] = df["Rating"].astype(float)
    print(f"\nThong ke target:")
    print(f"  Min:    {df['Target'].min()}")
    print(f"  Max:    {df['Target'].max()}")
    print(f"  Mean:   {df['Target'].mean():.3f}")
    print(f"  Median: {df['Target'].median()}")
    print(f"  Std:    {df['Target'].std():.3f}")

    df.to_csv(output_file, index=False)
    print(f"\nDa luu: {output_file}  ({len(df):,} dong)")


if __name__ == "__main__":
    main()
