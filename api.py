import io
import sys
import os
import joblib
import numpy as np
from minio import Minio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from src.preprocessing.text import preprocess_text

# ==========================
# CẤU HÌNH MINIO
# ==========================
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
MINIO_BUCKET_GOLD = "shopee-gold"

# Biến toàn cục lưu trữ mô hình trên RAM
vectorizers = {}
models = {}


def load_model_from_minio(client: Minio, bucket_name: str, object_name: str):
    """Đọc file .pkl trực tiếp từ MinIO bucket vào RAM qua Stream"""
    response = client.get_object(bucket_name, object_name)
    try:
        data = io.BytesIO(response.read())
        return joblib.load(data)
    finally:
        response.close()
        response.release_conn()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Tải toàn bộ mô hình từ MinIO (Lớp Gold) vào RAM khi FastAPI khởi động"""
    try:
        print("🔗 Đang kết nối tới MinIO Data Lake...")
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )

        if not minio_client.bucket_exists(MINIO_BUCKET_GOLD):
            raise RuntimeError(f"Bucket '{MINIO_BUCKET_GOLD}' không tồn tại trên MinIO!")

        print(f"📥 Đang kéo artifacts từ bucket '{MINIO_BUCKET_GOLD}' vào RAM...")

        # 1. Tải Vectorizers
        vectorizers["Linear"] = load_model_from_minio(
            minio_client, MINIO_BUCKET_GOLD, "rating_vectorizer_linear.pkl"
        )
        vectorizers["LightGBM"] = load_model_from_minio(
            minio_client, MINIO_BUCKET_GOLD, "rating_vectorizer_lightgbm.pkl"
        )

        # 2. Tải Models
        models["Linear"] = load_model_from_minio(
            minio_client, MINIO_BUCKET_GOLD, "rating_model_linear.pkl"
        )
        models["LightGBM"] = load_model_from_minio(
            minio_client, MINIO_BUCKET_GOLD, "rating_model_lightgbm.pkl"
        )

        print("✅ Đã nạp thành công toàn bộ mô hình từ MinIO vào RAM!")
        yield
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình từ MinIO: {e}")
        yield
    finally:
        # Giải phóng RAM khi tắt server
        vectorizers.clear()
        models.clear()


# ==========================
# KHỞI TẠO FASTAPI
# ==========================
app = FastAPI(
    title="Shopee Rating API (MinIO Integrated)",
    description="API phục vụ mô hình tải trực tiếp từ MinIO Object Storage",
    version="2.0.0",
    lifespan=lifespan
)


class PredictRequest(BaseModel):
    text: str
    model_type: str = "LightGBM"  # "Linear" hoặc "LightGBM"


class PredictResponse(BaseModel):
    processed_text: str
    raw_prediction: float
    clipped_prediction: float
    stars: int


# ==========================
# ENDPOINTS
# ==========================
@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái nạp mô hình"""
    return {
        "status": "online",
        "loaded_models": list(models.keys()),
        "source": f"MinIO @ {MINIO_ENDPOINT}/{MINIO_BUCKET_GOLD}"
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_rating(request: PredictRequest):
    model_name = request.model_type
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Mô hình '{model_name}' chưa sẵn sàng hoặc không hợp lệ.")

    # 1. Tiền xử lý
    processed = preprocess_text(request.text)
    if not processed:
        raise HTTPException(status_code=400, detail="Văn bản rỗng sau tiền xử lý.")

    # 2. Vector hóa
    vec = vectorizers[model_name]
    X = vec.transform([processed])

    # 3. Dự đoán
    model = models[model_name]
    raw_pred = float(model.predict(X)[0])
    clipped = float(np.clip(raw_pred, 1, 5))
    rounded_star = int(round(clipped))

    return PredictResponse(
        processed_text=processed,
        raw_prediction=raw_pred,
        clipped_prediction=clipped,
        stars=rounded_star
    )
