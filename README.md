# ⭐ Shopee Rating Prediction — MLOps Architecture

Hệ thống dự đoán điểm đánh giá (**Rating Prediction**) cho sản phẩm Shopee, với đầu ra trong khoảng **1.0 → 5.0**.

Dự án sử dụng các mô hình Machine Learning:

* **Linear Regression**
* **LightGBM**
* **TF-IDF** để vector hóa dữ liệu văn bản

Hệ thống được thiết kế theo kiến trúc **Microservices** và triển khai theo hướng **MLOps**, với pipeline dữ liệu:

```text
MinIO (Data Lake)
       ↓
FastAPI (AI Backend)
       ↕
Streamlit (Frontend)
```

Toàn bộ hệ thống được đóng gói bằng **Docker Compose**, giúp triển khai nhanh chóng và đơn giản.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────┐
                    │      MinIO       │
                    │    Data Lake     │
                    │ Bronze / Silver  │
                    │      / Gold      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     FastAPI      │
                    │    AI Backend    │
                    │                  │
                    │  ML Inference    │
                    │ Linear / LightGBM│
                    └────────┬─────────┘
                             │ REST API
                             ▼
                    ┌──────────────────┐
                    │    Streamlit     │
                    │     Frontend     │
                    └──────────────────┘
```

---

# 🚀 Quick Start

Hệ thống được phân tách thành các dịch vụ độc lập.

Khuyến nghị sử dụng **Docker Compose** để tự động triển khai toàn bộ hệ thống mà không cần cấu hình thủ công phức tạp.

---

## 🐳 Cách 1 — Docker Compose ⭐

Đây là cách triển khai **đơn giản và được khuyến nghị nhất**.

### Khởi động toàn bộ hệ thống

```bash
docker compose up -d --build
```

Sau khi quá trình build hoàn tất, các dịch vụ sẽ hoạt động tại:

| Service               | URL                        |
|-----------------------|----------------------------|
| 🎨 Streamlit Frontend | http://localhost:8501      |
| 🧠 FastAPI Swagger UI | http://localhost:8000/docs |
| 🗄️ MinIO Console     | http://localhost:9001      |

### 🔐 MinIO Credentials

```text
Username: admin
Password: password123
```

---

# 💻 Cách 2 — Chạy Local bằng Python

> ⚠️ Yêu cầu:
>
> * Đã cài Python và các dependencies.
> * MinIO phải đang chạy.
> * Cần mở **2 terminal song song** cho Backend và Frontend.

---

## 1️⃣ Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Khởi chạy Backend — FastAPI

Mở **Terminal 1**:

```bash
uvicorn api:app --reload --port 8000
```

Sau đó truy cập Swagger API:

```text
http://localhost:8000/docs
```

---

## 3️⃣ Khởi chạy Frontend — Streamlit

Mở **Terminal 2**:

```bash
streamlit run app.py
```

Sau đó truy cập giao diện web:

```text
http://localhost:8501
```

---

# 🔄 Full Training Pipeline

Nếu bạn thay đổi dữ liệu hoặc muốn huấn luyện lại mô hình từ đầu, hãy thực hiện pipeline sau.

```text
Raw Data
    ↓
Bronze Layer
    ↓
Text Preprocessing
    ↓
Silver Layer
    ↓
TF-IDF Vectorization
    ↓
Model Training
    ├── Linear Regression
    └── LightGBM
    ↓
Model Evaluation
    ↓
Gold Layer / Model Registry
    ↓
FastAPI Deployment
```

---

## 1️⃣ Cài đặt môi trường

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Text Preprocessing — Bronze → Silver

Chạy script để:

* Chuẩn hóa dữ liệu văn bản
* Làm sạch dữ liệu nhiễu
* Xử lý NLP
* Tách từ
* Chuẩn bị dữ liệu cho bài toán Regression

```bash
python -m src.preprocessing.prepare_regression
```

Sau khi xử lý, pipeline sẽ sinh ra file:

```text
shopee_reviews_for_regression.csv
```

---

## 3️⃣ Model Training — Silver → Gold

Chạy script huấn luyện:

```bash
python -m src.training.train_model.py
```

Hệ thống sẽ tự động:

* Vector hóa dữ liệu bằng **TF-IDF**
* Huấn luyện **Linear Regression**
* Huấn luyện **LightGBM**
* Đánh giá hiệu năng mô hình
* Tính toán các metrics
* Lưu mô hình đã train
* Xuất metadata và metrics

Các artifacts được lưu trong thư mục:

```text
models/
```

Ví dụ:

```text
models/
├── rating_model_lightgbm.pkl
├── rating_model_linear.pkl
├── rating_vectorizer_lightgbm.pkl
├── rating_vectorizer_linear.pkl
└── rating_metadata_lightgbm.json
└── rating_metadata_linear.json
```

---

## 4️⃣ Cập nhật Model Registry

Sau khi train xong, upload các artifacts mới lên MinIO.

Bucket sử dụng cho tầng Gold:

```text
shopee-gold
```

Các file cần cập nhật có thể bao gồm:

```text
*.pkl
*.json
```

Bạn có thể upload thủ công thông qua **MinIO Console** hoặc tự động hóa bằng script.

---

## 5️⃣ Reload Backend

Sau khi cập nhật model trên MinIO, khởi động lại Backend để API tải model mới vào RAM:

```bash
docker compose restart backend
```

---

# 🧠 Machine Learning Models

## Linear Regression

Linear Regression được sử dụng làm mô hình baseline cho bài toán dự đoán rating.

Pipeline:

```text
Review Text
     ↓
TF-IDF
     ↓
Linear Regression
     ↓
Predicted Rating
```

---

## LightGBM

LightGBM được sử dụng để xây dựng mô hình Gradient Boosting hiệu năng cao.

Pipeline:

```text
Review Text
     ↓
TF-IDF Features
     ↓
LightGBM Regressor
     ↓
Predicted Rating
```

---

# 🗄️ Data Lake Architecture

Dữ liệu được tổ chức theo mô hình Data Lake nhiều tầng:

```text
Bronze
   │
   │ Raw Data
   ▼
Silver
   │
   │ Cleaned & Processed Data
   ▼
Gold
   │
   │ ML Models
   │ Metrics
   │ Model Metadata
   ▼
Model Serving
```

| Layer     | Mô tả                                |
|-----------|--------------------------------------|
| 🥉 Bronze | Dữ liệu gốc (Raw Data)               |
| 🥈 Silver | Dữ liệu đã được làm sạch và xử lý    |
| 🥇 Gold   | Model artifacts, metrics và metadata |

---

# 🛠️ Technology Stack

| Technology     | Purpose                    |
|----------------|----------------------------|
| Python         | Programming Language       |
| FastAPI        | AI Backend & REST API      |
| Streamlit      | Web Frontend               |
| MinIO          | Object Storage & Data Lake |
| Docker         | Containerization           |
| Docker Compose | Service Orchestration      |
| Scikit-learn   | Machine Learning           |
| LightGBM       | Gradient Boosting Model    |
| TF-IDF         | Text Feature Extraction    |

---

# 📂 Project Workflow

```text
┌───────────────┐
│ Shopee Reviews│
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Bronze Layer  │
│   Raw Data    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Preprocessing │
│      NLP      │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Silver Layer  │
│ Processed Data│
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ TF-IDF        │
│ Vectorization │
└───────┬───────┘
        │
        ├───────────────┐
        ▼               ▼
┌──────────────┐ ┌──────────────┐
│ Linear Reg.  │ │   LightGBM   │
└──────┬───────┘ └──────┬───────┘
       │                │
       └────────┬───────┘
                ▼
        ┌───────────────┐
        │ Model Metrics │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ Gold / MinIO  │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │   FastAPI     │
        │ Model Serving │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │   Streamlit   │
        │     UI        │
        └───────────────┘
```

---

# 📌 Useful Commands

### Build và chạy hệ thống

```bash
docker compose up -d --build
```

### Kiểm tra containers

```bash
docker compose ps
```

### Xem logs

```bash
docker compose logs -f
```

### Restart Backend

```bash
docker compose restart backend
```

### Dừng hệ thống

```bash
docker compose down
```

---

# 🎯 Project Goal

Mục tiêu của dự án là xây dựng một hệ thống Machine Learning hoàn chỉnh theo hướng **MLOps**, bao gồm:

* Data Lake
* Data Processing Pipeline
* NLP Preprocessing
* Machine Learning Training
* Model Evaluation
* Model Registry
* REST API Model Serving
* Web Frontend
* Containerized Deployment

Hệ thống minh họa quy trình triển khai một mô hình Machine Learning từ **Raw Data → Training → Model Registry → API
Serving → User Interface** trong môi trường Microservices.
