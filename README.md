# ⭐ Shopee Rating Prediction

> Hệ thống dự đoán điểm đánh giá (rating 1.0 → 5.0) sản phẩm Shopee bằng **Linear / Ridge Regression + TF-IDF**.
> Pipeline ML đầy đủ: **tiền xử lý** (đặc thù tiếng Việt) → **huấn luyện** → **demo UI** → **đóng gói Docker**.

---

## 🚀 Quick Start (3 cách)

Model đã được train sẵn (`models/rating_model.pkl`) — chỉ cần start demo.

### Cách 1 — Docker Compose ⭐ (đơn giản nhất)

```bash
docker-compose up --build
```
→ Mở browser: **http://localhost:8501**

### Cách 2 — Docker thuần

```bash
docker build -t shopee-rating-prediction .
docker run -p 8501:8501 shopee-rating-prediction
```

### Cách 3 — Python local (không Docker)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔄 Full Pipeline (train lại từ đầu)

```bash
# 1. Cài thư viện
pip install -r requirements.txt

# 2. Tiền xử lý văn bản — giữ TẤT CẢ rating 1-5
python -m src.preprocessing.prepare_regression

# 3. Train: TF-IDF + grid search α (5-fold CV) + Ridge final + save .pkl
python -m src.training.train_ridge

# 4. Sinh 8 biểu đồ trực quan regression (scatter, residual, alpha curve...)
python -m src.reporting.plots

# 5. Chạy demo
streamlit run app.py
```
