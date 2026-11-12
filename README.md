# Bài Tập Lớn Học Máy (Machine Learning) - PTIT

## Đề tài: Dự đoán & Đánh giá Nguy cơ Bệnh Tim Mạch (Healthcare and Patient Risk Analysis)

## 1. Cấu trúc thư mục dự án

```text
cardiovascular-risk-ml-ptit/
│
├── .gitignore
├── healthcare_cvd_risk_pipeline.ipynb   # File Notebook chính, chạy end-to-end, nhúng sẵn kết quả
├── BAO_CAO_BTL.md                       # Thuyết minh báo cáo chi tiết & Kịch bản trả lời phản biện
├── requirements.txt                     # Danh sách thư viện và phiên bản ghim ổn định
│
├── data/
│   └── raw/
│       ├── framingham.csv               # Tập A: 4.240 bệnh nhân, Mỹ (Tỷ lệ bệnh 15.19%)
│       └── cardio_train.csv             # Tập B: 70.000 mẫu (68.619 sạch), Đông Âu (Tỷ lệ bệnh 49.47%)
│
└── figures/                             # 11 biểu đồ xuất bản độ phân giải cao (200 DPI)

```

---

## 2. Hướng dẫn cài đặt & Thực thi nhanh

### Bước 1: Khởi tạo môi trường ảo Python 3.12

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Bước 2: Chạy mã nguồn

Mở tệp `healthcare_cvd_risk_pipeline.ipynb` bằng VS Code hoặc Jupyter Lab/Notebook với Kernel `.venv` và chọn **Run All Cells**.
