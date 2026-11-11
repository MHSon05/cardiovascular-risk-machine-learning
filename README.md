# Bài Tập Lớn Học Máy (Machine Learning) - PTIT

## Đề tài: Dự đoán & Đánh giá Nguy cơ Bệnh Tim Mạch (Healthcare and Patient Risk Analysis)
**Nghiên cứu Đa Tập Dữ Liệu, Đánh Giá Xuyên Miền (Cross-Domain), Định Lượng Trôi Phân Phối (Data Drift) và Khung Dự Đoán Tin Cậy (Trustworthy AI)**

---

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
    ├── fig1_eda_distributions.png       # Phân phối 8 đặc trưng lâm sàng đối sánh A vs B
    ├── fig2_correlation_matrix.png      # Ma trận tương quan đặc trưng từng tập
    ├── fig3_roc_pr_curves_framingham.png# Đường cong ROC & PR trên tập Framingham
    ├── fig4_roc_pr_curves_cardio.png    # Đường cong ROC & PR trên tập Cardio
    ├── fig5_learning_curves.png         # Đường cong học tập Stratified 5-Fold (± 1 SD)
    ├── fig6_in_domain_comparison.png    # Biểu đồ cột so sánh ROC-AUC & F1-Score
    ├── fig7_adversarial_validation_drift.png # [Bonus 1] Kiểm định đối nghịch đo trôi dữ liệu
    ├── fig8_calibration_curves.png      # [Bonus 2] Hiệu chuẩn xác suất hậu nghiệm hai chiều (A->B & B->A)
    ├── fig9_shap_global_summary.png     # [Bonus 3] Đối sánh SHAP đa miền Mỹ vs Đông Âu
    ├── fig10_shap_local_false_negatives.png  # [Bonus 3] SHAP Waterfall bóc tách ca âm tính giả
    └── fig11_conformal_prediction_coverage.png # [Bonus 4] Khung dự đoán tin cậy Conformal (MAPIE)
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

---

## 3. Tóm tắt kết quả thực nghiệm nổi bật

1. **Hiệu năng Nội miền (In-Domain)**:
   - Trên Tập A (Framingham): **Logistic Regression** dẫn đầu với ROC-AUC **0.747** (PR-AUC **0.399**, Brier **0.113**), theo sát là **SVM RBF** (**0.732**) và **Random Forest** (**0.730**).
   - Trên Tập B (Cardio): Tổ hợp cây **XGBoost** (**0.798**) và **Random Forest** (**0.797**) đạt hiệu năng cao nhất; **Nystroem + SGD** (**0.795**) huấn luyện SVM trên gần 70.000 mẫu chỉ trong vài giây.
   - **Tối ưu ngưỡng chi phí y tế ($\text{Cost} = FP + 5FN$, $\text{Recall} \ge 0.80$)**: Giảm **9.5%–18.9%** tổng chi phí trên Tập A và **21.4%–34.8%** trên Tập B so với ngưỡng mặc định 0.50.

2. **4 Điểm cộng Sáng tạo ngoài bài giảng (Bonus Points)**:
   - **Adversarial Validation**: Phát hiện trôi dạt phân phối nghiêm trọng giữa 2 quần thể (**ROC-AUC = 0.9841**). Top đặc trưng trôi mạnh nhất: `chol_level`, `smoker`, `dia_bp`, `gluc_level`.
   - **Post-hoc Probability Calibration (2 chiều)**: Hạ Brier Score từ **0.3470 xuống 0.2123** (**-38.8%** chiều $A \to B$) và từ **0.3298 xuống 0.1235** (**-62.5%** chiều $B \to A$).
   - **SHAP Đa miền (Explainable AI)**: Đối sánh yếu tố nguy cơ Mỹ (`age` 0.571, `sys_bp` 0.262) vs Đông Âu (`sys_bp` 0.827, `age` 0.257, `chol_level` 0.196) và truy nguyên ca âm tính giả **#672** (35 tuổi, nam, hút thuốc, mô hình đoán nguy cơ chỉ **0.017**).
   - **Conformal Prediction (MAPIE, $\alpha = 0.10$)**: Đạt độ phủ **90.4%** trên Framingham (tách **16.63%** ca bất định $\{0, 1\}$) và **90.1%** trên Cardio (phân rõ **27.08%** $\{0\}$, **29.07%** $\{1\}$, **43.85%** $\{0, 1\}$).
