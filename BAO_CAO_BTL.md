# HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG (PTIT)
## BÁO CÁO BÀI TẬP LỚN HỌC MÁY (MACHINE LEARNING)
### ĐỀ TÀI: DỰ ĐOÁN VÀ ĐÁNH GIÁ NGUY CƠ BỆNH TIM MẠCH (HEALTHCARE AND PATIENT RISK ANALYSIS)
**Nghiên cứu Đa Tập Dữ Liệu, Đánh Giá Xuyên Miền (Cross-Domain), Định Lượng Trôi Phân Phối (Data Drift) và Khung Dự Đoán Tin Cậy (Trustworthy AI)**

---

## MỤC LỤC
1. [I. Giới Thiệu & Đặt Vấn Đề Nghiên Cứu](#i-giới-thiệu--đặt-vấn-đề-nghiên-cứu)
2. [II. Dữ Liệu Nghiên Cứu & Căn Chỉnh Lược Đồ (Schema Harmonization)](#ii-dữ-liệu-nghiên-cứu--căn-chỉnh-lược-đồ-schema-harmonization)
3. [III. Thiết Kế Thực Nghiệm Chống Rò Rỉ & 7 Thuật Toán Trên Lớp](#iii-thiết-kế-thực-nghiệm-chống-rò-rỉ--7-thuật-toán-trên-lớp)
4. [IV. Đánh Giá Hiệu Năng Nội Miền (In-Domain) & Tối Ưu Ngưỡng Y Tế](#iv-đánh-giá-hiệu-năng-nội-miền-in-domain--tối-ưu-ngưỡng-y-tế)
5. [V. Đánh Giá Xuyên Miền & 4 Điểm Cộng Sáng Tạo (Bonus Advanced ML)](#v-đánh-giá-xuyên-miền--4-điểm-cộng-sáng-tạo-bonus-advanced-ml)
   - [5.1. Thử nghiệm Zero-shot Cross-Domain](#51-thử-nghiệm-zero-shot-cross-domain)
   - [5.2. Kiểm định Đối nghịch (Adversarial Validation)](#52-kiểm-định-đối-nghịch-adversarial-validation)
   - [5.3. Hiệu chuẩn Xác suất Hậu nghiệm (Probability Calibration)](#53-hiệu-chuẩn-xác-suất-hậu-nghiệm-probability-calibration)
   - [5.4. Khám phá Cơ chế Ra Quyết định & Phân tích Âm tính giả (SHAP)](#54-khám-phá-cơ-chế-ra-quyết-định--phân-tích-âm-tính-giả-shap)
   - [5.5. Dự đoán Phù hợp Đảm bảo Độ phủ Thống kê (Conformal Prediction - MAPIE)](#55-dự-đoán-phù-hợp-đảm-bảo-độ-phủ-thống-kê-conformal-prediction---mapie)
6. [VI. Kết Luận & Khuyến Nghị Lâm Sàng](#vi-kết-luận--khuyến-nghị-lâm-sàng)
7. [VII. Kịch Bản Trả Lời Câu Hỏi Phản Biện (Defense Q&A)](#vii-kịch-bản-trả-lời-câu-hỏi-phản-biện-defense-qa)
8. [VIII. Hướng Dẫn Tái Hiện Kết Quả (Reproducibility Guide)](#viii-hướng-dẫn-tái-hiện-kết-quả-reproducibility-guide)

---

## I. GIỚI THIỆU & ĐẶT VẤN ĐỀ NGHIÊN CỨU

### 1.1. Bối cảnh y tế và động lực nghiên cứu
Bệnh tim mạch (Cardiovascular Diseases - CVD) là nguyên nhân gây tử vong hàng đầu trên thế giới, cướp đi sinh mạng của khoảng 17,9 triệu người mỗi năm (theo Tổ chức Y tế Thế giới - WHO). Việc phát hiện sớm bệnh nhân có nguy cơ cao giúp can thiệp lối sống hoặc điều trị nội khoa kịp thời, giảm thiểu tối đa các biến cố nguy hiểm như nhồi máu cơ tim hay đột quỵ.

Trong kỷ nguyên Trí tuệ Nhân tạo, việc ứng dụng Học Máy (Machine Learning) vào chẩn đoán lâm sàng đã đạt được nhiều thành tựu lớn. Tuy nhiên, một lỗ hổng nghiêm trọng tồn tại trong phần lớn các nghiên cứu học thuật là **sự ngộ nhận về hiệu năng nội miền (In-domain Overoptimism)**: Một mô hình được huấn luyện và kiểm thử trên một bộ dữ liệu duy nhất thường đạt độ chính xác rất cao, nhưng khi triển khai trên quần thể bệnh nhân tại bệnh viện khác hoặc quốc gia khác thì hiệu năng sụt giảm thảm hại.

### 1.2. Mục tiêu của đề tài
Nghiên cứu này được xây dựng bám sát 4 yêu cầu cốt lõi của môn học Học Máy (PTIT):
1. **Sử dụng từ 2 bộ dữ liệu y tế độc lập**: Phân tích đối sánh giữa nghiên cứu dịch tễ kinh điển (Framingham, Mỹ) và bộ dữ liệu chẩn đoán lâm sàng quy mô lớn (Cardiovascular Disease, Đông Âu).
2. **Tổng hợp toàn diện 7 mô hình học máy kinh điển** đã được học trên lớp (Slide 1 đến Slide 7).
3. **Phân tích lỗi chuyên sâu, đánh giá đúng chuẩn y tế (Model Assessment)**: Tối ưu ngưỡng quyết định y tế dựa trên hàm phạt bất đối xứng, tính khoảng tin cậy 95% bằng Bootstrap, và phân tích đường cong học tập (*Learning Curves*).
4. **Điểm cộng sáng tạo (Bonus ML ngoài chương trình)**: Khắc phục triệt để hiện tượng Trôi dữ liệu (*Data Drift*), sai lệch tỷ lệ nền (*Base-rate Shift*), tăng cường tính minh bạch (*Explainable AI với SHAP*) và bảo chứng an toàn lâm sàng (*Conformal Prediction với MAPIE*).

---

## II. DỮ LIỆU NGHIÊN CỨU & CĂN CHỈNH LƯỢC ĐỒ (SCHEMA HARMONIZATION)

### 2.1. Giới thiệu 2 bộ dữ liệu
- **Tập A: Framingham Heart Study (Mỹ)**:
  - Nguồn gốc: Nghiên cứu dịch tễ học tim mạch kéo dài nhiều thập kỷ tại thị trấn Framingham, Massachusetts.
  - Quy mô: 4.240 bệnh nhân, 16 thuộc tính.
  - Mục tiêu: Dự đoán biến cố bệnh tim mạch vành trong vòng 10 năm tới (`TenYearCHD`).
  - Đặc điểm dữ liệu: Dữ liệu mất cân bằng nghiêm trọng (Tỷ lệ dương tính chỉ **15.2%**, tương ứng 644 ca bệnh).
- **Tập B: Cardiovascular Disease Dataset (Đông Âu)**:
  - Nguồn gốc: Hồ sơ khám sức khỏe lâm sàng của Svetlana Ulianova.
  - Quy mô: 70.000 bệnh nhân, 13 thuộc tính.
  - Mục tiêu: Chẩn đoán sự hiện diện thực tế của bệnh lý tim mạch tại thời điểm khám (`cardio`).
  - Đặc điểm dữ liệu: Sau lọc giữ **68.619 bệnh nhân**, dữ liệu cân bằng (Tỷ lệ dương tính **49.47%**, tương ứng 33.945 ca bệnh).

### 2.2. Lọc dữ liệu ngoại lai sinh học bất khả thi (Cardio Dataset)
Tập Cardio chứa các bản ghi bị nhập lỗi vật lý do thao tác y tế (ví dụ: huyết áp tâm thu ghi thành 12.000 hoặc âm, huyết áp tâm thu thấp hơn tâm trương). Ta thiết lập bộ lọc sinh lý học nghiêm ngặt:
- $60 \le \text{ap\_hi (Huyết áp tâm thu)} \le 250$ mmHg.
- $30 \le \text{ap\_lo (Huyết áp tâm trương)} \le 160$ mmHg.
- Điều kiện sinh thái: $\text{ap\_hi} > \text{ap\_lo}$.
- Chiều cao: $120 \le \text{height} \le 220$ cm; Cân nặng: $30 \le \text{weight} \le 200$ kg.
- **Kết quả lọc**: Loại bỏ 1.381 dòng lỗi (1.97%), giữ lại **68.619 bệnh nhân** sạch sinh học.

### 2.3. Căn chỉnh về 8 đặc trưng lâm sàng dùng chung theo 3 nhóm chuẩn không gian metric
Để kiểm thử chéo và chuyển giao mô hình xuyên miền (*Cross-domain Transfer*), 8 đặc trưng dùng chung được phân tách thành 3 nhóm rõ ràng:
- **Nhóm biến số liên tục (`NUM_FEATURES`)**:
  1. `age`: Tuổi thực (Framingham tính theo năm; Cardio lấy ngày / 365.25).
  2. `bmi`: Chỉ số khối cơ thể $\text{BMI} = \text{weight (kg)} / (\text{height (m)})^2$ (Framingham khuyết 19 dòng = 0.45%).
  3. `sys_bp`: Huyết áp tâm thu (mmHg).
  4. `dia_bp`: Huyết áp tâm trương (mmHg).
- **Nhóm biến thứ bậc (`ORD_FEATURES`)**:
  5. `chol_level`: Phân tầng mức Cholesterol theo chuẩn AHA/NCEP ATP III (Framingham khuyết 50 dòng = 1.18%): 1 ($< 200$), 2 ($200\text{–}239$), 3 ($\ge 240$ mg/dL).
  6. `gluc_level`: Phân tầng mức Glucose máu lúc đói theo chuẩn ADA (Framingham khuyết 388 dòng = 9.15%): 1 ($< 100$), 2 ($100\text{–}125$), 3 ($\ge 126$ mg/dL).
- **Nhóm biến nhị phân (`BIN_FEATURES`)**:
  7. `is_male`: Giới tính nhị phân (1: Nam, 0: Nữ).
  8. `smoker`: Thói quen hút thuốc (1: Có, 0: Không).
- Biến mục tiêu `target`: 1 (Mắc bệnh/Biến cố nguy cơ), 0 (Khỏe mạnh/Không biến cố).

*(Xem hình vẽ đối sánh phân phối tại [`figures/fig1_eda_distributions.png`](figures/fig1_eda_distributions.png) và tương quan tại [`figures/fig2_correlation_matrix.png`](figures/fig2_correlation_matrix.png))*

---

## III. THIẾT KẾ THỰC NGHIỆM CHỐNG RÒ RỈ & 7 THUẬT TOÁN TRÊN LỚP

### 3.1. Phân chia dữ liệu phân tầng 60 / 20 / 20
Tuân thủ tuyệt đối quy tắc trong Slide *Model Assessment*:
- **60% Train Set**: Dùng để huấn luyện mô hình.
- **20% Validation Set**: Dùng để chọn siêu tham số, tối ưu ngưỡng quyết định y tế ($p^*$) và làm tập hiệu chuẩn xác suất.
- **20% Test Set**: Độc lập tuyệt đối, chỉ dùng để đánh giá hiệu năng cuối cùng.
- **Chống rò rỉ dữ liệu & Cô lập trạng thái (Data Leakage & State Isolation)**: `make_preprocessor()` tách riêng 3 nhánh: điền khuyết `median` + `StandardScaler` cho `num`; điền khuyết `most_frequent` + `StandardScaler` cho `ord` (tránh méo khoảng cách Euclidean trong KNN và phạt điều chuẩn $L_2$ trong Logistic/SVM); điền khuyết `most_frequent` giữ nguyên $\{0, 1\}$ cho `bin`. Mỗi mô hình sở hữu một `Pipeline` độc lập chỉ `fit` trên tập Train.

### 3.2. Cấu hình 7 thuật toán học máy kinh điển
1. **Dummy Baseline** (*Slide 1 & Model Assessment*): Chiến lược `most_frequent` làm mốc tham chiếu sàn.
2. **Logistic Regression** (*Slide 2 & 3*): Điều chuẩn L2 (`C=1.0`), giữ nguyên phân phối xác suất hậu nghiệm chuẩn để phục vụ tối ưu ngưỡng quyết định y tế và kiểm định hiệu chuẩn xuyên miền.
3. **K-Nearest Neighbors (KNN)** (*Slide 4*): $k=15$, trọng số khoảng cách (`weights='distance'`).
4. **Gaussian Naive Bayes** (*Slide 5*): Ước lượng mật độ xác suất theo phân phối chuẩn với giả thiết độc lập có điều kiện.
5. **Random Forest Classifier** (*Slide 6*): Tập hợp 100 cây quyết định, giới hạn độ sâu `max_depth=8`.
6. **XGBoost Classifier** (*Slide 6*): Gradient Boosting với `learning_rate=0.05`, `max_depth=4`, giữ phân phối xác suất gốc trung thực.
7. **Support Vector Machine (SVM)** (*Slide 7*):
   - Với Tập A (4.2k mẫu): `SVC(kernel='rbf', probability=True, class_weight='balanced')` (Platt scaling nội bộ chuẩn hóa lại xác suất).
   - Với Tập B (70k mẫu): Kernel SVM truyền thống có độ phức tạp $O(N^3)$, không khả thi với dữ liệu lớn. Áp dụng kỹ thuật trong Slide 7: **Xấp xỉ hạt nhân Nystroem (100 thành phần)** kết hợp **SGDClassifier(loss='log_loss')**, cho phép huấn luyện chỉ mất vài giây.

---

## IV. ĐÁNH GIÁ HIỆU NĂNG NỘI MIỀN (IN-DOMAIN) & TỐI ƯU NGƯỠNG Y TẾ

### 4.1. Hàm chi phí y tế và Tối ưu ngưỡng quyết định
Trong chẩn đoán y tế, lỗi **Âm tính giả (False Negative - FN)** làm mất đi cơ hội cứu sống bệnh nhân, nguy hiểm hơn nhiều so với lỗi **Dương tính giả (False Positive - FP)** (chỉ tốn thêm xét nghiệm kiểm tra lại). Do đó, hàm chi phí y tế được xác định theo Slide 3:
$$\text{Cost} = 1 \times FP + 5 \times FN$$

Áp dụng quy trình **Threshold Selection** chuẩn theo *Model Assessment (p. 34–36 & p. 51)*: Quét trên tập Validation để tìm ngưỡng $p^*$ thỏa mãn ràng buộc y tế $\text{Recall} \ge 0.80$ và tối đa hóa Precision (nếu Precision bằng nhau, chọn ngưỡng lớn nhất), sau đó cố định ngưỡng này để đánh giá trên tập Test độc lập.

### 4.2. Kết quả kiểm thử trên Tập A (Framingham Heart Study)
*Test Set: 848 bệnh nhân (129 ca bệnh thực tế)*

| Mô hình | ROC-AUC (95% CI) | PR-AUC | Ngưỡng $p^*$ | Recall | Precision | F1-Score | Cost mặc định (p=0.5) | Cost tối ưu ($p^*$) | Mức giảm chi phí | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | 0.500 [0.500, 0.500] | 0.152 | 0.500 | 0.000 | 0.000 | 0.000 | 645 | 645 | 0.0% | 0.152 |
| **Logistic Regression** | **0.747 [0.699, 0.793]** | **0.399** | **0.096** | **0.868** | 0.212 | 0.341 | 599 | **501** | **-16.4%** | **0.113** |
| **SVM (RBF Kernel)** | **0.732 [0.684, 0.779]** | 0.310 | 0.091 | 0.845 | 0.205 | 0.330 | 645 | 523 | **-18.9%** | 0.117 |
| **Random Forest** | 0.730 [0.680, 0.778] | 0.353 | 0.090 | **0.891** | 0.195 | 0.319 | 612 | 546 | -10.8% | 0.116 |
| **XGBoost** | 0.727 [0.678, 0.772] | 0.353 | 0.097 | 0.876 | 0.206 | 0.333 | 615 | 516 | -16.1% | 0.116 |
| **KNN** | 0.714 [0.666, 0.768] | 0.336 | 0.059 | 0.876 | 0.191 | 0.313 | 620 | 560 | -9.7% | 0.119 |
| **Gaussian NB** | 0.712 [0.662, 0.760] | 0.324 | 0.050 | 0.822 | **0.228** | **0.357** | 524 | **474** | -9.5% | 0.151 |

### 4.3. Kết quả kiểm thử trên Tập B (Cardiovascular Disease Dataset)
*Test Set: 13.724 bệnh nhân (6.789 ca bệnh thực tế)*

| Mô hình | ROC-AUC (95% CI) | PR-AUC | Ngưỡng $p^*$ | Recall | Precision | F1-Score | Cost mặc định (p=0.5) | Cost tối ưu ($p^*$) | Mức giảm chi phí | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | 0.500 [0.500, 0.500] | 0.495 | 0.500 | 0.000 | 0.000 | 0.000 | 33,945 | 33,945 | 0.0% | 0.495 |
| **XGBoost** | **0.798 [0.790, 0.805]** | **0.777** | **0.379** | 0.798 | 0.681 | 0.735 | 12,420 | **9,384** | **-24.4%** | **0.182** |
| **Random Forest** | **0.797 [0.790, 0.805]** | 0.776 | 0.384 | 0.799 | **0.682** | **0.736** | 13,159 | **9,353** | **-28.9%** | 0.183 |
| **SVM (Nystroem+SGD)** | 0.795 [0.787, 0.803] | 0.773 | 0.383 | 0.796 | **0.682** | 0.735 | 12,002 | 9,437 | -21.4% | 0.183 |
| **Logistic Regression** | 0.787 [0.779, 0.795] | 0.765 | 0.393 | 0.800 | 0.677 | 0.733 | 13,147 | 9,400 | **-28.5%** | 0.189 |
| **Gaussian NB** | 0.774 [0.766, 0.782] | 0.742 | 0.184 | 0.795 | 0.664 | 0.723 | 14,889 | 9,705 | **-34.8%** | 0.218 |
| **KNN** | 0.763 [0.754, 0.771] | 0.729 | 0.355 | **0.802** | 0.646 | 0.716 | 12,627 | 9,702 | -23.2% | 0.203 |

*(Xem hình vẽ ROC/PR tại [`fig3`](figures/fig3_roc_pr_curves_framingham.png), [`fig4`](figures/fig4_roc_pr_curves_cardio.png), biểu đồ học tập Stratified 5-Fold $\pm 1\sigma$ tại [`fig5`](figures/fig5_learning_curves.png) và biểu đồ cột so sánh tại [`fig6`](figures/fig6_in_domain_comparison.png))*

---

## V. ĐÁNH GIÁ XUYÊN MIỀN & 4 ĐIỂM CỘNG SÁNG TẠO (BONUS ADVANCED ML)

### 5.1. Thử nghiệm Zero-shot Cross-Domain
Khi lấy mô hình huấn luyện trên Tập A kiểm thử trực tiếp trên Tập B ($A \to B$) và ngược lại ($B \to A$), hiệu năng lâm sàng sụp đổ nghiêm trọng:
- **$A \to B$ tại ngưỡng $p = 0.50$**: Do Tập A chỉ có tỷ lệ bệnh nền 15.19%, xác suất dự đoán của mô hình gần như không vượt 0.50 khi đưa sang Tập B (49.47%), khiến **Recall tụt xuống chỉ còn 0.55% (Logistic Regression) đến 1.55% (XGBoost)**, chi phí y tế vọt lên **33.498 – 33.768**.
- **Brier Score tăng gấp 2–3 lần**: Từ mức `0.113–0.117` (In-domain A) nhảy vọt lên **0.347 – 0.357** ($A \to B$) và **0.328 – 0.334** ($B \to A$).
- **Nguyên nhân**:
  1. *Covariate Shift*: Sự sai lệch phân bố các yếu tố nguy cơ $P_A(X) \ne P_B(X)$.
  2. *Base-rate / Prior Shift*: Tỷ lệ mắc bệnh thực tế chênh lệch gấp hơn 3 lần ($15.19\%$ vs $49.47\%$).

---

### 5.2. Điểm cộng 1: Kiểm định Đối nghịch (Adversarial Validation)
- **Ý tưởng thuật toán**: Gộp không gian đặc trưng cân bằng giữa Tập A và Tập B, loại bỏ biến mục tiêu bệnh tim, gán nhãn miền:
  $$Y_{\text{domain}} = \begin{cases} 0 & \text{nếu mẫu thuộc Tập A (Framingham)} \\ 1 & \text{nếu mẫu thuộc Tập B (Cardio)} \end{cases}$$
- Tách tập Train/Test trước khi `fit` `Pipeline(preprocessor + XGBClassifier)` nhằm chống rò rỉ dữ liệu.
- **Kết quả thực nghiệm**: Mô hình đạt **ROC-AUC = 0.9841**.
- **Ý nghĩa toán học**: Hai quần thể bệnh nhân này khác biệt hoàn toàn về mặt thống kê.
- **Xếp hạng đặc trưng trôi dạt (Data Drift Ranking - Gain)**:
  1. `chol_level` (**0.4763**): Phổ cholesterol khác biệt rõ rệt giữa quần thể dịch tễ Mỹ và hồ sơ khám sức khỏe Đông Âu.
  2. `smoker` (**0.1143**): Tỷ lệ hút thuốc ở Framingham cao gấp nhiều lần so với Cardio.
  3. `dia_bp` (**0.1014**) và `gluc_level` (**0.0963**): Phân phối huyết áp tâm trương và đường huyết lúc đói lệch giữa hai nguồn đo lường.

*(Xem biểu đồ xếp hạng trôi dữ liệu tại [`figures/fig7_adversarial_validation_drift.png`](figures/fig7_adversarial_validation_drift.png))*

---

### 5.3. Điểm cộng 2: Hiệu chuẩn Xác suất Hậu nghiệm Hai chiều (Probability Calibration)
- Khi mô hình nguồn được chuyển sang bệnh viện mới có tỷ lệ nền khác biệt, ta không cần huấn luyện lại toàn bộ mô hình (tốn kém tài nguyên).
- **Phương pháp**: Bọc toàn bộ `Pipeline` đã huấn luyện trên miền nguồn bằng `FrozenEstimator`, dùng lát cắt Validation của miền đích để hiệu chuẩn theo **Platt Scaling (Sigmoid)** hoặc **Isotonic Regression** (`CalibratedClassifierCV`) cho **cả hai chiều $A \to B$ và $B \to A$**.
- **Kết quả thực nghiệm hai chiều**:
  - **Chiều $A \to B$ (Khắc phục *Under-confidence*, $15.19\% \to 49.47\%$):**
    - Raw XGBoost: **0.3470** $\to$ Sigmoid: **0.2195** (**Giảm 36.7%**) $\to$ Isotonic: **0.2123** (**Giảm 38.8%**).
  - **Chiều $B \to A$ (Khắc phục *Over-confidence*, $49.47\% \to 15.19\%$):**
    - Raw XGBoost: **0.3298** $\to$ Sigmoid: **0.1235** (**Giảm 62.5%**) $\to$ Isotonic: **0.1254** (**Giảm 62.0%**).
  - Đường cong tin cậy (*Reliability Diagram*) của cả hai chiều sau hiệu chuẩn áp sát đường chéo lý tưởng $y=x$.

*(Xem đường cong hiệu chuẩn hai chiều tại [`figures/fig8_calibration_curves.png`](figures/fig8_calibration_curves.png))*

---

### 5.4. Điểm cộng 3: Khám phá Cơ chế Ra Quyết định Đa miền & Phân tích Âm tính giả (SHAP)
- Sử dụng lý thuyết trò chơi hợp tác qua thư viện `SHAP (TreeExplainer)` và đồng bộ chính xác tên biến xuất ra từ `ColumnTransformer` (`get_feature_names_out()`) để so sánh mô hình XGBoost trên **cả Tập A (Test A) và Tập B ($N = 2.000$ mẫu ngẫu nhiên)**:
  - **Đối sánh Tầm quan trọng Toàn cục (Mỹ vs Đông Âu)**:
    - **Tập A (Framingham - Dự báo 10 năm)**: Tuổi tác (`age`, Mean $|\text{SHAP}| = \mathbf{0.571}$), Huyết áp tâm thu (`sys_bp`, **0.262**), Giới tính nam (`is_male`, **0.186**) và Hút thuốc (`smoker`, **0.119**) là 4 yếu tố hàng đầu; `chol_level` chỉ đóng góp **0.025**.
    - **Tập B (Cardio - Chẩn đoán lâm sàng)**: Huyết áp tâm thu (`sys_bp`, **0.827**) áp đảo tuyệt đối, tiếp theo là `age` (**0.257**) và **`chol_level` (0.196)** — mức ảnh hưởng cao gấp **7.7 lần** so với Framingham, trong khi `is_male` gần như trung tính (**0.005**).
  - **Phân tích Cục bộ Ca bệnh Bỏ sót (False Negative Case Study trên Tập A)**:
    - Trong 129 ca bệnh thực tế trên tập kiểm thử Framingham, mô hình bỏ sót 16 ca. Bệnh nhân **#672** là ca âm tính giả nghiêm trọng nhất với xác suất dự đoán cực thấp: **$p = 0.017$ (1.7%)** dù thực tế mắc bệnh ($Y=1$).
    - **Hồ sơ lâm sàng**: Nam giới (`is_male = 1`), còn rất trẻ (**35 tuổi**), huyết áp thấp/bình thường (**112 / 62.5 mmHg**), thể trạng gầy (**BMI = 21.03**), Cholesterol và Glucose bình thường (mức 1), nhưng **có hút thuốc lá** (`smoker = 1`).
    - **Giải thích từ SHAP Waterfall**: Dù giới tính nam (`+0.21`) và hút thuốc (`+0.14`) đẩy nguy cơ tăng lên, yếu tố tuổi trẻ (`age = -1.71` độ lệch chuẩn) kéo giá trị SHAP xuống mạnh tới **`-2.15`** cùng huyết áp tâm thu thấp (**`-0.29`**), lấn át hoàn toàn cảnh báo từ việc hút thuốc.

*(Xem biểu đồ đối sánh SHAP đa miền tại [`figures/fig9_shap_global_summary.png`](figures/fig9_shap_global_summary.png) và SHAP Waterfall ca bỏ sót tại [`figures/fig10_shap_local_false_negatives.png`](figures/fig10_shap_local_false_negatives.png))*

---

### 5.5. Điểm cộng 4: Dự đoán Phù hợp Đảm bảo Độ phủ Thống kê (Conformal Prediction - MAPIE)
- Trong y tế, việc ép máy tính đưa ra kết luận nhị phân cứng ($0$ hoặc $1$) ở vùng ranh giới bất định là rủi ro lớn.
- Áp dụng kỹ thuật **Split Conformal Prediction** qua thư viện `MAPIE` (mức ý nghĩa $\alpha = 0.10$) cho mô hình `Random Forest` trên **cả 2 tập dữ liệu** để đảm bảo bảo chứng toán học:
  $$P(Y \in C(X)) \ge 1 - \alpha = 90\%$$
- **Kết quả thực nghiệm đối sánh trên tập Test**:
  - **Tập A — Framingham (Mất cân bằng 15.19%, dự báo 10 năm)**:
    - Độ phủ thực nghiệm đạt: **90.4%** (vượt chuẩn 90%).
    - **Tập $\{0\}$ (An toàn loại trừ)**: **83.37%** bệnh nhân.
    - **Tập $\{1\}$ (Chắc chắn nguy cơ cao)**: **0.00%** (vì xác suất biến cố 10 năm hiếm khi tiệm cận >85% để tách đơn lẻ mà vẫn đảm bảo độ phủ 90%).
    - **Tập $\{0, 1\}$ (Vùng bất định - Chuyển bác sĩ)**: **16.63%** bệnh nhân được tự động gắn cờ chuyển chuyên khoa hội chẩn.
  - **Tập B — Cardio (Cân bằng 49.47%, chẩn đoán lâm sàng)**:
    - Độ phủ thực nghiệm đạt: **90.1%** (vượt chuẩn 90%).
    - **Tập $\{0\}$ (Chắc chắn khỏe mạnh)**: **27.08%** bệnh nhân $\to$ An toàn xuất viện/theo dõi định kỳ.
    - **Tập $\{1\}$ (Chắc chắn nguy cơ cao)**: **29.07%** bệnh nhân $\to$ Chỉ định điều trị can thiệp ngay.
    - **Tập $\{0, 1\}$ (Vùng bất định - Chuyển bác sĩ)**: **43.85%** bệnh nhân $\to$ Gắn cờ hội chẩn lâm sàng.
    - **Tập rỗng $\{\}$**: **0.00%** trên cả 2 tập.

*(Xem biểu đồ đối sánh Conformal Prediction tại [`figures/fig11_conformal_prediction_coverage.png`](figures/fig11_conformal_prediction_coverage.png))*

---

## VI. KẾT LUẬN & KHUYẾN NGHỊ LÂM SÀNG

1. **Hiệu năng thuật toán**:
   - Trên dữ liệu quy mô lớn cân bằng (Tập B, 68.6k mẫu), tổ hợp cây (**XGBoost** 0.798 và **Random Forest** 0.797) đạt hiệu năng cao nhất.
   - Trên dữ liệu dịch tễ cỡ vừa lệch lớp (Tập A, 4.2k mẫu), **Logistic Regression** dẫn đầu (ROC-AUC **0.747**, PR-AUC **0.399**, Brier **0.113**), chứng minh quan hệ tuyến tính mạnh của tuổi và huyết áp với nguy cơ 10 năm. Chuẩn hóa biến thứ bậc (`ord`) giúp KNN giảm thêm chi phí y tế (từ 7.1% lên 9.7%).
   - **Nystroem Kernel Approximation** giải quyết trọn vẹn bài toán huấn luyện SVM trên tập gần 70.000 dòng chỉ trong vài giây (ROC-AUC 0.795).
2. **Giá trị của việc tối ưu ngưỡng chi phí y tế**:
   - Giữ nguyên xác suất chuẩn không méo rồi tối ưu ngưỡng cắt $p^*$ theo $\text{Recall} \ge 0.80$ giúp giảm **9.5% – 18.9%** chi phí y tế trên Tập A và **21.4% – 34.8%** trên Tập B so với ngưỡng mặc định 0.50.
3. **Quy trình triển khai AI lâm sàng an toàn**:
   - Luôn cô lập `Pipeline` tiền xử lý 3 nhánh độc lập cho từng mô hình để ngăn rò rỉ dữ liệu và nhiễm chéo trạng thái.
   - Trước khi triển khai sang bệnh viện mới: Bắt buộc chạy **Adversarial Validation** kiểm tra trôi dữ liệu và dùng **Post-hoc Calibration** cho cả 2 chiều (giảm tới 38.8% Brier $A \to B$ và 62.5% $B \to A$) để hiệu chỉnh sai lệch tỷ lệ nền.
   - Áp dụng **Conformal Prediction** làm lớp bảo vệ ngoài cùng để kiểm soát tỷ lệ sai sót dưới $10\%$ ($\alpha = 0.10$) và chuyển các ca $\{0, 1\}$ cho bác sĩ chuyên khoa.

---

## VII. KỊCH BẢN TRẢ LỜI CÂU HỎI PHẢN BIỆN (DEFENSE Q&A)

Dưới đây là các câu hỏi trọng tâm mà Thầy/Cô hội đồng thường đặt ra và gợi ý trả lời chuẩn xác:

### Câu hỏi 1: Tại sao nhóm không sử dụng Deep Learning (Mạng nơ-ron sâu) hay TabPFN?
> **Trả lời**: 
> 1. Dữ liệu bảng (Tabular Data) dạng sinh lý học chỉ có 8–16 đặc trưng số và thứ bậc; các nghiên cứu chuẩn mực gần đây (Grinsztajn et al., NeurIPS 2022) đã chứng minh các mô hình Gradient Boosted Trees (như XGBoost, Random Forest) luôn vượt trội hoặc tương đương Deep Learning trên dữ liệu dạng bảng mà không bị hiện tượng overparameterization.
> 2. TabPFN bản chất là mô hình nền tảng Transformer yêu cầu tài nguyên GPU lớn và là hộp đen hoàn toàn, vi phạm mục tiêu của học phần là giúp sinh viên nắm vững bản chất toán học của các thuật toán học máy kinh điển và khả năng giải thích quyết định lâm sàng.

### Câu hỏi 2: Tại sao hàm chi phí y tế nhóm chọn $FP + 5FN$ mà không phải tỷ lệ khác?
> **Trả lời**:
> Trong sàng lọc bệnh tim mạch, lỗi Bỏ sót bệnh nhân ($FN$) đồng nghĩa với việc bệnh nhân không được cảnh báo và có thể đối mặt với nhồi máu cơ tim hoặc tử vong tại nhà. Ngược lại, lỗi Chẩn đoán nhầm ($FP$) chỉ dẫn đến việc bệnh nhân phải làm thêm các xét nghiệm chuyên sâu không xâm lấn (như đo điện tâm đồ ECG hoặc siêu âm tim). Theo các hướng dẫn đánh giá kinh tế y tế (Health Economics Evaluation), chi phí thiệt hại của một ca bỏ sót thường được ước tính gấp từ 5 đến 10 lần một ca báo động giả.

### Câu hỏi 3: Chỉ số ROC-AUC của Adversarial Validation lên tới 0.9841 có phải là lỗi quá khớp (Overfitting) không?
> **Trả lời**:
> Hoàn toàn không. ROC-AUC của Adversarial Validation được đo trên tập kiểm tra tách biệt (33% test split, trong đó `Pipeline` tiền xử lý được tách biệt hoàn toàn để chống rò rỉ dữ liệu). Điểm số cao tiệm cận 1.0 chứng minh rằng mô hình phân biệt miền dễ dàng nhận ra một mẫu đến từ Mỹ hay Đông Âu dựa vào phân phối mức cholesterol (`chol_level`), tỷ lệ hút thuốc (`smoker`), huyết áp tâm trương (`dia_bp`) và đường huyết (`gluc_level`). Đây là bằng chứng toán học rõ ràng nhất cho thấy dữ liệu bị trôi dạt phân phối nghiêm trọng (*Covariate Shift*).

### Câu hỏi 4: Conformal Prediction khác gì so với việc đặt 2 ngưỡng cắt (Cut-off Thresholds) thông thường?
> **Trả lời**:
> Đặt 2 ngưỡng cắt thủ công (ví dụ: $<0.3$ là âm tính, $>0.7$ là dương tính, ở giữa là nghi ngờ) hoàn toàn mang tính kinh nghiệm chủ quan và không có bảo chứng toán học. Conformal Prediction (MAPIE) cung cấp một bảo chứng thống kê chặt chẽ: Dù dữ liệu có phân phối phức tạp thế nào, xác suất nhãn thật nằm trong tập dự báo luôn được đảm bảo toán học $\ge 1 - \alpha$ (ở đây là $\ge 90\%$).

---

## VIII. HƯỚNG DẪN TÁI HIỆN KẾT QUẢ (REPRODUCIBILITY GUIDE)

1. **Khởi tạo môi trường ảo Python 3.12**:
   ```bash
   py -3.12 -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Cấu trúc thư mục chuẩn bị sẵn**:
   - Dữ liệu thô: `data/raw/framingham.csv` và `data/raw/cardio_train.csv`.
   - Toàn bộ pipeline đã được đóng gói và kiểm thử tại: [`healthcare_cvd_risk_pipeline.ipynb`](healthcare_cvd_risk_pipeline.ipynb).
   - Đồ thị xuất bản lưu tự động tại: [`figures/`](figures/).
3. **Thực thi**:
   - Mở Jupyter Notebook / VS Code và chọn Kernel `.venv`.
   - Chọn **Run All Cells** hoặc xem lại kết quả đã nhúng sẵn.
