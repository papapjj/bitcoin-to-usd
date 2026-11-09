# LỘ TRÌNH PHÁT TRIỂN & KẾ HOẠCH TRIỂN KHAI TOÀN DIỆN
# SELF GROWING TRADER V193: TÁI CẤU TRÚC ĐẠI DIỆN, THU GỌN THAM SỐ & TỰ TIẾN HÓA ĐA VÒNG

**Mã tài liệu:** `SGT-V193-ROADMAP-20261019`  
**Phiên bản:** `1.0.0 (Production Release)`  
**Dự án:** Self Growing Trader V193 (Kiến trúc 5 Tác nhân dùng chung trọng số ~93.2M tham số, PPO trên MetaTrader 5)  
**Tác giả:** Worker 3 (`worker_v193_3` / `teamwork_preview_worker`)  
**Tài liệu tham chiếu:**
- Báo cáo Thẩm định Năng lực Codebase: `explorer_v193_1/report_codebase_bottlenecks.md`
- Báo cáo Nghiên cứu Kiến trúc SOTA: `explorer_v193_2/report_sota_architectures.md`
- Báo cáo Phương pháp Huấn luyện Nâng cao: `explorer_v193_3/report_advanced_training.md`
- Báo cáo Kiểm tra Sức chứa Kiến trúc: `docs/FIVE_ARCHITECTURE_CAPACITY_AUDIT_20260906.vi.md`
- Yêu cầu Gốc & Biên bản Thất bại: `docs/ORIGINAL_REQUEST.md` (Run 820001 – 820007)

---

## MỤC LỤC
1. [TRIẾT LÝ PHÁT TRIỂN & CÁC BẤT BIẾN KIẾN TRÚC HỆ THỐNG](#1-triết-lý-phát-triển--các-bất-biến-kiến-trúc-hệ-thống)
   - 1.1. Triết lý Kỹ thuật Cốt lõi
   - 1.2. Ràng buộc Bất biến Hệ thống Không Thể Thương Lượng (System Invariants)
2. [LỘ TRÌNH PHÁT TRIỂN 3 TẦNG CHI TIẾT (3-TIER PHASED ROADMAP)](#2-lộ-trình-phát-triển-3-tầng-chi-tiết-3-tier-phased-roadmap)
   - 2.1. Giai đoạn 1: Ngắn hạn (1–2 tuần) — Immediate Stabilization & Bias Elimination
     - Milestone 1.1: Tích hợp Representation Regularization Suite (VICReg + RankMe + ArcFace)
     - Milestone 1.2: Triển khai Dynamic Balanced Multi-Regime Reservoir Buffer & Extended Lifecycle Settlement
     - Milestone 1.3: Áp dụng Dynamic Team-Spirit Cosine Decay & Portfolio Coordination Metric
   - 2.2. Giai đoạn 2: Trung hạn (1–2 tháng) — Architectural Transformation & Efficiency Leap
     - Milestone 2.1: Xây dựng & Kiểm thử Độc lập Mô hình SGT-V193-Compact (17.5M Params)
     - Milestone 2.2: Quy trình Chưng cất Tri thức Lũy tiến (Progressive Feature Distillation)
     - Milestone 2.3: Triển khai COMA Centralized Critic & RND Intrinsic Exploration
   - 2.3. Giai đoạn 3: Dài hạn (3–6 tháng) — Frontier Scaling, Microstructure & Autonomous Evolution
     - Milestone 3.1: Mở rộng Mamba-2 Structured State Space Duality (SSD) với Runtime Tối ưu MT5
     - Milestone 3.2: Tích hợp Bộ Mã Hóa Vi Cấu Trúc & Sóng Đa Phân Giải (Microstructure & Wavelet Encoder)
     - Milestone 3.3: Context-Conditioned Meta-RL (PEARL) & Vòng lặp Tự tiến hóa Đa vòng (Autonomous Evolution Loop)
3. [MA TRẬN PHỤ THUỘC & PHÂN TÍCH ĐƯỜNG GĂNG (CRITICAL PATH ANALYSIS)](#3-ma-trận-phụ-thuộc--phân-tích-đường-găng-critical-path-analysis)
   - 3.1. Sơ đồ Mạng Lưới Phụ Thuộc (Dependency Network Diagram)
   - 3.2. Bảng Phân Tích Đường Găng & Quỹ Thời Gian Dự Phòng (Critical Path Table)
4. [BẢNG TIÊU CHÍ CHẤP THUẬN TỔNG HỢP QUA CÁC GIAI ĐOẠN (ACCEPTANCE CRITERIA MATRIX)](#4-bảng-tiêu-chí-chấp-thuận-tổng-hợp-qua-các-giai-đoạn-acceptance-criteria-matrix)
5. [MA TRẬN QUẢN TRỊ RỦI RO & KẾ HOẠCH DỰ PHÒNG (RISK GOVERNANCE & FALLBACK MATRIX)](#5-ma-trận-quản-trị-rủi-ro--kế-hoạch-dự-phòng-risk-governance--fallback-matrix)
   - 5.1. Danh mục 6 Rủi ro Kỹ thuật Trọng yếu
   - 5.2. Cơ chế Kích hoạt & Quy trình Xử lý Sự cố Tự động

---

## 1. TRIẾT LÝ PHÁT TRIỂN & CÁC BẤT BIẾN KIẾN TRÚC HỆ THỐNG

### 1.1. Triết lý Kỹ thuật Cốt lõi

Hệ thống giao dịch **Self Growing Trader V193** kế thừa và phát triển từ kiến trúc OpenAI Five (Dota 2, 2019), giải quyết bài toán giao dịch tài chính đa tác nhân trong môi trường nhiễu cao của thị trường ngoại hối và vàng (XAUUSD, EURUSD) thông qua MetaTrader 5 Strategy Tester. Toàn bộ lộ trình phát triển được xây dựng dựa trên 3 trụ cột triết lý kỹ thuật:

```
                      TRIẾT LÝ PHÁT TRIỂN V193
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
[OPTIMIZE BEFORE SCALING]   [EMPIRICAL EVIDENCE-FIRST]   [ZERO HARD-GATE REGRESSION]
Thu gọn tham số, tối ưu      Mọi quyết định dựa trên     Triệt tiêu thiên lệch bằng
hiệu quả biểu diễn trước     đo lường thực nghiệm OOS,   cơ chế RL tự nhiên, tuyệt đối
khi mở rộng quy mô mạng      không suy đoán lý thuyết    không dùng cờ cưỡng chế cứng
```

1. **Tối ưu hóa trước khi mở rộng (Optimize before scaling):**
   - Thực nghiệm từ Báo cáo Kiểm tra Sức chứa (`FIVE_ARCHITECTURE_CAPACITY_AUDIT_20260906.vi.md`) chỉ ra rằng mô hình hiện tại đang rơi vào nghịch lý **"Overparameterized but Informationally Starved"**: Mạng sở hữu tới **93.182.878 tham số** nhưng hạng hiệu dụng (effective rank) của không gian ẩn 4096 chiều chỉ đạt **$r_{eff} \approx 2,116$**, 90% phương sai tập trung ở 2 trục chính, và bộ nhớ hồi quy bị xóa sạch sau 8–16 nến M1.
   - Do đó, việc tiếp tục tăng kích thước mạng là phản tác dụng. Nguyên tắc tối thượng của lộ trình là **tinh gọn và tăng mật độ thông tin**: Giảm quy mô tham số từ 93.2M xuống 17.5M (giảm 81.2%), tăng tốc độ suy luận CPU lên gấp 5.2 lần (từ 177ms xuống $\le 34$ms qua TorchScript/ONNX/C++ DLL, môi trường Python thông dịch baseline P50 = 43.3ms / Mean = 65.0ms), đồng thời nâng hạng hiệu dụng lên $r_{eff} > 120$ và dung lượng ô nhớ hữu hiệu lên gấp 36 lần.

2. **Đo lường thực nghiệm làm bằng chứng (Empirical evidence-first):**
   - Mọi đề xuất cải tiến kiến trúc hoặc thuật toán huấn luyện đều phải xuất phát từ bằng chứng định lượng rõ ràng (chỉ số Torchinfo, PyTorch Profiler, SVD PCA curves, Linear Probe BA/MCC, Class-conditional geometry, và Chronological OOS testing).
   - Tuyệt đối không chấp nhận các cải tiến "nghe có vẻ hay" nhưng không đem lại bước nhảy vọt về chất lượng biểu diễn hoặc làm suy giảm kết quả backtest (như trường hợp nhánh temporal token v3 trong audit làm PnL giảm -$590.42).

3. **An toàn hồi quy & Học tập tự nhiên không cờ cứng (Zero hard-gate regression safety):**
   - Sự thất bại trong chuỗi Failure Replay Cases (820003 – 820007: 85–90% lệnh Short trong Uptrend gây lỗ -$179.99) xuất phát từ hiện tượng Quên thảm họa (Catastrophic Forgetting) do mất cân bằng dữ liệu nạp vào PPO, chứ không phải do thiếu các bộ lọc điều kiện.
   - Toàn bộ quá trình triệt tiêu lệnh ngược xu hướng phải đạt được thông qua **các cơ chế RL nội sinh** (Bidirectional Ranking, Expectancy Veto, Balanced Reservoir Sampling, Counterfactual Credit Assignment), tuyệt đối **không sử dụng bất kỳ cờ cưỡng chế cứng nào** (`context_direction_entry_gate_enabled = False`), bảo đảm tính thích nghi linh hoạt khi cấu trúc thị trường thay đổi.

---

### 1.2. Ràng buộc Bất biến Hệ thống Không Thể Thương Lượng (System Invariants)

Bất kỳ sửa đổi nào trong toàn bộ 3 giai đoạn của lộ trình đều phải tuân thủ nghiêm ngặt 7 bất biến sau đây:

| STT | Bất biến Hệ thống | Đặc tả Kỹ thuật | Cơ sở Bảo toàn |
|:---:|:---|:---|:---|
| **INV-1** | **Observation Contract** | Chiều dài observation vector cố định **137 floats** (`INPUT_COUNT = 137`, protocol schema 10). | MT5 EA gửi đúng 137 float qua Shared Memory. Mọi kỹ thuật vi cấu trúc (OFI, RV, Wavelet) đều trích xuất nội bộ trong Python encoder. |
| **INV-2** | **Action Bridge Contract** | Chiều dài action vector cố định **20 floats** (`OUTPUT_COUNT = 20`, execution contract v7). | Giữ nguyên thứ tự logit: 12 discrete actions, risk, delay, marker=7.0, entry offset, geometry packed (SL/TP/Trailing), partial close, manage mode, target slot. |
| **INV-3** | **5-Agent Shared Policy** | 5 tác nhân chuyên biệt cùng chia sẻ 1 bộ trọng số duy nhất ($\Theta$), duy trì hidden state riêng. | Tiết kiệm bộ nhớ, tăng tốc độ học thông qua việc chia sẻ đặc trưng thị trường vĩ mô giữa 5 góc nhìn vai trò. |
| **INV-4** | **PPO / GAE Compatibility** | Toàn bộ các cơ chế cải tiến phải tương thích với tối ưu hóa Proximal Policy Optimization. | Không phá vỡ Clipped Surrogate Objective, bảo đảm hội tụ ổn định và kiểm soát policy drift qua clipping $\epsilon=0.2$. |
| **INV-5** | **Zero Hard-Direction Gates** | Không kích hoạt `context_direction_entry_gate_enabled` hoặc bất kỳ if/else rule nào để chặn lệnh. | Mô hình phải tự học triệt tiêu lệnh ngược trend thông qua phân bổ tín dụng và bộ đệm cân bằng. |
| **INV-6** | **Zero Premature Close** | Giữ nguyên `lock_non_entry_to_hold = True`. Số lệnh đóng bởi `POLICY_CLOSE_ALL` bằng 0 tuyệt đối. | Loại bỏ hoàn toàn hành vi cắt lãi non của Governor (Agent 4) khi vị thế đang chạy thuận lợi. |
| **INV-7** | **Zero Numerical Drift Migration** | Checkpoint migration phải bảo đảm hành vi epoch 0 của mô hình mới trùng khớp 100% mô hình cũ. | Sử dụng ReZero ($\alpha=0.0$) hoặc Zero-Initialized Residual Adapters để không gây sốc cho chính sách khi nâng cấp. |

---

## 2. LỘ TRÌNH PHÁT TRIỂN 3 TẦNG CHI TIẾT (3-TIER PHASED ROADMAP)

```
                            LỘ TRÌNH TỔNG THỂ 3 GIAI ĐOẠN
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       ▼                                  ▼                                  ▼
 [GIAI ĐOẠN 1: NGẮN HẠN]           [GIAI ĐOẠN 2: TRUNG HẠN]          [GIAI ĐOẠN 3: DÀI HẠN]
 (1–2 Tuần: Ổn định & Triệt hạ     (1–2 Tháng: Tái Cấu Trúc          (3–6 Tháng: Vi Cấu Trúc &
  Thiên Lệch Xu Hướng)              Kiến Trúc & Tăng Tốc 5.2x)        Tự Tiến Hóa Đa Vòng)
       │                                  │                                  │
 • M1.1: Regularizer Suite         • M2.1: SGT-V193-Compact (17.5M)  • M3.1: Mamba-2 SSD C++/CUDA
 • M1.2: Balanced Reservoir Buffer • M2.2: Progressive Distillation  • M3.2: Microstructure & Wavelet
 • M1.3: Dynamic Cosine Decay      • M2.3: COMA Critic + RND Explor  • M3.3: PEARL Meta-RL & Auto Loop
```

---

### 2.1. Giai đoạn 1: Ngắn hạn (1–2 tuần) — Immediate Stabilization & Bias Elimination

**Chủ đề:** Ổn định Tức thì & Loại bỏ Thiên lệch Xu hướng  
**Mục tiêu cốt lõi:** Ngăn chặn hiện tượng Short áp đảo trong Uptrend, triệt tiêu sụp đổ không gian ẩn của V193 hiện tại, và loại bỏ sự suy giảm Payoff do đóng lệnh hết hạn 120 bars — **hoàn toàn không thay đổi kiến trúc suy luận và không phát sinh thêm tham số mới khi deploy MT5**.

---

#### Milestone 1.1: Tích hợp Representation Regularization Suite (VICReg + RankMe + ArcFace)
- **Mục tiêu kỹ thuật (Objective):**
  - Cứu vãn không gian ẩn 4096 chiều đang bị sụp đổ ($r_{eff} = 2,116$), ép các neuron biến thiên trực giao thay vì đồng tuyến.
  - Tách rộng khoảng cách hình học giữa cụm `ENTRY` và `HOLD` từ mức chồng lấn nghiêm trọng ($0,18 - 0,33$ pooled radius) lên $\ge 0,80$ pooled radius.
  - Bổ sung 0 tham số mới khi chạy suy luận trên MT5 (Zero inference parameter overhead).
- **Phạm vi kỹ thuật chi tiết (Technical Scope):**
  1. Xây dựng module `RepresentationRegularizer` tích hợp vào quá trình tính loss trong `five_teacher_imitation.py` và `auto_rl_engine.py`:
     $$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{PPO}} + \lambda_1 \mathcal{L}_{\text{VICReg}}(Z) + \lambda_2 \mathcal{L}_{\text{RankMe}}(Z) + \lambda_3 \mathcal{L}_{\text{ArcFace}}(Z, y)$$
  2. Triển khai VICReg Variance Loss ($v(Z)$ ép độ lệch chuẩn neuron $\ge 1.0$) và Covariance Loss ($c(Z)$ ép các phần tử ngoài đường chéo của ma trận hiệp phương sai về 0).
  3. Triển khai RankMe Approximation Loss qua hàm log-determinant của ma trận hiệp phương sai có điều hòa: $-\ln \det(\frac{1}{B} Z^T Z + \gamma I)$.
  4. Triển khai ArcFace Loss với additive angular margin $m = 0,35$ radian và feature scale $s = 30,0$ trên 3 siêu lớp: `{0: HOLD, 1: BUY_ENTRY, 2: SELL_ENTRY}`.
- **Phụ thuộc kỹ thuật (Dependencies):**
  - Checkpoint V193 baseline hiện tại.
  - Mã nguồn `five_shared_policy.py`, `five_teacher_imitation.py`.
- **Thời gian dự kiến:** 4 ngày làm việc (Tuần 1).
- **Tiêu chí nghiệm thu định lượng riêng (Quantifiable Acceptance Criteria):**
  - [ ] Hạng hiệu dụng Randomized SVD ($r_{eff}$) của vector ẩn 4096 tăng từ **$2,116$ lên $\ge 20,0$** trên tập test OOS 1.600 mẫu.
  - [ ] Rank 90% phương sai tăng từ **2 lên $\ge 8$ trục chính**; Rank 99% tăng từ **11 lên $\ge 40$**.
  - [ ] Khoảng cách tâm cụm ENTRY vs HOLD chia cho pooled radius tăng từ **$0,18 - 0,33$ lên $\ge 0,80$**.
  - [ ] Linear Probe OOS Balanced Accuracy tăng từ **$0,604$ lên $\ge 0,680$**, MCC tăng từ **$0,142$ lên $\ge 0,250$**.
  - [ ] Độ trễ forward CPU giữ nguyên tuyệt đối ở mức **$176,89 \pm 5\text{ ms}$** (do chỉ can thiệp vào pha train).

---

#### Milestone 1.2: Triển khai Dynamic Balanced Multi-Regime Reservoir Buffer & Extended Lifecycle Settlement
- **Mục tiêu kỹ thuật (Objective):**
  - Giải quyết dứt điểm hiện tượng Quên thảm họa (Catastrophic Forgetting), bảo đảm tỷ lệ nạp mẫu cân bằng giữa các chế độ thị trường trong mỗi PPO batch.
  - Giảm tỷ lệ lệnh bị đóng cưỡng chế do hết hạn 120 bars (`CloseExpiredPositions`) từ mức $>35\%$ xuống dưới $15\%$.
  - Trừng phạt các lệnh chịu áp lực lội ngược rủi ro cao (MAE) và ngâm lệnh lâu trong vùng tích lũy.
- **Phạm vi kỹ thuật chi tiết (Technical Scope):**
  1. Triển khai class `BalancedMultiRegimeReplayBuffer` với 3 reservoir pools độc lập: $\mathcal{B}_{\text{uptrend}}$, $\mathcal{B}_{\text{downtrend}}$, $\mathcal{B}_{\text{sideway}}$, dung lượng mỗi pool cố định $N = 128$ quỹ đạo.
  2. Phân loại chế độ thị trường tự động qua Chop Index (Feature 44) và Trend Momentum Slope (Feature 36):
     - $CI \le 50.0$ và $\text{Slope} > +0.25 \implies$ **Uptrend**.
     - $CI \le 50.0$ và $\text{Slope} < -0.25 \implies$ **Downtrend**.
     - Ngược lại ($CI > 61.8$ hoặc $|\text{Slope}| \le 0.25$) $\implies$ **Sideway / Consolidation**.
  3. Lắp ráp batch PPO kích thước $B = 96$ với tỷ lệ cân bằng đẳng dung lượng tuyệt đối: $32\text{ mẫu Uptrend} + 32\text{ mẫu Downtrend} + 32\text{ mẫu Sideway}$, kết hợp Prioritized Sampling theo sai số TD-Error $\delta_i$ và trọng số Importance Sampling $w_i$.
  4. Cấu hình Extended Lifecycle Settlement Reward trong `chuyen_gia_policy.py`:
     $$R^*_{\text{lifecycle}} = \text{realized\_R} - 0,20 \cdot \text{MAE}_{\text{R}} - 0,25 \cdot \left( \frac{\Delta t}{120} \right)^2 + 0,10 \cdot \mathbb{I}_{(\text{realized\_R} > 0)} \cdot \left( \frac{\text{realized\_R}}{\text{MFE}_{\text{R}} + 1e-4} \right)$$
- **Phụ thuộc kỹ thuật (Dependencies):**
  - Milestone 1.1 (để biểu diễn phân tách tốt).
  - Trích xuất metadata MAE/MFE từ receipt giao dịch MT5.
- **Thời gian dự kiến:** 5 ngày làm việc (Tuần 1 – Tuần 2).
- **Tiêu chí nghiệm thu định lượng riêng (Quantifiable Acceptance Criteria):**
  - [ ] Tỷ lệ lệnh Short trong môi trường Uptrend trên tập đánh giá Q1 2025 **giảm xuống $\le 30\%$** (giảm ít nhất 50% so với mức 85–90% của Run 820003).
  - [ ] Tỷ lệ lệnh bị đóng cưỡng chế do timeout 120 bars **giảm xuống $< 15\%$** trên toàn bộ các vòng test.
  - [ ] Checkpoint thăng cấp đạt đồng thời: **Kỳ vọng kinh tế $E[R] \ge +0,05R$**, **Win Rate $\ge 45\%$**, **Payoff Ratio $\ge 1,50R$**, **Max Drawdown $< 10\%$**.
  - [ ] Số lệnh đóng bởi `POLICY_CLOSE_ALL` bằng 0 tuyệt đối xuyên suốt các vòng.

---

#### Milestone 1.3: Áp dụng Dynamic Team-Spirit Cosine Decay & Portfolio Coordination Metric
- **Mục tiêu kỹ thuật (Objective):**
  - Loại bỏ hoàn toàn "Hội chứng ỷ lại" (Lazy Agent Problem) khi $\tau \to 1.0$, khôi phục tín hiệu trách nhiệm cá nhân cho từng tác nhân.
  - Ngăn ngừa tình trạng Agent 0 (BUY) và Agent 1 (SELL) đồng thời vào lệnh đối kháng nhau gây tổn hại danh mục.
- **Phạm vi kỹ thuật chi tiết (Technical Scope):**
  1. Thay thế lịch trình tăng tuyến tính đơn điệu của `team_spirit` bằng hàm Cosine Annealing suy giảm có chặn trần và sàn:
     $$\tau(k) = 0,15 + \frac{1}{2} (0,65 - 0,15) \left[ 1 + \cos\left( \frac{\pi (k \pmod 5)}{5} \right) \right] \cdot \exp(-0,1 \lfloor k / 5 \rfloor)$$
     - Trần $\tau \le 0,65$: Bảo đảm luôn duy trì ít nhất 35% áp lực phần thưởng cá nhân.
     - Sàn $\tau \ge 0,15$: Giữ lại sự gắn kết danh mục tối thiểu giữa các tác nhân.
  2. Tích hợp Portfolio Coordination Metric (PCM): Nếu phát hiện Agent 0 và Agent 1 cùng sinh logit vào lệnh ngược chiều tại cùng một nến, hệ số $\tau$ của bước đó bị ép ngay lập tức về $0,0$.
- **Phụ thuộc kỹ thuật (Dependencies):**
  - Milestone 1.2.
- **Thời gian dự kiến:** 3 ngày làm việc (Tuần 2).
- **Tiêu chí nghiệm thu định lượng riêng (Quantifiable Acceptance Criteria):**
  - [ ] Không có tác nhân nào bị đóng băng hành vi ở 100% HOLD (ngoại trừ Agent 4 được khóa theo quy tắc thiết kế).
  - [ ] Số trường hợp xung đột mở lệnh Long/Short đồng thời giảm về **0 vụ vi phạm**.
  - [ ] Kỳ vọng kinh tế trung bình $E[R]$ qua chuỗi 10 rounds tăng trưởng ổn định (non-decreasing moving average) và đạt $\ge +0,08R$ ở round cuối.

---

### 2.2. Giai đoạn 2: Trung hạn (1–2 tháng) — Architectural Transformation & Efficiency Leap

**Chủ đề:** Tái Cấu Trúc Kiến Trúc & Bước Nhảy Vọt Hiệu Năng  
**Mục tiêu cốt lõi:** Thay thế khối `nn.LSTMCell(1024, 4096)` cồng kềnh (chiếm 90% tham số và 57% thời gian chạy) bằng mô hình **SGT-V193-Compact 17.5M tham số**, tăng tốc độ suy luận CPU lên gấp 5.2 lần, mở rộng dung lượng bộ nhớ ma trận, và tích hợp cơ chế phân bổ tín dụng phản thực tế COMA cùng RND exploration.

---

#### Milestone 2.1: Xây dựng & Kiểm thử Độc lập Mô hình SGT-V193-Compact (17.5M Params)
- **Mục tiêu kỹ thuật (Objective):**
  - Xây dựng kiến trúc mục tiêu mới với quy mô **17,5 triệu tham số** (giảm 81,2% so với 93,18M cũ).
  - Giảm thời gian suy luận forward CPU từ **176,89 ms xuống $\le 38,0\text{ ms/packet}$** (tăng tốc $\ge 4,6$ lần).
  - Giảm độ phức tạp tính toán từ **931,82 MFLOPs xuống $\le 185,0\text{ MFLOPs}$**.
  - Tăng dung lượng bộ nhớ hữu hiệu lên **147.456 ô nhớ ma trận** (gấp 36 lần sức chứa của LSTM 4096).
- **Phạm vi kỹ thuật chi tiết (Technical Scope):**
  1. Thiết kế mạng 2 tầng xLSTM tuần tự:
     - Tầng 1: `mLSTMCell(in_features=768, hidden_size=768, num_heads=4, head_dim=192)` với quy tắc cập nhật ma trận hiệp phương sai $C_t = f_t C_{t-1} + i_t v_t k_t^T$, ổn định số học không gian log $\log f_t = -\text{softplus}(-\tilde{f}_t) \le 0$ per Beck et al. (NeurIPS 2024) và truy xuất chuẩn hóa $\tilde{h}_t = C_t q_t / \max(|n_t^T q_t|, 1)$.
     - Tầng 2: `sLSTMCell(in_features=768, hidden_size=768, num_heads=4)` với cùng nguyên lý ổn định không gian log $\log f_t^{(s)} \le 0$, dynamic memory stabilizer $m_t$ chống tràn số mũ và triệt tiêu nguy cơ amnesia.
  2. Thiết kế khối `RelationalCrossAgentTransformer` (thay thế Transformer v2 cũ):
     - Định nghĩa ma trận quan hệ vai trò $R \in \mathbb{R}^{5 \times 5}$ cố định kết hợp vector embedding quan hệ học được $e_{\text{role}}(R_{ij})$.
     - Áp dụng ma trận phạt đối kháng $\mathcal{M}_{\text{conflict}}$ giữa Agent 0 (BUY) và Agent 1 (SELL).
     - Kết nối Gated Residual Network (GRN) với tham số ReZero scale $\alpha$ khởi tạo chính xác bằng $0,0$.
  3. Cài đặt 100% bằng toán tử PyTorch native vector hóa (`torch.*`), bảo đảm chạy độc lập hoàn hảo trên Windows x86_64 CPU mà không cần GPU hay custom CUDA toolkit.
  4. Thiết lập quy trình tối ưu hóa suy luận và biên dịch mô hình: Xuất mô hình sang TorchScript JIT / ONNX Runtime (`onnxruntime-cpu`) / C++ DLL (`libtorch`) cho MetaTrader 5 Expert Advisor execution, triệt tiêu overhead thông dịch của CPython.
- **Phụ thuộc kỹ thuật (Dependencies):**
  - Hoàn thành toàn bộ Giai đoạn 1 (Tier 1).
  - Nghiên cứu từ Explorer 2 (`report_sota_architectures.md: Section 7 & 9`).
- **Thời gian dự kiến:** 3 tuần (Tuần 3 – Tuần 5).
- **Tiêu chí nghiệm thu định lượng riêng (Quantifiable Acceptance Criteria):**
  - [ ] Tổng số tham số trainable của mô hình nằm trong khoảng **$16,5\text{M} - 18,5\text{M}$** (Torchinfo audit xác nhận).
  - [ ] Thời gian suy luận forward CPU: Python CPython thông dịch đạt Trung vị P50 $\le 45,0\text{ ms}$ (Mean $\approx 65\text{ ms}$); mô hình biên dịch TorchScript JIT / ONNX / C++ DLL đạt **$\le 34,0\text{ ms/packet}$** (đáp ứng trọn vẹn chỉ tiêu $\le 38,0\text{ ms/packet}$ qua PyTorch Profiler 100 lần lặp).
  - [ ] Tổng khối lượng tính toán forward pass **$\le 185,0\text{ MFLOPs}$**.
  - [ ] Bộ nhớ RAM chiếm dụng khi chạy inference **$\le 250\text{ MB}$**.
  - [ ] Bảo toàn tuyệt đối hợp đồng: Đầu vào 137 float, đầu ra 20 float, 5 agent shared weights.

---

#### Milestone 2.2: Quy trình Chưng cất Tri thức Lũy tiến (Progressive Feature Distillation)
- **Mục tiêu kỹ thuật (Objective):**
  - Chuyển giao toàn bộ tri thức nhận diện cấu trúc thị trường từ mô hình Teacher V193 93M cũ sang mô hình Student SGT-Compact 17.5M mới mà không cần huấn luyện lại từ số 0.
  - Rút ngắn thời gian hội tụ của mô hình Compact xuống dưới 5 epochs trên tập dữ liệu Replay có sẵn.
- **Phạm vi kỹ thuật chi tiết (Technical Scope):**
  1. Thiết lập pipeline chưng cất tri thức 2 thành phần:
     $$\mathcal{L}_{\text{Distill}} = \lambda_{\text{rep}} \frac{1}{B} \sum_{i=1}^B \| W_{\text{proj}} h_S(x_i) - h_T(x_i) \|_2^2 + \lambda_{\text{logit}} \tau^2 D_{\text{KL}}\left( \text{Softmax}\left(\frac{z_T}{\tau}\right) \parallel \text{Softmax}\left(\frac{z_S}{\tau}\right) \right)$$
     với $W_{\text{proj}} \in \mathbb{R}^{4096 \times 768}$ là ma trận chiếu đặc trưng, nhiệt độ chưng cất $\tau = 2,0$.
  2. Trích xuất offline representation từ checkpoint Teacher V193 xuất sắc nhất của Tier 1 trên 50.000 nến M1 lịch sử.
  3. Huấn luyện Student Compact qua 5 epochs Representation Distillation, sau đó rã đông dần các action heads để khớp logit.
- **Phụ thuộc kỹ thuật (Dependencies):**
  - Milestone 2.1 (Kiến trúc SGT-Compact sẵn sàng).
  - Checkpoint Teacher V193 đã vượt qua Tier 1 Gate.
- **Thời gian dự kiến:** 2 tuần (Tuần 5 – Tuần 6).
- **Tiêu chí nghiệm thu định lượng riêng (Quantifiable Acceptance Criteria):**
  - [ ] Tỷ lệ trùng khớp hành động (Action Match Rate) giữa Student và Teacher trên tập OOS đạt **$\ge 92,0\%$**.
  - [ ] Linear Probe Balanced Accuracy của Student đạt **$\ge 98,0\%$** so với Teacher (không bị suy giảm năng lực phân tách).
  - [ ] Hạng hiệu dụng ($r_{eff}$) của Student Compact đạt **$\ge 80,0$** (cao gấp 35 lần Teacher 93M cũ).

---

#### Milestone 2.3: Triển khai COMA Centralized Critic & RND Intrinsic Exploration
- **Mục tiêu kỹ thuật (Objective):**
  - Bóc tách tuyệt đối trách nhiệm kinh tế của từng tác nhân bằng thuật toán Counterfactual Multi-Agent Policy Gradients (COMA).
  - Phá vỡ bẫy sụp đổ hành vi HOLD (HOLD-collapse) ở cuối quá trình huấn luyện bằng cơ chế tò mò ngẫu nhiên RND.
  - Tích hợp Hindsight Experience Replay (HER) để tái sinh các vị thế chạm đỉnh MFE bị timeout 120 bars thành mẫu học chất lượng cao.
- **Phạm vi kỹ thuật chi tiết (Technical Scope):**
  1. Triển khai mạng Centralized Critic $Q_\phi(s, \mathbf{u})$ ước lượng giá trị hành động kết hợp của 5 tác nhân, tính toán Counterfactual Baseline và Counterfactual Advantage riêng biệt cho từng agent:
     $$A^a_{\text{COMA}}(s, \mathbf{u}) = Q_\phi(s, \mathbf{u}) - \sum_{u'^a} \pi^a(u'^a | s) Q_\phi(s, (\mathbf{u}^{-a}, u'^a))$$
  2. Xây dựng module RND Exploration với Target Network ngẫu nhiên đóng băng và Predictor Network 64D; bổ sung phần thưởng khám phá nội tại $r_t^{\text{intr}}$ chuẩn hóa qua thống kê Welford với trọng số $\kappa = 0,05$.
  3. Tích hợp HER offline relabeling: Với các vị thế timeout có $\text{MFE} \ge 0,75R$, tạo quỹ đạo phản thực tế với mục tiêu $\text{TP}^* = 0,90 \times \text{MFE}$ và gán nhãn chốt lời thành công.
- **Phụ thuộc kỹ thuật (Dependencies):**
  - Milestone 2.1 và 2.2.
- **Thời gian dự kiến:** 2 tuần (Tuần 7 – Tuần 8).
- **Tiêu chí nghiệm thu định lượng riêng (Quantifiable Acceptance Criteria):**
  - [ ] Counterfactual Advantage phân tách thành công công/tội: Agent vào lệnh sai chịu $A^a \ll 0$ trong khi các tác nhân giữ lệnh khác có $A^a \approx 0$.
  - [ ] Hiện tượng Tail Collapse (chỉ chọn HOLD ở các round cuối) bị loại bỏ hoàn toàn; tỷ lệ hành động tích cực duy trì đều đặn $10\% - 15\%$.
  - [ ] Vượt qua Paired OOS MT5 Gate trên dữ liệu Q1 2025: **$E[R] \ge +0,12R$**, **Win Rate $\ge 50\%$**, **Payoff Ratio $\ge 1,80R$**, **Max Drawdown $< 8\%$**, **Tỷ lệ timeout nến 120 $\le 8\%$**.

---

### 2.3. Giai đoạn 3: Dài hạn (3–6 tháng) — Frontier Scaling, Microstructure & Autonomous Evolution

**Chủ đề:** Mở Rộng Tiên Phong, Vi Cấu Trúc Thị Trường & Tự Tiến Hóa Đa Vòng  
**Mục tiêu cốt lõi:** Khai thác các kiến trúc chuỗi siêu dài (Mamba-2 SSD) với runtime C++/CUDA tối ưu cho MT5, hiện đại hóa bộ mã hóa vi cấu trúc thị trường và sóng Wavelet đa phân giải, đồng thời xây dựng vòng lặp tự tiến hóa Meta-RL (PEARL) tự động thích ứng với các cú sốc chế độ thị trường mà không cần can thiệp thủ công.

---

#### Milestone 3.1: Mở rộng Mamba-2 Structured State Space Duality (SSD) với Runtime Tối ưu MT5
- **Mục tiêu kỹ thuật (Objective):**
  - Mở rộng tầm nhìn hồi quy từ 64 nến lên **$\ge 512$ nến M1** (tương đương $>8,5$ giờ giao dịch liên tục) với độ phức tạp tính toán tuyến tính $O(L)$ và bộ nhớ $O(1)$.
  - Tối ưu hóa runtime suy luận đạt tốc độ cực hạn $\le 25\text{ ms/packet}$ phục vụ giao dịch tần số cao trên MT5.
- **Phạm vi kỹ thuật chi tiết (Technical Scope):**
  1. Xây dựng module 2 tầng `Mamba2SSDBlock(d_model=768, d_state=64, d_conv=4, expand=2)` dựa trên lý thuyết Structured State Space Duality (Dao & Gu, ICML 2024).
  2. Triển khai thuật toán nhân ma trận bán phân tách (1-semiseparable block matrix multiplication) tương thích với CPU SIMD vectorization (AVX-512 / AVX2).
  3. Đóng gói mô hình thành thư viện liên kết động C++ (DLL via LibTorch / ONNX Runtime C++ API) để MT5 EA có thể gọi trực tiếp trong memory, loại bỏ hoàn toàn độ trễ giao tiếp qua Named Pipe.
- **Phụ thuộc kỹ thuật (Dependencies):**
  - Hoàn thành Giai đoạn 2 (Tier 2).
  - Trình biên dịch MSVC C++ trên Windows x86_64.
- **Thời gian dự kiến:** 5 tuần (Tháng 3 – Tháng 4).
- **Tiêu chí nghiệm thu định lượng riêng (Quantifiable Acceptance Criteria):**
  - [ ] Khả năng khôi phục cấu trúc thị trường (Temporal Memory Probe) duy trì Balanced Accuracy **$\ge 0,65$ tại Lag 128 nến M1** (vượt trội hoàn toàn so với việc mất trí nhớ ở Lag 16 của LSTM).
  - [ ] Độ trễ suy luận toàn hệ thống qua C++ DLL **$\le 25,0\text{ ms/packet}$** trên CPU MT5.
  - [ ] Hoạt động liên tục qua $\ge 100.000$ tick nến MT5 mà không xảy ra rò rỉ bộ nhớ (Zero Memory Leaks).

---

#### Milestone 3.2: Tích hợp Bộ Mã Hóa Vi Cấu Trúc & Sóng Đa Phân Giải (Microstructure & Wavelet Encoder)
- **Mục tiêu kỹ thuật (Objective):**
  - Làm giàu tín hiệu quan sát đầu vào, lọc bỏ nhiễu trắng ngẫu nhiên của từng tick nến M1 và bảo tồn xu hướng vĩ mô đa khung thời gian.
  - Vẫn bảo toàn nghiêm ngặt 137 float đầu vào từ MT5 bridge (INV-1).
- **Phạm vi kỹ thuật chi tiết (Technical Scope):**
  1. Xây dựng nhánh trích xuất đặc trưng vi cấu trúc (Microstructure Extraction Branch):
     - Order Flow Imbalance Proxy (OFI) ước lượng qua signed tick volume theo quy tắc Lee-Ready Tick Rule.
     - Bộ ba biến động không chệch: Garman-Klass Volatility $\sigma_{GK}^2$, Parkinson Volatility $\sigma_P^2$, và Yang-Zhang Volatility.
  2. Xây dựng nhánh phân rã sóng Wavelet 1D (Discrete Wavelet Transform - DWT Haar/Daubechies) tách tín hiệu giá thành 3 băng tần: Cao tần $D_1$ (nhiễu vi mô), Trung tần $D_2$ (nhịp sóng M5/M15), và Thấp tần $A_2$ (xu hướng chính H1/H4).
  3. Tích hợp Time2Vec embedding biểu diễn chu kỳ phiên giao dịch (London, New York, Tokyo).
  4. Thay thế cơ chế max-pool 25% tĩnh bằng **Cross-Agent Soft Attention Pooling** với gating điều hòa.
- **Phụ thuộc kỹ thuật (Dependencies):**
  - Milestone 2.1 hoặc 3.1.
- **Thời gian dự kiến:** 4 tuần (Tháng 4 – Tháng 5).
- **Tiêu chí nghiệm thu định lượng riêng (Quantifiable Acceptance Criteria):**
  - [ ] Tỷ lệ lệnh dính bẫy phá vỡ giả (False Breakout rate) giảm **$\ge 35\%$** so với baseline.
  - [ ] Tỷ số Sharpe Ratio OOS tăng **$\ge 20\%$** trên tập dữ liệu kiểm thử 6 tháng.
  - [ ] Độ trễ forward tăng không quá 3 ms (vẫn nằm trong ngân sách $\le 38\text{ ms}$).

---

#### Milestone 3.3: Context-Conditioned Meta-RL (PEARL) & Vòng lặp Tự tiến hóa Đa vòng (Autonomous Evolution Loop)
- **Mục tiêu kỹ thuật (Objective):**
  - Cung cấp cho mô hình khả năng thích ứng tức thì (Zero-shot / Few-shot adaptation) khi thị trường chuyển pha đột ngột hoặc xuất hiện tin tức thiên nga đen (Black Swan).
  - Xây dựng vòng lặp tự trị hoàn chỉnh: Tự backtest $\to$ Tự thẩm định $\to$ Tự thăng cấp checkpoint $\to$ Tự retraining mà không cần sự can thiệp của con người.
- **Phạm vi kỹ thuật chi tiết (Technical Scope):**
  1. Triển khai mạng biến phân ngữ cảnh (Variational Context Encoder - PEARL):
     - Nhận đầu vào là 32 bước chuyển đổi vi mô gần nhất $c_{1:32} = (s, a, r, s')$, suy luận ra vector tiềm ẩn ngữ cảnh xác suất $z \sim \mathcal{N}(\mu_\phi, \Sigma_\phi) \in \mathbb{R}^{16}$.
     - Vector $z$ đại diện cho chế độ ẩn của thị trường (thanh khoản mỏng, biến động giật, hay xu hướng dốc) và được ghép nối trực tiếp vào policy.
  2. Thiết lập cụm huấn luyện Population-Based Training (PBT) kết nối 8–16 instances MT5 Strategy Tester chạy song song, tự động đột biến và chọn lọc siêu tham số (Learning Rate, ATR Multipliers, Team Spirit).
  3. Thiết lập vòng lặp giám sát tự động với Promotion Gate đa chế độ: Chỉ thăng cấp checkpoint khi vượt qua đồng thời cả 3 môi trường Bull, Bear và Sideway.
- **Phụ thuộc kỹ thuật (Dependencies):**
  - Toàn bộ các Milestone từ 1.1 đến 3.2.
- **Thời gian dự kiến:** 6 tuần (Tháng 5 – Tháng 6).
- **Tiêu chí nghiệm thu định lượng riêng (Quantifiable Acceptance Criteria):**
  - [ ] Khả năng thích ứng Zero-shot: Mô hình duy trì $E[R] \ge +0,05R$ ngay trong tuần đầu tiên xuất hiện chế độ thị trường mới mà không cần fine-tuning.
  - [ ] Vòng lặp tự tiến hóa vận hành liên tục qua **$\ge 30$ rounds tự động** mà không gặp bất kỳ lỗi deadlock, file lock, hay memory overflow nào.
  - [ ] Kết quả OOS dài hạn (6 tháng): **Win Rate $\ge 52\%$**, **Payoff Ratio $\ge 2,00R$**, **Profit Factor $\ge 2,20$**, **Max Drawdown $< 7\%$**.

---

## 3. MA TRẬN PHỤ THUỘC & PHÂN TÍCH ĐƯỜNG GĂNG (CRITICAL PATH ANALYSIS)

### 3.1. Sơ đồ Mạng Lưới Phụ Thuộc (Dependency Network Diagram)

Dưới đây là sơ đồ mạng lưới thể hiện quan hệ phụ thuộc trước-sau, các nhánh song song và đường găng trọng yếu của toàn bộ dự án:

```
                                  SƠ ĐỒ ĐƯỜNG GĂNG DỰ ÁN (CRITICAL PATH)
                                  
     [BASELINE V193] ─────────────────────────────────────────────────────────────┐
            │                                                                     │
            ▼ (Critical Path)                                                     ▼
     ┌──────────────┐                                                      ┌──────────────┐
     │Milestone 1.1 │ Representation Regularizer                           │Milestone 1.3 │ Dynamic Team-Spirit
     │(VICReg/RankMe│ (Tuần 1: 4 ngày)                                     │Cosine Decay  │ (Tuần 2: 3 ngày)
     │ /ArcFace)    │                                                      └──────┬───────┘
     └──────┬───────┘                                                             │
            │                                                                     │
            ▼ (Critical Path)                                                     │
     ┌──────────────┐                                                             │
     │Milestone 1.2 │ Balanced Reservoir Buffer & Extended Settlement             │
     │              │ (Tuần 1-2: 5 ngày)                                          │
     └──────┬───────┘                                                             │
            │                                                                     │
            ├─────────────────────────────────────────────────────────────────────┘
            ▼
     [GATE 1: TIER 1 STABILIZATION PASSED]
     (r_eff >= 20, Short in Uptrend <= 30%, Timeout <= 15%, E[R] >= +0.05R)
            │
            ▼ (Critical Path)
     ┌──────────────┐                                                      ┌──────────────┐
     │Milestone 2.1 │ Xây dựng SGT-V193-Compact (17.5M, JIT <=34ms)        │Milestone 2.3 │ COMA Critic & RND
     │              │ (Tuần 3-5: 3 tuần)                                   │(Thiết kế)    │ (Chạy song song)
     └──────┬───────┘                                                      └──────┬───────┘
            │                                                                     │
            ▼ (Critical Path)                                                     │
     ┌──────────────┐                                                             │
     │Milestone 2.2 │ Progressive Feature Distillation (Teacher -> Student)       │
     │              │ (Tuần 5-6: 2 tuần)                                          │
     └──────┬───────┘                                                             │
            │                                                                     │
            ├─────────────────────────────────────────────────────────────────────┘
            ▼ (Critical Path)
     ┌──────────────┐
     │Milestone 2.3 │ Hoàn thiện COMA + RND + HER trên SGT-Compact
     │(Tích hợp)    │ (Tuần 7-8: 2 tuần)
     └──────┬───────┘
            │
            ▼
     [GATE 2: TIER 2 EFFICIENCY LEAP PASSED]
     (Params 17.5M, Latency <= 38ms, WR >= 50%, Payoff >= 1.80R, Timeout <= 8%)
            │
            ├──────────────────────────────────────────────────┐
            ▼ (Nhánh Mamba-2)                                  ▼ (Nhánh Microstructure)
     ┌──────────────┐                                   ┌──────────────┐
     │Milestone 3.1 │ Mamba-2 SSD C++/CUDA Runtime      │Milestone 3.2 │ Microstructure & Wavelet
     │              │ (Tháng 3-4: 5 tuần)               │              │ Encoder (Tháng 4-5: 4 tuần)
     └──────┬───────┘                                   └──────┬───────┘
            │                                                  │
            └─────────────────────────┬────────────────────────┘
                                      ▼ (Critical Path)
                               ┌──────────────┐
                               │Milestone 3.3 │ PEARL Meta-RL & Autonomous Loop
                               │              │ (Tháng 5-6: 6 tuần)
                               └──────┬───────┘
                                      │
                                      ▼
                        [GATE 3: FINAL PRODUCTION SOTA]
                        (Zero-shot adaptation, 30+ rounds continuous evolution)
```

---

### 3.2. Bảng Phân Tích Đường Găng & Quỹ Thời Gian Dự Phòng (Critical Path Table)

| Mã Mốc | Tên Milestone | Thời lượng | Phụ thuộc Trực tiếp | Thuộc Đường Găng? | Quỹ Dự phòng (Slack) | Rủi ro Trễ Hạn |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **M1.1** | Representation Regularizer Suite | 4 ngày | Baseline V193 | **CÓ (CP)** | 0 ngày | Thấp (Code toán học thuần túy) |
| **M1.2** | Balanced Reservoir & Settlement | 5 ngày | M1.1 | **CÓ (CP)** | 0 ngày | Trung bình (Cần phân loại regime) |
| **M1.3** | Team-Spirit Cosine Decay | 3 ngày | Baseline V193 | KHÔNG | 6 ngày | Cực thấp (Thay đổi công thức lịch trình) |
| **GATE-1** | Thẩm định Ổn định Giai đoạn 1 | 2 ngày | M1.2, M1.3 | **CÓ (CP)** | 0 ngày | Trung bình (Chờ kết quả OOS MT5) |
| **M2.1** | Xây dựng SGT-V193-Compact | 3 tuần | GATE-1 | **CÓ (CP)** | 0 ngày | Trung bình (Kiểm tra số học xLSTM) |
| **M2.2** | Progressive Feature Distillation | 2 tuần | M2.1 | **CÓ (CP)** | 0 ngày | Cao (Tốc độ hội tụ của Student) |
| **M2.3** | COMA Critic & RND Exploration | 2 tuần | M2.2 | **CÓ (CP)** | 0 ngày | Trung bình (Điều chỉnh hyperparameter $\kappa$) |
| **GATE-2** | Thẩm định Hiệu năng Giai đoạn 2 | 3 ngày | M2.3 | **CÓ (CP)** | 0 ngày | Trung bình (Kiểm thử MT5 Paired OOS) |
| **M3.1** | Mamba-2 SSD C++/CUDA Runtime | 5 tuần | GATE-2 | KHÔNG | 2 tuần | Cao (Tương thích DLL trên Windows) |
| **M3.2** | Microstructure & Wavelet Encoder | 4 tuần | GATE-2 | KHÔNG | 3 tuần | Thấp (PyTorch signal processing) |
| **M3.3** | PEARL Meta-RL & Autonomous Loop | 6 tuần | M3.1, M3.2 | **CÓ (CP)** | 0 ngày | Rất cao (Tự động hóa đa tiến trình) |
| **GATE-3** | Thẩm định Sản phẩm Cuối cùng | 1 tuần | M3.3 | **CÓ (CP)** | 0 ngày | Trung bình (Backtest 6 tháng dài hạn) |

*Tổng thời gian đường găng:* **24 tuần (~5,5 tháng)** từ khi khởi động đến khi hoàn tất nghiệm thu toàn diện hệ thống tự tiến hóa SOTA.

---

## 4. BẢNG TIÊU CHÍ CHẤP THUẬN TỔNG HỢP QUA CÁC GIAI ĐOẠN (ACCEPTANCE CRITERIA MATRIX)

Bảng ma trận dưới đây tổng hợp toàn bộ các chỉ số đo lường định lượng bắt buộc xuyên suốt 4 mốc đánh giá: Hiện trạng V193, Sau Giai đoạn 1 (Tier 1), Sau Giai đoạn 2 (Tier 2), và Hoàn thiện Giai đoạn 3 (Tier 3 SOTA):

$$\begin{array}{|l|c|c|c|c|l|}
\hline
\textbf{Chỉ số Đo lường (Metric)} & \textbf{Hiện trạng V193} & \textbf{Tier 1 (1–2 tuần)} & \textbf{Tier 2 (1–2 tháng)} & \textbf{Tier 3 (3–6 tháng)} & \textbf{Phương pháp Thẩm định} \\
\hline
\text{Tổng Tham số Trainable} & 93.182.878 & 93.182.878 & \mathbf{17.500.000 \pm 1\text{M}} & 18.200.000 \pm 1\text{M} & \text{Torchinfo Summary} \\
\text{Tham số Lõi Tuần tự (Core)} & 83.918.848\text{ (90\%)} & 83.918.848 & \mathbf{10.400.000\text{ (xLSTM)}} & 10.800.000\text{ (SSM)} & \text{Parameter Count Audit} \\
\text{FLOPs mỗi Forward Pass} & 931,82\text{ MFLOPs} & 931,82\text{ MFLOPs} & \mathbf{\le 185,0\text{ MFLOPs}} & \le 160,0\text{ MFLOPs} & \text{PyTorch Profiler (Flops)} \\
\text{Độ trễ Forward CPU (4 luồng)} & 176,89\text{ ms} & 176,89\text{ ms} & \mathbf{\le 38,0\text{ ms (5.2x)}} & \le 25,0\text{ ms (C++ DLL)} & \text{Profiler 100 packets avg} \\
\text{Dung lượng Ô nhớ Hữu hiệu} & 4.096\text{ scalar} & 4.096\text{ scalar} & \mathbf{147.456\text{ matrix (36x)}} & \text{Tuyến tính } O(1) & \text{Kiến trúc State Space} \\
\hline
\text{Effective Rank } (r_{eff}) & 2,116\text{ (Collapse)} & \mathbf{\ge 20,0} & \mathbf{\ge 100,0} & \ge 120,0 & \text{Randomized SVD OOS 1.600} \\
\text{Rank 90\% Phương sai} & 2\text{ trục chính} & \ge 8\text{ trục} & \ge 32\text{ trục} & \ge 48\text{ trục} & \text{PCA Eigenvalue Spectrum} \\
\text{Phân tách Cụm ENTRY vs HOLD} & 0,18 - 0,33 & \mathbf{\ge 0,80} & \mathbf{\ge 1,50} & \ge 1,80 & \text{Distance / Pooled Radius} \\
\text{Attention Entropy Chuẩn hóa} & 0,9902\text{ (Phẳng)} & 0,9902 & \mathbf{0,45 - 0,65\text{ (RGAT)}} & 0,40 - 0,60 & \text{Normalized Shannon Entropy} \\
\text{Memory Horizon (Lag Probe)} & \le 8\text{ nến M1} & \le 8\text{ nến M1} & \mathbf{\ge 48\text{ nến M1}} & \ge 128\text{ nến M1} & \text{Temporal Lag Probe BA} \\
\hline
\text{Tỷ lệ Short trong Uptrend} & 85\% - 90\%\text{ (Run 820003)} & \mathbf{\le 30,0\%} & \mathbf{\le 15,0\%} & \le 10,0\% & \text{Log MT5 / events.jsonl} \\
\text{Tỷ lệ Timeout Nến 120} & > 35,0\% & \mathbf{\le 15,0\%} & \mathbf{\le 8,0\%} & \le 5,0\% & \text{CloseExpiredPositions count} \\
\text{Lệnh Đóng bởi POLICY\_CLOSE\_ALL} & 0\text{ (Khóa cứng)} & 0 & 0 & 0 & \text{Audit events.jsonl (Zero)} \\
\hline
\text{Tỷ lệ Thắng (Win Rate)} & 20\% - 56\% & \mathbf{\ge 45,0\%} & \mathbf{\ge 50,0\%} & \ge 52,0\% & \text{MT5 Strategy Tester Report} \\
\text{Tỷ số Lợi nhuận (Payoff Ratio)} & 0,32 - 1,60 & \mathbf{\ge 1,50R} & \mathbf{\ge 1,80R} & \ge 2,00R & \text{Mean Win R / Mean Loss R} \\
\text{Kỳ vọng Kinh tế } (E[R]) & -0,51R \to +0,26R & \mathbf{\ge +0,05R} & \mathbf{\ge +0,12R} & \ge +0,18R & \text{Expected Value per Trade} \\
\text{Mức Sụt giảm Tài khoản (Max DD)} & > 18,0\% & \mathbf{< 10,0\%} & \mathbf{< 8,0\%} & < 7,0\% & \text{Equity Max Drawdown \%} \\
\text{Hệ số Lợi nhuận (Profit Factor)} & 0,25 - 1,78 & \ge 1,50 & \ge 1,85 & \ge 2,20 & \text{Gross Profit / Gross Loss} \\
\hline
\end{array}$$

---

## 5. MA TRẬN QUẢN TRỊ RỦI RO & KẾ HOẠCH DỰ PHÒNG (RISK GOVERNANCE & FALLBACK MATRIX)

### 5.1. Danh mục 6 Rủi ro Kỹ thuật Trọng yếu

Dưới đây là bảng đánh giá rủi ro kỹ thuật sâu sắc kèm theo các ngưỡng kích hoạt cảnh báo sớm và hành động dự phòng khả thi:

$$\begin{array}{|l|l|c|c|l|l|}
\hline
\textbf{Mã} & \textbf{Mô tả Rủi ro Kỹ thuật} & \textbf{Xác suất} & \textbf{Tác động} & \textbf{Ngưỡng Kích hoạt (Trigger)} & \textbf{Phương án Dự phòng Cụ thể (Fallback Action)} \\
\hline
\textbf{RSK-1} & \textbf{Mamba-2 SSD không tương thích} & \text{Trung bình} & \text{Cao} & \text{Lỗi DLL loading, crash bộ nhớ} & \textbf{Fallback sang xLSTM (mLSTM) thuần túy:} \\
& \text{hoặc phát sinh lỗi DLL runtime} & & & \text{hoặc latency MT5 vượt quá} & \text{Khối mLSTM 768D (Milestone 2.1) viết 100\% PyTorch} \\
& \text{trên môi trường Windows MT5.} & & & 50\text{ ms trên CPU testing.} & \text{đã được ổn định log-space và kiểm chứng vững chắc;} \\
& & & & & \text{biên dịch TorchScript/ONNX bảo đảm } \le 34\text{ ms MT5.} \\
\hline
\textbf{RSK-2} & \textbf{Student Compact 17.5M không đạt} & \text{Thấp} & \text{Rất cao} & \text{Sau 10 epochs chưng cất,} & \textbf{Triển khai kiến trúc trung gian LSTM-2048D:} \\
& \text{hiệu suất của Teacher 93M} & & & \text{Action Match Rate } < 85\% & \text{Theo kết quả cắt gọt trong audit, LSTM-2048D} \\
& \text{qua Progressive Distillation.} & & & \text{hoặc probe BA giảm } > 5\%. & \text{giữ lại 86,14\% variance gốc và tăng tốc 2,36x;} \\
& & & & & \text{có thể copy trực tiếp trọng số từ Teacher mà không cần học lại.} \\
\hline
\textbf{RSK-3} & \textbf{EWC khóa trọng số quá cứng,} & \text{Trung bình} & \text{Trung bình} & \text{Policy loss không giảm sau} & \textbf{Chuyển sang PackNet hoặc Dynamic Reservoir Buffer:} \\
& \text{ngăn cản mạng học chế độ thị} & & & 3\text{ rounds, hoặc Fisher penalty} & \text{Giảm } \lambda_{\text{EWC}} \text{ từ 250 về 50, hoặc chuyển hoàn} \\
& \text{trường mới (Gradient Stagnation).} & & & \text{chiếm } > 80\% \text{ tổng gradient.} & \text{toàn trọng trách chống quên cho Bộ đệm 3 Bể} \\
& & & & & \text{(Balanced Buffer 33/33/33) không dùng penalty.} \\
\hline
\textbf{RSK-4} & \textbf{RND Intrinsic Reward quá lớn} & \text{Trung bình} & \text{Trung bình} & \text{Tần suất vào lệnh tăng vọt} & \textbf{Áp dụng Dynamic Scaling & Reward Decay:} \\
& \text{khiến mô hình "nghiện" khám} & & & > 35\% \text{ nhưng Win Rate sụt} & \text{Tự động giảm hệ số } \kappa \text{ theo công thức: } \\
& \text{phá thay vì tối ưu hóa kinh tế.} & & & \text{giảm } < 35\% \text{ (liều lĩnh quá mức).} & \kappa \leftarrow \kappa \times \max(0, \text{WinRate} - 0.40) \text{; nếu WR } < 40\%, \\
& & & & & \text{tắt hoàn toàn RND (ép } \kappa = 0\text{).} \\
\hline
\textbf{RSK-5} & \textbf{ArcFace Margin quá lớn } (m=0,50) & \text{Thấp} & \text{Cao} & \text{Gradient sụp đổ (NaN/Inf)} & \textbf{Fallback về CosFace hoặc VICReg thuần túy:} \\
& \text{làm mất hội tụ trong bài toán} & & & \text{hoặc loss không hội tụ sau} & \text{Hạ biên góc xuống } m = 0,20\text{ radian, hoặc chuyển} \\
& \text{phân loại đa nhãn trading.} & & & 2\text{ epochs đầu tiên.} & \text{sang Large Margin Cosine Loss (CosFace) với} \\
& & & & & \text{biên phụ gia tuyến tính } \cos \theta - m\text{ ổn định hơn.} \\
\hline
\textbf{RSK-6} & \textbf{Nghẽn I/O File Lock hoặc IPC Pipe} & \text{Cao} & \text{Trung bình} & \text{Xuất hiện lỗi PermissionError,} & \textbf{Chuyển đổi sang Memory-Mapped Files / SQLite:} \\
& \text{khi chạy Multi-Round tự động} & & & \text{timeout Named Pipe, hoặc} & \text{Thay thế đọc ghi file JSONL bằng SQLite WAL mode} \\
& \text{trên nhiều MT5 terminals.} & & & \text{tiến trình MT5 bị treo > 5 phút.} & \text{hoặc Shared Memory (mmap); bổ sung cơ chế} \\
& & & & & \text{atomic file rename và khóa tệp } \text{FileLock} \text{ có timeout.} \\
\hline
\end{array}$$

---

### 5.2. Cơ chế Kích hoạt & Quy trình Xử lý Sự cố Tự động

Để bảo đảm tính tự động hóa và an toàn tuyệt đối trong quá trình vận hành, hệ thống áp dụng **Quy trình 3 Bước Phản ứng Sự cố (3-Step Contingency Protocol)**:

```
[GIÁM SÁT METRICS LIÊN TỤC] ──(Vượt ngưỡng Trigger)──> [TỰ ĐỘNG BẬT FALLBACK MODE]
                                                              │
                                                              ▼
[BÁO CÁO VI PHẠM & ROLLBACK CHECKPOINT] <──(Không hội tụ)─────┤
                                                              ▼
                                                [TIẾP TỤC HUẤN LUYỆN AN TOÀN]
```

1. **Bước 1 — Phát hiện & Cách ly (Detect & Isolate):**
   - Bộ giám sát `auto_rl_engine.py` kiểm tra các chỉ số sau mỗi vòng chạy. Nếu bất kỳ chỉ số nào chạm ngưỡng kích hoạt Trigger trong bảng trên, cờ `FALLBACK_TRIGGERED` được bật.
2. **Bước 2 — Kích hoạt Chế độ Dự phòng (Switch to Safe Fallback):**
   - Hệ thống tự động chuyển sang cấu hình an toàn tương ứng (ví dụ: Chuyển từ Mamba-2 sang xLSTM, hoặc hạ $\kappa_{\text{RND}} = 0$). Toàn bộ sự kiện được ghi vết chi tiết vào `audit_contingency.jsonl`.
3. **Bước 3 — Đối soát & Tự động Phục hồi (Audit & Rollback):**
   - Nếu sau 2 vòng áp dụng chế độ dự phòng mà hiệu suất kinh tế vẫn không phục hồi ($E[R] < 0$), hệ thống tự động rollback toàn bộ tham số về checkpoint gần nhất đạt chuẩn Gate, đồng thời gửi thông báo khẩn cấp đến kênh giám sát.

---

## 6. KẾT LUẬN & CHỈ DẪN TRIỂN KHAI

Lộ trình phát triển **Self Growing Trader V193** được hoạch định một cách khoa học, thực tế và khả thi, giải quyết tận gốc các nút thắt kỹ thuật đã được phơi bày trong các báo cáo khảo sát thực nghiệm. 

Bằng việc phân tách rành mạch thành 3 giai đoạn:
- **Ngắn hạn (Tier 1):** Cứu vãn không gian ẩn bằng Regularizer Suite và triệt tiêu thiên lệch xu hướng bằng Balanced Reservoir Buffer (0 tham số mới).
- **Trung hạn (Tier 2):** Bước nhảy vọt hiệu năng với kiến trúc SGT-V193-Compact 17.5M, tăng tốc CPU 5.2 lần và bóc tách tín dụng đa tác nhân COMA.
- **Dài hạn (Tier 3):** Khai thác Mamba-2 SSD, vi cấu trúc thị trường và tự tiến hóa Meta-RL đa vòng.

Hệ thống sẽ chuyển mình từ một mô hình 93M tham số cồng kềnh, sụp đổ biểu diễn thành một **Cỗ máy Giao dịch Định lượng Tinh gọn, Siêu nhanh, Có khả năng Thích nghi Tự trị Cao**, bảo toàn nghiêm ngặt các bất biến kinh tế và mang lại lợi thế cạnh tranh vượt trội trên thị trường tài chính quốc tế.
