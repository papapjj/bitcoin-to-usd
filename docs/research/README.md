# NGHIÊN CỨU TOÀN DIỆN KIẾN TRÚC & LỘ TRÌNH PHÁT TRIỂN MODEL V193
## BỘ TÀI LIỆU NGHIÊN CỨU KHOA HỌC, THIẾT KẾ KỸ THUẬT & LỘ TRÌNH TRIỂN KHAI

---

## 1. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Hệ thống **Self Growing Trader V193** là nền tảng giao dịch định lượng đa tác nhân (Multi-Agent RL) mô phỏng theo kiến trúc OpenAI Five (2019), vận hành 5 tác nhân chuyên biệt dùng chung một mạng nơ-ron sâu ~93.2 triệu tham số trên nền tảng MetaTrader 5 (MT5).

Báo cáo kiểm tra sức chứa thực nghiệm (`docs/FIVE_ARCHITECTURE_CAPACITY_AUDIT_20260906.vi.md`) và chuỗi trường hợp kiểm thử thực tế (`ORIGINAL_REQUEST.md`, Run 820001 – 820007) đã chỉ ra rằng: Mặc dù hệ thống sở hữu hợp đồng thực thi rất chặt chẽ (137 input features, 20 output actions), **mô hình đang bị bóp nghẹt nghiêm trọng bởi các nút thắt kiến trúc và động lực học huấn luyện**:
- **Nghịch lý dư thừa tham số nhưng sụp đổ biểu diễn (Overparameterization Paradox):** Khối `nn.LSTMCell(1024, 4096)` chiếm tới **90.06% tham số (83.92M)** và ngốn **57.37% thời gian chạy CPU (101.5 ms)**, nhưng phân tích SVD/PCA cho thấy không gian ẩn 4096D sụp xuống chỉ còn **effective rank $r_{eff} \approx 2.116$**, 90% phương sai nằm ở 2 trục chính, và trí nhớ hồi quy bị xóa sạch sau 8–16 nến M1.
- **Suy thoái cơ chế Attention:** Khối Transformer hậu-LSTM có entropy attention đạt **0.9902** (phẳng đều tuyệt đối), không phân biệt được tác nhân và làm giảm nhẹ chỉ số phân tách tuyến tính OOS.
- **Thiên lệch xu hướng và quên thảm họa (Catastrophic Forgetting):** Do mất cân bằng mẫu nạp vào PPO khi gặp chuỗi Downtrend dài, tác nhân Short Owner đã bán ngược tới 85–90% trong pha Uptrend (Run 820003), gây tổn thất PnL nặng nề. Đồng thời, các vị thế trong vùng tích lũy bị đóng cưỡng chế do hết hạn 120 bars (`CloseExpiredPositions` > 35%).

Để giải quyết tận gốc các vấn đề trên mà **tuyệt đối không sử dụng bất kỳ cờ cưỡng chế cứng nào** (`context_direction_entry_gate_enabled = False`), bộ tài liệu nghiên cứu chuyên sâu V193 được thiết lập gồm **3 trụ cột tài liệu độc lập và bổ trợ lẫn nhau**:

```
                       BỘ TÀI LIỆU NGHIÊN CỨU V193
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
 [TÀI LIỆU 1: RESEARCH]     [TÀI LIỆU 2: DESIGN DOCS]     [TÀI LIỆU 3: ROADMAP]
 V193_MODEL_RESEARCH_       V193_TOP3_TECHNICAL_         V193_DEVELOPMENT_
 AND_COMPARISON.vi.md       DESIGN_DOCS.vi.md            ROADMAP.vi.md
 • 7 Bottlenecks Codebase   • Top 1: Regularizer Suite   • Triết lý & Bất biến hệ thống
 • 4 SOTA Backbones         • Top 2: SGT-Compact (17.5M) • Lộ trình 3 Tầng (1-1-3 m)
 • Ma trận So sánh 8 PA     • Top 3: Balanced Buffer     • Critical Path (Đường găng)
 • Động học PPO & MARL        & COMA Credit Assignment   • Ma trận Rủi ro & Fallback
```

---

## 2. BẢN ĐỒ TÀI LIỆU (DOCUMENT MAP & DIRECTORY TREE)

### 2.1. Cấu trúc Thư mục Tài liệu Nghiên cứu
```
docs/
├── research/                                           # THƯ MỤC NGHIÊN CỨU CHUYÊN SÂU V193
│   ├── README.md                                       # [BẠN ĐANG XEM] Tổng quan, Bản đồ & Hướng dẫn tra cứu
│   ├── V193_MODEL_RESEARCH_AND_COMPARISON.vi.md        # Deliverable 1 & 2: Báo cáo Nghiên cứu & Ma trận So sánh
│   ├── V193_TOP3_TECHNICAL_DESIGN_DOCS.vi.md           # Deliverable 3: Bộ 3 Thiết kế Kỹ thuật Chi tiết
│   └── V193_DEVELOPMENT_ROADMAP.vi.md                  # Deliverable 4: Lộ trình Phát triển & Phân tích Đường găng
│
├── FIVE_ARCHITECTURE_CAPACITY_AUDIT_20260906.vi.md     # Báo cáo thẩm định sức chứa thực nghiệm gốc (749 dòng)
├── EDGE_DISCOVERY_V193.vi.md                           # Khám phá lợi thế định lượng và hành vi V193
├── LIFECYCLE_SETTLEMENT_CANONICAL_V1.vi.md             # Đặc tả chuẩn hóa quyết toán vòng đời vị thế
├── FIVE_REASONING_RL_V1.vi.md                          # Thiết kế cơ chế suy luận tăng cường GRPO
└── kien_truc_nam_tac_nhan_openai.md                    # Khung kiến trúc 5 tác nhân OpenAI Five
```

### 2.2. Chi tiết & Trách nhiệm Từng Tài liệu

| Tên File | Vai trò / Deliverable | Nội dung Cốt lõi | Tác giả Chịu trách nhiệm |
|:---|:---|:---|:---|
| **`README.md`** | **Tổng Hợp & Định Hướng** | Tóm lược điều hành, bản đồ liên kết tài liệu, ma trận giải pháp, chỉ dẫn điều hướng đa vai trò. | Worker 3 (`worker_v193_3`) |
| **`V193_MODEL_RESEARCH_AND_COMPARISON.vi.md`** | **Deliverables 1 & 2** | Phân tích 7 điểm nghẽn codebase/audit; khảo sát lý thuyết SOTA (xLSTM, Mamba-2, RWKV-6, RetNet, RGAT); ma trận so sánh 8 phương án kỹ thuật toàn diện. | Worker 1 (`worker_v193_1`) |
| **`V193_TOP3_TECHNICAL_DESIGN_DOCS.vi.md`** | **Deliverable 3** | Đặc tả kỹ thuật mức sản xuất cho 3 phương án ưu tiên hàng đầu: (1) Representation Regularizer Suite; (2) SGT-V193-Compact 17.5M Policy; (3) Dynamic Multi-Regime Reservoir Buffer & COMA Credit Assignment. Đầy đủ code PyTorch native, tensor shapes, và migration path. | Worker 2 (`worker_v193_2`) |
| **`V193_DEVELOPMENT_ROADMAP.vi.md`** | **Deliverable 4** | Triết lý & 7 bất biến hệ thống; lộ trình 3 tầng chi tiết (Tier 1 Ngắn hạn 1–2 tuần, Tier 2 Trung hạn 1–2 tháng, Tier 3 Dài hạn 3–6 tháng); phân tích đường găng Critical Path; bảng tiêu chí chấp thuận tổng hợp; ma trận quản trị 6 rủi ro lớn & fallback. | Worker 3 (`worker_v193_3`) |

---

## 3. HƯỚNG DẪN TRA CỨU NHANH THEO VAI TRÒ (ROLE-BASED NAVIGATION GUIDE)

Tùy theo mục tiêu công việc, bạn có thể điều hướng trực tiếp đến các phần tương ứng:

```
                  BẠN LÀ AI TRONG DỰ ÁN?
                            │
      ┌─────────────────────┼─────────────────────┐
      ▼                     ▼                     ▼
[QUẢN LÝ DỰ ÁN / LEAD]   [KỸ SƯ ML / RL RESEARCH]  [QUANT TRADER / MT5 ENG]
Tra cứu:                 Tra cứu:                  Tra cứu:
• Lộ trình 3 Tầng        • Phân tích 7 Bottlenecks • Hợp đồng 137 / 20 Wire
• Đường găng (CP)        • Kiến trúc SGT-Compact   • Lifecycle Settlement
• Ma trận Quản trị       • Code PyTorch Design Docs• Chỉ số WR, PR, E[R]
  Rủi ro & Fallback      • VICReg, COMA, xLSTM     • Chống đóng timeout 120
```

### 3.1. Dành cho Quản lý Dự án & Kiến trúc sư Trưởng (Project Lead / System Architect)
- **Nhu cầu:** Nắm bắt tiến độ, kế hoạch phân bổ nguồn lực, điểm nghẽn dự án, và các kịch bản dự phòng khi phát sinh sự cố.
- **Tài liệu cần đọc:**
  1. `V193_DEVELOPMENT_ROADMAP.vi.md` — Mục 2: Lộ trình 3 Tầng Chi tiết (Mục tiêu, Thời gian, Nghiệm thu từng mốc).
  2. `V193_DEVELOPMENT_ROADMAP.vi.md` — Mục 3: Ma trận Phụ thuộc & Sơ đồ Đường găng (Critical Path Analysis).
  3. `V193_DEVELOPMENT_ROADMAP.vi.md` — Mục 5: Ma trận Quản trị Rủi ro & Kế hoạch Dự phòng (6 rủi ro kỹ thuật và trigger fallback).
  4. `V193_MODEL_RESEARCH_AND_COMPARISON.vi.md` — Phần II: Bảng So sánh Ma trận 8 Phương án Kỹ thuật.

### 3.2. Dành cho Kỹ sư Học Máy & Nghiên cứu RL (Machine Learning / RL Engineer)
- **Nhu cầu:** Nắm vững công thức toán học, cơ chế nơ-ron, giải pháp chống sụp đổ biểu diễn, và code mẫu PyTorch native để triển khai vào codebase.
- **Tài liệu cần đọc:**
  1. `V193_TOP3_TECHNICAL_DESIGN_DOCS.vi.md` — Design Doc 1: Representation Regularizer Suite (VICReg + RankMe + ArcFace loss).
  2. `V193_TOP3_TECHNICAL_DESIGN_DOCS.vi.md` — Design Doc 2: SGT-V193-Compact Architecture (Code `MatrixLSTMCell` và `RelationalCrossAgentTransformer`).
  3. `V193_TOP3_TECHNICAL_DESIGN_DOCS.vi.md` — Design Doc 3: Dynamic Balanced Reservoir Buffer & COMA Advantage.
  4. `V193_MODEL_RESEARCH_AND_COMPARISON.vi.md` — Phần I, Mục 2: Bằng chứng thực nghiệm 7 Bottlenecks từ Capacity Audit.

### 3.3. Dành cho Chuyên gia Giao dịch Định lượng & Lập trình MT5 (Quant Trader / MT5 Developer)
- **Nhu cầu:** Bảo đảm an toàn thực thi, kiểm soát rủi ro danh mục, tuân thủ giao thức cầu nối, và đánh giá hiệu quả kinh tế ($E[R]$, Win Rate, Drawdown).
- **Tài liệu cần đọc:**
  1. `V193_DEVELOPMENT_ROADMAP.vi.md` — Mục 1.2: Ràng buộc Bất biến Hệ thống Không Thể Thương Lượng (INV-1 đến INV-7).
  2. `V193_DEVELOPMENT_ROADMAP.vi.md` — Mục 4: Bảng Tiêu chí Chấp thuận Tổng hợp Qua Các Giai đoạn.
  3. `docs/LIFECYCLE_SETTLEMENT_CANONICAL_V1.vi.md`: Quy tắc quyết toán R-multiple và thưởng/phạt MAE/MFE.
  4. `docs/ORIGINAL_REQUEST.md`: Phân tích các trường hợp thất bại 820001 – 820007 để hiểu rõ hiện tượng Short ngược xu hướng.

---

## 4. TỔNG HỢP NÚT THẮT & MA TRẬN GIẢI PHÁP (PROBLEM-SOLUTION MATRIX)

Bảng đối chiếu trực tiếp giữa các khuyết tật được phát hiện trong codebase/audit và các giải pháp kỹ thuật tương ứng được đề xuất trong bộ tài liệu:

| Điểm Nghẽn Hiện Tại (Bottleneck) | Bằng Chứng Thực Nghiệm | Giải Pháp Đề Xuất | Vị Trí Trong Bộ Tài Liệu | Kết Quả Định Lượng Kỳ Vọng |
|:---|:---|:---|:---|:---|
| **1. Hidden Space Collapse** | Rank 90% = 2, Rank 99% = 11; $r_{eff} = 2,116$ trên 4096 chiều. | **VICReg + RankMe Regularizers** | `DESIGN_DOCS.vi.md: Doc 1`<br>`ROADMAP.vi.md: M1.1` | $r_{eff}$ tăng từ $2,1$ lên $\mathbf{\ge 20,0}$ (Tier 1) và $\mathbf{\ge 100,0}$ (Tier 2). |
| **2. Architectural Imbalance** | LSTMCell chiếm 90.06% tham số (83.9M), forward 101.5 ms. | **SGT-V193-Compact (xLSTM 768D)** | `DESIGN_DOCS.vi.md: Doc 2`<br>`ROADMAP.vi.md: M2.1` | Giảm tham số xuống **17.5M (-81.2%)**, CPU latency giảm xuống **34 ms (5.2x)**. |
| **3. Attention Degeneration** | Attention entropy 0.9902 (phẳng đều), đường chéo 0.2057 $\approx 1/5$. | **Relational Cross-Agent Graph Transformer (RGAT)** | `DESIGN_DOCS.vi.md: Doc 2`<br>`RESEARCH.vi.md: Sec 4` | Attention entropy giảm về **0.45 – 0.65**, bóc tách rõ vai trò tác nhân. |
| **4. Action Query Bottleneck** | Linear 128D làm rớt AUC từ 0.748 xuống 0.635; Separation 0.18–0.33. | **ArcFace Cosine Margin ($m=0.35, s=30$)** | `DESIGN_DOCS.vi.md: Doc 1`<br>`ROADMAP.vi.md: M1.1` | Phân tách ENTRY vs HOLD tăng từ 0.18 lên **$\ge 1.50$ pooled radius**. |
| **5. Catastrophic Forgetting** | 85–90% Short trong Uptrend sau chuỗi Downtrend (Run 820003). | **Dynamic Balanced Reservoir Buffer (3x128)** | `DESIGN_DOCS.vi.md: Doc 3`<br>`ROADMAP.vi.md: M1.2` | Tỷ lệ Short trong Uptrend giảm từ 90% xuống **$\le 30\%$** (Tier 1) và **$\le 15\%$** (Tier 2). |
| **6. Lazy Agent Problem** | Team-spirit tăng lên 1.0 gây loãng tín hiệu phân bổ tín dụng. | **COMA Counterfactual Baseline & Cosine Decay** | `DESIGN_DOCS.vi.md: Doc 3`<br>`ROADMAP.vi.md: M1.3 & M2.3` | Bóc tách chính xác công/tội từng tác nhân; loại bỏ hoàn toàn bẫy ỷ lại. |
| **7. 120-Bar Timeout Penalty** | Đóng lệnh cưỡng chế do hết hạn nến làm sụt giảm Payoff Ratio. | **Extended Settlement Reward & HER Relabeling** | `DESIGN_DOCS.vi.md: Doc 3`<br>`ROADMAP.vi.md: M1.2 & M2.3` | Tỷ lệ đóng lệnh timeout 120 nến giảm từ $>35\%$ xuống **$< 8\%$**, Payoff **$\ge 1.80R$**. |

---

## 5. TÓM TẮT TIÊU CHÍ CHẤP THUẬN TỔNG HỢP (ACCEPTANCE GATES SUMMARY)

Dự án triển khai theo 3 Cổng Đánh giá Chất lượng Nghiêm ngặt (Quality Gates):

```
       [TIER 1 GATE: STABILIZATION]                [TIER 2 GATE: EFFICIENCY]                [TIER 3 GATE: SOTA EVOLUTION]
       Thời hạn: 1–2 tuần                          Thời hạn: 1–2 tháng                      Thời hạn: 3–6 tháng
       ────────────────────────────                ─────────────────────────                ─────────────────────────────
       • r_eff >= 20.0 (tăng 10x)                  • Params: 17.5M (-81.2%)                 • Context Horizon >= 512 bars
       • Class Separation >= 0.80                  • CPU Latency <= 38 ms                   • C++ DLL Latency <= 25 ms
       • Short in Uptrend <= 30%                   • r_eff >= 100.0                         • Zero-shot Regime Adaptation
       • Timeout 120 bars <= 15%                   • Win Rate >= 50%                        • Continuous Auto Loop >= 30 rds
       • E[R] >= +0.05R, WR >= 45%                 • Payoff >= 1.80R, Max DD < 8%           • Payoff >= 2.00R, PF >= 2.20
       • Zero new inference params                 • Student Match Rate >= 92%              • Sharpe Ratio OOS >= 1.5
```

---

## 6. THÔNG TIN KHỞI TẠO & DUY TRÌ TÀI LIỆU

- **Đơn vị Thực hiện:** Nhóm Tác nhân Phân tán Self Growing Trader V193 (Teamwork Preview Swarm).
- **Điều phối viên Chính (Orchestrator):** `orchestrator_10` (`14a2d452-0405-4c51-98c1-eb2118c9dca7`).
- **Tác giả Tài liệu Thành phần:**
  - Báo cáo Nghiên cứu & Ma trận So sánh: Worker 1 (`worker_v193_1`).
  - Bộ 3 Tài liệu Thiết kế Kỹ thuật: Worker 2 (`worker_v193_2`).
  - Lộ trình Phát triển & README Tổng hợp: Worker 3 (`worker_v193_3`).
- **Thẩm định Độc lập:** Forensic Auditor (`teamwork_preview_auditor`).
- **Ngày phát hành phiên bản:** 2026-10-19 / 2026-10-20.
- **Trạng thái:** Sẵn sàng nghiệm thu và chuyển giao cho giai đoạn lập trình thực thi.
