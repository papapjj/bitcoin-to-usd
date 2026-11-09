# CẨM NANG HƯỚNG DẪN TRIỂN KHAI THỰC TẾ TRÊN METATRADER 5 (MT5)
## HỆ THỐNG GIAO DỊCH TỰ ĐỘNG SELF GROWING TRADER V193 — CHAMPION PHA 2

> **Phiên bản tài liệu:** 1.0 (Production Release)  
> **Mã định danh mô hình:** Champion Pha 2 V193 (`best_promoted_checkpoint_phase2.pkl` / `champion_phase2_round_06.pkl`)  
> **Mã băm toàn vẹn SHA256:** `F8D214C6998B0B8C259C0C82B7BD94B175719312B506FFD3DE40C286E0013079`  
> **Kích thước file:** 1,273,950,613 bytes (~1.27 GB)  
> **Ngôn ngữ:** Tiếng Việt (Vietnamese)

---

## MỤC LỤC
1. [Tổng Quan Kiến Trúc & Mô Hình 5-Agent](#1-tổng-quan-kiến-trúc--mô-hình-5-agent)
2. [Yêu Cầu Môi Trường Phần Cứng & Hệ Thống](#2-yêu-cầu-môi-trường-phần-cứng--hệ-thống)
3. [Cấu Trúc Thư Mục & Chuẩn Bị Artifacts](#3-cấu-trúc-thư-mục--chuẩn-bị-artifacts)
4. [Hướng Dẫn Biên Dịch Expert Advisor (EA)](#4-hướng-dẫn-biên-dịch-expert-advisor-ea)
5. [Thiết Lập Tùy Chọn Terminal MT5](#5-thiết-lập-tùy-chọn-terminal-mt5)
6. [Thiết Lập Biểu Đồ & Cấu Hình Thông Số Đầu Vào (.set)](#6-thiết-lập-biểu-đồ--cấu-hình-thông-số-đầu-vào-set)
7. [Khởi Chạy Runtime Python Bridge Inference](#7-khởi-chạy-runtime-python-bridge-inference)
8. [Quy Trình Giám Sát Telemetry Thời Gian Thực](#8-quy-trình-giám-sát-telemetry-thời-gian-thực)
9. [Nguyên Tắc Quản Trị Rủi Ro Sản Xuất](#9-nguyên-tắc-quản-trị-rủi-ro-sản-xuất)
10. [Giải Đáp Sự Cố Thường Gặp (FAQ)](#10-giải-đáp-sự-cố-thường-gặp-faq)

---

## 1. Tổng Quan Kiến Trúc & Mô Hình 5-Agent

Self Growing Trader V193 là hệ thống giao dịch thuật toán áp dụng Học tăng cường sâu (Deep Reinforcement Learning - RL) dựa trên kiến trúc OpenAI Five đa tác nhân dùng chung mạng neural (Shared Policy Network).

### 1.1. Thông số Kỹ thuật Cốt lõi
- **Tổng số tham số:** `117,350,081` tham số (117,062,010 tham số có thể huấn luyện; 288,071 tham số đóng băng làm tham chiếu KL divergence trong Reasoning RL).
- **Phân bổ phân lớp mạng:**
  - `FiveObservationEncoder`: 843,278 tham số — ánh xạ 137 đặc trưng đầu vào thành không gian ẩn 1024 chiều.
  - `LSTM Core`: 83,918,848 tham số — `torch.nn.LSTMCell` (input=1024, hidden=4096), duy trì trạng thái ngữ cảnh thị trường liên tục.
  - `PostLstmCrossAgentTransformer`: 12,610,050 tham số — cơ chế self-attention đa tác nhân và mã hóa quan hệ thời gian đa khung độ (2, 8, 32, 128 nến).
  - `FactorizedActionHeads`: 19,401,763 tham số — tính toán logits hành động rời rạc, tham số hình học vào lệnh và giá trị phê bình (Critic value heads).
  - `FiveReasoningHead` & `FiveReasoningReferenceHead`: 288,071 tham số mỗi nhánh.
- **Tiêu chuẩn giao thức:**
  - Bridge Schema Version: **10**
  - Execution Contract: **v7**
  - Execution Ledger: **v17**

### 1.2. Phân Quyền 5 Tác Nhân (Five-Agent Authority)
Mô hình điều phối 5 tác nhân chuyên biệt, đảm bảo tính phân tách trách nhiệm tuyệt đối:
1. **Agent 0 (Long Entry Owner):** Quản lý lệnh BUY. Quyền hạn: `HOLD (0)`, `MARKET_BUY (1)`, `BUY_LIMIT (3)`, `BUY_STOP (5)`.
2. **Agent 1 (Short Entry Owner):** Quản lý lệnh SELL. Quyền hạn: `HOLD (0)`, `MARKET_SELL (2)`, `SELL_LIMIT (4)`, `SELL_STOP (6)`.
3. **Agent 2 (Position Manager):** Quản lý vị thế mở. Được **khóa cứng ở HOLD (0)** trong cấu hình sản xuất để triệt tiêu việc can thiệp đóng lệnh sớm.
4. **Agent 3 (Pending Manager):** Quản lý lệnh chờ (`CANCEL_ORDERS`). Được **khóa cứng ở HOLD (0)**.
5. **Agent 4 (Portfolio Governor):** Điều hành danh mục. Được **khóa cứng ở HOLD (0)** (`POLICY_CLOSE_ALL == 0`), đảm bảo 100% vị thế mở tuân thủ chốt lời/cắt lỗ theo hình học cố định.

### 1.3. Cấu Trúc Vector Đặc Trưng Đầu Vào (137 chiều) & Đầu Ra (20 chiều)
- **137 Inputs (Trích xuất tại M1 bar boundary):**
  - Chiều 0–28 (29 dims): Đặc trưng nến, động lượng đa khung, EMA spread, ATR percentile, RSI, MACD, Bollinger Bands.
  - Chiều 29–68 (40 dims): 5 vị thế mở gần nhất × 8 thuộc tính (trạng thái, chiều, khoảng cách ATR di chuyển, tuổi nến, SL/TP dist, PnL R).
  - Chiều 69–76 (8 dims): Đặc trưng tài khoản (số vị thế, rủi ro tích lũy, drawdown hiện tại, tiến độ giao dịch trong ngày).
  - Chiều 77–105 (29 dims): Đặc trưng đa khung thời gian MTF (M5, M15, H1 EMA & ATR), Choppiness Index 60 nến, Range compression, Wyckoff breakout.
  - Chiều 106–116 (11 dims): Spread/ATR hiện thời và cự ly SL/TP chuẩn hóa theo ATR.
  - Chiều 117–120 (4 dims): Số nến trôi qua từ đỉnh/đáy 20 nến và 60 nến.
  - Chiều 121–136 (16 dims): Chỉ báo khối lượng nâng cao và dấu chân tích lũy/phân phối Wyckoff (Volume Spring/Upthrust score, Rolling volume z-scores).
- **20 Outputs (Mã hóa hành động & Hình học):**
  - 0–11: Lựa chọn hành động (`HOLD`, `MARKET_BUY`, `MARKET_SELL`, `BUY_LIMIT`, `SELL_LIMIT`, `BUY_STOP`, `SELL_STOP`, `CANCEL_ORDERS`, `CLOSE_WORST`, `CLOSE_BEST`, `CLOSE_ALL`, `MANAGE_POSITION`).
  - 12: Tỷ lệ rủi ro vốn (`RiskPct`).
  - 13: Độ trễ thực thi theo tick (`ActionDelay`).
  - 14: Factorized action marker (`>= 4.5` kích hoạt hình học nhân tố hóa).
  - 15: Cự ly đặt lệnh chờ (`EntryOffsetAtr`).
  - 16: Discrete Geometry packed index (trong cấu hình cố định được ghi đè bởi SL/TP cố định).
  - 17–19: Tỷ lệ chốt từng phần, chế độ quản lý và chỉ số vị thế mục tiêu.

---

## 2. Yêu Cầu Môi Trường Phần Cứng & Hệ Thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị sản xuất |
| :--- | :--- | :--- |
| **Hệ điều hành** | Windows 10 64-bit hoặc Windows Server 2019 | Windows 11 64-bit hoặc Windows Server 2022 |
| **CPU** | 4 Cores (Intel Core i5 gen 8+ / AMD Ryzen 5) | 8+ Cores (Intel Core i7/i9 hoặc AMD Ryzen 7/9) |
| **RAM** | 16 GB | 32 GB RAM |
| **Ổ đĩa cứng** | SSD còn trống tối thiểu 20 GB | NVMe SSD PCIe Gen 3/4 còn trống $\ge 50$ GB |
| **Nền tảng giao dịch** | MetaTrader 5 Build 4000+ (64-bit), Portable Mode | MetaTrader 5 Build 4400+ (64-bit) Portable Mode |
| **Môi trường Python** | Python 3.10 – 3.14 (64-bit) | Python 3.11 hoặc 3.12 (64-bit) |
| **Thư viện Deep Learning** | PyTorch 2.2.0+ (CPU hoặc CUDA) | PyTorch 2.4.0+ CPU (Inference CPU chỉ ~177ms) |
| **Tài khoản sàn** | Demo hoặc Real Hedge Account (Đòn bẩy $\ge 1:100$) | Tài khoản Raw/ECN Spread thấp, Đòn bẩy 1:200 (`BTCUSDm`) |

---

## 3. Cấu Trúc Thư Mục & Chuẩn Bị Artifacts

Cấu trúc thư mục chuẩn trên máy chủ giao dịch:
```text
c:\Users\Admin\Desktop\new-chat\
├── mt5_v2_runner\                          # Thư mục MT5 Portable
│   ├── terminal64.exe                     # File thực thi MT5 Terminal
│   ├── metaeditor64.exe                   # Trình biên dịch MQL5
│   ├── MQL5\
│   │   ├── Experts\
│   │   │   ├── SelfGrowing_MT5_Trainer_EA.mq5   # Mã nguồn EA
│   │   │   └── SelfGrowing_MT5_Trainer_EA.ex5   # File thực thi EA sau biên dịch
│   │   ├── Include\
│   │   │   ├── compute_features.mqh             # Thư viện tính toán 137 đặc trưng
│   │   │   └── self_growing_execution_contract.mqh # Thư viện contract v7
│   │   └── Files\                              # Thư mục IPC trao đổi dữ liệu với Python
│   └── Common\
│       └── Files\
├── outputs\he_chuyen_gia\v193_runs\
│   ├── best_promoted_checkpoint_phase2.pkl      # Checkpoint Champion Pha 2 chính thức
│   └── production_registry\                    # Hồ sơ đăng ký sản xuất
│       ├── registry_phase2_champion.json
│       ├── README.md
│       └── phase2_analysis_report.md
├── configs\
│   ├── he_chuyen_gia_reasoning_rl_v1_candidate.yaml
│   └── experiments\reasoning_rl_v1_active_candidate.yaml
└── scripts\
    ├── preflight_mt5.py
    ├── verify_mt5_contract_static.py
    ├── verify_production_registry_phase2.py
    └── run_tensor_rule_v186_candidate.py
```

### Kiểm Tra Mã Băm Checkpoint Trước Khi Chạy
Mở PowerShell tại thư mục gốc và chạy:
```powershell
python -c "
import hashlib
path = 'outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint_phase2.pkl'
h = hashlib.sha256()
with open(path, 'rb') as f:
    while c := f.read(16*1024*1024):
        h.update(c)
digest = h.hexdigest().upper()
print('Checksum:', digest)
assert digest == 'F8D214C6998B0B8C259C0C82B7BD94B175719312B506FFD3DE40C286E0013079', 'HASH MISMATCH!'
print('CHECKPOINT XÁC THỰC THÀNH CÔNG 100% BIT-FOR-BIT.')
"
```

---

## 4. Hướng Dẫn Biên Dịch Expert Advisor (EA)

File EA mã nguồn tại: `mt5_v2_runner/MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5`.

### Cách 1: Biên dịch tự động qua dòng lệnh (CLI - Khuyến nghị)
Mở PowerShell và chạy lệnh:
```powershell
cd c:\Users\Admin\Desktop\new-chat\mt5_v2_runner
.\metaeditor64.exe /compile:MQL5\Experts\SelfGrowing_MT5_Trainer_EA.mq5 /log:compile.log
Get-Content compile.log -Tail 10
```
*Kết quả hợp lệ:* Dòng cuối file `compile.log` ghi nhận: `0 errors, 0 warnings`. File `SelfGrowing_MT5_Trainer_EA.ex5` được sinh ra tại `MQL5/Experts/`.

### Cách 2: Biên dịch qua giao diện MetaEditor GUI
1. Mở MT5, nhấn phím `F4` để mở MetaEditor.
2. Tại cây thư mục Navigator bên trái, mở `Experts` -> Nhấp đúp vào `SelfGrowing_MT5_Trainer_EA.mq5`.
3. Nhấn phím `F7` (hoặc nút **Compile** trên thanh công cụ).
4. Quan sát cửa sổ **Toolbox - Errors**: Xác nhận `0 errors, 0 warnings`.

---

## 5. Thiết Lập Tùy Chọn Terminal MT5

Khởi chạy MetaTrader 5 ở chế độ **Portable**:
```powershell
.\mt5_v2_runner\terminal64.exe /portable
```

Vào menu: **Tools** -> **Options** (hoặc nhấn `Ctrl + O`):

1. **Tab "Expert Advisors" (Cực kỳ quan trọng):**
   - Đánh dấu chọn: `[x] Allow Algo Trading` (Cho phép giao dịch tự động).
   - Bỏ chọn: `[ ] Allow DLL imports` (Hệ thống tuyệt đối **không dùng DLL**, đảm bảo bảo mật và tính ổn định).
   - Đánh dấu chọn: `[x] Allow WebRequest for listed URL` (Cho phép các endpoint nội bộ nếu cần thu thập telemetry).
2. **Tab "Charts":**
   - Thiết lập `Max bars in chart`: `100000` (đảm bảo đủ lịch sử nến cho các chỉ báo 60 nến và ATR 240 nến).
3. **Tab "Trade":**
   - Kiểm tra `Deviation`: Chọn `Default` (hoặc theo quy định sàn).
4. Nhấn **OK** để lưu cài đặt.

---

## 6. Thiết Lập Biểu Đồ & Cấu Hình Thông Số Đầu Vào (.set)

### 6.1. Chuẩn Bị Biểu Đồ
1. Trong cửa sổ **Market Watch**, tìm mã giao dịch: `BTCUSDm` (hoặc mã Bitcoin tương ứng của sàn như `BTCUSD`, `BTCUSDT`).
2. Nhấp chuột phải -> chọn **Chart Window**.
3. Chuyển khung thời gian biểu đồ sang: **M1** (1 Phút).

### 6.2. Gắn EA và Cài Đặt Input Parameters
Kéo `SelfGrowing_MT5_Trainer_EA` từ Navigator vào biểu đồ `BTCUSDm, M1`. Trong hộp thoại cài đặt:

- **Tab "Common":**
  - Đánh dấu: `[x] Allow Algo Trading`.
- **Tab "Inputs":** Thiết lập chính xác các tham số chuẩn sản xuất sau:

| Tên tham số EA | Giá trị thiết lập | Ý nghĩa vận hành |
| :--- | :---: | :--- |
| `BridgeFile` | `self_growing_bridge.csv` | Tệp đồng bộ trạng thái cầu nối |
| `RequestFile` | `self_growing_request.csv` | Tệp ghi nhận 137 đặc trưng M1 từ MT5 |
| `ActionFile` | `self_growing_action.csv` | Tệp nhận chỉ thị hành động từ Python |
| `ExecutionFile` | `self_growing_execution.csv` | Tệp sổ cái ghi nhận khớp lệnh thực tế |
| `FeatureMtfScalingMode` | `legacy_m1_atr_v1` | Chuẩn co giãn đặc trưng đa khung thời gian |
| `FixedSlAtr` | `5.0` | Stop Loss cố định = 5.0 × ATR(14) |
| `FixedTpAtr` | `10.0` | Take Profit cố định = 10.0 × ATR(14) ($RR = 2.0$) |
| `MaxHoldingBars` | `120` | Đóng lệnh bảo hiểm sau 120 nến M1 (2 giờ) |
| `PendingExpiryBars` | `30` | Hủy lệnh chờ nếu không khớp sau 30 nến M1 |
| `MaxPositions` | `0` | 0 = Không giới hạn hạn mức phần mềm; vốn sàn quản lý |
| `MagicNumber` | `123456` | Số nhận diện duy nhất của hệ thống |
| `ActionWaitMs` | `15000` | Thời gian chờ tối đa phản hồi Python (15 giây) |

Nhấn nút **Save** để lưu thành file cấu hình: `MQL5/Profiles/Tester/v193_production.set`.  
Nhấn **OK** để kích hoạt EA.

### 6.3. Kiểm Tra Banner Khởi Động
Quan sát tab **Experts** trong cửa sổ Terminal, bạn phải nhìn thấy dòng thông báo:
```text
Self-Growing MT5 Trainer EA feature-v10 | Inputs=137 Outputs=20
Bridge initialized successfully. Waiting for Python inference bridge...
```

---

## 7. Khởi Chạy Runtime Python Bridge Inference

Hệ thống giao tiếp thông qua cơ chế tệp IPC nguyên tử (Atomic file swap). Mở một cửa sổ PowerShell riêng biệt và thực thi lệnh:

```powershell
cd c:\Users\Admin\Desktop\new-chat
$env:PYTHONPATH=".;src"

python scripts/run_tensor_rule_v186_candidate.py `
  --mode mt5-smoke `
  --config configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml `
  --profile configs/experiments/reasoning_rl_v1_active_candidate.yaml `
  --seed-artifact outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint_phase2.pkl `
  --runner-source mt5_v2_runner `
  --output-dir outputs/live_production_run `
  --allow-unverified-imitation-seed `
  --skip-paired-validation
```

### Các Tham Số Runtime Quan Trọng:
- `--seed-artifact`: Trỏ trực tiếp vào Checkpoint Champion Pha 2.
- `--runner-source`: Trỏ vào thư mục runner `mt5_v2_runner`.
- `--output-dir`: Nơi ghi nhận log, chẩn đoán và telemetry thời gian thực.
- `--skip-paired-validation`: Chạy trực tiếp suy luận chính sách (Inference Only) mà không cần bước đối đầu mẫu cặp.

Khi tiến trình khởi động thành công, màn hình console sẽ hiển thị:
```text
[Candidate Runner] Loaded checkpoint: best_promoted_checkpoint_phase2.pkl
[Candidate Runner] Parameter verification: 117,350,081 parameters OK.
[Candidate Runner] Bridge service active. Polling self_growing_request.csv...
```

---

## 8. Quy Trình Giám Sát Telemetry Thời Gian Thực

### 8.1. Kiểm Tra Nhịp Quyết Định M1 (Heartbeat)
Khi một cây nến M1 đóng cửa:
1. MT5 xuất 137 đặc trưng vào `self_growing_request.csv`.
2. Python đọc dữ liệu, chạy forward pass qua mạng neural (~177 ms trên CPU), kiểm tra các bộ lọc phân quyền và ghi lệnh vào `self_growing_action.csv.tmp` rồi đổi tên nguyên tử thành `self_growing_action.csv`.
3. MT5 đọc lệnh, gửi lên sàn và ghi nhận kết quả vào `self_growing_execution.csv`.
4. Console Python xuất log:
   ```text
   [BridgeFive] served=... seq=... packet=[a0=..., a1=..., a2=0, a3=0, a4=0]
   ```
   *Lưu ý an toàn:* Xác nhận `a4=0` (Agent 4 luôn là HOLD, đảm bảo không có lệnh đóng sớm `POLICY_CLOSE_ALL`).

### 8.2. Giám Sát Sức Khỏe Bằng Script Tự Động
Chạy công cụ chẩn đoán trực tiếp:
```powershell
python scripts/export_mt5_live_diagnosis.py --runner mt5_v2_runner --output outputs/live_production_run
```
Theo dõi các chỉ số sinh tồn tại `outputs/live_production_run/mt5_diagnostics.json`:
- **Current Balance & Equity**: Số dư và tài sản ròng cập nhật theo từng tick.
- **Current Drawdown Pct**: Sụt giảm vốn hiện tại. Ngưỡng cảnh báo: `> 8.0%`.
- **Win Rate & Profit Factor**: Đánh giá hiệu suất rolling 50 lệnh gần nhất.
- **Reconciliation Status**: Đảm bảo tất cả vị thế đóng đều được hạch toán đầy đủ trong sổ cái `self_growing_execution.csv`.

---

## 9. Nguyên Tắc Quản Trị Rủi Ro Sản Xuất

1. **Rủi Ro Cố Định 1.0% Mỗi Lệnh:**
   EA tự động tính toán khối lượng lệnh (lot size) dựa trên vốn chủ sở hữu (`Equity`) và khoảng cách Stop Loss thực tế (5 ATR):
   $$\text{Volume (Lots)} = \frac{\text{Equity} \times 1.0\%}{\text{Cự ly SL (Points)} \times \text{Tick Value}}$$
   *Tuyệt đối không tăng `FixedRiskPct` lên quá 2.0%* để duy trì khả năng hấp thụ biến động theo kiểm định OOS.
2. **Tỷ Lệ Lợi Nhuận/Rủi Ro (Payoff Asymmetry):**
   Với SL = 5 ATR và TP = 10 ATR ($RR = 2.0$), hệ thống có lợi thế toán học dương ngay cả khi tỷ lệ thắng chỉ ở mức 40%:
   $$E[R] = (0.40 \times 2.0R) - (0.60 \times 1.0R) = +0.20R > 0$$
3. **Cơ Chế Thoát Hiểm Khẩn Cấp (Circuit Breaker):**
   - Nếu Equity Drawdown vượt quá **10.0%**, người vận hành nên tạm dừng Python bridge bằng `Ctrl + C`. Khi Python ngắt kết nối, EA tự động chuyển về chế độ an toàn (`NOOP_HOLD`), không mở thêm lệnh mới và để các vị thế hiện hữu tự động chạm TP/SL hoặc hết hạn 120 nến.
4. **Hủy Lệnh Chờ Tự Động:**
   Tất cả lệnh chờ Limit/Stop không khớp sau 30 nến M1 (30 phút) sẽ được tự động thu hồi để giải phóng ký quỹ.

---

## 10. Giải Đáp Sự Cố Thường Gặp (FAQ)

### Câu hỏi 1: Terminal MT5 báo lỗi `timeout` khi chờ chỉ thị từ Python?
- **Nguyên nhân:** Python bridge chưa được bật, hoặc tiến trình bị treo, hoặc đường dẫn file bridge trong MT5 bị lệch.
- **Khắc phục:**
  1. Kiểm tra cửa sổ console Python xem script có đang chạy hay không.
  2. Kiểm tra tham số `ActionWaitMs` trong EA (phải đặt tối thiểu 15,000 ms).
  3. Kiểm tra xem MT5 và Python có đang trỏ cùng thư mục `MQL5/Files` hay không.

### Câu hỏi 2: Có hiện tượng lệnh bị đóng sớm trước khi chạm SL hoặc TP không?
- **Trả lời:** **Tuyệt đối KHÔNG.** Hệ thống V193 Champion Pha 2 cam kết kỹ thuật `POLICY_CLOSE_ALL == 0`. Agent 2 (Position Manager) và Agent 4 (Governor) đều bị khóa cứng ở HOLD. Lệnh chỉ có thể kết thúc do:
  - Chạm Stop Loss (5 ATR).
  - Chạm Take Profit (10 ATR).
  - Hết thời gian tối đa 120 nến M1 (`EXPERT close`).

### Câu hỏi 3: Mô hình xử lý thế nào khi sàn giao dịch mất kết nối mạng?
- **Trả lời:** EA được thiết kế theo nguyên lý **Fail-Closed Safe Mode**. Khi mất kết nối hoặc không nhận được tín hiệu hợp lệ từ Python trong 15 giây, hàm `RequestPythonActionBatch` trả về `timeout` và EA giữ nguyên trạng thái vị thế (không đặt thêm bất kỳ lệnh thị trường nào). Khi có mạng trở lại, hệ thống tự động đồng bộ lại từ nến mới.

### Câu hỏi 4: Có thể giao dịch các cặp tiền khác (EURUSD, XAUUSD) bằng model này không?
- **Trả lời:** Checkpoint Champion Pha 2 V193 được huấn luyện và tối ưu hóa đặc thù trên động lực học giá của Bitcoin (`BTCUSDm`) khung thời gian M1. Nếu muốn áp dụng trên tài sản khác, cần tiến hành chạy kiểm định OOS độc lập và tối thiểu 10 vòng Fine-tuning với dữ liệu của tài sản đó trước khi đưa vào sản xuất.

### Câu hỏi 5: Làm sao để kiểm tra lại tính toàn vẹn của hệ thống bất kỳ lúc nào?
- **Trả lời:** Chạy script kiểm định tự động đi kèm trong gói bàn giao:
  ```powershell
  python scripts/verify_production_registry_phase2.py
  ```
  Script sẽ tự động đối soát toàn bộ SHA256 của các checkpoint, kiểm tra bit-for-bit equivalence, xác thực file registry metadata, và rà soát lại toàn bộ cam kết kỹ thuật.
