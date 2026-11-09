# Kiến trúc năm tác nhân dùng chung

## Một bộ não, năm dòng trạng thái

Năm tác nhân dùng chung một mạng. Mỗi tác nhân giữ riêng:

- trạng thái ẩn 4096 chiều;
- ô nhớ 4096 chiều;
- hành động trước;
- lịch sử số bước.

Chỉ có một bản của các ma trận học được.

## Hợp đồng phân quyền

Năm dòng nhớ không còn có quyền hành động giống nhau:

- tác nhân 0: entry mua thị trường/giới hạn/dừng;
- tác nhân 1: entry bán thị trường/giới hạn/dừng;
- tác nhân 2: đóng vị thế tốt/xấu và `MANAGE_POSITION`;
- tác nhân 3: hủy lệnh chờ;
- tác nhân 4: `CLOSE_ALL` ở cấp danh mục.

Tất cả đều có `HOLD`. Mỗi action khác `HOLD` chỉ có một chủ. Python áp dụng mặt
nạ quyền trước lấy mẫu, bridge giữ `agent_id/trace_id`, EA kiểm tra lại quyền và
ghi receipt nếu vi phạm. `CLOSE_ALL` là phủ quyết duy nhất; không còn bộ chọn một
hướng entry thắng. Cơ chế này không thêm tham số hoặc thay đổi hình dạng tensor,
nên checkpoint đúng kiến trúc hiện tại vẫn nạp được.

## Đầu vào

Mỗi quan sát có 137 giá trị. Encoder phiên bản 2 chia chúng thành bốn họ:

- cấu trúc thị trường 78 chiều → hai lớp tích chập 1D → 256 chiều;
- tài khoản 8 chiều → MLP riêng → 64 chiều;
- trạng thái thực thi 11 chiều → MLP riêng → 64 chiều;
- năm vị thế × 8 chiều → hai lớp dùng chung, véc-tơ từng vị thế và max-pool 64 chiều.

Nhận dạng tác nhân và hành động trước có embedding 32 chiều riêng. Sáu nhánh tạo 512 chiều rồi được chiếu thành 1024 chiều. Chuẩn hóa liên tục thuộc biên Python của cầu nối và không bị lặp lại trong encoder.

25% đầu, tức 256 chiều, được lấy cực đại chéo giữa năm tác nhân rồi trả lại cho cả năm. 768 chiều còn lại giữ riêng theo từng tác nhân.

## Tầng nhớ và Transformer hậu-LSTM

Mỗi tác nhân đi qua cùng một tầng nhớ dài-ngắn có cổng 4096 phần tử. Trọng số
dùng chung, trạng thái hidden/cell hồi tiếp vẫn riêng. Năm hidden hiện thời sau
đó tạo thành năm token cho đúng một `TransformerEncoderLayer`: projection
`4096→512`, self-attention 8 head, FFN `512→2048→512`, projection `512→4096`
và residual học được. Transformer cấp representation cho toàn bộ action,
target, parameter, value và reasoning head; output đã trộn không được ghi ngược
vào recurrent memory, nên reset episode của một agent không nhận lịch sử của
agent khác.

Policy architecture version hiện là 3 và observation architecture version là 4.
Checkpoint ngoài các version migration được hỗ trợ bị từ chối fail-fast.
Seed khởi tạo mới dùng cho UI imitation nằm tại
`outputs/he_chuyen_gia/kho_chuyen_gia_transformer_v3.pkl`. Nếu seed chưa tồn tại,
UI tự tạo bằng cấu hình Five khóa cứng rồi mới chạy teacher imitation.

## Đầu ra

Từ trạng thái ẩn có các nhánh học được cho:

- loại hành động;
- mục tiêu;
- độ trễ 0–3 nhịp;
- mức rủi ro;
- khoảng vào lệnh;
- kiểu quản lý vị thế;
- tỷ lệ đóng vị thế;
- giá trị trạng thái.

## Phần thưởng và học

Đường năm tác nhân dùng bộ học chung. Phản hồi một dấu vết không được phép đi trực tiếp vào đường học cũ. Lớp điều phối phải gom phản hồi theo chỉ số tác nhân rồi mới gọi đường học năm tác nhân.

### Candidate Reasoning RL V1

Five có thêm một head giả thuyết nhiều bước candidate-only. Head sinh chuỗi
`HOLD + sáu entry`, nhận reward từ `action_q_values_r` của path 48-M1 đã mature
và học bằng group-relative PPO/GRPO với KL tới reference đóng băng. Future path
không đi vào inference. Config nền giữ `reasoning_action_scale=0` và
`reasoning_encoder_credit_scale=0`, nên có thể chạy telemetry-only đối chứng.
Profile active của UI đặt action scale 0.25 có clip, chỉ tác động bảy logit
HOLD/entry và vẫn giữ encoder credit 0; đây là candidate research để kiểm tra
nhân quả, chưa phải promotion cho production. Luồng dạy bảy lựa chọn giữ thân mạng cố
định và điều kiện cận dưới lợi nhuận trước PPO được mô tả tại
`docs/SEVEN_CHOICE_FROZEN_HEAD_TEACHER.vi.md`. Hợp đồng đầy đủ nằm trong
`docs/FIVE_REASONING_RL_V1.vi.md`.

UI V193 không tạo tab Reasoning riêng. **Train candidate** mặc định chọn config
Reasoning RL cô lập + profile active action/optimizer và vẫn cho phép chọn V193
không reasoning trong cùng form.
Trong chế độ Reasoning RL, ô **Bộ nhãn bảy lựa chọn** ghép cố định dữ liệu với
cấu hình `sl500`, gồm thời gian giữ 12, 24 hoặc 48 nến. UI truyền nguyên cặp đã
ghép để không chạy nhầm checkpoint bằng quy tắc cắt lỗ, chốt lời hoặc thời gian
giữ khác với lúc tạo nhãn. Cấu hình cũ vẫn được giữ như lựa chọn riêng để tái
hiện các lượt trước.
Hợp đồng Five/Bridge được nhập vào Candidate; tab kiến trúc read-only và tab
**Đầu ra tư duy** của ReasoningGraph cũ đã bị xóa. Live Decision đọc trường
`five_reasoning` causal gồm chuỗi hypothesis, hypothesis cuối, action scale,
residual clip và cờ future-label để không nhầm trace cũ với GRPO mới.

## Backend parity Model 1 tùy chọn

`environment.backend=mt5` vẫn là mặc định. Khi bật tường minh `python_model1`, Python đọc oracle NPZ và phát request/lifecycle nhân quả vào cùng production decision/learning loop. Tape chứa account và ký quỹ MT5 nguyên bản; backend không tự mô phỏng lại broker margin. Kiến trúc và runbook đầy đủ nằm trong `docs/MODEL1_PYTHON_PARITY_BACKEND.vi.md`.

## Biên dữ liệu imitation

`BottomTop_Teacher_EA` v4 nằm ngoài runtime policy, không phải writer của
checkpoint và không giao dịch. Dataset sản xuất schema 9 dời BUY/SELL từ chính
bar pivot sang observation đầu tiên sau xác nhận cấu trúc, tính lại Bid/Ask và
giữ toàn bộ HOLD/hard-negative theo split thời gian. Accepted entry giữ SL cấu
trúc 1,50 ATR ngoài pivot và TP tại pivot đối diện; target và PnL tương lai không
đi vào 137 observation. Offline trainer mặc định replay cửa sổ recurrent 64 bước
(UI cho chọn 2–64) bằng BPTT và cho gradient đi qua observation encoder, toàn bộ LSTM, Transformer
hậu-LSTM, projection action-query rồi tới adapter residual phi tuyến 128 chiều có điều kiện role
BUY/SELL. Pairwise ranking loss học thứ tự
ENTRY/HOLD; ngưỡng role chỉ được hiệu chỉnh trên validation và luôn bị chặn bởi
entry-rate ceiling. Geometry target có trong schema 9 nhưng geometry imitation
không thuộc workflow action-only hiện hành.
Residual chung tác động MARKET/LIMIT/STOP cùng phía, giữ nguyên thứ hạng kiểu
lệnh trong chính residual. Trọng số riêng action embedding, value, target và
geometry head giữ nguyên hash; output dùng chung recurrent representation có
thể đổi và phải được kiểm tra lại bằng MT5/OOS. MT5 PPO giữ toàn bộ đường
teacher action cố định 32 update đầu. Từ update 33 đến 256, encoder/LSTM/Transformer và
base-action vẫn cố định để hidden space không trôi; calibrator và geometry head
teacher được thích nghi dưới neo khi checkpoint có marker tương ứng.

Độ dài BPTT imitation và PPO online có buffer độc lập, nhưng hiện cùng đặt ở 64
bước. `FIVE_LSTM_UNROLL=64` điều khiển rollout, GAE và tần suất optimizer của
MT5 PPO; trường sequence trong UI chỉ điều khiển BPTT imitation.

Pause/resume action imitation là file-control ngoài model graph. UI ghi
stop-request; trainer chỉ quan sát sau optimizer batch, rồi atomically serialize
system cùng optimizer, RNG, epoch, permutation/cursor, best-state, patience và
history. Loader fail-closed nếu source/dataset/config/hyperparameter/output hoặc
frozen/recurrent hash khác. Resume artifact có lifecycle `paused`, không đi vào
Candidate/MT5 và bị xóa sau khi candidate cuối cùng reload thành công.
Trainer đồng thời tích lũy confusion counts và confidence ENTRY ngay trong
forward batch, không chạy thêm train replay. Cuối epoch nó phát train metric
online, validation snapshot, fit-status heuristic và tỷ lệ confidence ENTRY
đúng trên false-entry penalty pressure. Java chỉ render các trường có provenance
này vào tab Chẩn đoán học; tab không tạo reward mới và không tham gia optimizer.
Hợp đồng đầy đủ nằm
trong `docs/BOTTOM_TOP_TEACHER_IMITATION.vi.md`.

Candidate MT5 tạo thêm `Mt5DiagnosticsMonitor` chỉ đọc trong một thread riêng.
Monitor tail tăng dần `self_growing_execution.csv`, `self_growing_bridge.csv` và
`*.execution_trace_observatory.jsonl`, rồi replace nguyên tử
`mt5_diagnostics.json`. UI Java chỉ poll file compact này, không quét lặp ledger
lớn và không có đường ghi ngược vào MT5/model. `run_mt5_ablation_v193.py` dùng
cùng monitor cho ba runner clone tuần tự và tổng hợp `ablation_summary.json`;
checkpoint nguồn được đối chiếu SHA-256 sau mỗi arm.

Runner clone có thể chứa file CSV rỗng trước khi tác nhân Strategy Tester tạo
ledger thật trong `Tester/Agent-*/MQL5/Files`. Monitor tiếp tục dò đường dẫn mới
nhất trong suốt lượt chạy. Nhờ vậy trạng thái giao dịch không còn bị kẹt ở 0 khi
file tác nhân xuất hiện muộn, và vẫn theo được lượt chuyển sang tác nhân thử
nghiệm mới mà không mất tổng đã đọc. Dòng tiêu đề được ghi lại sau khi tệp bị
tạo lại không bị tính nhầm thành một sự kiện.

## Số tham số

Tổng policy mặc định chưa gắn reasoning head: **93.182.878**.

Encoder: **795.026**. Calibrator LONG/SHORT có **558.212** tham số, gồm adapter
phi tuyến role-conditioned **33.536** tham số; migration của adapter là
output-neutral.

Tầng nhớ 1024→4096: **83.918.848**.

Transformer hậu-LSTM: **7.351.297**. Phần còn lại thuộc các action/value head.

## Không còn trong đường hiện hành

Các thành phần sau chỉ còn mã tương thích hoặc tài liệu lịch sử, không được khởi tạo cho cấu hình năm tác nhân:

- mạng hồi tiếp trực tiếp nhiều tầng cũ;
- mạng thời điểm riêng;
- các đầu mục tiêu phụ cũ;
- chủ sở hữu biến động riêng;
- chín chuyên gia theo ngữ cảnh;
- mạng tò mò;
- mạng ích lợi;
- mạng chi phí rủi ro;
- đồ thị suy luận;
- head Reasoning RL V1 không thuộc nhóm legacy graph ở trên; nó chỉ được bật
  bởi profile candidate tường minh và mặc định không có action authority;
- ngân hàng luật học được.
