# Tổng quan kiến trúc hiện tại

Cấu hình đang dùng là hệ năm tác nhân dùng chung một bộ trọng số.

## Luồng quyết định

Mỗi tác nhân nhận 137 giá trị quan sát. Phần quan sát được xử lý có cấu trúc trước khi đi vào tầng nhớ. Năm tác nhân dùng chung toàn bộ trọng số nhưng giữ trạng thái ẩn, ô nhớ và hành động trước riêng.

Luồng chính:

1. 137 giá trị quan sát.
2. Tách 78 giá trị cấu trúc thị trường cho mạng tích chập 1D hai lớp, 8 giá trị tài khoản và 11 giá trị thực thi cho hai MLP riêng.
3. Xử lý tập 5 vị thế bằng cùng hai lớp, giữ véc-tơ từng vị thế và lấy cực đại theo từng chiều.
4. Ghép các embedding thị trường, tài khoản, thực thi, vị thế, tác nhân và hành động trước; biến đổi thành 1024 chiều.
5. Chia sẻ 25% đầu của biểu diễn giữa năm tác nhân bằng phép lấy cực đại chéo.
6. Một tầng nhớ dài-ngắn có cổng 4096 phần tử dùng chung trọng số.
7. Một Transformer hậu-LSTM duy nhất: bottleneck 512, 8 attention head, FFN 2048, trả residual về 4096.
8. Các nhánh hành động, mục tiêu, thời điểm, rủi ro, khoảng vào, quản lý vị thế và giá trị trạng thái.

Không còn đường chạy 4×64, chú ý trước từng tầng, mạng thời điểm riêng, mạng tò mò, ngân hàng luật, đồ thị suy luận hoặc chín chuyên gia riêng trong cấu hình hiện hành.

## Số tham số

Tổng số tham số học được của policy mặc định là **93.182.878**.

Riêng tầng nhớ 1024→4096 có **83.918.848** tham số và Transformer hậu-LSTM có
**7.351.297** tham số. Bottleneck 512 giữ block mới ở khoảng 8,8% kích thước
LSTM; attention trực tiếp ở width 4096 không được dùng vì quá lớn.

Encoder có **795.026** tham số và các đầu ra có **1.117.707** tham số.

Năm tác nhân không nhân năm số tham số vì cùng dùng một bộ trọng số.

## Cầu nối nền tảng giao dịch

Bridge v10 mang packet năm action cho cùng một quan sát. Mỗi action có `agent_id` và `trace_id` riêng; EA từ chối packet thiếu hoặc trùng agent. Cả năm memory tiến cùng một bước và năm execution receipt được ghép lại thành một transition đội đồng bộ.

Ngoài backend MT5 mặc định, project có backend `python_model1` opt-in để replay một episode Model 1 đã khóa bằng NPZ qua chính production policy loop. Hợp đồng này kiểm tra tuyệt đối 137 input, 20 output, action, account/margin, lệnh, lifecycle và learner state; chỉ độ trễ fill được miễn. Chi tiết tại `docs/MODEL1_PYTHON_PARITY_BACKEND.vi.md`.

Năm proposal được phân quyền cố định: tác nhân 0 sở hữu entry mua, 1 sở hữu entry
bán, 2 quản lý vị thế, 3 quản lý lệnh chờ và 4 giám sát danh mục. Hai hướng entry
có thể cùng hoạt động; riêng `CLOSE_ALL` của tác nhân 4 phủ quyết packet. EA trả
receipt riêng cho từng `trace_id` và từ chối rõ action sai quyền. V193 đặt
`MaxPositions=0`, nên không chặn entry theo quota số vị thế/lệnh chờ; broker và
ký quỹ quyết định lệnh có được nhận hay không. Risk trong cùng packet được chia
đều giữa các entry. Vector 137 vẫn có cửa sổ năm vị thế chi tiết, còn side pipe
biến độ dài giữ toàn bộ vị thế cho lifecycle/PnL.

## Teacher EA cho imitation learning

Project có `BottomTop_Teacher_EA` v4 làm historical labeler, không đặt lệnh.
BUY tại swing low thoát ở swing high kế tiếp; SELL tại swing high thoát ở swing
low kế tiếp. Giá thực thi dùng Bid/Ask, trừ spread và kiểm tra SL cấu trúc trước
khi chấp nhận nhãn có net PnL dương. `sample_time` tách khỏi
`label_finalized_time`; pivot đối diện và PnL chỉ là target/metadata, không vào
137 input. Builder kiểm tra action authority, execution geometry, PnL identity,
độ trễ nhãn và chia dataset theo thời gian. Chi tiết tại
`docs/BOTTOM_TOP_TEACHER_IMITATION.vi.md`.

Dataset imitation schema 9 giữ toàn bộ dòng hợp lệ và dời nhãn tới observation
sau xác nhận pivot trước khi replay cửa sổ recurrent 64 bước. Trainer học mapping
phi tuyến từ action-query 128 chiều tới prior HOLD/ENTRY riêng theo role
LONG/SHORT, có pairwise ranking loss và validation rate ceiling. Loss dùng BPTT
thật để cập nhật observation encoder, toàn bộ LSTM, Transformer hậu-LSTM và projection action-query;
trọng số riêng của action-style/value/target/geometry head vẫn khóa. Vì các head
này dùng chung recurrent representation, logits đầu ra của chúng có thể đổi dù
trọng số head không đổi. Dataset có SL cấu trúc 1,50 ATR và TP tại pivot đối
diện, nhưng action-only không tối ưu loss geometry. MT5 warm-up bảo vệ action prior khỏi PPO ghi đè ngay,
nhưng candidate vẫn phải qua frozen OOS và MT5 trước khi có thể promotion.
Manifest workflow công khai provenance imitation của seed, nên hiệu lực prior
không còn được suy đoán chỉ từ tên checkpoint.

Action imitation hỗ trợ pause/resume đúng batch boundary từ UI. Artifact tạm
giữ cả system, Adam, epoch/permutation cursor, RNG, best-state và patience; chỉ
được nạp lại khi source, dataset, config, hyperparameter, output và các hash
trọng số đều khớp. Artifact pause không phải candidate và luôn non-promotable.
UI còn có tab Chẩn đoán học nhận metric cuối epoch: train BA/MCC online,
validation BA/precision/recall/MCC, generalization gap, cảnh báo underfit/
overfit và tỷ lệ confidence ENTRY đúng trên áp lực false-entry. Tỷ lệ cuối là
proxy loss imitation, không phải reward RL hoặc lợi nhuận giao dịch.

Chẩn đoán MT5 online nằm ở phần riêng và đọc execution/bridge/optimizer trace
thật: PF, expectancy, payoff, WR hòa vốn, equity DD, PPO update/loss, reward
density, action entropy/collapse và chất lượng khớp lệnh. Workflow Ablation chạy
matched `frozen_policy`, `full_online`, `reasoning_off` trên bản sao cùng seed
và xuất delta PF/expectancy/DD; một cửa sổ ablation không thay thế frozen OOS.

## Reward và cạnh tranh

Bridge khóa `trace_id` với agent sở hữu, rồi reward lấy lợi nhuận đã chốt từ execution ledger, lợi nhuận đang mở từ position snapshot và chi phí giao dịch đúng một lần. Năm agent nhận bonus cạnh tranh zero-sum theo mức cải thiện PnL mới so với trung bình, trong khi reward đội vẫn phạt drawdown/risk và dùng terminal `equity_R - max_drawdown`. Công thức và giới hạn kiểm chứng được mô tả tại `docs/FIVE_COMPETITIVE_REWARD_V1.vi.md`.

## Tài liệu lịch sử

Kiến trúc cũ được lưu tại `docs/lich_su/` chỉ để tra cứu và không mô tả cấu hình hiện hành.
