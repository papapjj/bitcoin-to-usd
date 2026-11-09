# V193 — Observation Pattern Tree V1

## Mục tiêu

Encoder Five v4 không còn chỉ ghép phẳng các embedding ngữ nghĩa. Nó xây một cây
neural phân cấp từ đúng observation 137 chiều để học quan hệ giữa cấu trúc giá,
vị thế, tài khoản và trạng thái thực thi. Cây phục vụ hai việc:

1. root pattern đi vào pre-LSTM qua một residual học được;
2. các head auxiliary học outcome bất biến sau khi future của dataset sample hoặc
   horizon MT5 đã kết thúc.

Future outcome không phải feature và không xuất hiện trong time/live/MT5 inference.

## Kiến trúc

```text
Observation 137
├─ market 78 ───────────→ market leaf 64 ─┐
├─ positions 5 x 8 ─────→ position leaf 64 ┴→ price node 128 ─┐
├─ account 8 ───────────→ account leaf 64 ─┐                  ├→ root 128
└─ execution 11 ────────→ execution leaf 64┴→ context node 128┘
                                                               ├→ utility [long, short]
                                                               ├→ excursion [long MFE, long adverse,
                                                               │              short MFE, short adverse]
                                                               ├→ no-edge logit
                                                               ├→ return H4/H8/H16/H32/H48 + mask
                                                               ├→ time-to-MAE/MFE, underwater, recovery,
                                                               │  positive-persistence cho LONG/SHORT
                                                               ├→ SL-first/TP-first/neither cho LONG/SHORT
                                                               ├→ 6 action x [fill, fill-delay, terminal,
                                                               │  post-fill MFE, post-fill adverse]
                                                               └→ 6 action x [unfilled, neither, SL, TP]

base pre-LSTM 1024 + pattern residual(root) → cross-agent pooling → LSTM 4096
→ Transformer hậu-LSTM 512/8 heads/FFN 2048 → action/value/reasoning heads
```

Các utility target được clip theo `five_pattern_tree_utility_clip_atr` rồi
chuẩn hóa vào `[-1, 1]`. MFE và độ lớn MAE được chuẩn hóa vào `[0, 1]`.
`no_edge=1` khi utility tốt nhất không vượt ngưỡng edge. Return nhiều horizon dùng
`tanh` thay vì clip cứng, và mỗi horizon có mask riêng; path chưa đủ 32/48 bar không
được lấy bar cuối làm nhãn giả. Nếu một bar M1 chạm cả SL lẫn TP mà không có tick,
nhãn dùng quy tắc bảo thủ `SL-first`.

Nhãn theo action giữ riêng MARKET/LIMIT/STOP và BUY/SELL. Pending còn ghi có khớp
hay không, bar khớp và excursion chỉ tính sau khi khớp. Với dataset cũ, trainer tái
dựng path 48 bar từ market tape chung nên không cần đào lại shard; pending chuẩn
hóa cách neo 1 ATR và dùng geometry mặc định. MT5 dùng đúng quote, entry price,
SL/TP và path đã đóng trong `EntryPreferenceRecord`.

Không có head trend/range/breakout/pullback giả: dataset hiện không có ground-truth
đáng tin cậy cho các tên đó. Node price/context là latent representation; chỉ các
đầu ra có nhãn outcome thực mới nhận supervised loss.

## Loss và writer

Mỗi batch gồm 16 anchor độc lập. PPO actor re-forward từng anchor bằng đúng hidden,
cell, previous action và action mask đã cache. Observation Tree dùng observation tại
anchor và các label terminal:

```text
utility_loss   = 0,5 * (SmoothL1(long, short) + SmoothL1(long-short contrast))
excursion_loss = SmoothL1(MFE/adverse của hai hướng)
no_edge_loss   = BCEWithLogits có positive weight giới hạn [1, 4]
multi_horizon  = SmoothL1 có horizon mask
path_dynamics  = SmoothL1(time/extreme/underwater/recovery/persistence)
first_hit      = CrossEntropy(SL-first/TP-first/neither)
action_path    = SmoothL1 riêng 6 kiểu entry
action_outcome = CrossEntropy riêng 6 kiểu entry
direction_loss = -log P(HOLD | no-edge) hoặc -log P(LONG/SHORT | edge)
hold_ceiling   = max(P(HOLD) - 0,20, 0)^2, chỉ khi có edge

aux_loss = auxiliary_weight * weighted_mean(các loss trên)
total    = PPO_actor_only + aux_loss
```

LONG gom `{MARKET_BUY, BUY_LIMIT, BUY_STOP}` và SHORT gom ba action tương ứng cho
loss hướng; các head action-path/action-outcome không gộp ba kiểu khớp. HOLD chỉ là target
dương khi cả hai utility không vượt edge. Trên anchor có edge, direction loss và
ceiling loss vẫn kéo HOLD xuống ngay cả khi cả năm agent cùng chọn HOLD và PPO
advantage chuẩn hóa không phân biệt được chúng.

Economic actor credit dùng biên nhận của hành động gốc. HOLD có vị thế học theo
delta marked-R của chính owner; HOLD khi trống bị loại khỏi toàn bộ loss kinh tế,
không được ăn reward của tác nhân khác trong cùng packet. Pattern actor credit chỉ
dùng tại anchor terminal để thưởng HOLD no-edge hoặc phạt HOLD bỏ lỡ edge.

Chỉ `FiveAdam` hiện hữu được quyền ghi trọng số. Auxiliary loss không đi qua value
head; economic critic không học privileged target. Tree và semantic encoders nhận
gradient auxiliary, còn PPO tự học trọng số residual để quyết định mức sử dụng root.

## Chống leakage

- observation tại anchor chỉ chứa dữ liệu có sẵn ở thời điểm quyết định;
- dataset oracle chỉ được tạo trong `run_dataset()` và chạy sau `_close_all()`;
- MT5 oracle chỉ được tạo từ path ledger sau 48 nến M1 đóng, rồi join lại packet
  anchor bằng `sequence` và `trace_id`;
- label không được lưu vào recurrent memory, action trace đầu vào hay bridge;
- time mode, frozen/OOS và live inference không có auxiliary target; MT5 online
  train có target trễ nhưng không có future trong observation/action request;
- mỗi anchor là một recurrent state độc lập, không ghép thành chuỗi LSTM giả.

## Checkpoint

Architecture version là `4`. Checkpoint encoder v2 được thêm tree và residual một
cách trung tính: trọng số residual khởi tạo bằng 0 nên output encoder cũ không đổi
tại thời điểm migrate. Checkpoint v3 giữ nguyên tree/root/residual và được gắn năm
head path v4 có trọng số/bias bằng 0; action/value output không đổi tại thời điểm
migrate. Optimizer giữ state tham số cũ và đăng ký thêm tham số mới. Checkpoint
trước v2 hoặc sai topology vẫn bị từ chối và phải dùng output mới.

## Cấu hình V193

```yaml
five_pattern_tree_auxiliary_loss_weight: 0.25
five_pattern_tree_min_edge_atr: 0.10
five_pattern_tree_mfe_weight: 0.15
five_pattern_tree_mae_weight: 0.25
five_pattern_tree_utility_clip_atr: 3.0
five_pattern_tree_utility_loss_weight: 1.0
five_pattern_tree_excursion_loss_weight: 0.50
five_pattern_tree_no_edge_loss_weight: 0.50
five_pattern_tree_direction_loss_weight: 1.0
five_pattern_tree_edge_hold_ceiling: 0.20
five_pattern_tree_hold_ceiling_loss_weight: 1.0
five_pattern_tree_horizon_loss_weight: 0.50
five_pattern_tree_path_dynamics_loss_weight: 0.50
five_pattern_tree_first_hit_loss_weight: 0.50
five_pattern_tree_action_path_loss_weight: 0.50
five_pattern_tree_action_outcome_loss_weight: 0.50
```

Các trọng số này chỉ có tác dụng khi `five_pattern_reward_enabled: true` và trainer
có dataset outcome hoặc MT5 path đã đóng đủ horizon.

## Telemetry và đánh giá

`pattern_learning` xuất target ATR, prediction normalized, từng auxiliary loss,
utility/excursion MAE, no-edge accuracy, direction/ceiling loss, xác suất HOLD/LONG/
SHORT và số anchor được supervise. Đây là train
telemetry, không phải bằng chứng edge. Promotion vẫn cần paired baseline cùng dataset
hash/seed/order và frozen walk-forward OOS sau chi phí.
