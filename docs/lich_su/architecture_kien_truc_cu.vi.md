> **Lưu ý:** tài liệu này mô tả kiến trúc cũ trước chế độ năm tác nhân dùng chung. Các đoạn `4×64`, chú ý trước từng tầng, mạng thời điểm riêng và các đầu phụ không còn mô tả đường chạy hiện tại. Kiến trúc hiện hành được mô tả tại `docs/kien_truc_nam_tac_nhan.md`.

# Kiến trúc

## Data và feature

`data.py`, `features.py`, `features_mql5.py`, `orders.py`, `environment.py` vẫn giữ vai trò xử lý dữ liệu, feature, action và backtest Python nội bộ. Backtest Python dùng cho evaluate/audit nhanh, không dùng để train chính.

Candidate V153 thêm `features.mtf_scaling_mode=native_tf_atr_v2`: EMA
spread/slope/gap M5/M15/H1 được biểu diễn bằng ATR đúng timeframe, còn ATR
relative dùng `(ATR_tf-ATR_M1)/(ATR_tf+ATR_M1)`. Nó chặn sai lệch đơn vị đã
làm H1 EMA/volatility dính soft-clip của ratio M1; seed v13 giữ
`legacy_m1_atr_v1`. Candidate `groupwise_mtf_v3` dùng `1.2*tanh` riêng cho
block MTF native 80..91 sau projection để H1 mạnh vẫn có phân bố liên tục,
trong khi v13 giữ normalizer cũ. Sau 137 input bridge bất biến và hai token context nhân
quả, candidate có thể thêm tám interaction trend/location/volatility và wick
có hướng chỉ vào Direct-RNN entry. Volatility không có score hướng độc lập,
management bị chặn toàn bộ tail market này, và Full-RL là writer duy nhất.
Chi tiết biểu diễn, quarantine input và ablation A/B/C/D ở
`docs/ENTRY_INPUT_REPRESENTATION_V153.vi.md`.

V157/V175/V187 bổ sung nhãn phản thực tế trễ cho đúng surface `HOLD` và sáu entry,
không thêm gate hay head inference. EA lưu quote/ATR/SL/RR/entry price đã
broker-clamp cùng 48 M1 đóng sau **mọi quyết định HOLD hoặc entry** vào
`self_growing_entry_preference.csv`; Python replay bảo thủ path đó rồi đưa CE
target vào cùng Full-RL AdamW/cosine batch. Khi account phẳng row có scope
`flat_v1`; khi còn exposure, `entry_marginal_v2` chỉ đánh giá P/L thị trường
của proposal mới và không được dùng làm account-PnL/risk credit. Counterfactual
không dùng trailing hay policy exit sau đó. Chi tiết contract, telemetry và
giới hạn bằng chứng ở `docs/DIRECT_ENTRY_PREFERENCE_V157.vi.md`.

V176 sửa contract chạy headless của chính EA, không đổi công thức feature hay
surface inference: `iBars()` không được dùng làm clock/gate vì tester có thể
nạp trước cả history và giữ count không đổi. EA lấy `TimeCurrent()` làm trần
M1 để phát đúng một request mỗi nến, prewarm indicator handles có timeout hữu
hạn ở `OnInit`, rồi vẫn kiểm tra `FeatureBuffersReady()` trước từng request.
Vì vậy non-visual không còn chạy hết ngày với zero action, nhưng một buffer
không hợp lệ vẫn fail-closed thay vì gửi input bão hoà vào policy.

## Self-growing layer

V162 cố định taxonomy owner theo regime M15. Context branch tự sinh từng làm
489 decision tách thành 14 expert, khiến mỗi owner chỉ nhận vài batch AdamW;
branch/nursery đã bị xóa và artifact cũ bị migrate để bỏ owner branch cùng
selector state. Đây không phải gate hay action prior: mọi lifecycle hợp lệ của
cùng regime nay tích lũy vào đúng policy owner.

`self_growing_runner.py` chứa core mặc định `SelfGrowingPolicy`. Policy có một hoặc nhiều module:

- `w_in`, `b_hidden`: encoder input sang hidden.
- `w_rec`: hồi tiếp từ hidden state trước đó, tương ứng `m_k(t-1)` trong tài liệu gốc.
- `deep_w`, `deep_rec`, `deep_b`, `deep_state`: các tầng ẩn sâu bổ sung sau tầng hidden đầu tiên. `model.hidden_layers` và `model.hidden_size` quyết định chính xác topology expert mới; profile context-expert hiện là 4 tầng × 64 neuron. Mỗi hidden block có một feature-attention ngay trước phép biến đổi recurrent: `input_attention_q/b` cho tầng đầu và `deep_attention_q/b` cho từng tầng sâu. Attention dùng `N*softmax(q*x+b)` để tái trọng số kênh mà không đổi chiều; khởi tạo 0 cho trọng số bằng 1 nên migration checkpoint cũ giữ nguyên hành vi ban đầu. Số attention được suy ra trực tiếp từ hidden depth: config thêm tầng hoặc `expand_module_depth()` tự thêm attention trung tính, còn `expand_module()` resize attention theo width. Tất cả tensor attention nằm trong cùng Full-RL AdamW và immutable forward trace, không tạo writer thứ hai. Artifact cũ một tầng vẫn được chạy như stack rỗng để tương thích. Candidate train có thể gọi `expand_module()` để thêm neuron đến `max_hidden_neurons_per_module=512` và `expand_module_depth()` để thêm tầng ẩn mới khi `representation_need` vượt ngưỡng `hidden_depth_growth_threshold`; mặc định cap depth là `max_hidden_layers_per_module=24`, và MT5 promotion/reject quyết định neuron/tầng mới có được giữ hay không. Resume checkpoint có width/depth khác config bị từ chối thay vì resize tensor đã học.
- Sau mỗi block hidden, runtime **bắt buộc** chuẩn hoá và có dropout xác định. `hidden_normalization_mode` được chọn giữa `rms_v1` và `layernorm_v1`, nhưng `hidden_normalization_enabled` và mọi `hidden_normalization_layers` phải là `true`. `hidden_dropout_enabled` và `hidden_dropout_train_only` cũng phải là `true`; `hidden_dropout_rates` cho phép schedule dương theo từng block (danh sách ngắn lặp rate cuối khi tăng depth). Parser từ chối `none`, `false` và rate 0; migration/load/copy/forward sửa checkpoint cũ hoặc thuộc tính bị sửa tay về contract hợp lệ. Mọi Full-RL MT5 forward, kể cả frozen/OOS, lấy mask xác định từ seed + bước chạy; runtime reset đưa bước về 0 để paired frozen/online cùng seed dùng đúng cùng action surface. Entry cache **chính forward đã mask** cho lifecycle backprop; telemetry ghi `hidden_dropout_training_active`, step trước và số step tăng theo module, đồng thời runtime dừng nếu một module không tăng step. Recurrent state lưu hidden chưa mask, còn hidden đã mask chỉ dùng cho forward output và lifecycle credit. Frozen không tắt dropout/chuẩn hoá, nhưng cấm mọi writer outcome-derived: AdamW, geometry/audit, advantage EMA, TimingNet, selector, expert fitness/history và neuron utility EMA; chỉ trace runtime được dọn sau settlement.
- M15 regime routing không dùng vector đã chuẩn hoá cho policy. `BridgeRequest` giữ `raw_inputs` song song với `inputs`; runner chuẩn hoá chỉ `inputs`, rồi facade đưa raw sidecar vào `MarketRegimeClassifier` và TimingNet cho EMA/ATR, M15/H1 native ratio và TimingNet `soft_clip_v1` đúng một lần. Direct-RNN, memory và Full-RL vẫn chỉ thấy policy view chuẩn hoá. Timing output được cache theo signature M15 nên M1 request không tự thành clock RNN/label; telemetry ghi hai regime-score view để phát hiện saturation/collapse.
- `w_out`, `b_out`: hidden cuối cùng của stack sang 20 output hành động theo action schema v4.
- `gate_w`, `gate_b`: chọn trọng số tham gia của module.
- `plasticity`, `score`, `utility`, `age`, `lifecycle`: trạng thái phục vụ neural proposal/growth/consolidation/pruning.
- `memory_decay`: tỷ lệ giữ working memory ngắn hạn của module.

Runtime của model sâu dùng fast-path compatibility cache sau khi artifact đã được migrate. `ChuyenGiaPolicy.ensure_compatible()` chỉ bỏ qua lần kiểm tra lặp khi signature input/output, module, hidden stack, M1 graph type và schema không đổi; `ContextualExpectancyLedger` và `ReasoningGraphSystem` cũng cache migration/invariant theo record/topology signature. Khi cấu trúc đổi, cache tự mất hiệu lực và chạy ensure đầy đủ. `_query_memory` có cache trong phạm vi một `ContextFrame` để tránh query cùng memory/expectancy nhiều lần trong một decision, nhưng không chia sẻ qua request MT5 mới. Khi lifecycle đóng cập nhật expectancy, cache decision/causal bị xóa ngay; nếu không, context lặp lại sẽ giữ bias trước-credit và bỏ qua bằng chứng mới.

Trong profile `direct_rnn`, bias legacy của memory chỉ được phép sửa các head tham số (SL/TP/trailing). V13 vì vậy xuất thêm `expectancy_action_delta` để đưa LCB âm/dương đã xác thực vào đúng logit BUY/SELL/LIMIT/STOP. Delta được giới hạn và chỉ áp dụng theo action trong context đang truy vấn; không biến expectancy âm của một action thành global HOLD hay mask toàn bộ entry. Hard LCB entry mask là contract tách riêng: profile có thể khai báo `context_expectancy_entry_gate_allowed_contexts`; khi đã khai báo, chỉ owner context nằm trong allowlist mới được veto action. Profile candidate hiện chỉ allowlist `CTX_VOLATILE_CHOP`; các owner khác vẫn ghi telemetry LCB và nhận delta mềm, nhưng không thể bị hard mask bởi exact hay aggregate fallback.

TimingNet trong `timing_network.py` là nguồn timing học được duy nhất. Regime router độc lập đọc tuple feature của **nến M15 đã đóng** và đổi owner trực tiếp, không còn candidate M1/hysteresis/countdown. Runtime materialize rolling tensor `32 x 78` (oldest-to-newest, 58 market + 4 timing + 16 Wyckoff), rồi `soft_clip_v1` từng hàng và unroll recurrent encoder trên đúng suffix hợp lệ; M1 request cùng signature chỉ tái dùng tensor/forward đã cache. Nó không đọc position/account/execution. `timing_network_action_scale=0` và không nhận lifecycle-P/L writer. `scripts/train_timingnet_isolated.py` train capsule tách biệt trên split thời gian 70/15/15 và frozen MT5 snapshot, dùng cùng tensor 32 M15 và bỏ 31 sample warm-up khỏi supervised transition loss. Capsule có thể mang adapter thời gian nhân quả 12 chiều và tự khai báo input size, nên không được nạp vào runtime 78 chiều nếu chưa có integration experiment. Candidate chỉ được phép đổi entry thành `HOLD` sau OOS gate Brier/AUC/precision/support; khi chưa đủ bằng chứng, hệ chỉ telemetry và management không bị chặn. Artifact v13 cũ vẫn đọc được; seed file không bị ghi đè.

Candidate `he_chuyen_gia.context_history_inputs: 2` nối hai token context M15 đã chốt gần nhất (`t-1`, `t-2`) sau vector bridge. Profile active còn nối 13 interaction entry nhân quả, nên Direct-RNN nhận 152 chiều nội bộ trong khi EA/CSV và feature parity vẫn đúng 137 chiều. Snapshot được lấy trước proposal và context hiện tại chỉ được push sau toàn bộ decision; do đó không rò nhãn hiện tại/future. Full-RL cache đúng vector 152 chiều rồi AdamW/cosine học các cột local; 13 interaction chỉ đi route entry, còn semantic routing v2 cho entry lẫn management đọc hai token. `context_history_inputs: 0` và enrichment legacy giữ v13 nguyên trạng. Xem `docs/CONTEXT_HISTORY_INPUTS_V1.vi.md`.

`semantic_input_routing_mode=role_isolated_v1` là contract candidate tách quyền đọc feature của Direct-RNN: entry chỉ nhận 58 market + timing/Wyckoff để chọn `BUY/SELL/HOLD`; management chỉ nhận 40 position + 11 execution để close/trailing/cancel; account context chỉ đặt risk budget và hard safety veto, không có đường tới BUY/SELL logit. Nếu bật history candidate, cả hai route còn nhận hai token context M15 `t-1/t-2` đã phân loại tường minh; đó là metadata thị trường nhân quả, không phải position/account leakage. Hidden state, deep recurrent state và TBPTT history cũng tách theo route, vì mask input đơn lẻ không ngăn management state rò sang entry ở bước sau. Full-RL AdamW vẫn là writer duy nhất: management-only có support lệnh riêng nhưng bị đánh dấu control-only, không vào actor/value batch entry; chỉ hard account-risk veto đổi action thực thi thành HOLD và đánh dấu trace actor-ineligible, không tạo gradient giả. Duplicate-entry output mask bị tắt trong candidate vì nó có thể đẩy argmax từ side bị chặn sang side đối ứng. Trường bridge cũ `open_risk_ratio` hiện là capacity exposure, nên candidate dùng ngưỡng 0.60/0.90 thay vì diễn dịch sai thành 3%/5% tiền rủi ro. Seed v13 giữ mode `legacy`; chi tiết contract, telemetry và regression ở `docs/SEMANTIC_INPUT_ROUTING_V2.vi.md`.

V185 tách nhãn entry trên immutable 48-M1 path thành terminal-R, MFE/MAE,
fill/time-to-positive, direction, timing, geometry và spread-cost. Candidate dùng
`path_quality_v1`; artifact cũ mặc định `terminal_r_v1`. Queue chọn round-robin
theo context/action/outcome và hard-negative nhưng vẫn yêu cầu tối thiểu nửa
batch là lifecycle MT5. Nhãn này chỉ đi qua Full-RL AdamW, không tạo inference
gate hay writer thứ hai. Xem `docs/ENTRY_ACTION_SEPARATION_V185.vi.md`.

Policy còn có `ExperienceSuperNetwork`, là siêu mạng ký ức phân cấp được lưu trực tiếp trong artifact `.pkl`:

- `MemoryNeuron`: một ký ức transition/action gồm embedding observation, action, next-state, reward EMA, TD delta EMA, prediction error, utility, strength, forget gate, plasticity và recurrent state.
- `MemoryCluster`: pattern memory gom các ký ức gần nhau thành cụm centroid/action prototype, tương ứng tầng trừu tượng đầu tiên.
- `StrategyMemory`: gom nhiều cluster thành một memory chiến lược có context/action prototype, confidence, plasticity và recurrent state.
- `MetaStrategyMemory`: gom nhiều strategy thành meta-strategy có trust, growth bias và prune bias.
- `MemoryConnection`: cạnh ký ức có loại `remember`, `forget`, `plastic`, `recurrent`, `lateral`, kèm weight, plasticity, utility và age.
- Mỗi `observe_transition()` hấp thụ transition vào siêu mạng, tự tăng node/cạnh khi có kinh nghiệm mới, tự gom node thành cluster, cluster thành strategy, strategy thành meta-strategy, tự decay/quên, cập nhật trọng số theo utility/eligibility và prune khi vượt giới hạn.
- `ActionRecoveryLearner` dùng schema V12 và định danh lifecycle theo `position_ticket` (fallback `deal/order ticket`). Vì bridge close và final execution ledger có thể gọi hai đường phản hồi khác nhau, cả hai đường đều quy về cùng một lifecycle key và chỉ ghi một mẫu real. Artifact recovery V10/V11 được migrate bằng cách giữ prototype/mean nhưng giảm một nửa support/variance để loại ảnh hưởng duplicate; marker lifecycle được làm mới và không bị xoá khi reset intraday. Policy core, graph và expectancy V14 không bị xoá.
- Kết nối gồm cả ngang và sâu: `lateral/recurrent/plastic/remember/forget` ở tầng ký ức, `abstract` đi lên tầng cao hơn và `down` đưa ảnh hưởng từ tầng cao xuống tầng thấp.
- Khi `act()` hoặc `activate()` chạy, siêu mạng query trạng thái hiện tại, tìm cluster/strategy/meta-strategy gần nhất và tạo memory context/action bias/gate bias nhỏ có clamp để ký ức tốt có thể ảnh hưởng quyết định nhưng không lấn át policy gốc.
- Memory schema v10 không so cosine trực tiếp trên vector bridge. Runtime MT5 phát 137 input: 58 market, 40 position, 8 account, 11 execution, 4 timing và 16 Wyckoff. `MarketEncoder` giữ key 32 chiều nhưng dành riêng 8 chiều pooled cho Wyckoff, 16 chiều cho market/timing lõi và 8 chiều cho execution; `AccountEncoder` tạo 10 chiều, `PositionEncoder` tạo 6 chiều, tổng `MemoryKey` là 48 chiều. Việc bump schema làm journal/memory cũ không được trộn vào không gian mới. Khi không có position, similarity ưu tiên market/account; khi có position, key bổ sung lifecycle position để ký ức học đúng hoàn cảnh vào lệnh, Spring/Upthrust, retest và kết quả sau đó.
- Audit parity của MT5 model 4 vẫn fail-closed cho toàn bộ vector. Vì CSV tham chiếu là bar-level còn MT5 real-tick có volume động, chỉ sáu Wyckoff volume features đầu tiên (index 121–126: relative volume, volume trend, effort/result và absorption liên quan) được áp dụng biên tuyệt đối có tên `dynamic_volume_parity_atol=2.5e-4`; mọi feature khác vẫn dùng `atol=1e-6`, `rtol=1e-5`, và lệch vượt biên vẫn loại candidate.
- Reconciliation lấy lifecycle đã đóng từ execution ledger. Khi tester hết kỳ còn position mở, mark cuối trong bridge cung cấp số position và floating PnL để đối chiếu report; mark này không được đưa vào expectancy/memory như một lifecycle policy vì chưa có receipt đóng vị thế.
- Context-expert settlement chỉ có một đường canonical: facade truyền trace entry bất biến cùng action thực thi thật từ receipt MT5 vào `ChuyenGiaPolicy.apply_execution_lifecycle_credit()`. Receipt `trace + ticket + close + R` được khử trùng trước credit; sau đó network, expectancy action-local, reasoning, causal ledger, memory, micro-growth và action-recovery chạy đúng một lần và phát `lifecycle_credit_ack`. Với SL/TP `SYSTEM`, runner giữ route owner ngay từ receipt entry `ACCEPTED/PLACED` và fallback bằng `position_owner_trace_id`, nên close trace rỗng không làm mất lifecycle online. Chuỗi wrapper lifecycle V11/V13/V18/V19/V24/V9 cũ đã bị xóa, nên không còn đường lồng nhau có thể nuốt lỗi hoặc ghi credit hai lần.
- Account context chỉ chứa feature chuẩn hoá như `equity_to_balance`, `floating_pnl_r`, equity/balance drawdown, `open_risk_ratio`, equity return/volatility, losing streak, time-under-water, position count và day-trade pressure. Không lưu raw balance/equity history làm memory key.
- `M1ThinkingModule` nhận context này vào nhiều điểm: working-memory candidate, predictor next-state, critic/selector score, hypothesis action candidate và selected action. Như vậy supernet không chỉ bias output cuối mà trở thành nguồn memory chính cho module tư duy.
- Khi `propose()`/growth chạy, depth/meta-strategy trong siêu mạng tăng nhẹ representation/growth pressure; nếu ký ức xấu chiếm ưu thế thì prune pressure tăng nhẹ. Mọi thay đổi cấu trúc vẫn phải qua MT5 fitness/promotion.
- Journal MT5 chỉ là nguồn chống mất khi tắt giữa chừng; lần train sau journal được replay và hấp thụ vào `ExperienceSuperNetwork`, không còn chỉ là replay rời rạc.
- Trong lúc position đang mở, Python bridge giữ tracker theo `position_ticket` cho từng expert owner. Tracker cập nhật điểm thuận lợi nhất và bất lợi nhất theo `unrealized_r` cùng snapshot observation/context tại hai điểm đó. Khi market context đổi trước khi lệnh đóng, hoặc khi position biến mất khỏi snapshot, bridge emit event `position_context_extrema_until_context_change` về đúng expert sở hữu lệnh, không gửi cho expert active mới. Điểm bất lợi nhất sau entry được dùng như mốc entry tốt hơn: BUY học context gần đáy hơn, SELL học context gần đỉnh hơn, với `entry_timing_gain_r=max(0,-low_r)` thay vì phạt âm. Entry timing chỉ được ghi thành evidence dương khi có hồi xác nhận sau extreme (`recovery_after_low_r >= 0.15`); event cũng lưu `bars_from_segment_start_to_low`, `bars_from_segment_start_to_high`, `giveback_from_high_r` và parameter hint như `prefer_limit_entry`, `entry_offset_reference_r`, `wait_bars_reference`. Khi policy gặp lại observation tương tự và đang chọn entry, memory chỉ ghi soft evidence/preferred pending action vào trace để policy tự học; nó không đổi action, không đổi offset và không chặn hướng entry. Điểm thuận lợi nhất được ghi như evidence quản lý TP/trailing (`MANAGE_POSITION`). Các bản ghi `memory_extrema` là derived evidence/audit, không được tính như lifecycle policy trong expectancy fitness. Đây là bias cục bộ theo context tương tự, không phải rule chặn entry toàn cục.
- Mọi market-regime expert (`TANG/GIAM MANH`, `TANG/GIAM YEU`, `RANGING`, và ba nhánh `VOLATILE_*`) đều nhìn thấy đủ 6 action entry BUY/SELL market và pending. Nhãn xu hướng chỉ là input để học, không phải BUY/SELL gate; context debounce cũng chỉ được ghi telemetry và không loại bỏ lượt học đầu tiên. Quyền management/close vẫn tách theo owner position, còn strong-trend guard chỉ bảo vệ action thoát sớm, không chặn entry. Position cùng chiều được bảo vệ khỏi `CLOSE_BEST`/`CLOSE_ALL` khi chưa có bằng chứng thoát; `CLOSE_WORST` cũng bị loại nếu không có lệnh lệch chiều hoặc lỗ khẩn cấp. Quyền close được mở lại khi context chuyển, position lệch chiều, `unrealized_r <= trend_manh_lo_r_cho_phep_thoat`, gần TP, hoặc giveback đạt `trend_manh_giveback_r_de_thoat` trong ít nhất `trend_manh_bars_sau_mfe_de_thoat` bar. Khi guard đang giữ trend và policy chọn `MANAGE_POSITION`, output được giới hạn thành trailing-only để không biến quản lý thành partial/full close. Một close vi phạm điều kiện này được gắn tag `premature_strong_trend_exit`; mistake replay tạo replacement `MANAGE_POSITION` thay vì tiếp tục củng cố `CLOSE_*`.

Timing source trong context audit luôn ghi `timing_source=TimingNet` khi mạng được bật. Nếu profile tắt mạng, audit ghi `TimingNet_disabled` và trả tín hiệu trung tính; không có fallback rule-based. Vì vậy q/direction chỉ được kết luận cùng với `timing_network_update_count`, trace TimingNet và frozen MT5 A/B, không suy ra từ một nhóm context đơn lẻ.

`M1` hiện có metadata các thành phần theo tài liệu gốc: `state_encoder`, `working_memory`, `hypothesis_bank`, `predictor`, `critic`, `selector`. Runtime hiện thực hóa ở mức module recurrent gated; việc tách từng thành phần thành sub-module độc lập vẫn là bước tiếp theo.

`neural_graph.py` là graph runtime experimental của M1; cấu hình/CLI production giữ legacy làm baseline cho đến khi paired MT5 A/B chứng minh graph tốt hơn. `NeuronNode`, `ConnectionEdge`, `NeuralRegion` và `NeuralGraph` giữ activation/gate clamp, budget neuron/edge/recurrent/lateral/cross-module/M1-module, whitelist kết nối, invariant, deep-copy, serialization migration, rewire/split/merge và `GraphActivationTrace`. Whitelist runtime hiện mở cho các vùng học được `encoder/memory/hypothesis/predictor/critic/selector/meta/output`; `input` vẫn là sensor từ MT5 nên không phải target ghi trực tiếp. Runtime activation không tự thay topology.

Trong `ModuleState`, input vẫn giữ contract raw v13 mặc định. STP có thêm `learning.stp_input_clip`: giá trị `50` tương đương đường cũ, còn candidate anti-overfit có thể giới hạn tail trước khi cập nhật `p_stp`, giảm khuếch đại tail và bão hòa hidden mà không fit thống kê trên OOS. Giá trị này phải được ghi trong run contract và candidate artifact riêng; không được sửa seed khóa.

Sau action head, `ChuyenGiaPolicy` có lớp output tùy chọn `rms_v1` hoặc `layernorm_v1`. `rms_v1` chỉ giảm RMS của 12 điểm action khi vượt target; `layernorm_v1` center rồi scale 12 điểm action theo thống kê của chính forward pass hiện tại. Cả hai đều giữ nguyên 8 tham số SL/TP/quản trị, không lưu running statistic và không dùng kết quả trade để fit chuẩn hóa; `raw` là mặc định tương thích v13. Đây là lớp ổn định biên độ/độ tin cậy giữa expert, không phải cơ chế sửa hướng BUY/SELL: vì phép biến đổi dùng cùng scale dương cho cả action block nên không được coi là bằng chứng rằng action âm expectancy đã được sửa.

Selector có thêm hai núm penalty theo loss streak: penalty theo utility key/expert-side và penalty theo policy-group/role-side. Chúng chỉ trừ điểm khi cùng scope tiếp tục thua, không xóa action khỏi contract; CLI audit candidate mặc định để 0 để cô lập thí nghiệm, còn run muốn kiểm tra khả năng giảm chọn expert thua phải ghi rõ hai giá trị vào `selector_contract`. Gate expectancy action-local là một thí nghiệm riêng, không được gộp với chuẩn hóa khi đánh giá nguyên nhân.

Expectancy cũng tách lifecycle entry khỏi exit/management. Realized R của cả vị thế là bằng chứng trực tiếp cho entry, nhưng không đủ để kết luận `CLOSE_*` hoặc `MANAGE_POSITION` tốt/xấu nếu chưa có counterfactual replay. Vì vậy credit của action `exit/manage` được ghi với origin `management_shadow`: vẫn xuất hiện trong audit và thống kê riêng, nhưng không tạo `expectancy_action_bias` hay inhibition lên policy entry. Điều này ngăn một lệnh entry thua làm hỏng toàn bộ chuyên gia quản lý lệnh.

Backend mặc định là `direct_rnn`: `SelfGrowingPolicy.act()` lấy output 20 chiều trực tiếp từ RNN/deep recurrent stack, sau đó đi qua memory/risk/action-parameter heads và validation trước khi gửi MT5. `M1GraphModule` vẫn giữ API `forward/copy/reset_state/resize_hidden` cho backend tương thích `m1_graph`, nhưng khi backend là `direct_rnn` thì graph không chạy `forward()`, không ghi active neuron/edge vào `StepTrace` và không nhận lifecycle graph credit. Action bias từ memory bị giới hạn; không backend nào được bypass EA hard risk/action contract. Trước khi chọn action, regime expert có thể thêm `context_direction_prior_strength` tối đa 0.25 vào score: `TANG_*` ưu tiên nhẹ BUY, `GIAM_*` ưu tiên nhẹ SELL; riêng `VOLATILE`, timing direction khác 0 từ q tối thiểu `volatile_timing_direction_min_quality` cũng được dùng như evidence mềm vì ledger V13 cho thấy subgroup timing q=0 âm rõ rệt. Đây chỉ là prior mềm, không phải mask; cả sáu entry BUY/SELL market và pending vẫn hợp lệ và telemetry ghi rõ delta đã áp dụng.

Mỗi lần train hoặc frozen OOS trên MT5, runner phải rebind toàn bộ `he_chuyen_gia.cau_hinh` hiện hành vào facade, classifier, registry và context owner trước khi gọi `configure_learning_integrity()`. Nếu chỉ cấu hình backend graph hoặc chỉ cập nhật learning trên direct-RNN, artifact pickle có thể giữ `context_direction_prior_strength`, topology flag hay action-recovery threshold của phiên cũ; khi đó hai A/B dùng cùng candidate nhưng thực tế chạy hai policy contract khác nhau. Candidate chỉ được so sánh sau khi telemetry xác nhận profile runtime đã được áp dụng.

Action-recovery có hai cờ tách biệt: `action_recovery_enabled` chỉ cho phép evidence rerank/thay action trong runtime, còn `action_recovery_learning_enabled` quyết định policy có tiếp tục hấp thụ shadow/counterfactual và lifecycle evidence hay không. Anti-overfit run có thể tắt thay action nhưng vẫn học evidence; frozen OOS phải tắt cả hai để không mutation policy.

`GraphPlasticityEngine` chạy trong `SelfGrowingPolicy.observe_transition()` cho graph backend. Credit từ reward/cost/TD/prediction error/account stress/lifecycle tag chỉ tác động node/edge active. Negative structural pressure chỉ disable/prune trên candidate mode; baseline chỉ nhận audit/prior feedback. Với backend `direct_rnn`, outcome âm dùng pairwise margin cục bộ trên đúng hidden neuron của decision: giảm logit action gây lỗi và tăng action thay thế theo nguyên nhân, ví dụ `CLOSE_ALL -> MANAGE_POSITION`; không hạ action đó trên toàn bộ context. Loss của pending entry đầu tiên không còn tăng `HOLD`: nếu runner-up tại chính decision đó cũng là entry hợp lệ thì credit tăng runner-up làm counterfactual gần nhất, kể cả khi nó khác side/style; nếu không mới fallback sang đổi cùng-side `LIMIT <-> STOP`. Nếu entry bổ sung được chọn khi đã có position rồi lifecycle đó lỗ, replacement credit chuyển sang `MANAGE_POSITION` với gain cục bộ `3x` để model thoát quán tính xếp chồng đã tích lũy trong artifact; đây là nhãn học theo outcome, không phải runtime mask/veto, nên mọi entry vẫn hợp lệ ở quyết định tiếp theo. Short-term recurrent plasticity có decay/clamp, utility cao được consolidation và invariant/finite-state gate chặn candidate lỗi trước promotion.

`GraphSupernetworkRuntime` ghép `ModuleErrorTrace`, error cluster, `ModuleNeedSignal`, `ModuleSpec`, `GraphDevelopmentController`, `CrossModuleConnector` và `ModuleRolePrior`. Nó đọc trace thay vì rule `winrate thấp -> module X`, tạo specialist module/node cùng cross-module edge yếu trên **candidate copy**, ghi `candidate_id/need_reason/created_by/src_module_id/dst_module_id`, rồi để replay + MT5 workflow quyết định promote/reject. `CrossModuleConnector` không chỉ nối theo role guide hẹp: nó thêm fanout dẻo tới nhiều vùng học được, peer fanout hai chiều tới module candidate/active khác và self-loop hồi tiếp yếu cho module mới, trong khi output/action bias vẫn bị clamp bằng bound riêng. Role prior/cooldown chỉ cập nhật sau outcome; graph baseline không bị mutation bởi candidate reject.

Tổng số neuron và connection của graph không còn bị chặn bởi `max_neurons` hoặc `max_connections`; hai field này chỉ còn để tương thích artifact/config cũ. Các guard chuyên biệt như recurrent, cross-module, M1-module và cadence grow mỗi trial vẫn còn để tránh kết nối nguy hiểm hoặc tăng trưởng quá dày trong một vòng.

`TrialMicroGrowthController` chạy bên trong graph candidate trial sau khi `observe_transition()` nhận outcome đã đủ điều kiện action feedback. Nó tạo micro-neuron yếu với `context_prototype`, nối bằng cùng connector dẻo/hồi tiếp, chỉ tăng support khi context sau đủ gần và có credit dương; context xa không làm fade. Context gần nhưng inactive/credit âm sẽ làm gate/weight fade và có thể prune. Baseline policy không gọi micro-growth.

Candidate cấu trúc context có outcome âm được chuyển sang lifecycle hợp lệ `pruning` và giảm `fade/gate`; `LAM_MO` chỉ là trạng thái audit của candidate, không phải lifecycle topology. Graph invariant không chấp nhận chuỗi lifecycle tùy ý như `fading`, nhờ đó credit cuối lượt không thể làm hỏng checkpoint trước khi lưu.

`reasoning_graph.py` cung cấp `ReasoningGraphSystem`, một graph workspace chung cho các mạng suy luận phụ. Graph này có node/edge cho `curiosity`, `development`, `risk`, `action_head`, `utility`, gồm cạnh input->encoder, cạnh controller->output, self-loop recurrent và lateral peer links giữa controller. `SelfGrowingPolicy` lưu graph này trong artifact, deep-copy cho candidate, tự migrate khi load policy cũ, dùng nó để enrich context trước khi gọi các controller và credit utility/gate sau `observe_transition()` hoặc `consolidate()`. Reasoning graph không trực tiếp chọn action và không chạm EA/hard risk/action schema.

`graph_metrics.py` ghi graph size, cross-module/M1-module edges, utility/gate, recurrent health, plasticity health, module-need score và outcome role/cluster vào graph trial-history JSONL. `graph_supernetwork_runtime.py` tổng hợp growth accept rate và role/cluster outcomes; metric MT5 chỉ có giá trị kết luận khi report tester thật tồn tại.

`supernetwork_blueprint.py`, `adaptive_supernetwork.py`, `lineage.py` và `child_supernetwork_manager.py` xử lý child không clone. Blueprint chỉ chứa prior/rule/budget/objective; spawn tạo id, seed, topology, graph/memory/controller/plasticity riêng. Child phải qua diversity/lineage budget, warm-up từ journal schema v6 đã lọc, replay và MT5 promotion contract trước khi selector nhìn thấy nó. `PolicyEnsembleSelector` chỉ chọn policy id có trace/fallback M1; không override EA/risk gate.

Backend graph là opt-in và có thể chọn rõ bằng `--m1-backend graph`; production config mặc định legacy cho đến khi graph thắng paired MT5 A/B. `SelfGrowingPolicy.ensure_compatible()` giữ artifact thiếu backend ở legacy, còn policy khai báo graph sẽ repair/migrate M1 legacy nhất quán. MT5 smoke thật vẫn là bằng chứng deploy bắt buộc.

Growth controller đã có `GrowthSignal(error, novelty, conflict, representation_need, complexity)` để quyết định clone specialist hoặc mở rộng chiều sâu hidden, thay vì chỉ random. `prune_low_utility` có sẵn cơ chế loại module lifecycle `pruning`, nhưng merge/split/ablation utility vẫn chưa hoàn chỉnh.

`SpatialWorldPolicy` là core optional, bật bằng CLI `--policy-core world`. Core này kế thừa interface của `SelfGrowingPolicy` để MT5 runner vẫn gọi `act -> observe_transition -> develop` như cũ, nhưng thêm `WorldState`:

- Mỗi module là một `NationState` có `pos3` Euclidean, `hpos` hyperbolic, `energy`, `knowledge`, `trust`, `utility`.
- Neuron ẩn được biểu diễn bằng `NeuronNode`; tuyến trao đổi giữa module là `EdgeRoute`.
- `WorldState.message_weight()` kết hợp khoảng cách 3D, tương đồng hyperbolic, energy và trust để tạo ảnh hưởng liên quốc gia.
- World influence điều chỉnh gate và action trace; transition update energy/knowledge/trust qua TD delta, reward và prediction error.
- Core này không đổi EA và không đổi bridge schema.

`compare-cores` là orchestration riêng để chạy hai core như hai thế giới độc lập:

- Core `current` và `world` chạy song song bằng hai MT5 runner-dir khác nhau.
- Mặc định từ `mt5_v2_runner` sinh `mt5_v2_runner_current` và `mt5_v2_runner_world` nếu chưa có.
- Sau train, hai core chỉ trao đổi positive replay experience; không copy module, neuron, edge, gate hay world state qua nhau.
- Replay candidate phải qua MT5 validation của chính core đó trước khi được giữ.

## MT5 bridge

`mt5_runner.py` ghi `SelfGrowing_MT5_Trainer_EA.set`, tạo `.ini` tester riêng, chạy terminal và phục vụ action cho EA:

```powershell
terminal64.exe /portable /config:<ini>
```

Runner có chế độ đóng MT5 qua `--mt5-close-scope trial|round|never`. Ở scope `trial`, runner ghi `ShutdownTerminal=1` để Strategy Tester đóng sau mỗi trial. Ở scope `round`, từng trial ghi `ShutdownTerminal=0`, runner giữ terminal trong round rồi cuối round gọi terminate/kill fallback nếu MT5 không tự thoát. Scope `never` giữ MT5 mở để debug một lượt đơn lẻ. Riêng Visual train nhiều pass sẽ override `round`/`never` về effective `trial`, vì MT5 portable single-instance có thể nuốt `/config` kế tiếp và không ghi report.

EA đóng vai trò bridge thuần. Chiều MT5 sang Python là `self_growing_request.csv` schema v8 gồm `schema_version`, `event_id`, `sequence`, account state, `drawdown_pct`, `spread_points`, position state, order state, execution pipe và input vector. Chiều Python sang MT5 là `self_growing_action.csv` schema v8 gồm `schema_version`, `event_id`, `trace_id`, action, score, confidence, `risk_pct`, output vector 20 chiều và metadata hiển thị `market_context`, `expert_id`, `policy_id`, `expert_state`, `context_reason`, `no_trade_reason`. EA dùng metadata này để hiển thị trạng thái thị trường và chuyên gia đang hoạt động trực tiếp trên chart Strategy Tester. Runner chỉ gọi `act()` một lần cho mỗi request; action gửi cho EA lấy từ đúng `StepTrace` đó. Khi request kế tiếp tới, runner dùng observation kế tiếp làm `next_obs`. Nếu MT5 xác nhận lifecycle đã đóng, feedback online dùng trực tiếp `net P/L / initial_risk` có clip `[-2R, 2R]`, cùng đơn vị với final lifecycle credit. Nếu acknowledgement `CLOSE_*` đến trước và chưa có `initial_risk`, runner ghép đúng `position_ticket` với final `DEAL_ADD` receipt rồi mới cấp credit, không settle sớm bằng `profit / account_equity`. Spread, commission, swap và drawdown do chính lệnh đã nằm trong net P/L nên không bị trừ lần hai; runner chỉ gọi `observe_transition()` cho exit trace và entry-owner trace tương ứng. Lifecycle không được mở/cắt edge, thay đổi alpha-gate hay topology.

Sau mỗi transition, runner append realtime vào `outputs/mt5_experience_journal.jsonl` hoặc file đặt bằng `--experience-journal`, gồm `trace_id`, action, `action_details`, `positions_pipe`, `account_context`, reward, cost, TD delta, prediction error, next observation và snapshot `StepTrace`. Nếu MT5/tester bị tắt giữa chừng, các transition đã flush vẫn còn trong journal. Lần train sau replay tối đa `--replay-journal-max` transition có reward hoặc TD delta dương vào policy trước khi tạo trial; replay này cập nhật action/predictor/critic và được hấp thụ vào `ExperienceSuperNetwork`. Với nguồn `mt5_online`, replay chỉ nhận entry có `memory_key_schema=6` để không trộn key space v6 với journal/artifact cũ. Promote policy vẫn phải qua MT5 validation/promotion như các thay đổi khác.

Profile MT5 hiện có `learning.mt5_online_update_scale` (YAML mặc định `0.35`): scale này chỉ làm chậm plasticity/weight/head/selector update trên cửa sổ train ngắn; lifecycle, replay và evidence vẫn ghi đầy đủ để policy còn học mà một phiên may mắn không ghi đè seed.

Plasticity dài hạn của `ModuleState` dùng correlation contract riêng, tách khỏi
feature contract của forward. Candidate chuẩn hóa pre bằng `soft_clip_v1`,
post qua `tanh`, rồi giới hạn norm mỗi update và bound `w_in` trong `[-4, 4]`.
Hebbian/positive Oja drive là phần reward-gated; anti-Hebbian, Oja norm
correction, L2 decay và gradient của loss giữ dấu ổn định nên reward âm không
thể biến cơ chế hãm thành khuếch đại. `plasticity_last_stats` lưu norm từng
thành phần cùng `update_count`, EMA và variance để truy nguyên overfit theo
module; `GraphPlasticityReport` xuất tổng norm tương ứng cho graph backend.
Legacy artifact được migrate về mode `raw`, còn candidate phải ghi rõ
`learning.plasticity_input_normalization_mode` và `learning.plasticity_update_clip`.

Từ candidate V152, `groupwise_v2` là chuẩn hóa không thống kê và nhân quả:
market/timing/Wyckoff vẫn soft-clip theo score ATR, còn position/account/
execution được nén signed-log; `chop_index` và hai counters `bars_since_*`
được đưa về miền có nghĩa. Seed v13 vẫn `raw`. Với candidate legacy chưa bật
full RL, Direct-RNN có thể dùng encoder credit entry-only theo batch module ×
action, phải qua ngưỡng advantage và sign coherence. Khi bật
`direct_rnn_full_rl_enabled`, đường này bị tắt hoàn toàn: lifecycle canonical
tạo batch policy/value/entropy và backprop qua chính cache forward của toàn bộ
sáu hidden block.

Từ candidate sau V152, đường học online của `direct_rnn` có đúng một writer
cho các ma trận train thường (`w_in`, `w_rec`, `w_out`, `b_hidden`, `b_out`,
gate và value head, cùng `deep_w/deep_rec/deep_b`):
`ModuleState.apply_rnn_adamw_cosine()`. Candidate full-RL chỉ gọi writer này
sau batch lifecycle đã settle; các đường plasticity định kỳ, fitness credit,
lifecycle action-margin, lifecycle encoder và rejection attribution bị từ chối
ở optimizer để không có writer thứ hai. Forward cache lưu đúng normalization,
dropout, gate và recurrent state của entry. Loss policy/value/entropy backprop
đủ chiều sâu sáu block nhưng dùng TBPTT=1 theo thời gian; lateral link,
TimingNet, memory và overlay execution là stop-gradient. Nếu overlay đổi
winner action, chỉ value loss được học để tránh credit-routing sai. Optimizer
giữ moment Adam, bias correction, global-gradient clip, decoupled weight decay
và cosine annealing theo số optimizer step. Với config candidate hiện tại,
learning rate bắt đầu ở `0.0005`, giảm mượt về `20%` trong chu kỳ 256 step rồi
restart; telemetry lifecycle ghi raw-gradient norm, clip factor, cosine factor,
effective learning rate, update norm và `direct_rnn_full_rl_batch`. Khi
`direct_rnn_adamw_enabled=false`, artifact v13 được bảo vệ không thực hiện
fallback update tay của encoder. Structural proposal/causal mutation vẫn là
một nhánh candidate-only riêng và phải qua replay + MT5 promotion, không phải
update online của policy đang chạy.

Candidate V193 tiếp tục dùng Transformer Q/K/V nhân quả và TensorRuleBank độc lập trong từng expert từ V191,
từng expert. Summary 64 chiều được factor thành structure/timing/regime/risk,
sau đó đi qua sáu action adapter riêng. Lifecycle settled chỉ sửa adapter và
posterior của action đã thực thi; encoder credit mặc định bằng 0. Counterfactual
sáu action và auxiliary delayed mới được cập nhật encoder, trong đó gradient
được route theo họ nghĩa. Reconstruction không chạy trên sample chỉ có PnL.
nhưng đưa TensorRule về `shadow` trong curriculum timing. Profile bật behavior
exploration entry 10%, tắt `train_entry_probe`; rule chưa được sửa live logits.
AdamW Full-RL vẫn là writer duy nhất. Chi tiết ở
`docs/EDGE_DISCOVERY_V193.vi.md`.

Delayed credit của profile V193 được hòa giải với policy hiện tại trước khi vào
batch. Direct-RNN re-forward immutable decision input và causal memory token
bằng trọng số hiện tại; lifecycle actor dùng clipped current/behavior importance
ratio cộng JS-drift gate. Executed action chỉ học action/value head, không cập
nhật encoder chung. Counterfactual HOLD + 6 entry dùng target full-information
`counterfactual_consensus_v3`, không tái sử dụng action do policy cũ chọn và chỉ
truyền tối đa 0,25 gradient vào encoder. Artifact cũ mặc định tắt cơ chế này.

Từ TensorRule schema 6/artifact 18, head auxiliary tăng từ 73 lên 91 target và
thêm `enter_now_probability`, `timing_advantage_r`, `optimal_delay_fraction`
cho mỗi sáu action. Preference trưởng thành cuối tester vẫn được
TensorRule pre-accumulate độc lập theo microbatch 32. Effective batch vẫn là 64
và direction có norm budget 0,50; không có optimizer thứ hai. Actor batch vẫn
giữ quota lifecycle, còn tensor direction chỉ được writer AdamW lifecycle kế
tiếp tiêu thụ. Cờ `tensor_rule_preaccumulated` ngăn cùng evidence cập nhật bank
lần hai.

Control surface Java V193 giữ sáu command contract thành các tab độc lập. Mọi
workflow có optimizer writer (candidate, fresh, champion-shadow và train fold
walk-forward) bị khóa vào `configs/he_chuyen_gia_latest_ui_v193.yaml`; candidate bắt buộc profile
`edge_discovery_v193_candidate.yaml`. Config người dùng chỉ còn tác
động lên seed/frozen/admin không học. Runtime preflight kiểm tra lại invariant
này trước khi tạo process. Mọi workflow ghi vẫn buộc output mới hoặc bản copy
khác source; direct online-resume không được mở lại. Live 3D tiếp tục chỉ đọc
telemetry. Build Java chạy command-contract self-test trước khi đóng gói EXE.

Mỗi lần AdamW thật sự ghi, event `direct_rnn_full_rl_batch` còn chứa
`direct_rnn_block_updates` theo module. Mỗi block có fingerprint trước/sau,
`delta_l2`, `delta_linf`, tỉ lệ delta theo norm cũ, số phần tử đổi và số phần
tử của block. Các block gồm `encoder` (`w_in/w_rec/b_hidden`), từng
`hidden_layer_N` (`deep_w/deep_rec/deep_b`), `action_head` (rows 0..11),
`risk_head` (row 12), `geometry_compatibility_head` (rows 13/14),
`execution_parameter_head` (rows 15+), `value_head`, `gate`, lateral và
conditioned-representation head khi có. Fingerprint này được tạo từ giá trị
tensor thực, không dùng `weight_hash` của reasoning graph legacy. Vì V13 fixed
geometry đóng băng rows 13/14, fingerprint trước/sau và mọi delta của
`geometry_compatibility_head` phải bằng nhau/0; sai lệch là lỗi contract.

Mọi owner gốc dùng cùng `direct_rnn_rl_batch_size=8`. Không còn override batch
riêng theo context và không còn saturation regularizer cục bộ: việc có ít hơn
một batch hoàn chỉnh không được phép làm một expert bị bỏ lại chỉ vì nó mang
nhãn regime khác. Hidden normalization và dropout ở mọi tầng đã cấu hình vẫn bắt buộc trong
forward/backprop; observability dùng trace action/lifecycle chuẩn thay vì một
nhánh debug làm đổi cadence học.

Khi cả `direct_rnn_full_rl_enabled=true` và `direct_rnn_adamw_enabled=true`,
đây là một ranh giới cứng chứ không phải quy ước: chỉ
`apply_rnn_adamw_cosine(..., source="full_lifecycle_rl")` có thể ghi vào
`w_in/w_rec/w_out`, sáu hidden block, `value_w/value_b` và `gate_w/gate_b`.
`mutate`, Hebbian/Oja/STDP, homeostasis, fitness credit, bridge feedback,
causal-memory credit, curiosity proposal và parameter-replacement lifecycle
đều dừng trước khi ghi Direct-RNN. Điều này vẫn đúng khi `_learning_enabled`
được tắt: Full-RL lúc đó đóng băng thay vì rơi về writer legacy. Trong online
Full-RL, transition chỉ được giữ như execution-trace tạm thời cho canonical
settlement; `action_recovery`, selector outcome, expert history/fitness,
reasoning graph, curiosity, causal/archive, ExperienceSuperNetwork,
partitioned memory và trading-world model đều không được ghi. Telemetry chỉ
ghi snapshot/event độc lập, không trở thành gradient hay state ra quyết định.

`NeuralActionParameterHead` và `NeuralRiskCostGate` không có forward cache hay
VJP trong loss Full-RL hiện tại. Vì vậy chúng bị tắt cả cập nhật độc lập lẫn
overlay runtime trong Full-RL; output 20 chiều của Direct-RNN đi thẳng tới
normalizer/decode. Không coi risk gate cũ là một regularizer vì một recurrent
overlay không nhận gradient có thể đổi lệnh mà optimizer không nhìn thấy. Nếu
cần khôi phục risk gate sau này, nó phải được đưa vào cache forward và
backpropagated như một phần loss batch Full-RL, không được bật lại qua
`observe_transition()`.

TimingNet V6 có thêm tensor rolling 32 M15 và giữ đầu multi-horizon `(3, 6, 12)` dự báo **confirmed active
context** ở tương lai. TimingNet vẫn chỉ đọc market/timing/Wyckoff; nhãn
context xuất hiện sau H quyết định mới settle cross-entropy, nên không rò
future bar/PnL vào action tại t. Xác suất horizon gần nhất chỉ block entry khi
calibration nhãn trễ đạt đủ support, precision và Brier theo config; chưa đủ
bằng chứng thì chỉ telemetry. Gate đặt sau selector, không đảo BUY/SELL và
không tác động management/close. Hồ sơ quyết định direct-RNN cũng
ghi attribution local `w_in × action-margin` trên đúng input 137 chiều;
không còn gán nhầm map controller nội bộ của graph thành giá trị feature MT5.

`ExperienceSuperNetwork` ngoài các tầng `MemoryNeuron -> MemoryCluster -> StrategyMemory -> MetaStrategyMemory` còn có `causal_nodes` cho trí nhớ nhân quả theo lệnh. Mỗi transition có detail sẽ tạo chuỗi `state memory -> action node -> position node -> outcome node`, với các cạnh `causes_action`, `targets_position`, `leads_to_outcome` và `credit`. Action node giữ `action_name`, `management_mode`, `target_slot`, partial/trailing/SL/TP params và account context trước action; position node giữ ticket/side/volume/entry/SL/TP/unrealizedR/age; outcome node giữ reward, cost, TD delta, prediction error, utility, account context trước/sau lifecycle và memory schema.

Neuron action credit cũng nhận account stress. Nếu neuron thúc đẩy entry trong lúc drawdown/exposure/floating loss/equity volatility cao và lifecycle sau đó xấu, risk penalty tăng phần credit âm cho neuron đó; quản lý vị thế trong stress tốt sẽ có cơ hội nhận credit thuận lợi hơn qua reward/TD/lifecycle.

Khi tạo trial candidate **không dùng Full-RL**, policy có thể chạy
`apply_causal_memory_plasticity()` trên bản copy trước khi MT5 validate. Hàm
này đọc causal memory xấu như `near_tp_missed`, `gave_back_profit`, utility/TD
âm, rồi chỉnh nhỏ `w_out`, `gate_w`, `w_rec`, `w_lat` và M1
`hypothesis_bank.w_decode`. Trong Full-RL, nhánh này chỉ ghi sự kiện
`direct_rnn_full_rl_legacy_credit_skipped`, không được chỉnh weight; mọi credit
thực hiện phải chờ batch lifecycle Full-RL.

`NeuralDevelopmentController`, alpha edge-gate và lifecycle structural-development hook đã bị loại bỏ. Direct-RNN giữ topology cố định trong train; lifecycle settlement chỉ cập nhật các trọng số được cho phép bởi contract học hiện hành. `CuriosityNeuronSystem` vẫn là mạng recurrent neural dùng để tạo evidence/memory có giới hạn, nhưng không được mở/cắt edge hoặc thay đổi topology của policy đang chạy.

Đường action runtime còn có các mạng phụ sinh học hơn:
- `NeuralActionParameterHead` nhận state, output thô, memory context và gate để sinh lại nhóm output 11..18 trước khi EA decode risk/SL/TP/trailing/partial/mode/slot.
- `NeuralRiskCostGate` nhận observation, output, gate và causal memory để giảm ảnh hưởng action mở lệnh khi risk/cost xấu, đồng thời tăng hold/close nhẹ.
- `NeuralUtilityNetwork` học utility window/structure sau mỗi MT5 fitness và cung cấp structural signal mềm cho module utility/development controller.
- `NeuralPositionOutcomeClassifier` nằm trong `ExperienceSuperNetwork`, đọc lifecycle position features để tạo `lifecycle_tags`; rule MFE/MAE/nearTP/giveback chỉ còn được lưu thành `audit_lifecycle_tags` và làm teacher ban đầu.

Hai head action/risk ở trên chỉ hoạt động trong profile legacy hoặc graph;
Full-RL Direct-RNN vô hiệu hóa chúng theo contract một-writer ở đoạn trước.

Từ V171, ranh giới này được kiểm tra cả ở facade, không chỉ trong
`ChuyenGiaPolicy`: lifecycle online Full-RL không gọi `ChuyenGiaDocLap.ghi_ket_qua`
và không gọi `PolicyEnsembleSelector.observe_outcome`. Selector chỉ được dọn
pending trace và phát event audit `selector_update_skipped_full_rl`; vì thế
không thể biến outcome mới thành ranking làm đổi action ở quyết định sau.
Runner cũng không được ghi position-extrema vào `ExperienceSuperNetwork` khi
Full-RL: `record_position_context_extrema()` trả audit
`full_rl_position_extrema_memory_skipped` và không tạo contextual-expectancy
record/neural cell. Điều này đóng đường writer gián tiếp vốn chỉ lộ ra khi
đối chiếu artifact online V170 với seed.
TimingNet là auxiliary learner tách biệt: nó chỉ nhận label timing/regime trễ,
không có overlay action trong Full-RL; nó không phải writer của Direct-RNN,
value head hay module gate.

Action entry và management có credit khác nhau nhưng cùng writer: entry nhận
final R đúng một lần; trace `POLICY_CLOSE_*` chỉ nhận delayed benefit so với
giữ lệnh thêm horizon. Counterfactual này vào queue Full-RL manager riêng rồi
gọi cùng AdamW/batch của Direct-RNN. Facade chỉ route trace và receipt, không
được biến benefit đó thành selector utility, expert fitness/history hay memory
credit legacy. EA đánh dấu receipt HORIZON là emitted **trước** khi ghi ledger,
nhờ đó `post_exit_horizon_bars`/`post_exit_hold_r` là các giá trị nhân quả thật;
parser chỉ có fallback `age_bars` cho ledger schema 12 cũ đã ghi nhầm horizon 0.
Trong Full-RL, CLOSE_WORST/CLOSE_BEST/CLOSE_ALL đã broker-confirmed được giữ
trong execution-trace cache tạm thời kể cả khi không đủ điều kiện entry-memory,
vì chúng cần sống tới post-exit horizon. Cache này không vào ExperienceSuperNetwork
và không có gradient riêng. Một receipt đơn lẻ có `credit=0` chỉ có nghĩa queue
chưa đủ batch AdamW; execution-trace observatory, không phải callback live kế
tiếp, là nguồn tuyệt đối để đối chiếu ticket → delayed receipt.

`PolicyEnsembleSelector.last_trace`, fallback policy hiện hành và cờ
`_selection_has_non_cold_candidate` là runtime routing/observability, không phải
bằng chứng học. `HeChuyenGiaMotContextMotPolicy.reset_runtime_state()` luôn xóa
chúng cùng trace map trước MT5 session hoặc frozen replay; các map learned
utility/count/win/loss streak không bị reset bởi thao tác này. Nhờ vậy checkpoint
không truyền fallback của tester trước sang bề mặt action của session sau.

Runner còn duy trì `experience_starvation`: `no_trade_streak`, `unfilled_streak`, `cadence_deficit`, `starvation`, `intrinsic_reward`. Một policy `HOLD` khi tài khoản flat và không có pending order được ghi nhận trong telemetry như tín hiệu khám phá, nhưng không được ghi thành lifecycle/reward và tự nó không tạo entry action. Baseline cũ có thể bật `train_entry_probe`, nhưng candidate V188 tắt probe này và dùng behavior exploration trực tiếp trong entry actor để action phát ra có behavior probability/credit rõ ràng. Tín hiệu starvation vẫn được set vào policy trước `act()`, đi vào learning-drive/telemetry và không bỏ qua safety gate MT5/risk.

Policy còn có `outcome_starvation` lưu trong file model: `failed_goal_streak`, `non_improve_streak`, `goal_deficit`, `trial_regression`, `outcome_starvation`, `last_fitness_delta`. Trong `train-mt5`, trial được sinh tuần tự để trial sau nhận áp lực từ trial trước; nếu fitness không cải thiện so với trial trước hoặc candidate không được promote, starvation tăng. Trong `improve` và `train-to-goal`, policy output luôn được ghi lại trạng thái đói; nếu goal chưa đạt hoặc trial_score không hơn baseline_score thì vòng sau dùng policy đói hơn, còn khi đạt goal/cải thiện thì áp lực giảm dần.

Ngoài starvation, policy lưu `rejection_memory` ngay trong artifact `.pkl`. Khi candidate/trial bị reject, `absorb_rejection_feedback()` phân loại lỗi từ reason và metric MT5 rồi cập nhật mạnh nhưng có clamp: lỗi không có lệnh giảm bias HOLD và tăng áp lực entry/khám phá; lỗi lỗ, drawdown hoặc execution tăng quản lý/close/risk patience và giảm entry; đồng thời cập nhật `NeuralRiskCostGate`, `CuriosityNeuronSystem`, `NeuralDevelopmentController` và credit `ReasoningGraphSystem`. Nó còn đọc các `StepTrace` xấu trong rejected policy để phạt trực tiếp active graph edge/node cùng id trên baseline, giảm utility/gate/weight và tăng eligibility/plasticity; với module recurrent thường, nó chỉnh trực tiếp hidden->selected-action `w_out`. Đây là bài học hành vi trên baseline, không phải promote/copy topology của candidate lỗi.

Sau khi MT5 tạo report HTML, runner parse net profit, profit factor, total trades, win rate, average win/loss và drawdown rồi tính fitness. Runner cũng đếm số dòng bridge để xác nhận EA đã gửi telemetry.

Model runtime, recurrent state, gate/module activation, world-core nếu bật, và action selection đều nằm ở Python. EA không đọc weight, không activate model và không tự chọn action.

Từ V179, path `entry_preference` không chỉ tạo cross-entropy cho bảy logits
`HOLD + 6 entry`. Mỗi path M1 đã mature cũng tạo mục tiêu chất lượng mềm cho
từng **entry hợp lệ** theo `(context, action, side)`: `sigmoid(R / temperature)`.
Nó cập nhật `conditioned_quality_*` và encoder sáu tầng qua đúng batch
Full-RL/AdamW hiện hành; các alternative không được chọn vì thế không còn phải
chờ một lifecycle thật mới xuất hiện trong representation. Thử nghiệm V180
từng đưa quality logit vào final action logits, nhưng MT5 cho thấy objective
48-bar đó không ổn định với final lifecycle: dù maturity guard loại random
initialization, action có utility dương giảm và VOLATILE_CHOP xấu đi. Toàn bộ
residual/action-value runtime và support state đã được gỡ bỏ. Vì vậy quality
head vẫn chỉ là auxiliary representation dưới Full-RL/AdamW duy nhất, không
cộng vào logits, không tạo mask/gate/risk rewrite và không tự biến một label
ngắn hạn thành lệnh. Gradient auxiliary được trung bình theo số alternative
của record và vẫn đi qua PCGrad với actor khi chúng đối nghịch. Telemetry batch
tách loss, số candidate quality, positive candidate, gradient L1 và bucket
condition để không nhầm "head đã đổi" với "actor đã có edge".

Artifact candidate còn có thư mục đồng hành `<candidate>.runtime/`. Runner
chụp report, execution ledger, raw feature audit, bridge/request/action và
entry-preference CSV ngay sau MT5 kết thúc, trước khi run kế tiếp ghi đè
`Common\\Files` của portable terminal. Archive này chỉ là bằng chứng forensic
và không được đọc lại bởi inference hay optimizer.

## EA

`SelfGrowing_MT5_Trainer_EA.mq5` chỉ tính input từ MT5, chờ action từ Python, thực thi action và ghi ledger. Contract V13 đặt SL `FixedSlAtr × ATR M1` và TP `FixedTpAtr × ATR M1` tại entry, sau đó khóa tuyệt đối: EA không có trailing, breakeven, partial close hay `TRADE_ACTION_SLTP`. `CLOSE_WORST`, `CLOSE_BEST`, `CLOSE_ALL` và `TIME_SAFETY` vẫn đóng được position; `MANAGE_POSITION` giữ index 11 cho artifact cũ nhưng bị mask/no-op. Schema bridge vẫn 20 output, trong đó rows 13/14 và 16..19 không tham gia execution hoặc learning.

Nếu Strategy Tester kết thúc khi vẫn còn position, report MT5 đã mark-to-market floating PnL nhưng EA có thể chưa kịp ghi final-close receipt. Runner đọc `positions_pipe` ở bridge terminal mark, ghép `unrealized_r`, owner trace, age/MFE/MAE với `initial_risk` và entry provenance trong execution ledger, rồi tạo lifecycle `TESTER_TERMINAL_MARK` cho đúng policy trước khi save candidate. Reconciliation report vẫn dùng ledger đóng + terminal open PnL để không double-count; synthetic lifecycle chỉ lấp feedback bị thiếu và được ghi nguồn `mt5_terminal_mark` rõ ràng.

Facade hiện dùng contract `FIXED_CONTEXT_OWNER_V32`. Với context M15 đã có owner, entry chỉ được phát từ policy của chính owner đó; không có broad `CTX_VOLATILE` parent challenger trong committee của `CTX_VOLATILE_CHOP/SPIKE/EXPANSION`, nên trace, lifecycle và AdamW luôn cùng một policy. Profile active tắt các entry-style specialist; nếu một profile thí nghiệm bật chúng, chúng vẫn là policy tách biệt và phải qua replay/OOS riêng. Position-management specialist chỉ được dùng cho vị thế đang mở, không thay entry owner. V31 evidence leaf/back-off, correlated exposure và trace pending vẫn là telemetry/ranking hỗ trợ, nhưng không có quyền đổi owner. Artifact cũ được migration xóa selector evidence của role `VOLATILE_PARENT_ENTRY_GENERALIST` trước khi resume.

Artifact `.pkl` cạnh registry là checkpoint có thẩm quyền. Registry chỉ là index/audit JSON và không còn ghi lại từng policy vào `outputs/kho_chuyen_gia`; cơ chế cũ không có reader, trộn candidate qua cùng đường dẫn và nhân đôi toàn bộ memory khi serialize. Trước khi lưu candidate, quota learning được áp lại cho cả expert tạo lười và cache trace trùng được compact.

Artifact v13 cũ từng chứa controller từ module `phat_trien_cau_truc_context`, nhưng module và đường runtime `develop()` đã bị loại bỏ. `legacy_pickle_migration.py` chỉ ánh xạ type cũ sang carrier tạm trong lúc deserialise; `HeChuyenGiaMotContextMotPolicy.ensure_compatible()` loại carrier khỏi từng policy trước khi dùng hoặc lưu lại. Vì vậy loader không khôi phục structural growth/alpha gate, còn checkpoint được lưu sau migration không còn phụ thuộc module đã xóa. Fast compatibility cache có version riêng và bao gồm shape của `ModuleState.value_w`; một cache cũ không thể bỏ qua migration value-head trước khi role specialist fork module từ seed v13.

Position owner và decision policy là hai định danh khác nhau. MT5 vẫn ghi owner theo market-context expert để quyền quản lý position không trôi sang context-role specialist; `decision_policy_id`, `decision_role` và proposal id giữ policy thực sự đã tạo action. Trace map ưu tiên decision policy khi học, còn lifecycle của position vẫn truy từ owner trace. Manager chỉ nhận lifecycle R khi trace quản lý mới nhất được liên kết đúng owner trace; shadow proposal bị xóa khỏi execution cache và không được credit. Selector utility có hai tầng: leaf `market_context + exposure_stage + microstate + action_side + proposal_id` để giữ khác biệt cục bộ, và family `decision_role + action_side` làm prior back-off. Gradient policy STOP của `GIAM_MANH` vẫn không sửa weight/memory policy STOP của `VOLATILE_CHOP`; chỉ outcome thống kê cấp family được chia sẻ để hai specialist không lặp cùng một cold-start mù.

Trong đề xuất của position-management specialist, owner guard còn lọc theo trạng thái thật của position đích: `CLOSE_BEST` chỉ mở khi position tốt nhất đạt tối thiểu `quan_ly_doi_context_min_profit_r`, `CLOSE_ALL` chỉ mở khi toàn bộ position của owner đều đạt ngưỡng đó, còn `CLOSE_WORST` chỉ mở ở mức lỗ khẩn cấp. Giới hạn này chỉ thuộc capability management; union của context generalist và ba entry-style specialist vẫn giữ đủ sáu entry action.

Entry volume không còn lấy từ input EA `Lots` hoặc `.set`. Python action wire ghi `risk_pct` đã decode từ output policy; EA tính volume từ `risk_pct`, equity, khoảng cách SL, tick size/value và broker `SYMBOL_VOLUME_MIN/MAX/STEP`. EA chỉ normalize/validate volume rồi ledger reject nếu policy risk tạo volume không hợp lệ.

Trong V32, strong-trend/context-switch exit guard mở sớm thoát lệnh khi lỗ đạt `-0.35R` (hoặc các điều kiện giveback/gần TP đã có). Đây chỉ là quyền thoát được mở rộng để xử lý cụm SL kéo dài; không chặn entry và không sửa risk head hay volume của entry.

`technical_min_rr` là validation của config Python: cặp fixed `fixed_tp_atr / fixed_sl_atr` phải không thấp hơn floor. EA không decode RR từ policy; runtime active dùng `2.5/5.0`, tức RR cố định `2.0`. `StrategyMinRR` vẫn là ngưỡng mục tiêu/báo cáo, không phải gate entry.

Execution ledger V13 giữ `initial_risk` theo lifecycle ticket để Python tính R theo chính lệnh đóng, không suy từ equity toàn tài khoản. EA lưu `initial_entry_price`, `initial_sl_price`, `initial_tp_price` và `initial_risk_distance_price` bất biến, cùng `geometry_mode=fixed_atr_v1`, `fixed_sl_atr`, `fixed_tp_atr`, `fixed_rr`. Không có grid replay, target geometry, `L_geometry`, PCGrad geometry hay mutation SL/TP. Lifecycle entry nhận final R đúng một lần; `POLICY_CLOSE_*` nhận counterfactual hold riêng.

Khi request MT5 sang ngày giao dịch mới, runner gọi `reset_intraday_state(day)` trên hệ chuyên gia. Hook này xóa cache/query ngắn hạn, working memory runtime, account/position context tạm và memory cực trị trong ngày, nhưng không xóa long-term `ExperienceSuperNetwork` hay trọng số policy. Mục tiêu là không để bias của ngày trước kéo sang ngày sau khi regime đầu ngày đổi. Debounce context không còn chặn action ở đầu ngày; candidate expert vẫn phát entry để có dữ liệu học thật.

Từ Retention V2/schema 5, mỗi context owner có một `ExpertRetentionStore` bounded lưu
deep-copy của lifecycle/entry-path đã settle theo recent, diverse và anchor.
Store không có optimizer; nó chỉ đưa replay copy vào batch Full-RL sau khi
batch live đã đạt quota lifecycle, và AdamW vẫn là writer Direct-RNN duy nhất.
Bảy logits entry được tối ưu/ra quyết định theo TRADE/HOLD rồi
action-given-TRADE bằng `logmeanexp`, nên sáu lớp entry không còn lợi thế 6:1
so với HOLD. Target `hierarchical_on_policy_trade_v2` đánh giá TRADE bằng
utility của action policy đã chọn; HOLD chưa có action thực thi dùng mean của
các entry hợp lệ. Hợp đồng `max` hindsight cũ bị loại khỏi pending queue và
retention store khi migrate, nên một entry thua không còn được dạy TRADE chỉ
vì alternative không chọn sau đó thắng. Replay tính lại action head hiện tại
trên cached causal hidden. Candidate V187 còn snapshot HOLD và giữ bucket
preference action `0`, vì nếu chỉ thu sample entry thì chính việc giảm TRADE sẽ
ngắt supervision ở vùng state vừa học và cho phép xác suất TRADE bật tăng lại.
Neural HOLD dùng resolver preference riêng để không bị firebreak
`flat_idle_non_feedback` loại nhầm; transport no-op và safety rewrite vẫn bị
loại. Strong HOLD, hard-negative và lifecycle dương đều có thể thành anchor;
anchor capacity được cân bằng theo bucket. Hai mươi phần trăm anchor được split
ổn định theo context frame thành `audit_only`: chúng không đi vào replay
gradient ở các batch sau. Trước mỗi optimizer step có snapshot tensor cùng
AdamW moments; non-finite, update quá lớn, mean KL hoặc bất kỳ held-out-anchor
KL regression quá ngưỡng làm rollback nguyên tử cả weight, moments và step
counter. Contract đầy đủ nằm trong
`docs/DIRECT_RNN_RETENTION_V1.vi.md`; profile/seed không opt-in vẫn giữ hành vi
legacy.

Retention ở cấp optimizer không được xem là retention của trading edge. Tầng
deployment dùng transaction `champion–shadow matrix` trong
`champion_retention.py`: champion được hash và giữ bất biến; mỗi candidate seed
được train thành artifact riêng; chính từng candidate hash phải qua mọi cửa sổ
frozen OOS tuyệt đối và paired với champion. Promotion yêu cầu ma trận hoàn
chỉnh theo `promotion_min_oos_windows × promotion_min_seeds`, rồi chọn candidate
theo policy-origin LCB tệ nhất. Luồng walk-forward không còn được gom row của
những artifact khác nhau để hợp thức hóa artifact tốt nhất. Contract vận hành
và audit nằm ở `docs/CHAMPION_EDGE_RETENTION.vi.md`.

Context router chốt state từ nến M15 đã đóng; nó cache tuple M15 chuẩn hoá ổn định gồm EMA-spread/ATR-M15, EMA-slope/ATR-M15, gap/ATR-M15 và ATR-M15/ATR-H1 giữa các request M1. Việc chia tỉ lệ này triệt tiêu mẫu số ATR-M1 của wire MTF cũ, nên dao động M1 không thể đổi owner. Không có countdown, fast-path volatile hay trạng thái context-đang-xác-nhận. Nhãn volatile chỉ dành cho biến động M15 cao nhưng hướng gần trung tính, với `volatile_max_directional_strength=0.18`; trend yếu dương/âm được route sang `CTX_TANG_YEU`/`CTX_GIAM_YEU`. Sau đó volatile được tách thành ba owner độc lập: `CTX_VOLATILE_SPIKE` (gap M15 lớn), `CTX_VOLATILE_EXPANSION` (ATR M15 cao) và `CTX_VOLATILE_CHOP` (biến động cao nhưng giằng co). Mỗi owner có policy, memory, lifecycle credit và fitness riêng; policy con được khởi tạo fresh thay vì copy action bias của parent volatile broad, còn artifact cũ được migrate một lần. Cả ba giữ BUY/SELL market và pending; `MANAGE_POSITION` chỉ là row tương thích bị mask, owner guard chỉ lọc `CLOSE_*` non-emergency khi đang có position.

Fitness của hệ chuyên gia đọc expectancy đã gộp từ các ledger riêng của owner, không coi thiếu trường tổng hợp là LCB bằng 0. Scope context/action âm vẫn là hard gate cho promotion; phần trừ trong rank được giới hạn để feedback học không bị bão hòa. `management_shadow` có LCB riêng để audit chất lượng thoát lệnh nhưng không đi vào policy-origin gate/bias nếu chưa có replay. Ở decision-time, LCB âm đủ mẫu được chuyển thành `expectancy_negative_inhibition` chỉ trên đúng context × action entry tương ứng để M1 tránh lặp lại hành động thua; không có suppression toàn cục hay rule no-trade. Khi checkpoint, full `StepTrace` chỉ được giữ trong cửa sổ replay gần nhất; evidence học và audit lifecycle không phụ thuộc vào trace archive này.

EA ghi `self_growing_bridge.csv` với mỗi bar gồm `schema_version`, `event_id`, `trace_id`, `time`, `bar`, `balance`, `equity`, `drawdown_pct`, `spread_points`, `positions`, `day_trades`, `action`, `action_score`, `action_confidence`, `trades_before`, `trades_after`, `inputs_pipe`, `outputs_pipe`, `response_status`, `positions_pipe`. `positions_pipe` chứa snapshot theo ticket để journal truy vết lifecycle. Execution ledger V13 tách intent, entry owner, fixed geometry và exit controller; entry trace nhận final lifecycle R đúng một lần, còn `POLICY_CLOSE_*` dùng `POST_EXIT_OBSERVATION` counterfactual riêng. Rows 13/14 là compatibility-only: không decode, không gradient và delta AdamW luôn bằng 0. Ledger vẫn giữ audit broker clamp, nhưng không tạo replay/target geometry. Deal SL, TP hoặc stop-out broker-triggered luôn để trống intent; credit cuối bám đúng decision-policy owner đã phát action thật.

EA giữ lifecycle tracker theo ticket trong lúc test, cập nhật MFE/MAE, khoảng cách gần nhất tới TP/SL, số bar từ đỉnh lời và mức giveback. Runner gắn cả `positions_pipe` trước action và `positions_pipe_after` ở request kế tiếp vào `StepTrace`, nhờ đó `ExperienceSuperNetwork` nhận biết các pattern như `near_tp_missed`, `near_tp_giveback`, `gave_back_profit`, `closed_after_near_tp_bad` và `should_have_partial_or_trail`.
