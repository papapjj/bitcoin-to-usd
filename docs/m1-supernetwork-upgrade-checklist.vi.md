# Checklist nâng cấp M1 thành siêu mạng tự sinh

> **Cập nhật triển khai: 2026-06-23.**  
> `[x]` = đã có mã nguồn và test/kiểm tra cục bộ; `[~]` = đã có đường code hoặc contract nhưng còn phụ thuộc replay/MT5/EA thật để xác nhận vận hành; `[ ]` = chưa được phép hoặc chưa thể kết luận trong archive. Snapshot cũ phản ánh coverage mã/test, **không phải bằng chứng PnL/MT5 A/B**.  
> Backend graph có trace/credit/candidate/specialist/child path, nhưng `configs/self_growing.yaml` và CLI phát hành hiện giữ `m1_backend=direct_rnn` làm baseline production. Graph chỉ bật qua `--m1-backend m1_graph` hoặc alias `graph`, và chỉ được promote khi report MT5 đối soát execution ledger, paired A/B không xấu hơn baseline, và mọi gate telemetry/risk pass. Không mục nào yêu cầu MT5 live được đánh dấu `[x]` nếu không có terminal, bridge telemetry và report tester thật.

Mục tiêu là biến M1 thành graph neuron/connection có khả năng tự thích nghi, tự phát triển và sinh child supernetwork độc lập mà không clone raw weight/memory/topology, đồng thời vẫn giữ EA bridge, action schema và MT5 promotion gate.

## 0. Nguyên tắc giữ cứng

- [x] Giữ `INPUT_COUNT=119`, `OUTPUT_COUNT=20`, action schema v4 và bridge request/action schema v2 cho đến khi có migration riêng.
- [x] Giữ EA là bridge thuần: EA chỉ tính input, execute action và ghi telemetry; runtime model vẫn nằm ở Python.
- [x] Giữ MT5 Strategy Tester là promotion gate cuối cho policy/candidate được lưu.
- [x] Không promote candidate chỉ vì replay/Python audit tốt nếu MT5 không tốt hơn baseline.
- [x] Không copy nguyên weight, neuron id, connection id, recurrent state hoặc memory state từ parent sang child nếu mục tiêu là "không clone".
- [x] Không cho child/grandchild tự sinh vô hạn; mọi sinh mạng mới phải qua budget, diversity và validation gate.
- [x] Không xoá lifecycle telemetry như MFE/MAE/nearTP/giveback vì đây là nguồn credit assignment và audit lỗi lệnh.
- [x] Không trộn memory key schema cũ với schema v6; journal online phải có `memory_key_schema=6` mới replay vào memory hiện tại.
- [x] Không đưa raw balance/equity tuyệt đối vào memory key; chỉ dùng account context đã chuẩn hoá.
- [x] Không thay thế `mt5_fitness()` bằng mạng neural; fitness vẫn là thước đo khách quan/window-level và safety gate.

## 1. Khảo sát trước khi sửa

- [x] Đọc `.understand-anything/knowledge-graph.json` để xác nhận layer, flow và constraint hiện tại.
- [x] Đọc `docs/architecture.vi.md`, `docs/domain-flow.vi.md`, `docs/overview.vi.md` và checklist này trước khi sửa kiến trúc.
- [x] Chạy search các tên liên quan: `M1ThinkingModule`, `ExperienceSuperNetwork`, `NeuralDevelopmentController`, `ModuleState`, `SpatialWorldPolicy`, `StepTrace`.
- [x] Lập danh sách file chịu ảnh hưởng trực tiếp: `self_growing_runner.py`, `growth.py`, `plasticity.py`, `mt5_runner.py`, tests và docs.
- [x] Xác định phần nào là runtime trading, phần nào là candidate/proposal, phần nào là validation/promotion để tránh sửa nhầm best policy trực tiếp.
- [x] Nếu thay schema artifact `.pkl`, định nghĩa version/migration trước khi sửa.
- [x] Nếu thay action/output bridge, tạo migration riêng và cập nhật cả Python, EA, tests, docs; không làm chung trong phase M1 graph nếu chưa cần.

## 2. Primitive neuron/connection graph

- [x] Tạo module mới `src/self_growing_trader/neural_graph.py`.
- [x] Định nghĩa `NeuronNode` với tối thiểu: `id`, `region`, `bias`, `activation`, `state`, `utility`, `plasticity`, `eligibility`, `age`, `lifecycle`.
- [x] Định nghĩa `ConnectionEdge` với tối thiểu: `id`, `src`, `dst`, `weight`, `gate`, `kind`, `utility`, `eligibility`, `plasticity`, `age`, `enabled`.
- [x] Định nghĩa `NeuralRegion` để nhóm neuron theo role: `encoder`, `memory`, `hypothesis`, `predictor`, `critic`, `selector`, `meta`, `output`.
- [x] Định nghĩa `NeuralGraph` với API: `activate()`, `reset_state()`, `copy()`, `add_neuron()`, `add_connection()`, `remove_neuron()`, `remove_connection()`, `prune_disabled()`.
- [x] Hỗ trợ connection kind tối thiểu: `forward`, `recurrent`, `lateral`, `memory`, `predictive`, `critic`, `selector`, `meta`; foundation cũng có `gating` và `plastic`.
- [x] Hỗ trợ clamp activation/gate để tránh activation nổ.
- [x] Hỗ trợ guard an toàn theo loại: `max_recurrent_edges`, `max_lateral_edges`, cross-module và M1-module edge; không còn hard cap tổng neuron/connection bằng `max_neurons` hoặc `max_connections`.
- [x] Hỗ trợ invariant check: không edge trỏ tới neuron thiếu, không duplicate id, shape output đúng khi truyền `output_size`.
- [x] Hỗ trợ serialization/pickle ổn định.
- [x] Thêm unit tests cho create/copy/activate/add/invariant/pickle và budget cơ bản.

## 3. Mapping M1 hiện tại sang graph

- [x] Viết adapter đọc `M1ThinkingModule` hiện tại và tạo `NeuralGraph` tương đương ban đầu ở `src/self_growing_trader/m1_graph_adapter.py`.
- [x] Map `state_encoder.w/b` thành edge input -> encoder và bias encoder.
- [x] Map `working_memory` thành memory region với recurrent/self-state edges.
- [x] Map `hypothesis_bank` thành hypothesis region và hypothesis -> output/action decode edges.
- [x] Map `predictor` thành predictive region hoặc predictive edges từ state/memory/action.
- [x] Map `critic` thành critic region chấm predicted state/action.
- [x] Map `selector` thành selector region với lateral/recurrent inhibition edges.
- [x] Giữ output behavior gần nhất có thể với M1 cũ trong test parity nhỏ ở mức shape/finite output; parity số học chính xác chưa đặt vì graph activation là foundation riêng.
- [x] Thêm cờ artifact/runtime để biết policy đang dùng M1 legacy hay M1 graph: `SelfGrowingPolicy.m1_backend`.
- [x] Khi load artifact cũ, tự migrate hoặc tạo `M1Graph` từ legacy M1 mà không crash; artifact cũ giữ legacy, graph backend có explicit conversion và load compatibility.
- [x] Không xoá legacy M1 ngay trong phase đầu; chỉ xoá khi graph backend đã pass parity, train smoke và docs đã cập nhật.

## 4. Runtime `M1GraphModule`

- [x] Tạo `M1GraphModule` có API tương thích: `forward(x, goal, context)`, `copy()`, `resize_hidden()`, `reset_state()`.
- [x] Cho memory context từ `ExperienceSuperNetwork` đi vào memory/predictor/critic/selector như M1 cũ; foundation hiện mới dùng `action_bias`.
- [x] Cho action bias từ memory supernet ảnh hưởng nhẹ vào selector/output, có clamp.
- [x] Tính và lưu trace activation của neuron/connection cho credit assignment.
- [x] Trả về output action vector, confidence/score và các error/novelty/conflict tương đương để runner không mất telemetry ở mức wrapper API.
- [x] Giữ recurrent state riêng theo policy/candidate; không share state giữa candidate và baseline qua `copy()`.
- [x] Đảm bảo `copy()` deep-copy graph, memory state và eligibility; không alias numpy array.
- [x] Test parity với seed cố định: legacy M1 và graph M1 không cần giống tuyệt đối nhưng phải cùng shape, finite output, không crash qua nhiều step.

## 5. Credit assignment theo neuron/connection

- [x] Mở rộng `StepTrace` hoặc trace nội bộ để lưu `active_neuron_ids`, `active_edge_ids`, gate, selected hypothesis, critic score và memory match.
- [x] Khi `observe_transition()` nhận reward/cost/TD/prediction_error, phân bổ credit dương/âm về neuron/connection đã tham gia action ở foundation `GraphPlasticityEngine`.
- [x] Gắn credit theo lifecycle outcome: `near_tp_missed`, `gave_back_profit`, `closed_after_near_tp_bad`, `should_have_partial_or_trail`; `GraphCreditSignal.lifecycle_tags` được parse thành penalty có attribution.
- [x] Phạt entry neuron/edge khi account stress cao và lifecycle xấu ở mức `account_stress` và `lifecycle_penalty`.
- [x] Thưởng management neuron/edge khi giảm drawdown, bảo vệ profit, hoặc close/partial/trail đúng lúc ở mức reward/TD dương.
- [x] Cập nhật `utility`, `eligibility`, `plasticity` theo credit; `forget_gate` chưa có trong `NeuronNode`.
- [x] Không update trực tiếp best policy bằng credit xấu nếu đang ở candidate proposal; sửa trên candidate trước rồi MT5 validate.
- [x] Thêm test cho credit không ghi nhầm vào node không active.

## 6. Plasticity nhanh nhưng có kiểm soát

- [x] Tái dùng `plasticity.py`: Hebbian, anti-Hebbian, Oja, gradient-like update, homeostatic bias/scale; `GraphPlasticityEngine` gọi cùng primitive và test finite/active-only.
- [x] Thêm `GraphPlasticityEngine` áp dụng update trên `ConnectionEdge`.
- [x] Gating plasticity theo utility và risk: phần rủi ro cao học chậm hơn hoặc bị dampen qua credit âm/account stress.
- [x] Thêm short-term plasticity cho recurrent/memory edges nhưng phải decay và clamp.
- [x] Thêm consolidation: edge/neuron có utility cao qua nhiều MT5-promoted runs thì giảm plasticity để chống quên ở foundation `consolidate_high_utility()`.
- [x] Thêm decay/prune pressure cho edge/neuron utility thấp lâu ngày ở foundation disable edge utility thấp trong candidate mode.
- [x] Phân biệt update online/replay với update candidate; online chỉ ghi memory/trace, structural change vẫn đi qua candidate.
- [x] Test finite weight sau nhiều update; không NaN/Inf.

## 7. Development controller thao tác trực tiếp lên graph

- [x] Development controller xuất action cấp graph `grow_neuron`, `grow_edge`, `prune_edge`, `consolidate`, `rewire_edge`, `split_region`, `merge_region` với invariant/budget guard.
- [x] Input controller nhận error, novelty, conflict, uncertainty, utility, growth/prune pressure, memory evidence, causal outcome và starvation qua `ModuleNeedSignal`/trace cluster.
- [x] Controller chọn region mục tiêu: encoder/memory/hypothesis/predictor/critic/selector/meta ở mức role -> region map.
- [x] Controller chọn edge kind mục tiêu: forward/recurrent/lateral/predictive/critic/selector/meta ở mức role -> edge kind map.
- [x] Khi thêm neuron, khởi tạo bằng signal role/spec, utility và seed riêng; candidate lifecycle/gate thấp được giữ cho đến promotion.
- [x] Khi thêm connection, chọn source/destination theo utility; `NeuralRoleController` đọc vector need và role prior/cooldown để chọn target.
- [x] Khi prune, chỉ disable trước, xoá thật sau khi candidate được reject/promote rõ ràng.
- [x] Khi split/merge region, vẫn dùng similarity/gradient conflict làm guard an toàn.
- [x] Thêm event log cho structural action: reason, target, before/after neurons/edges và details.
- [x] Test structural action không phá invariant graph.

## 8. M1 biết cần tạo module gì

- [x] Không dùng rule cứng kiểu `winrate thấp -> tạo module X` làm cơ chế chính; rule nếu có chỉ là teacher/bootstrapping ban đầu.
- [x] Mỗi action phải ghi trace đủ để truy lỗi: market state, account stress, position state, memory match, selected hypothesis, critic score, selector confidence, active neurons, active edges, output action, reward, cost, TD delta, prediction error và lifecycle tags.
- [x] Thêm `ModuleNeedSignal` hoặc cấu trúc tương đương để gom tín hiệu thiếu năng lực: `prediction_error`, `novelty`, `conflict`, `critic_miscalibration`, `selector_uncertainty`, `memory_match_quality`, `risk_stress_loss`, `position_giveback`, `cadence_deficit`.
- [x] Gom lỗi thành cluster theo regime và lifecycle, không chỉ theo reward tổng: `near_tp_missed`, `gave_back_profit`, `closed_after_near_tp_bad`, `stress_entry_loss`, `low_memory_match`, `high_prediction_error`, `hypothesis_conflict`.
- [x] Tính `need_predictor` từ prediction error cao, future-state uncertainty và critic score sai sau entry.
- [x] Tính `need_memory` từ novelty cao, memory match thấp, repeated unknown regime và prediction error lặp lại trong cùng key space.
- [x] Tính `need_critic` từ critic score cao nhưng reward/TD âm, hoặc critic quá phạt action tốt.
- [x] Tính `need_selector` từ conflict cao, selector confidence thấp, hoặc nhiều hypothesis tốt nhưng chọn action xấu.
- [x] Tính `need_position_manager` từ near-TP miss, giveback profit, late close loss, partial/trailing không kích hoạt khi nên kích hoạt.
- [x] Tính `need_risk_gate` từ entry trong drawdown/account stress/exposure cao rồi thua.
- [x] Tính `need_cadence_explorer` từ cadence deficit, quá ít lệnh hợp lệ, hoặc unfilled/no-trade streak cao.
- [x] Tính `need_entry_quality` từ nhiều lệnh nhưng expectancy thấp, average loss xấu hoặc entry bị đảo chiều sớm.
- [x] Tính `need_exit_quality` từ win có giveback, loss giữ quá lâu, hoặc close không theo lifecycle tốt.
- [x] Tính `need_memory_critic` khi memory được dùng nhiều nhưng utility không cải thiện hoặc memory bias kéo action xấu.
- [x] Cho `NeuralDevelopmentController` đọc vector need scores và chọn role module bằng neural output, thay vì switch-case cố định.
- [x] Ghi `module_need_reason` vào trial history để audit module mới được tạo vì cụm lỗi nào.
- [x] Khi module type được MT5 promote trong tình huống tương tự, tăng prior/xác suất chọn type đó ở foundation `ModuleRolePrior`.
- [x] Khi module type bị reject, giảm prior/xác suất hoặc tăng ngưỡng tạo lại trong cùng regime ở foundation `ModuleRolePrior`.
- [x] Không cho một cụm lỗi tạo nhiều module cùng role liên tục nếu chưa có promote; dùng cooldown theo cluster/role ở foundation `ModuleRolePrior`.

## 9. ModuleSpec cho module mới

- [x] Chuẩn hoá `ModuleSpec` hoặc cấu trúc tương đương với: `role`, `reason`, `input_sources`, `output_targets`, `initial_gate_bias`, `initial_plasticity`, `budget`, `validation_goal`.
- [x] Hỗ trợ role `predictor_specialist` cho lỗi dự đoán tương lai/state/action outcome.
- [x] Hỗ trợ role `memory_specialist` cho regime lạ, memory match thấp và thiếu context chuỗi.
- [x] Hỗ trợ role `critic_specialist` cho chấm điểm sai reward/risk/cost.
- [x] Hỗ trợ role `selector_specialist` cho xung đột hypothesis và confidence thấp.
- [x] Hỗ trợ role `position_management_specialist` cho trailing, partial close, breakeven, target slot và lifecycle lệnh.
- [x] Hỗ trợ role `risk_gate_specialist` cho drawdown, exposure, spread, losing streak và account stress.
- [x] Hỗ trợ role `cadence_exploration_specialist` cho thiếu lệnh/trải nghiệm nhưng vẫn phải qua risk gate.
- [x] Hỗ trợ role `entry_quality_specialist` cho cải thiện điểm vào lệnh.
- [x] Hỗ trợ role `exit_quality_specialist` cho cải thiện điểm thoát/giữ/chốt.
- [x] Hỗ trợ role `memory_critic_specialist` để đánh giá ký ức nào nên tin, ký ức nào nên prune.
- [x] Với `position_management_specialist`, input nên gồm position state, account context, causal memory, lifecycle tags, critic state; output nên gồm trailing gate, partial close ratio, management mode, target slot, close bias.
- [x] Với `risk_gate_specialist`, input nên gồm drawdown, open risk ratio, floating PnL R, spread, losing streak, equity volatility; output nên gồm entry gate, hold bias, close/reduce risk bias.
- [x] Với `memory_specialist`, input nên gồm market key, account key, position key, causal memory summary; output nên gồm memory context, action bias, critic bias.
- [x] Với `predictor_specialist`, input nên gồm encoder state, working memory, chosen action, recent outcomes; output nên gồm predicted state/outcome embedding và uncertainty.
- [x] Với `critic_specialist`, input nên gồm predicted state, memory context, account stress, position lifecycle; output nên gồm action utility/risk/cost score.
- [x] Với `selector_specialist`, input nên gồm hypothesis activations, critic scores, conflict, confidence; output nên gồm selection gate và fallback choice.
- [x] Module mới phải bắt đầu với gate thấp để không lấn át policy chính trước khi chứng minh utility.
- [x] Module mới có plasticity cao lúc warmup; consolidation sau promote nhiều lần vẫn cần nối vào runtime.
- [x] ModuleSpec không được yêu cầu đổi EA bridge hoặc action schema nếu chưa có migration riêng.

## 10. Kết nối giữa module mới, module khác và M1

- [x] Module mới không được là hộp kín bắt buộc; neuron của nó có thể nối với neuron cùng module, module khác và vùng M1 nếu qua whitelist/budget/validation.
- [x] Module mới có `flexible_peer_module_fanout` để tạo edge yếu hai chiều với module candidate/active khác, không chỉ đi vòng qua M1.
- [x] Mọi edge mới từ module mới sang M1 hoặc module khác phải bắt đầu yếu: gate thấp, weight nhỏ, utility bằng 0 hoặc thấp, lifecycle `candidate`.
- [~] Chỉ tăng gate/weight/consolidation sau khi candidate được MT5 promote hoặc có utility ổn định qua nhiều validation.
- [x] Hỗ trợ edge `forward` giữa module: ví dụ `memory_specialist -> predictor_specialist`, `predictor_specialist -> critic_specialist`, `critic_specialist -> selector_specialist`.
- [x] Hỗ trợ edge `recurrent` trong module và giữa module: ví dụ `memory_specialist(t-1) -> memory_specialist(t)`, `predictor_specialist(t-1) -> predictor_specialist(t)`, `position_manager(t-1) -> critic_specialist(t)`.
- [x] Hỗ trợ edge `lateral` để cạnh tranh/hợp tác giữa module cùng vai trò hoặc đối nghịch: ví dụ `entry_quality_specialist <-> risk_gate_specialist`, `hypothesis_A <-> hypothesis_B`, `memory_specialist_A <-> memory_specialist_B`.
- [x] Hỗ trợ edge `gating/modulatory` để module này điều khiển độ mạnh của module/edge khác, không cần xoá thẳng connection.
- [x] Hỗ trợ edge `critic/evaluation` để module chấm điểm hypothesis/output/module khác.
- [x] Hỗ trợ edge `memory/context` để module ký ức cấp context cho predictor, critic, selector hoặc position manager.
- [x] Hỗ trợ edge `predictive` để module dự đoán state/outcome của action hoặc position lifecycle.
- [x] Hỗ trợ edge `selector` để module ảnh hưởng lựa chọn action cuối bằng score/gate/fallback, không ghi đè trực tiếp không kiểm soát.
- [x] Hỗ trợ edge `meta/development` để module báo utility/conflict/novelty/growth pressure cho development controller.
- [x] Whitelist runtime cho module -> M1 mở trên vùng học được: `encoder`, `memory`, `hypothesis`, `predictor`, `critic`, `selector`, `meta`, `output`; riêng `output` chỉ là bias/gate nhỏ có bound.
- [x] Whitelist runtime cho M1 -> module mở trên vùng học được: `encoder`, `memory`, `hypothesis`, `predictor`, `critic`, `selector`, `meta`, `output` nếu qua guard/budget.
- [x] Whitelist ban đầu cho memory -> module: `ExperienceSuperNetwork context -> memory_specialist/predictor/critic/position_manager`.
- [x] Không cho module ghi trực tiếp vào raw input, EA bridge schema, hard risk limit, OrderSend hoặc memory key schema.
- [x] Không cho child/module ghi thẳng weight M1 production; mọi thay đổi phải qua candidate copy và promotion.
- [x] Không cho module tự xoá module khác ngoài candidate rollback/prune path có audit.
- [x] Kết nối hai chiều được phép nhưng phải có decay/clamp/iteration limit để tránh self-amplifying confidence.
- [~] Vòng `selector -> critic -> selector` nếu có phải được kiểm soát bằng gate, clamp, trace và MT5 validation riêng.
- [x] Vòng `memory -> predictor -> memory` nếu có phải ghi prediction error và memory utility để biết vòng này giúp hay gây nhiễu.
- [x] Mỗi cross-module edge phải có `src_module_id`, `dst_module_id`, `edge_kind`, `created_by`, `need_reason`, `candidate_id`.
- [x] Mỗi cross-module edge phải được credit assignment riêng; edge nào tham gia action xấu nhiều lần phải giảm utility/gate hoặc bị prune ở foundation `GraphPlasticityEngine`.
- [x] Budget cross-module riêng: giới hạn số edge liên module, recurrent liên module và lateral liên module.
- [x] Diversity guard: không tạo nhiều module giống nhau rồi nối dày đặc làm thành clone topology.
- [x] Nếu module mới output trực tiếp vào action logits/parameters, output đó phải là bias/gate nhỏ, không thay thế output chính ngay.

## 11. Kết nối khuyến nghị theo từng specialist

- [x] `risk_gate_specialist` nên nhận từ M1 encoder, account stress, drawdown, spread, exposure, losing streak; nên xuất tới M1 selector, entry output gate, hold/close bias và critic.
- [x] `risk_gate_specialist` chủ yếu dùng edge `forward`, `gating/modulatory`, `critic/evaluation`.
- [x] `position_management_specialist` nên nhận position lifecycle, M1 memory, causal memory, account context; nên xuất tới trailing output, partial close output, management mode, target slot, close bias và critic.
- [x] `position_management_specialist` chủ yếu dùng edge `forward`, `memory/context`, `gating/modulatory`, `critic/evaluation`, `recurrent`.
- [x] `memory_specialist` nên nhận M1 encoder, memory key, account key, position key và ExperienceSuperNetwork summary; nên xuất tới predictor, critic, selector và action bias.
- [x] `memory_specialist` chủ yếu dùng edge `memory/context`, `forward`, `recurrent`, `gating/modulatory`.
- [x] `predictor_specialist` nên nhận M1 encoder, M1 working memory, chosen action và recent outcomes; nên xuất predicted state/outcome/uncertainty tới critic và development controller.
- [x] `predictor_specialist` chủ yếu dùng edge `predictive`, `forward`, `recurrent`, `critic/evaluation`, `meta/development`.
- [x] `critic_specialist` nên nhận predictor output, memory context, account stress, position lifecycle; nên xuất utility/risk/cost score tới selector và output gate.
- [x] `critic_specialist` chủ yếu dùng edge `critic/evaluation`, `forward`, `gating/modulatory`, `selector`.
- [x] `selector_specialist` nên nhận hypothesis activations, critic scores, conflict, confidence và child policy scores; nên xuất selection gate/fallback choice tới final selector.
- [x] `selector_specialist` chủ yếu dùng edge `selector`, `lateral`, `gating/modulatory`.
- [x] `entry_quality_specialist` nên nhận encoder state, memory context, spread/cost, trend/range features; nên xuất nhỏ tới buy/sell logits và critic.
- [x] `entry_quality_specialist` phải bị `risk_gate_specialist` gate khi account stress/drawdown cao.
- [x] `exit_quality_specialist` nên nhận position lifecycle, MFE/MAE, distance TP/SL, bars since MFE, giveback; nên xuất tới close/partial/trailing/management output.
- [x] `memory_critic_specialist` nên nhận memory matches, causal match utility, support count và outcome quality; nên xuất gate/prune pressure tới memory edges và memory context.

## 12. Shadow candidate và rollback

- [x] Mọi grow/prune/rewire/split/merge chạy trên candidate copy, không chạy thẳng trên baseline.
- [x] Candidate có `candidate_id`, `parent_policy_id`, `parent_fitness`, `structural_diff`, `created_at`.
- [x] Chạy fast replay validation trước MT5 để loại candidate rõ ràng tệ.
- [x] Candidate chỉ đi MT5 nếu không vi phạm invariant, budget, action schema và risk hard gate.
- [~] Nếu MT5 tốt hơn baseline và pass drawdown/trade-count/telemetry gate, promote candidate.
- [x] Nếu fail, rollback toàn bộ structural diff và chỉ giữ audit/journal kinh nghiệm nếu hợp lệ.
- [x] Lưu reject reasons đủ cụ thể: `no_trade`, `missing_telemetry`, `drawdown`, `fitness_regression`, `invalid_graph`, `budget_exceeded`.
- [x] Bài học reject phải tác động lớn lên baseline qua `rejection_memory`, starvation, output bias, risk/curiosity/development heads và reasoning graph, nhưng không copy topology candidate lỗi.
- [x] Test rollback không alias lại state của rejected candidate vào baseline.

## 13. Blueprint để sinh siêu mạng con không clone

- [x] Tạo module `src/self_growing_trader/supernetwork_blueprint.py`.
- [x] Định nghĩa `SuperNetworkBlueprint`: `blueprint_id`, `parent_id`, `role`, `regions`, `connection_rules`, `plasticity_profile`, `growth_policy`, `risk_policy`, `memory_schema`, `objective`.
- [x] Blueprint không chứa raw weight/memory/recurrent state của parent.
- [x] Blueprint chỉ chứa prior/thiết kế: số neuron theo region, loại edge cho phép, budget, bias học, mục tiêu.
- [x] Tạo `BlueprintGenerator` nhận M1 graph summary, memory summary, failure clusters, winning causal patterns để sinh blueprint; chỉ lưu compressed teaching signals, không copy raw state.
- [x] Tạo `spawn_supernetwork_from_blueprint(blueprint, seed)` sinh graph mới với neuron/connection id mới và weight seed riêng.
- [x] Tính `diversity_score` giữa child và parent: topology distance, region ratio, edge-kind distribution, action bias distance.
- [x] Reject child nếu diversity quá thấp, vì đó là clone trá hình.
- [x] Cho child inherit schema/learning rule nhưng không inherit memory state trực tiếp.
- [x] Nếu dùng kinh nghiệm parent, chỉ dùng journal schema v6 đã lọc hoặc compressed teaching signals; child spawn/warm-up không copy memory neuron/state.

## 14. Adaptive child supernetwork

- [x] Tạo module `src/self_growing_trader/adaptive_supernetwork.py`.
- [x] Định nghĩa `AdaptiveSuperNetwork` với API: `act()`, `observe_transition()`, `develop()`, `copy_candidate()`, `promote_candidate()`, `rollback_candidate()`, `spawn_child()`.
- [x] Child có `NeuralGraph` riêng.
- [x] Child có memory graph riêng, schema v6, và warm-up journal đã lọc; không copy memory state từ parent.
- [x] Child có `GraphDevelopmentController` và role prior riêng; không alias controller của parent.
- [x] Child có `PlasticityEngine` riêng; foundation dùng `GraphPlasticityEngine` riêng trong `observe_transition()`.
- [x] Child có utility/promotion history riêng.
- [x] Child có lineage metadata riêng: parent, generation, role, blueprint, birth seed.
- [x] Child có guard riêng: typed edge guard, max trials và max children ở mức graph/lineage foundation; tổng neuron/connection không còn hard cap.
- [x] Child tự warmup bằng replay/journal hợp lệ trước khi vào MT5.
- [x] Child tự phát triển qua shadow candidate; replay + MT5 promotion contract được enforced trước khi child selectable.
- [x] Child không phụ thuộc M1 để ra action sau khi đã spawn; M1 chỉ có thể cung cấp prior/teacher signal.

## 15. Lineage và kiểm soát sinh sản

- [x] Tạo module `src/self_growing_trader/lineage.py`.
- [x] Định nghĩa `NetworkLineage`: `network_id`, `parent_id`, `root_id`, `generation`, `blueprint_id`, `role`, `created_at`, `promotion_count`, `failure_count`.
- [x] Chỉ cho mạng sinh child nếu `promotion_count >= N`, utility ổn định và không vượt budget; foundation kiểm soát promotion/generation/children budget.
- [x] Giới hạn `max_generation`, `max_children_per_network`, `max_alive_networks`.
- [x] Có diversity gate bắt buộc giữa parent-child và sibling-sibling ở foundation spawn/diversity.
- [x] Có stale pruning: mạng con không improve sau nhiều trial bị archive hoặc loại khỏi selector.
- [x] Không xoá artifact/report trước khi ghi lineage/audit; trial history/lineage record được giữ trong child manager và JSONL contract.
- [x] Test lineage không tạo vòng parent-child.

## 16. Meta-selector giữa nhiều mạng

- [x] Thiết kế `PolicyEnsembleSelector` hoặc tương đương để chọn M1/child theo regime ở foundation `policy_ensemble.py`.
- [x] Selector input gồm memory query, account stress, spread, drawdown, open positions, child confidence, recent utility; foundation hiện có memory_match/account_stress/drawdown/open_positions/confidence/recent_utility.
- [x] Selector không được override safety gate EA/risk; foundation chỉ chọn policy id, không đụng EA/risk.
- [x] Selector phải ghi trace: policy selected, reason/score, fallback policy.
- [x] Selector hiện deterministic và chỉ chọn policy id; mọi selector neural trong tương lai vẫn phải qua candidate + MT5 validation, không được deploy bằng điểm nội bộ.
- [x] Có fallback về baseline M1 nếu child output invalid/low confidence.
- [~] Test multi-policy không làm sai `trace_id` và action/event_id trong MT5 bridge.

## 17. MT5 validation và reporting

- [x] Thêm metric `growth_accept_rate`: structural candidates được MT5 promote / candidates tested.
- [x] Thêm metric `child_survival_rate`: child qua MT5 / child spawned.
- [x] Thêm metric `diversity_score`: khác biệt child-parent ở foundation `diversity_score()`.
- [x] Thêm metric `graph_size`: neurons, connections, recurrent, lateral, disabled ở foundation `graph_metrics()`.
- [x] Thêm metric `cross_module_edges`: số edge module-module theo kind ở foundation `graph_metrics()`.
- [x] Thêm metric `m1_module_edges`: số edge M1-module theo chiều M1->module và module->M1 ở foundation `graph_metrics()`.
- [x] Thêm metric `cross_module_edge_utility`: utility/gate trung bình theo edge kind ở foundation `graph_metrics()`.
- [x] Thêm metric `recurrent_loop_health`: loop count, max recurrent gain, NaN/Inf count ở foundation `graph_metrics()`.
- [x] Thêm metric `plasticity_health`: mean/min/max gate/plasticity/utility và NaN/Inf count trong `graph_metrics()`.
- [x] Thêm metric `credit_attribution`: số active nodes/edges nhận credit theo action/outcome.
- [x] Thêm metric `module_need_scores`: predictor, memory, critic, selector, position manager, risk gate, cadence, entry, exit ở foundation `module_need_metrics()`.
- [x] Thêm metric `module_role_accept_rate`: tỉ lệ promote/reject theo từng role module.
- [x] Thêm metric `cluster_to_module_outcome`: cụm lỗi nào tạo module gì và MT5 outcome ra sao.
- [x] Thêm metric `prediction_error_trend`, `memory_usefulness`, `selector_switch_rate`.
- [x] Ghi các metric vào trial history JSONL và report summary.
- [~] Không coi run hợp lệ nếu thiếu telemetry bridge hoặc report MT5.
- [ ] Smoke MT5 tối thiểu: `rounds=1`, `trials=1`, bridge có dòng, report parse được, output `.pkl` load lại được.

## 18. CLI và config

- [x] Thêm config/field cho decision backend: `SelfGrowingPolicy.m1_backend=direct_rnn|m1_graph`, CLI `--m1-backend direct_rnn|m1_graph|graph`; production config mặc định direct RNN, graph bật tường minh bằng `--m1-backend m1_graph`.
- [x] Thêm config cho child spawning: enable, max children, max generation, diversity threshold, budget.
- [x] Thêm config module need thresholds/cooldown: prediction, memory, risk, position manager, cadence.
- [x] Thêm config `module_role_budget` để giới hạn số module theo role.
- [x] Thêm config cross-module connectivity: enable, max cross-module edges, max M1-module edges, allowed edge kinds, recurrent loop limit, flexible M1 fanout và peer module fanout.
- [x] Thêm config whitelist theo vùng học được: module mới có thể nối hai chiều với `encoder/memory/hypothesis/predictor/critic/selector/meta/output`; raw input/EA/hard risk không phải target ghi trực tiếp.
- [x] Thêm CLI flag tương ứng cho M1 backend: `--m1-backend graph`; `--enable-child-supernet`, `--max-child-networks` chưa thêm.
- [x] UI Java có dropdown `M1 backend` và truyền backend tường minh cho các workflow MT5; giá trị production hiện hành là `direct_rnn`.
- [x] Cập nhật `configs/self_growing.yaml` hoặc config mới khi behavior đã có code.
- [x] Search lại để xoá config/flag cũ nếu bị thay thế.

## 19. Test bắt buộc

- [x] Unit test `NeuralGraph` invariant/copy/activation.
- [x] Unit test migration legacy M1 -> graph M1.
- [x] Unit test graph structural action grow/prune/consolidate không phá shape; rewire chưa làm.
- [x] Unit test credit assignment chỉ tác động active node/edge.
- [x] Unit test error clustering nhận đúng cluster near-TP/giveback/stress-entry/high-prediction-error.
- [x] Unit test module need scoring tăng đúng role khi đưa trace lỗi tương ứng.
- [x] Unit test `ModuleSpec` không chứa yêu cầu phá bridge/action schema.
- [x] Unit test role cooldown chặn tạo lặp module cùng role khi chưa có promote.
- [x] Unit test promote/reject feedback cập nhật prior module role đúng chiều.
- [x] Unit test cross-module edge whitelist cho phép đúng edge hợp lệ và reject edge nguy hiểm.
- [x] Unit test module -> M1 edge chỉ tạo output bias/gate nhỏ, không ghi đè output chính.
- [x] Unit test recurrent cross-module edge có clamp/decay và không tạo NaN qua nhiều step.
- [x] Unit test edge metadata có `src_module_id`, `dst_module_id`, `edge_kind`, `created_by`, `need_reason`, `candidate_id`.
- [x] Unit test credit assignment cho cross-module edge xấu làm giảm utility/gate.
- [x] Unit test risk gate có thể gate entry specialist khi stress cao.
- [x] Unit test blueprint không chứa weight/memory raw của parent.
- [x] Unit test child spawn tạo id/topology/seed khác parent.
- [x] Unit test diversity gate reject clone trá hình.
- [x] Unit test child develop dùng controller/plasticity riêng.
- [x] Unit test rollback candidate không làm bẩn baseline.
- [x] Unit test artifact pickle load/save qua version mới.
- [x] Integration test CLI smoke không MT5 nếu có thể mock evaluator.
- [ ] MT5 smoke thật với `rounds=1`, `trials=1` trước khi kết luận deploy được.
- [ ] Nếu thay EA/bridge, compile EA và chạy MT5 report thật.

## 20. Tài liệu và graph sau khi triển khai

- [x] Cập nhật `docs/overview.vi.md` khi M1 graph backend thật sự chạy.
- [x] Cập nhật `docs/architecture.vi.md` với layer `NeuralGraph`, `BlueprintGenerator`, `AdaptiveSuperNetwork`, `NetworkLineage`.
- [x] Cập nhật `docs/domain-flow.vi.md` với flow spawn child, child warmup, child MT5 promotion.
- [x] Cập nhật docs để mô tả rõ M1 suy ra module mới từ trace -> error cluster -> need score -> ModuleSpec -> candidate -> MT5 promote/reject.
- [x] Cập nhật docs để mô tả rõ module mới có thể nối với module khác và với M1 bằng edge kind nào, whitelist nào và guard nào.
- [x] Cập nhật checklist này, chuyển từng mục đã xong từ `[ ]` sang `[x]` kèm ghi chú nếu cần.
- [x] Cập nhật `.understand-anything/knowledge-graph.json` bằng `/understand --language vi` nếu lệnh có sẵn; nếu không có, cập nhật fallback inspection như đang dùng trong repo.
- [x] Nếu thêm dashboard/visual lineage, chạy hoặc cập nhật understand-dashboard tương ứng nếu công cụ khả dụng.
- [x] Tài liệu phải nói rõ phần nào là default, phần nào là experimental.

## Ghi chú kiểm chứng còn lại

- `[~]` không phải trạng thái deploy: các đường graph/candidate/child đã có test cục bộ nhưng cần MT5 Strategy Tester thực để chứng minh bridge telemetry, report parser, trade count, drawdown và promotion.
- Các mục section 21 là **cố ý không xoá migration path**: graph đã là default cho policy/config mới nhưng legacy vẫn phải tồn tại khi artifact cũ còn được hỗ trợ. Không xoá compatibility path chỉ để tăng số checkbox.
- Bốn checkbox `[ ]` còn lại đều là các bản ghi lặp của hai bằng chứng external bắt buộc: compile EA và smoke `rounds=1,trials=1` với report parse/telemetry/output pickle. Chạy `python scripts/run_graph_mt5_smoke.py --runner-dir <MT5_RUNNER>` để tạo bằng chứng đó.

## 21. Xoá phần cũ khi thay thế

- [~] Sau khi M1 graph backend thay legacy M1 hoàn toàn, search và xoá code legacy không còn dùng.
- [~] Xoá import/export/registry entry của legacy backend nếu không còn route sử dụng.
- [~] Xoá config/flag legacy nếu đã bị thay bằng config mới.
- [~] Xoá tests/mock/fixtures chỉ phục vụ behavior cũ nếu không còn cần.
- [~] Xoá text UI/docs mô tả M1 cố định nếu M1 graph đã là default.
- [~] Search lại các tên cũ sau khi xoá: `M1ThinkingModule`, `m1_components`, `state_encoder`, `hypothesis_bank`, `legacy`.
- [~] Chỉ giữ legacy migration code nếu artifact cũ vẫn cần load; ghi rõ là migration path, không phải runtime chính.

## 22. Tiêu chí hoàn thành cuối

- [x] M1 chạy bằng graph neuron/connection thống nhất cho policy mới; legacy chỉ là explicit migration/testing backend.
- [x] M1 tự grow/prune/rewire/consolidate qua candidate, không mutate random mù.
- [x] M1 tự suy ra cần module gì từ trace, cụm lỗi, need scores và feedback promote/reject, không phụ thuộc rule cứng.
- [x] Module mới có role/spec/input/output/validation goal rõ ràng và bắt đầu bằng gate thấp.
- [x] Neuron của module mới có thể nối với neuron module khác và neuron/vùng M1 qua whitelist edge kind, flexible M1 fanout, peer module fanout, recurrent self-loop, budget, trace và candidate validation; MT5 là promotion gate external.
- [x] Cross-module/M1-module edge có credit assignment riêng và có thể bị prune nếu gây hại.
- [x] M1 sinh child supernetwork từ blueprint + seed riêng, không clone weight/memory/topology.
- [x] Child có graph, memory, plasticity, development controller, promotion gate và lineage riêng.
- [x] Child tự thích nghi/phát triển qua `observe_transition()` và `develop()`.
- [~] Child chỉ được giữ nếu qua replay validation và MT5 promotion gate.
- [x] Multi-network selector có trace và fallback an toàn.
- [ ] MT5 smoke thật pass, report parse được, telemetry bridge đủ dòng.
- [x] Tests pass.
- [x] Docs và knowledge graph phản ánh đúng code hiện tại.
