# Luồng nghiệp vụ

## Train MT5

1. CLI nhận `train-mt5` cùng trials/round.
2. Python tạo `SelfGrowingPolicy` seed hoặc load `--seed-policy`; nếu CLI dùng `--policy-core world` thì tạo/chuyển sang `SpatialWorldPolicy`.
3. Mỗi round sinh trial bằng causal memory plasticity, `ReasoningGraphSystem`, `CuriosityNeuronSystem` và `NeuralDevelopmentController`; `mutation_sigma` chỉ scale biến đổi trên action rows/output nodes đã có bằng chứng gây lỗ từ StepTrace hoặc causal memory xấu. Khi trace có attribution, nó chỉ sửa các hidden connection hoặc graph edge active đã góp phần tạo action lỗ, không nhiễu toàn bộ action/gate/recurrent.
4. Growth controller tính error/novelty/conflict/representation/complexity để quyết định clone specialist trong giới hạn module. Với world-core, `WorldState` còn điều chỉnh gate/action qua energy/knowledge/trust và mixed geometry.
5. MT5 Strategy Tester chạy EA `SelfGrowing_MT5_Trainer_EA`.
6. EA tính input/account/risk state từ MT5 và ghi request schema v2 vào `self_growing_request.csv`, kèm `event_id`, `drawdown_pct`, `spread_points` và `positions_pipe` để truy vết từng ticket/slot.
7. Python runner đọc request, gọi đúng một lần `policy.act()` để tạo `StepTrace`, action/output, `trace_id`, rồi ghi `self_growing_action.csv` kèm metadata `market_context`, `expert_id`, `policy_id`, `expert_state`, `context_reason`, `no_trade_reason` cho EA hiển thị trên chart.
8. Khi request kế tiếp tới, Python dùng input kế tiếp làm `next_obs`, tính reward từ log-equity change đã clip, trừ cost spread/drawdown/confidence, rồi gọi `observe_transition()` cho trace trước đó. Topology Direct-RNN giữ cố định trong lifecycle.
9. EA đọc action cùng `event_id/trace_id`, execute trong Strategy Tester và ghi telemetry schema v2 vào `self_growing_bridge.csv`. Các action quản lý lệnh hiện phân biệt `CLOSE_WORST`, `CLOSE_BEST`, `CLOSE_ALL`; chỉ `MANAGE_POSITION` được dùng output vector để breakeven, modify SL/TP, trailing hoặc partial close. `HOLD` là no-op, còn output chọn target slot đang tạm thời bị tắt nên runtime ép slot quản lý về 0.
10. Python đọc bridge telemetry, parse report và tính fitness.
11. Trial tốt nhất được consolidate và lưu ra output `.pkl`.

## Paired train MT5 Model 1 và Python

1. Hai nhánh bắt đầu từ cùng seed bất biến; hash seed được kiểm tra trước/sau MT5.
2. MT5 Model 1 tạo input, action, account/margin snapshot và lifecycle ledger.
3. Capture được kiểm tra và biên dịch thành NPZ; Python không dùng CSV trong hot loop.
4. Python replay nhân quả NPZ qua đúng production policy loop.
5. Lệnh paired verifier chỉ thành công khi input/output/action, account/margin, execution contract và learner-state hash giống tuyệt đối. Thời gian fill là ngoại lệ duy nhất.

Runbook: `docs/MODEL1_PYTHON_PARITY_BACKEND.vi.md`.

## Improve

`improve` chấm baseline seed bằng MT5, train trial mới, rồi promote chỉ khi trial tốt hơn, có trade, net profit dương, profit factor đạt yêu cầu và drawdown trong giới hạn.

## Train-to-goal

`train-to-goal` lặp `improve` đến khi baseline seed hoặc trial đạt profit factor, winrate, expectancy, drawdown và số lệnh/ngày mục tiêu. Nếu seed hiện tại đã đạt goal sau lượt MT5 baseline, workflow copy seed đó ra output và dừng, thay vì bắt mutate thêm một trial kém hơn.

Trong vòng improve nội bộ, `train-to-goal` không dùng promotion gate cứng như deploy, nhưng không còn ghi đè `winner.pkl` bằng trial hồi quy. Khi gate cứng tắt, workflow tính `goal_progress_score` từ fitness, net profit, profit factor, expectancy, drawdown, sample count và execution quality; trial chỉ được promote tiếp nếu score này tốt hơn baseline. Trial tệ vẫn bị dùng làm feedback reject để tăng bài học, nhưng baseline tốt hơn được giữ lại.

Nếu `--seed-policy` chưa tồn tại, `train-to-goal` không dừng bằng `FileNotFoundError`; nó chạy một lượt bootstrap `train-mt5` fresh ra đúng `--output`, ghi entry bootstrap vào log JSONL, rồi dùng policy đó làm seed cho vòng goal tiếp theo. `auto-improve` dùng cùng cơ chế bootstrap này.

Khi MT5 report không có lệnh thua, parser coi profit factor là rất cao và expectancy bằng avg win/expected payoff thay vì 0. Điều này tránh reject sai các run có winrate 100%.

## Walk-forward

`walk-forward` chia khoảng ngày thành nhiều cửa sổ MT5, train từng cửa sổ, ghi JSONL và copy policy tốt nhất thành `walk_forward_best.pkl`.

## Compare-cores

`compare-cores` chạy `current` và `world` như hai thế giới riêng. Mỗi core có runner-dir MT5 riêng để không đè request/action/bridge/report. Sau khi train xong, Python trao đổi positive replay experience theo hai chiều, validate lại từng candidate bằng MT5, rồi chọn `best.pkl` từ policy có fitness tốt nhất sau guarded exchange.




## M1 graph supernetwork candidate (graph opt-in)

1. Cấu hình/CLI production dùng `m1_backend=direct_rnn` làm baseline; output quyết định lấy từ RNN/deep recurrent stack 20 chiều. `m1_graph` chỉ là backend tương thích khi bật tường minh cho artifact/candidate cũ. EA request/action giữ 137 input, 20 output, bridge schema hiện hành và action schema v4.
2. `act()` query `ExperienceSuperNetwork`, đưa context chuẩn hoá vào graph memory/predictor/critic/selector, chạy graph và lưu `GraphActivationTrace` vào `StepTrace` cùng active neuron/edge, gate, memory match và lifecycle evidence.
3. `observe_transition()` đọc reward/cost/TD/prediction error/account stress/lifecycle tag; `GraphPlasticityEngine` chỉ credit active node/edge. Baseline không bị structural prune/grow bởi credit xấu.
4. `propose()` copy sâu policy trước. `GraphSupernetworkRuntime` gom trace thành failure cluster/need score, chọn `ModuleSpec`, rồi grow/prune/rewire/consolidate và tạo cross-module edge yếu trên candidate. Module mới được nối bằng role guide cộng fanout dẻo tới các vùng học được của M1, peer fanout tới module khác và self-loop hồi tiếp yếu khi phù hợp; raw input/EA/hard risk/action schema không phải target ghi trực tiếp. Candidate phải pass invariant, finite-state, budget, bridge/action schema và replay gate.
5. `ReasoningGraphSystem` enrich context cho `curiosity/development/risk/action_head/utility` và nhận credit từ outcome/consolidate. Nó chỉ điều chế controller context, không tự gửi action sang EA.
6. Trial MT5 tốt hơn baseline mới được promote. Promote cập nhật role prior/consolidation/audit; reject rollback toàn bộ structural diff nhưng không bị bỏ phí: `absorb_rejection_feedback()` lưu bài học vào `rejection_memory`, tăng outcome/starvation, chỉnh mạnh có clamp lên bias HOLD/entry/manage, risk gate, curiosity, neural development và credit `ReasoningGraphSystem`. Với trace xấu, nó phạt trực tiếp graph `active_neuron_ids`/`active_edge_ids` hoặc hidden->action links của module đã tạo action sai. Nhờ vậy trial sau chịu áp lực hành vi lớn từ lỗi trước mà vẫn không copy topology lỗi vào baseline. Trial history graph được ghi JSONL.
7. Sau promotion hợp lệ, `ChildSupernetworkManager` có thể sinh child candidate từ blueprint/seed độc lập, kiểm tra diversity/lineage budget, warm-up journal schema v6 rồi chạy replay + MT5 promotion contract. Child chưa promote không được vào ensemble selector.
8. `PolicyEnsembleSelector` ghi policy-selection trace và fallback về M1 khi child invalid/low-confidence. EA và hard-risk gate vẫn là authority cuối.

**Bằng chứng hiện có:** các bước Python trên có unit/integration test. Archive đóng gói này không có MT5 terminal/runner report thật, vì vậy bước MT5/EA compile/smoke phải chạy tại môi trường MT5 và chỉ khi đó mới được coi là deploy evidence.
