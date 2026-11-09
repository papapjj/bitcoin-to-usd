# Checklist curiosity neural và neural candidate proposal

## Mục tiêu

- [x] Không dùng công thức curiosity cố định để quyết định tò mò.
- [x] Thêm `CuriosityNeuronSystem` là mạng nơ-ron recurrent nhỏ có state riêng.
- [x] Curiosity nhận context từ state, action, memory query và causal memory xấu/tốt.
- [x] Curiosity sinh proposal neural cho action head, gate, recurrent và growth.
- [x] Candidate được chỉnh bằng neural proposal trước khi MT5 validate.
- [x] Random mutation không còn là cơ chế proposal chính; residual noise nếu có do `NeuralDevelopmentController` xuất ra.
- [x] Causal memory xấu vẫn chỉnh trực tiếp candidate weight qua `apply_causal_memory_plasticity()`.
- [x] `observe_transition()` cho curiosity tự học từ reward, TD delta và prediction error.
- [x] `act()` dùng curiosity action delta như bias nhỏ, không thay EA bridge.
- [x] Có `experience_starvation` gồm `no_trade_streak`, `unfilled_streak`, `cadence_deficit`, `starvation`, `intrinsic_reward`.
- [x] MT5 runner đưa starvation vào policy trước `act()`, vào `StepTrace`, journal và replay.
- [x] Curiosity gain tăng theo starvation nhưng vẫn clip để không vượt safety gate.
- [x] Có `outcome_starvation` trong policy để lưu `failed_goal_streak`, `non_improve_streak`, `goal_deficit`, `trial_regression`.
- [x] Trial không cải thiện so với trial trước làm tăng starvation cho trial/round sau.
- [x] `train-to-goal` chưa đạt mục tiêu làm policy output đói hơn để vòng kế tiếp không quay lại trạng thái thờ ơ.
- [x] Trial history ghi `curiosity_proposal` và `causal_plasticity` để audit.
- [x] Trial history ghi `neural_development_proposal` để audit structural action/spec.

## Guard

- [x] Curiosity/proposal chỉ sửa candidate, không sửa thẳng best policy trong train candidate.
- [x] Candidate vẫn phải qua MT5 fitness, trade-count gate và promotion/rollback.
- [x] Policy cũ load được vì `ensure_compatible()` tự tạo curiosity system nếu thiếu.
- [x] `SpatialWorldPolicy` copy/chuyển core giữ nguyên curiosity system.

## Còn lại

- [ ] Realized PnL/commission/slippage theo deal đóng vẫn cần bổ sung để credit assignment nhiều bar chính xác hơn.
- [ ] Có thể thêm UI hiển thị `curiosity_proposal.drive/risk_dampen` từ JSON history.
