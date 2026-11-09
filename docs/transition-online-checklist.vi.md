# Checklist transition và vòng học online

Nguồn đối chiếu: `C:\Users\Admin\Downloads\huong_dan_transition_va_vong_hoc_online_chi_tiet.txt`.

## P0

- [x] Mỗi event MT5 chỉ gọi một lần `SelfGrowingPolicy.act()`.
- [x] Action gửi cho EA lấy từ đúng `StepTrace.selected_action` vừa tạo.
- [x] `StepTrace` có `trace_id`, timestamp, raw observation, encoded state, memory snapshot, candidate action/scores, selected action, log-prob tối thiểu, prediction, critic value, confidence/novelty/conflict và gate snapshot.
- [x] `observe_transition()` dùng request kế tiếp làm `next_obs`, không dùng lại `obs_t`.
- [x] Predictor được update từ actual next state và action trace.
- [x] Critic dùng TD target `reward - cost + gamma * V(next_state)`.
- [x] Policy head nhận update theo advantage từ TD delta.
- [x] MT5 path gọi `observe_transition()` trong lúc Strategy Tester đang chạy; không có lifecycle structural-development hook.
- [x] Toy/unit tests khóa predictor phân biệt BUY/SELL action, critic TD signal và policy head weight thay đổi sau transition.

## P1

- [x] Bridge EA/Python dùng một contract schema v2.
- [x] Request có `schema_version`, `event_id`, account state, drawdown, spread, position state và observation.
- [x] Response có `schema_version`, `event_id`, `trace_id`, selected action, output vector và confidence.
- [x] Bridge telemetry giữ lại `event_id/trace_id` từ response.
- [x] Reward online dùng log-equity change có clip và cost từ spread/drawdown/confidence.
- [x] Window fitness dùng dạng `J = R - lambda_dd*DD - lambda_tail*CVaR - lambda_cost*Cost - lambda_turn*Turnover - lambda_cmp*Complexity`, và không cho nghiệm quá ít lệnh thắng nhờ PF/WR ảo.
- [x] Promotion gate bắt buộc đủ số lệnh tối thiểu theo `min_daily_trades * backtest_days`.
- [x] Trial fail không promote cấu trúc; transition có TD delta dương hoặc reward tốt chỉ được replay vào candidate, rồi phải qua MT5 validation tốt hơn baseline mới được giữ, nếu không rollback về seed/best trước đó.
- [x] Request/bridge có `positions_pipe` để truy vết position ticket, side, volume, entry, SL, TP, unrealized R và age theo slot.
- [x] `ExperienceSuperNetwork` hấp thụ `action_details` + `positions_pipe` thành causal memory `action -> position -> outcome` với cạnh credit ngược về state memory.
- [x] EA theo dõi lifecycle theo ticket: MFE/MAE, gần TP/SL nhất, bars-since-MFE và giveback; supernetwork tag được `near_tp_missed` / `gave_back_profit`.
- [x] Causal memory xấu có quyền sửa trực tiếp weight trên candidate: `w_out`, `gate_w`, `w_rec`, `w_lat` và M1 `hypothesis_bank.w_decode`, nhưng chỉ promote nếu qua MT5 validation.
- [ ] Telemetry trade-level vẫn còn thiếu realized PnL, commission và slippage tách riêng theo deal đóng.
- [ ] Credit assignment mới là one-step bar-level; chưa có open-trace assignment hoặc N-step/GAE cho lệnh giữ nhiều bar.
- [ ] Python-MT5 parity test không còn phù hợp dạng cũ vì EA không chạy model local; thay bằng contract/telemetry parity test.

## P2

- [ ] Structural reward theo rolling window.
- [ ] Rollback structural changes khi utility xấu.
- [ ] Recruit/prune dormant pool đầy đủ theo edge/neuron-level score.
- [x] Consolidation/replay chống quên: runner ghi `outputs/mt5_experience_journal.jsonl` realtime, replay positive transition khi train sau khởi động lại và hấp thụ chúng vào `ExperienceSuperNetwork`.
- [x] Siêu mạng ký ức có 4 tầng `MemoryNeuron -> MemoryCluster -> StrategyMemory -> MetaStrategyMemory`, cạnh ngang/hồi tiếp/dẻo/nhớ/quên và cạnh sâu `abstract/down`, tự decay/prune theo utility.
- [x] Siêu mạng ký ức kết hợp sâu với `M1ThinkingModule`: context đi vào working memory, predictor, critic/selector, hypothesis candidate và selected action.
- [ ] Walk-forward/OOD adaptation trước paper trading.
