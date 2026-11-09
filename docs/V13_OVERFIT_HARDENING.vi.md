# V13 — hardening chống overfit nhưng vẫn giữ learning

## Mục tiêu

Training vẫn được phép cập nhật weight, selector, replay, memory và tạo
specialist. Anti-overfit chỉ chặn **promotion**, không chặn việc thu thập bằng
chứng hay rollback học tập.

## Các thay đổi

1. Frozen OOS promotion gate fail-closed:
   - phải `fitness_eligible=true`;
   - phải reconciliation PnL/ledger hợp lệ;
   - tối thiểu 30 lifecycle OOS;
   - drawdown tối đa 10%;
   - policy expectancy LCB không âm;
   - không còn negative context scope;
   - feature parity và real-tick evidence phải hợp lệ.
2. Walk-forward chỉ truyền các row đã qua OOS gate vào
   `promotion_stability_report`. Giá trị fitness `-100` nhưng finite không còn
   bị hiểu nhầm là OOS thành công.
3. Profile `he_chuyen_gia_mot_context_mot_policy` yêu cầu ma trận frozen OOS
   `3 windows x 3 seeds`, real-tick evidence và feature parity.
4. Structural growth vẫn bật nhưng chậm và cần bằng chứng lặp lại:
   lifecycle tạo/giữ candidate tăng, threshold tăng, cooldown tăng, tối đa một
   candidate mỗi lifecycle. Các expert vẫn tiếp tục học từ lifecycle thật.
5. Runtime fitness guard được bật để dừng run đang suy giảm quá sâu mà không
   xóa memory hoặc seed.
6. Credit âm của lifecycle được tách thành hai hệ số trong config candidate:
   `negative_context_credit_multiplier` và
   `negative_entry_replacement_gain`. Seed v13 giữ giá trị lịch sử `8.0/3.0`;
   candidate chống overfit có thể hạ hai hệ số để một loss không đảo action
   margin quá mạnh. Event `neuron_action_replacement_credit` ghi lại đúng hệ
   số đã dùng để audit.
7. Candidate MT5 có thể bật riêng hai directional gate:
   `context_direction_entry_gate` chặn entry trend/volatile/ranging khi timing
   direction chưa xác nhận (candidate mặc định yêu cầu `timing_quality >= 0.25`);
   `context_direction_side_entry_gate` chặn entry ngược phía sau khi direction
   đã xác nhận. Cả hai vẫn tắt ở seed/baseline v13 và phải được ghi trong
   report candidate.
8. Candidate có thêm hidden normalization trước mỗi `tanh` của module và
   deep layer: `none` (baseline), `rms_v1` hoặc `layernorm_v1`. Chuẩn hóa chỉ
   dùng thống kê của forward hiện tại, không có running-stat/outcome leakage;
   mode và ngưỡng phải được ghi trong report candidate.
9. TimingNet là nguồn tạo tín hiệu timing duy nhất. `timing_quality` phải được
   kết hợp từ quality/entry-probability/uncertainty của mạng; không được khôi
   phục rule detector cũ hoặc dùng `timing_q/dir` thiếu `timing_source`.
10. Plasticity dài hạn được tách dấu an toàn: Hebbian và positive Oja drive mới
    nhận reward-gate; anti-Hebbian, Oja norm correction, L2 regularization và
    loss-gradient descent không bị đảo dấu khi reward âm. Pre của plasticity
    dùng `soft_clip_v1` riêng (không đổi raw feature contract của forward), mỗi
    lần update có norm cap, weight bound và telemetry `update_count/EMA/variance`
    cùng norm của từng thành phần. Legacy pickle được migrate về `raw`, còn
    candidate phải ghi rõ `learning.plasticity_input_normalization_mode`.

## Cách train đúng

Train candidate vẫn có thể chạy như trước, nhưng không dùng trực tiếp output
đó để deploy. Sau đó chạy workflow walk-forward với ít nhất 3 cửa sổ và 3 seed:

```powershell
python -m self_growing_trader.cli --config configs/he_chuyen_gia_mot_context_mot_policy.yaml walk-forward-he-chuyen-gia-mt5 `
  --output outputs/he_chuyen_gia/walk_forward_v13_strict `
  --train-window 15 --test-window 5 --step 5 `
  --seed-count 3 --model 4 `
  --from-date 2025.05.01 --to-date 2025.07.01 `
  --runner-dir mt5_v2_runner --keep-mt5-open
```

Artifact `walk_forward_candidate_unpromoted.pkl` chỉ là candidate để audit.
Chỉ `walk_forward_best.pkl` khi summary có `promotion_eligible=true` mới được
xem là artifact có thể promote; nếu không, seed/champion cũ vẫn nguyên vẹn.

## Diễn giải kết quả

- Train tốt nhưng OOS fail: giữ lại evidence để học tiếp, không promote.
- OOS một cửa sổ tốt nhưng ma trận 3x3 fail: coi là regime luck/overfit.
- OOS qua đủ gate nhưng feature/real-tick fail: vẫn không promote.
- Chỉ khi nhiều cửa sổ và seed cùng qua gate mới coi là edge có khả năng lặp lại.
