# Báo cáo kiểm thử V186 — 2026-08-03

> Báo cáo lịch sử của schema 1. Kiến trúc hiện hành là V191 schema 2/artifact 14;
> xem `docs/ACTION_CONDITIONED_ABSTRACTION_V191.vi.md`. Các con số 112.392 và
> artifact 12 bên dưới chỉ mô tả candidate V186 tại thời điểm kiểm thử.

## Kết luận

Implementation V186 đã qua Python debug, migration từ seed artifact thật, toàn bộ regression suite và một MT5 telemetry smoke trên runner clone. Bằng chứng hiện tại xác nhận data-flow, expert isolation, sole-writer và telemetry authority; chưa xác nhận edge hoặc đủ điều kiện promotion.

## Python và migration

- Full suite: `857 passed in 111.59s`.
- V186/retention targeted suite: `32 passed`.
- Synthetic abstraction: hai motif có cùng temporal form nhưng utility trái dấu hội tụ vào hai rule khác nhau; action ranking đúng trên temporal fold thứ ba chưa thấy.
- Seed nguồn: `outputs/he_chuyen_gia/debug/v189_edge_baseline_seed20260802.pkl`.
- SHA256 seed nguồn trước/sau: `BA4130C943BA4A852DD9B82D79D3B091C4555AF28FA3BA0DC89EDE3E2CA82255`, không đổi.
- SHA256 config nguồn trước/sau: `9AACF05B6C7795732F0967616BD99DEBF04925A7E26E3137840CB1ACC7D6A441`, không đổi.
- Candidate migration có 9 expert, 9 bank object và parameter fingerprint độc lập; mỗi bank 112.392 tham số, artifact v12.
- `authority=telemetry` cho output actor giống chính xác bản copy có tensor rule disabled.
- Manifest: `outputs/he_chuyen_gia/debug/tensor_rule_v186_seed_clone_debug/manifest.json`.

## MT5 telemetry smoke

Run: `outputs/he_chuyen_gia/experiments/tensor_rule_v186_mt5_smoke_20260803_v2`.

- Model 1, window 2025.01.01 → 2025.01.02, wall-time 30 giây.
- Bridge phục vụ 58 decision; run cố ý dừng bởi `max_wall_sec`, nên artifact được ghi là `.incomplete.pkl` và không phải candidate promotion.
- 47 feedback trace có TensorRule payload: 10 `insufficient_history`, 37 `low_membership`.
- Cả 47 trace đều `tensor_rule_action_applied=false`; MT5 action vẫn là actor output.
- CTX_RANGING nhận 3 tensor batches với lần lượt 24, 31 và 32 valid samples.
- Effective accumulation 24 → 55 → ready; batch thứ ba nhập rule gradient vào cùng `full_lifecycle_rl_adamw` write.
- `TensorRuleBank.update_count=1`, rule support cuối khoảng 40,71 và 46,29; `unknown_rate=1.0`, nên chưa có rule đủ ổn định để trao authority.
- Có 19 lifecycle settlements, 11 entry-preference receipts, contrastive pair và membership supervision thực sự đi qua batch cuối.
- Source config/seed hash không đổi; không có storage-gate failure; output khoảng 179 MB; runner copy mới được xóa ở lần cleanup đầu tiên.

Run này chứng minh MT5 thật đã tạo tensor trace và ít nhất một AdamW rule update. Nó không có report hoàn tất, frozen baseline pair, policy-origin OOS trades hoặc promotion statistics; do đó tuyệt đối không được dùng để kết luận edge.

## Lỗi cleanup đã sửa

Smoke đầu tiên dùng helper cũ hard-link `metatester64.exe`; bốn agent MT5 có sẵn giữ handle trên cùng file khiến Windows từ chối xóa hard-link clone và làm mất manifest. V186 runner hiện copy executable riêng, cleanup là best-effort và luôn ghi manifest trước khi trả lỗi. Smoke v2 xác nhận runner mới được xóa thành công mà không dừng bốn agent cũ.

## Bước promotion còn thiếu

- shadow comparison cùng seed/window/model;
- real-tick paired OOS 3 seed × 3 window không chồng lấn;
- tối thiểu 30 policy-origin trades mỗi window;
- calibration theo temporal fold;
- policy expectancy LCB95, paired delta LCB, context-action scope, half-window stability và DD gates.

## Bổ sung V190 — 2026-08-04

Candidate profile hiện có thêm State-Aliasing Memory riêng cho mỗi expert và artifact v13. Targeted regression kết hợp Memory, TensorRule, Full-RL, exclusive-writer, Direct-RNN backend, hidden attention và deep hidden đạt `81 passed`; riêng test Memory đạt `13 passed`.

Full regression workspace sau khi tích hợp đạt `874 passed in 179.94s`.

Python candidate debug `outputs/he_chuyen_gia/experiments/state_alias_memory_v190_debug_warm/manifest.json` tạo 9 expert, mỗi expert có 18.009 tham số memory và fingerprint độc lập. Tất cả memory trở thành ready sau warm-up; bridge vẫn 137×20; source config hash trước/sau không đổi. Đây chưa phải MT5 smoke hoặc OOS edge proof.
