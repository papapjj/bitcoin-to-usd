# Checklist kỹ thuật V6 — Một context, một chuyên gia, một policy

## A. Contract

- [x] Config bắt buộc chế độ `mot_context_mot_chuyen_gia`.
- [x] Cấm policy chung và cấm trộn policy.
- [x] CLI chỉ lộ các pipeline context-owner.
- [x] UI chỉ tạo command context-owner.
- [x] Artifact ngoài contract bị từ chối khi nạp.

## B. Owner và thực thi

- [x] Context chưa ổn định → `KHÔNG_GỬI_LENH`, không học.
- [x] Trace phải có context, chuyên gia, policy, phiên bản và cờ ổn định.
- [x] Bridge fail-closed nếu thiếu owner.
- [x] Một request chỉ gọi một policy owner.
- [x] Không voting, blending, fallback hoặc bootstrap action.
- [x] `FLAT_IDLE`, timeout và EA từ chối không tạo credit neural.

## C. Cách ly chuyên gia

- [x] Có 5 expert gốc độc lập.
- [x] Mỗi expert có policy, graph, memory, fitness và growth riêng.
- [x] Context con có owner policy độc lập.
- [x] Lifecycle credit chỉ về expert owner.
- [x] Child candidate không được trở thành policy live thứ hai.

## D. Quan sát

- [x] Hồ sơ quyết định ghi context → expert → policy provenance.
- [x] Log ghi feature, neuron, edge và tham số action.
- [x] Trace thiếu owner hiển thị `KHÔNG_GỬI_LENH`.
- [x] Attribution chỉ dựa trên trace trực tiếp của owner.
- [x] Structural journal gắn context-owner và snapshot before/after.

## E. Xác minh tại Windows/MT5

- [ ] Chạy `KHOI_DONG_UI_CONTEXT_EXPERT_V6.cmd`.
- [ ] Kiểm tra tiêu đề có `V6 Context-Expert · Một context, một chuyên gia`.
- [ ] Xác minh command preview chỉ chứa pipeline context-owner.
- [ ] Compile EA và ghi hash source/EX5 vào contract.
- [ ] Chạy MT5 smoke với reset telemetry.
- [ ] Xác minh `hop_dong_owner.hop_le=true` trước mọi order ticket.
- [ ] Chạy feature parity 119 input và walk-forward OOS.
