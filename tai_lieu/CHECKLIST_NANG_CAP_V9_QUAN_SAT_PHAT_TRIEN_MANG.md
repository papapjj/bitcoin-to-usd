# Checklist kỹ thuật V9 — Quan sát đầy đủ sự phát triển mạng

## A. Hợp đồng action và context-owner

- [x] Mỗi record quyết định có `ma_context`, `ma_chuyen_gia`, `ma_policy` hoặc bị chặn fail-closed.
- [x] Context chưa ổn định/thiếu owner được ghi `KHÔNG_GỬI_LENH`; không chuyển thành HOLD để học.
- [x] Hồ sơ mới không còn `hanh_dong_policy`, `hanh_dong_hop_le`, `policy_action_index`, `legal_action_index`.
- [x] UI chỉ công bố action nghĩa vụ, action phát bridge và action wire MT5.
- [x] Semantic/emitted/wire thiếu hoặc lệch → contract không hoàn chỉnh; không được hiển thị `-1` như action.
- [x] Console bridge không in `policy_action=-1`.
- [ ] Xác minh Strategy Tester: `request_id → semantic_action → wire_action → ticket → fill → close` cho mỗi lifecycle.

## B. Không giới hạn hiển thị và telemetry

- [x] `max_event_bytes=0`: không compact JSONL.
- [x] `max_active_nodes=0`, `max_active_edges=0`: activity đầy đủ.
- [x] `max_state_nodes=0`, `max_state_edges=0`: delta state đầy đủ.
- [x] `max_responsible_neurons=0`, `max_responsible_edges=0`: attribution đầy đủ.
- [x] `LiveEventTailer` Java nhận 0 như không giới hạn dòng/byte.
- [x] `LiveSnapshot` Java không cắt timeline, structural history hay network-development history.
- [x] Canvas 3D không chọn sample node/edge; render toàn bộ topology nhận được.
- [x] Python `LiveEventReader` mặc định không giữ `deque(maxlen=600)` nữa.
- [x] UI khóa `sample=1`, quota neuron=0, quota edge=0 để không ghi thiếu do setting cũ.
- [ ] Kiểm tra heap JVM/RAM khi graph vượt 100k neuron hoặc 1M edge; đây là năng lực máy, không phải quota V9.

## C. Theo dõi phát triển mạng

- [x] Ghi snapshot baseline khi `khoi_tao`.
- [x] Ghi snapshot `bat_dau_thu_nghiem` và `ket_thuc_thu_nghiem`.
- [x] Ghi snapshot tại `quyet_dinh`, `feedback`, `phat_trien`, `vong_doi_lenh`.
- [x] Mỗi snapshot có topology đầy đủ (node/edge), activity, synapse delta, growth event, owner và action contract.
- [x] Nhật ký structural riêng chỉ nhận event topology thật; `graph_credit_attribution` không giả làm “mọc cấu trúc”.
- [x] UI có tab **Theo dõi phát triển mạng** để đọc sâu mọi snapshot.
- [x] UI có tab **Phát triển cấu trúc** cho event tạo/cắt/nối/fade cụ thể.
- [x] 3D ưu tiên highlight neuron/cạnh attribution nhưng không cắt các phần còn lại.
- [x] 3D đánh dấu tím neuron/cạnh vừa thay đổi cấu trúc/state; Inspector có filter `Thay đổi gần nhất`.
- [ ] Xác minh event thật có `node_count_before != node_count_after` hoặc `edge_count_before != edge_count_after` sau growth candidate được giữ.

## D. Kiểm thử đã chạy trong gói V9

- [x] Python compile: `python -m compileall src chay_cli_context_expert_v9.py`.
- [x] Java compile toàn bộ source Swing.
- [x] Test action contract/context owner, explainability, observatory: `23 passed`.
- [x] Test quota=0 giữ toàn bộ attribution.
- [x] Test log `graph_credit_attribution` rỗng không tạo structural event giả.
- [x] Test stream `theo_doi_phat_trien_mang` giữ full topology/activity.
- [ ] Compile EA MQL5 thành `.ex5` trong MT5 MetaEditor.
- [ ] Strategy Tester end-to-end với execution ledger và feature parity 119 input.

## E. Điều kiện nghiệm thu trên máy chạy MT5

- [ ] Tiêu đề UI chứa `V9 Context-Expert`.
- [ ] Command preview chứa `chay_cli_context_expert_v9.py`.
- [ ] Command preview chứa `--live-network-development outputs/theo_doi_phat_trien_mang.jsonl`.
- [ ] Command preview chứa `--live-max-responsible-neurons 0` và `--live-max-responsible-edges 0`.
- [ ] `ho_so_quyet_dinh_chi_tiet.jsonl` không có key `hanh_dong_policy`/`hanh_dong_hop_le`.
- [ ] Không có giá trị action `-1` trong `quyet_dinh.semantic_action_index`, `emitted_action_index`, `wire_action_index`; khi bị block phải là `null` + `KHÔNG_GỬI_LENH`.
- [ ] `theo_doi_phat_trien_mang.jsonl` có snapshot giai đoạn `khoi_tao`, `quyet_dinh`, và nếu có train: `feedback`/`phat_trien`/`vong_doi_lenh`.
- [ ] Số node/edge trong UI bằng số node/edge của `topology_day_du` trong JSONL.
