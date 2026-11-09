# Báo cáo nâng cấp log giải thích đã thực hiện

## Phạm vi đã làm

Đã thêm hệ audit chỉ đọc cho kiến trúc một context — một chuyên gia — một policy:

```text
context
→ chuyên gia owner
→ policy owner
→ candidate/action
→ action mask / risk / bridge
→ feature
→ neuron
→ liên kết
→ MT5 lifecycle
```

## Thành phần thêm hoặc sửa

```text
src/self_growing_trader/nhat_ky_giai_thich.py
src/self_growing_trader/live_observatory.py
src/self_growing_trader/he_chuyen_gia_ngu_canh.py
src/self_growing_trader/self_growing_runner.py
src/self_growing_trader/graph_development.py
src/self_growing_trader/graph_supernetwork_runtime.py
src/self_growing_trader/cli.py
configs/he_chuyen_gia_mot_context_mot_policy.yaml
tests/test_nhat_ky_giai_thich.py
tai_lieu/LOG_GIAI_THICH_QUYET_DINH_VA_PHAT_TRIEN_CAU_TRUC.md
```

## Log mới

```text
outputs/ho_so_quyet_dinh_chi_tiet.jsonl
outputs/nhat_ky_phat_trien_neuron_lien_ket.jsonl
```

## Bảo đảm kỹ thuật đã thêm

```text
- Gắn metadata context/chuyên gia/policy trước khi policy emit telemetry.
- Không để log quyết định làm thay đổi action hoặc route.
- Context chưa ổn định ghi no-policy/no-learning, không biến thành HOLD để học.
- Mỗi decision profile có top neuron, edge và feature theo đóng góp trace.
- Mỗi event structural giữ lý do, before/after, event ID, snapshot neuron/edge tạo mới.
- Có CLI cấu hình riêng cho decision profile, structural event và số lượng neuron/edge lưu.
- Không thay action bridge, risk governor, policy weight, memory hay MT5 execution vì telemetry.
```

## Kiểm thử đã chạy

Lệnh:

```text
PYTHONPATH=src pytest -q \
  tests/test_nhat_ky_giai_thich.py \
  tests/test_live_observatory.py \
  tests/test_he_chuyen_gia_ngu_canh.py \
  tests/test_graph_development.py \
  tests/test_graph_supernetwork_runtime.py \
  tests/test_flat_idle_firebreak_v27.py \
  tests/test_walk_forward_protocol_v10.py
```

Kết quả:

```text
48 passed
```

## Giới hạn còn lại của archive hiện tại

Archive ZIP hiện tại không chứa thư mục `mt5_v2_runner`, `mt5_v2_runner_current` hoặc source EA `.mq5`. Vì vậy không thể trong môi trường này:

```text
- compile EA;
- chạy Strategy Tester;
- xác minh feature parity từ EA;
- xác minh ledger MT5 thật sau lệnh.
```

Một số test `test_mt5_runner.py` cần các runner/source EA này nên không thể chạy trên archive hiện tại. Đây là thiếu artifact MT5 trong ZIP, không phải lỗi của log Python mới.

## Bổ sung UI Java Swing — V16

Đã cập nhật `ui-java/src/com/selfgrowingtrader/control/LiveNeuralPanel.java` để UI không còn chỉ đọc telemetry tổng quát:

- tail và render `neural_live.ho_so_quyet_dinh.jsonl` trong tab **Giải thích quyết định**;
- tail và render `neural_live.phat_trien_cau_truc.jsonl` trong tab **Phát triển cấu trúc**;
- tô cam neuron/liên kết attribution trong canvas 3D;
- ưu tiên render attribution khi graph bị giới hạn node/edge;
- bổ sung filter **Chịu trách nhiệm quyết định** và attribution detail trong inspector neuron;
- build lại `ui-java/build/SelfGrowingTraderControl.jar`.

Xác nhận kỹ thuật:

```text
javac --release 17: pass
UI explainability smoke: pass
pytest phần explainability/context/live observatory: 35 passed
```
