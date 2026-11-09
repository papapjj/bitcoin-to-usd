# Bản vá Live Neuro-Development V15

## Mục tiêu

Bản vá này làm cho cửa sổ Live phản ánh đúng quá trình phát triển của graph candidate trong lúc train MT5:

```text
MT5 order close
→ execution ledger settlement
→ lifecycle credit
→ candidate micro-growth / retain / fade / prune
→ telemetry lifecycle + synapse-state delta
→ Java Live UI cập nhật đồ thị và Timeline
```

Baseline vẫn không được phép mọc trực tiếp. Mọi thay đổi cấu trúc chỉ xảy ra trên `graph_trial_mode=true` để candidate phải vượt promotion gate mới trở thành baseline.

---

## Các lỗi đã sửa

### 1. Lifecycle MT5 không gọi micro-growth

Trước bản vá, `apply_execution_lifecycle_credit()` chỉ gửi credit vào `GraphPlasticityEngine`. Weight/gate/utility có thể đổi, nhưng `GraphSupernetworkRuntime.learn_in_trial()` không được gọi sau khi order đóng.

Đã bổ sung `_live_observatory_execution_micro_growth()`:

- chạy sau khi ledger xác nhận `realized_r`;
- chỉ chạy trên candidate có `graph_trial_mode=true` và `graph_trial_id`;
- dùng outcome thật để tạo `reward`, `cost`, TD surprise và prediction-error proxy;
- outcome âm tạo **replacement pressure**, không giảm action cũ toàn cục;
- ghi các event:
  - `micro_neuron_grown`
  - `micro_neuron_retained`
  - `micro_neuron_fading`
  - `micro_neuron_pruned`
  - `graph_lifecycle_micro_growth_evaluated`

### 2. Lifecycle telemetry không mang growth event

Event `lifecycle` giờ có:

```json
{
  "growth_events": ["..."],
  "synapse_state": {
    "nodes": [],
    "edges": [],
    "removed_node_ids": [],
    "removed_edge_ids": []
  }
}
```

Điều này cho phép Timeline hiển thị chính xác neuron vừa sinh, giữ, mờ hoặc prune ngay sau lúc lệnh được settlement.

### 3. UI chỉ refresh khi node/edge mới xuất hiện

Trước đây `topology_id` cố ý chỉ hash cấu trúc. Vì vậy thay đổi `weight`, `gate`, `utility`, `fade`, `plasticity` không làm UI nhận topology mới.

Đã thêm `synapse_state` delta, bao gồm:

| Node | Edge |
|---|---|
| `bias`, `utility`, `plasticity`, `fade`, `lifecycle` | `weight`, `gate`, `utility`, `plasticity`, `fade`, `enabled`, `lifecycle` |

Delta được giới hạn băng thông bởi:

```yaml
observability:
  max_state_nodes: 512
  max_state_edges: 1024
  state_change_epsilon: 0.000001
```

### 4. Java Live UI không hợp nhất synapse update

`LiveNeuralPanel` giờ merge `synapse_state` trực tiếp vào snapshot đang vẽ:

- edge width/weight được cập nhật;
- inspector hiển thị utility/gate/fade mới;
- candidate/pruning state không phải chờ full topology snapshot;
- Timeline hiển thị `type` hoặc `kind` của micro-growth event.

JAR `ui-java/build/SelfGrowingTraderControl.jar` đã được build lại.

### 5. Active trace dễ gây hiểu nhầm khi topology/tail lệch

Khi bật **Chỉ trace hoạt hóa**, canvas chỉ vẽ node/edge active trong trace mới nhất. Layout 3D xếp nhiều neuron cùng module/region thành vòng Y/Z, nên một cụm M1 đang fire mạnh có thể nhìn như một vòng xanh dày; đây là layout của viewer, không phải model tự biến thành một topology hình vòng.

`LiveNeuralPanel` giờ hiển thị active theo dạng `mapped/raw` nếu trace mới nhất chứa ID chưa map được vào topology snapshot đang vẽ:

- `mapped`: số active đang tồn tại trong topology hiện tại và có thể render/inspect.
- `raw`: số active thô từ event decision/feedback/lifecycle mới nhất.

Nếu hai số khác nhau, legend dưới canvas cũng báo `active mapped/raw ...` để phân biệt lỗi/tình trạng telemetry-viewer với trạng thái học thật của graph.

---

## Cách chạy đúng

### Chạy từ Java UI

1. Bấm **Bật xem Live** trước khi bấm **Run**.
2. UI tự bật **Ghi telemetry khi train**.
3. Bảo đảm `Live events` là:

```text
outputs/neural_live.jsonl
```

4. Trong cửa sổ Live, xem tab **Timeline** và tab thông tin neuron.

### Chạy bằng CLI

Thêm tối thiểu:

```text
--live-observatory --live-events outputs/neural_live.jsonl --live-reset
```

Ví dụ, giữ nguyên lệnh train của bạn và thêm phần này ở cuối:

```bat
python -m self_growing_trader.cli --config configs/default.yaml train-to-goal ... --live-observatory --live-events outputs/neural_live.jsonl --live-reset
```

`--mt5-visual` chỉ mở Visual Strategy Tester của MT5. Nó không thay thế `--live-observatory`.

---

## Những gì phải xuất hiện khi hệ hoạt động đúng

Sau khi một order **đã đóng** và execution ledger gửi lifecycle hợp lệ, `outputs/neural_live.jsonl` phải có event:

```json
{
  "event_type": "lifecycle",
  "payload": {
    "trace_id": "...",
    "realized_net_r": -1.0,
    "growth_events": [
      {"type": "micro_neuron_grown", "source": "mt5_execution_lifecycle"},
      {"type": "graph_lifecycle_micro_growth_evaluated"}
    ],
    "synapse_state": {
      "node_change_count": 1,
      "edge_change_count": 6
    }
  }
}
```

Trong Timeline, bạn sẽ thấy dạng:

```text
lifecycle · micro_neuron_grown, graph_lifecycle_micro_growth_evaluated
```

Không phải mọi lifecycle đều sinh node. Khi growth budget đã đầy hoặc learning signal chưa vượt threshold, Timeline vẫn phải có:

```text
lifecycle · graph_lifecycle_micro_growth_evaluated
```

Điều này chứng minh lifecycle đã đi tới growth controller.

---

## Các điều kiện khiến vẫn chưa thấy mạng phát triển

1. **Không có `actions > 0`**: EA/bridge không gửi decision request, nên không có trace để attribution.
2. **Không có lifecycle thật**: lệnh chưa đóng hoặc execution ledger thiếu `trace_id`.
3. **Đang chạy baseline/seed trial**: baseline được bảo vệ, không mọc trực tiếp.
4. **Candidate không được tạo**: cần ít nhất một trial từ `propose()`; nếu chạy seed trial đầu tiên, trial đầu có thể không grow.
5. **Đang mở Live giữa lượt train đã bắt đầu không có telemetry**: bật toggle chỉ có hiệu lực cho lượt train sau.
6. **Sai path**: Java UI và Python phải đọc/ghi cùng `outputs/neural_live.jsonl` trong cùng workspace.

---

## Kiểm thử đã chạy

```text
PYTHONPATH=src pytest -q \
  tests/test_live_observatory.py \
  tests/test_graph_observability.py \
  tests/test_trial_micro_growth.py \
  tests/test_graph_supernetwork_runtime.py \
  tests/test_desktop_observatory_ui.py

20 passed
```

Đã thêm regression test mô phỏng đúng đường MT5:

```text
cache filled-entry trace
→ final lifecycle settlement
→ candidate micro neuron grows
→ lifecycle telemetry contains growth_events + synapse_state
```
