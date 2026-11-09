# Checklist neural development controller

## Giữ cứng vì là contract/an toàn

- [x] Giữ `INPUT_COUNT` / `OUTPUT_COUNT` và schema EA bridge.
- [x] Giữ MT5 Strategy Tester là promotion gate cuối.
- [x] Giữ hard reject cho report lỗi, không có trade, thiếu telemetry và drawdown vượt ngưỡng.
- [x] Giữ action enum để EA hiểu lệnh rõ ràng.

## Đã neural hóa thêm

- [x] Thay rule `diagnose_to_spec()` bằng `NeuralDevelopmentController`.
- [x] Controller neural sinh structural action: `hold/grow/prune/split/merge/consolidate`.
- [x] Controller neural sinh role/spec module mới: role, input sources, output targets, gate bias, plasticity.
- [x] `propose()` không còn gọi `module.mutate()` hoặc `dev_controller.mutate()` để tạo trial chính.
- [x] Residual noise trong curiosity không còn lấy trực tiếp từ `mutation_sigma`; nó do neural development output điều khiển.
- [x] `mutation_sigma` trong proposal chỉ biến đổi action rows/output nodes có bằng chứng gây lỗ từ StepTrace hoặc causal memory xấu; nếu trace có attribution thì chỉ sửa hidden connection hoặc graph edge active tạo action lỗ, không nhiễu global gate/recurrent/output.
- [x] Growth module mới dùng neural spec và biến đổi weight/gate có kiểm soát, không dùng random mutate.
- [x] Connection growth dùng score từ developmental RNN thay vì chọn random closed edge.
- [x] Split module chỉ chạy khi neural development chọn action `split`; không mutate module split.
- [x] Policy cũ load lại tự tạo `NeuralDevelopmentController` nếu thiếu.
- [x] `SpatialWorldPolicy` copy/chuyển core giữ controller neural mới.
- [x] Thêm `NeuralActionParameterHead` để sinh nhóm output tham số risk/SL/TP/trailing/partial/mode/slot bằng neuron trước khi đi qua EA schema.
- [x] Thêm `NeuralRiskCostGate` để dampen action mở lệnh và tăng hold/close khi mạng cảm nhận cost/risk xấu.
- [x] Thêm `NeuralPositionOutcomeClassifier` cho lifecycle tags; hard lifecycle rule chỉ còn là audit/teacher ban đầu.
- [x] Thêm `NeuralUtilityNetwork` để học utility window/structure, dùng làm structural signal mềm cho module utility và neural development.

## Còn cố ý giữ hoặc làm sau

- [ ] `mt5_fitness()` vẫn là công thức cứng vì đang là thước đo khách quan/window-level và gate an toàn.
- [ ] `decode_outputs()` vẫn giữ range/slot cố định để không phá EA bridge; output tham số trước đó đã do neural head sinh.
- [ ] Lifecycle telemetry MFE/MAE/nearTP/giveback vẫn được giữ làm audit vì cần giải thích lỗi lệnh.
- [ ] Split/merge vẫn dùng similarity metric để chọn cặp module an toàn; quyết định gọi split/merge đã chuyển sang neural.
