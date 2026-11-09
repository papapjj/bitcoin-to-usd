# Patch manifest — Live Neural Neuron Inspector V15.1

## File thay đổi

```text
ui-java/src/com/selfgrowingtrader/control/LiveNeuralPanel.java
```

## File thêm

```text
docs/LIVE_NEURAL_NEURON_INSPECTOR_V15_1.vi.md
docs/LIVE_NEURAL_NEURON_INSPECTOR_V15_1_PATCH_MANIFEST_2026-06-27.vi.md
docs/LIVE_NEURAL_NEURON_INSPECTOR_V15_1_TEST_REPORT_2026-06-27.vi.md
README_APPLY_PATCH_V15_1.txt
```

## Cách áp dụng

1. Đóng mọi cửa sổ `SelfGrowingTraderControl.exe` hoặc JAR cũ.
2. Giải nén patch vào root dự án:
   `C:\Users\Admin\Desktop\new-chat`.
3. Chấp nhận ghi đè `LiveNeuralPanel.java`.
4. Build lại bằng:

```bat
ui-java\build-ui.cmd
```

5. Mở EXE mới trong `ui-java\dist\SelfGrowingTraderControl\SelfGrowingTraderControl.exe`.

Không chép đè source mà mở EXE/JAR cũ sẽ không thấy nút `Thông tin neuron`.
