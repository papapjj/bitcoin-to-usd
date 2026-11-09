# Patch manifest — Java Swing Live Neural UI V14.3

## Mục tiêu

Đưa toggle `Bật xem Live / Tắt xem Live` và panel quan sát neural trực tiếp vào Java Swing UI gốc `ui-java`.

## File thêm hoặc thay thế

```text
ui-java/src/com/selfgrowingtrader/control/Main.java
ui-java/src/com/selfgrowingtrader/control/LiveNeuralPanel.java
ui-java/src/com/selfgrowingtrader/control/MiniJson.java
ui-java/build-ui.ps1
ui-java/build-ui.cmd
ui-java/README.md
ui-java/build/SelfGrowingTraderControl.jar
```

## Cách áp dụng

1. Giải nén patch vào root workspace `C:\Users\Admin\Desktop\new-chat` và cho phép overwrite `ui-java`.
2. Cài JDK 17+ nếu máy chưa có `javac`/`jpackage`.
3. Mở PowerShell tại workspace:

```powershell
.\ui-java\build-ui.cmd
```

4. Chạy UI mới:

```powershell
.\ui-java\dist\SelfGrowingTraderControl\SelfGrowingTraderControl.exe
```

Hoặc chạy JAR trực tiếp:

```powershell
java -jar .\ui-java\build\SelfGrowingTraderControl.jar C:\Users\Admin\Desktop\new-chat
```

## Dùng Live

- Tick `Ghi telemetry khi train`.
- Giữ `Live events = outputs/neural_live.jsonl` hoặc nhập path khác.
- Bấm `Bật xem Live`.
- Chạy pipeline. Command preview phải có `--live-observatory`.
- Tab `Mạng sống` sẽ cập nhật ngay khi trainer ghi event.

## Lưu ý

- Nếu telemetry không được bật lúc Python process đã start, toggle UI không thể bật trace giữa run đó. Dừng run hoặc chờ run tiếp theo.
- `Tắt xem Live` không dừng process Python/MT5 và không xóa trace.
- `actions=0` nghĩa EA/bridge chưa tạo decision request; trong tình trạng đó telemetry chỉ có run/topology/trial, không có activation live.
