# Patch manifest — Live Neural 3D V15

## Tệp thay đổi

| Tệp | Thay đổi |
|---|---|
| `ui-java/src/com/selfgrowingtrader/control/LiveNeuralPanel.java` | Thay canvas 2D bằng projection 3D tương tác, thêm layout semantic X/Y/Z, perspective, drag rotate, pan, zoom, auto rotate, reset camera, depth sorting, node/edge active glow và inspector XYZ. |
| `ui-java/src/com/selfgrowingtrader/control/Main.java` | Đổi tên/tooltip cửa sổ Live thành 3D, tăng kích thước tối thiểu và mặc định của cửa sổ Live. |
| `ui-java/README.md` | Cập nhật cách dùng cửa sổ Live 3D và gesture camera. |
| `docs/LIVE_NEURAL_3D_V15.vi.md` | Hướng dẫn vận hành, semantics và giới hạn. |

## Cách áp patch

1. Đóng mọi instance `SelfGrowingTraderControl.exe` hoặc JAR UI đang chạy.
2. Giải nén patch vào root workspace, ví dụ `C:\Users\Admin\Desktop\new-chat`, cho phép ghi đè các tệp cùng tên.
3. Mở CMD ở workspace và chạy:

```bat
ui-java\build-ui.cmd
```

4. Mở file mới tạo:

```bat
ui-java\dist\SelfGrowingTraderControl\SelfGrowingTraderControl.exe
```

Hoặc chạy JAR:

```bat
java -jar ui-java\build\SelfGrowingTraderControl.jar C:\Users\Admin\Desktop\new-chat
```

Không mở shortcut/EXE build cũ ở thư mục khác.
