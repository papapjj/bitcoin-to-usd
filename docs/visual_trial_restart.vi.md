# Visual MT5: tự khởi động lại cho trial kế tiếp

## Vấn đề

MT5 portable chỉ hoạt động ổn định như một instance. Với `--mt5-visual --mt5-close-scope round` hoặc `--mt5-visual --keep-mt5-open`, terminal của trial trước có thể vẫn sống sau khi người dùng đóng cửa sổ Visual. Khi Python launch trial tiếp theo bằng `/config:...ini`, Windows chuyển lời gọi vào terminal cũ; file INI mới không được áp dụng đầy đủ, Strategy Tester có thể chạy nền hoặc MT5 có thể thoát trước khi ghi report.

## Bản vá

Khi `visual=True` và caller yêu cầu `mt5_close_scope="round"` hoặc effective scope `"never"` từ `--keep-mt5-open`, `train_self_growing_mt5()` tự chuyển effective scope sang `trial` và ghi log:

```text
[MT5] visual mode overrides mt5_close_scope=<round|never> -> trial so each subsequent trial reopens Visual.
```

Điều này làm mỗi trial:

1. chạy từ một `terminal64.exe` sạch;
2. ghi `ShutdownTerminal=1` và `Visual=1` vào file INI;
3. đóng/dọn terminal sau trial;
4. khởi động Visual lại cho trial tiếp theo, kể cả khi người dùng đã đóng Visual ở trial hiện tại.

`--keep-mt5-open` vẫn hữu ích cho một lượt debug đơn lẻ, nhưng trong workflow train nhiều lượt với Visual nó không còn được giữ làm scope `never`; runner phải đóng terminal sau từng pass để lượt kế tiếp có report riêng.

## Tương thích bridge

EA trong `mt5_v2_runner` chấp nhận action response schema v2. Python vẫn đọc request schema v3 và execution acknowledgement nếu EA hỗ trợ, nhưng action file trả về v2 để EA v2 không bỏ qua quyết định.

## Sử dụng

Bạn có thể giữ nguyên command cũ có:

```bat
--mt5-visual --mt5-close-scope round
```

Sau bản vá, runtime tự dùng lifecycle trial. Không cần đổi command chỉ để Visual mở lại cho trial tiếp theo.

Nếu command có thêm `--keep-mt5-open`, runtime cũng tự dùng lifecycle trial trong lúc train. Muốn giữ MT5 mở để xem thủ công, dùng một lệnh debug/evaluate đơn lẻ thay vì workflow nhiều round/attempt.
