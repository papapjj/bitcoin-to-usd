# Công cụ khai thác cấu trúc ZigZag cho bộ não năm tác nhân

Công cụ quét dữ liệu nền tảng giao dịch bằng ZigZag **nhân quả đã xác nhận**, tìm số lượng lớn trường hợp của một cấu trúc xác định và tạo tập dữ liệu để huấn luyện bộ não năm tác nhân.

## Hai chế độ

### `mine`

Giữ dạng dữ liệu độc lập cũ để nghiên cứu cấu trúc. Mỗi mẫu chứa cửa sổ giá quá khứ và tương lai.

### `mine-five`

Đây là chế độ nên dùng với dự án hiện tại. Chế độ này đọc trực tiếp giao ước của dự án và bảo đảm:

- đúng `137` giá trị đầu vào của `TradingEnvironment`;
- đúng `5` tác nhân;
- đúng phiên bản sơ đồ đặc trưng;
- đúng chế độ chuẩn hóa đầu vào của cấu hình;
- ATR dùng trung bình động đơn giản của biên độ thực, khớp phần đặc trưng của dự án;
- dữ liệu tương lai không đi vào phần ngữ cảnh của bộ não;
- các trạng thái vị thế, tài khoản và thực thi được **môi trường tạo động sau mỗi hành động**, không được đóng băng từ lúc khai thác dữ liệu.

### `mine-five-selected`

Dùng cho thư mục raw có `manifest.json` do `mt5-fetch-selected` tạo. Lệnh khai thác
từng symbol thành một tape độc lập rồi ghi `dataset_collection.json`; tuyệt đối không
nối các timeline khác symbol. Trainer đọc collection và lấy episode luân phiên giữa
các symbol để giới hạn `--max-episodes` không thiên lệch về symbol đầu tiên.

## Vì sao không lưu sẵn `[thời gian, 5, 137]` cho mọi mẫu?

Trong 137 giá trị có 40 giá trị vị thế, 8 giá trị tài khoản và 11 giá trị thực thi. Các giá trị này phụ thuộc vào hành động mà chính sách thực hiện. Nếu công cụ ghi cứng chúng trước khi huấn luyện thì dữ liệu sẽ không còn khớp môi trường.

Vì vậy chế độ `mine-five` tạo một **băng dữ liệu thị trường dùng chung** chứa các đặc trưng thị trường đã được tính nhân quả trên toàn chuỗi. Mỗi mẫu chỉ lưu:

- chỉ số bắt đầu phần quá khứ;
- chỉ số điểm neo;
- chỉ số kết thúc phần tương lai;
- thông tin cấu trúc;
- kết quả hậu nghiệm để đánh giá.

Khi huấn luyện, `TradingEnvironment` lấy đoạn tương ứng từ băng dữ liệu và tự dựng đủ 137 giá trị theo trạng thái vị thế/tài khoản thực tế.

Cách này cũng phù hợp hơn với mục tiêu hàng chục hoặc hàng trăm triệu cấu trúc vì không nhân bản cùng một dãy nến cho từng mẫu.

## Lấy dữ liệu trực tiếp từ nền tảng giao dịch

```bash
python -m zigzag_miner.cli mt5-fetch \
  --runner-dir mt5_v2_runner \
  --symbol XAUUSD \
  --timeframe M1 \
  --bars 5000000 \
  --output XAUUSD_M1.npz
```

Khi có `--runner-dir`, công cụ mở đúng `terminal64.exe` của bộ chạy ở chế độ portable và tái sử dụng hồ sơ tài khoản đã lưu, giống đường huấn luyện MT5. Nếu terminal đó chưa từng được đăng nhập và lưu mật khẩu, cần đăng nhập một lần trong chính terminal portable rồi quét lại. Công cụ không lưu mật khẩu mới trong dự án.

Dữ liệu lấy trực tiếp có `spread`, khối lượng nhịp và khối lượng thật nếu nền tảng cung cấp. Đây là nguồn nên dùng cho `mine-five`.

Manifest raw nhiều symbol còn lưu contract MT5 của từng mã (`digits`, `point`,
contract size, tick size/value, calculation mode và currencies). Khi khai thác,
`point`/`digits` tương ứng được gắn vào tape và dùng lại khi tính `spread_price`,
feature fingerprint và môi trường. Không dùng `symbol.point_size` chung của config
cho các symbol khác nhau.

## Kiểm tra trước khi khai thác

```bash
python -m zigzag_miner.cli check-five \
  --project-root C:/duong_dan/new-chat \
  --config configs/he_chuyen_gia_latest_ui_v193.yaml \
  --input XAUUSD_M1.npz
```

Lệnh này kiểm tra số đầu vào, số tác nhân, phiên bản sơ đồ, dấu vân tay, chế độ chuẩn hóa, cột chênh lệch giá và dữ liệu khối lượng trước khi tạo tập lớn.

## Tạo tập dữ liệu đúng giao ước dự án

```bash
python -m zigzag_miner.cli mine-five \
  --input XAUUSD_M1.npz \
  --pattern examples/impulse_pullback.yaml \
  --output datasets/XAUUSD_impulse \
  --project-root C:/duong_dan/new-chat \
  --config configs/he_chuyen_gia_latest_ui_v193.yaml
```

Với nhiều symbol:

```bash
python -m zigzag_miner.cli mine-five-selected \
  --input-dir outputs/datasets/mt5_raw_YYYYMMDD_HHMMSS \
  --pattern examples/impulse_pullback.yaml \
  --output outputs/datasets/five_zigzag_multi_YYYYMMDD_HHMMSS \
  --project-root C:/duong_dan/new-chat \
  --config configs/he_chuyen_gia_latest_ui_v193.yaml
```

Nếu dữ liệu ngoài không có cột `spread`, lệnh sẽ dừng thay vì âm thầm gán bằng 0. Chỉ dùng `--default-spread-points` khi bạn thực sự biết mức chênh lệch muốn giả định.

Nếu không có dữ liệu khối lượng, lệnh cũng dừng vì 16 đặc trưng Wyckoff sẽ không thể tái tạo đúng. Có thể dùng `--allow-neutral-volume` cho thử nghiệm, nhưng không nên dùng để tạo tập huấn luyện chính.

## Cấu trúc thư mục đầu ra của `mine-five`

```text
datasets/XAUUSD_impulse/
  dataset_summary.json
  five_market_tape/
    environment_values.npy
    time_ns.npy
    manifest.json
  shard_00000.npz
  shard_00000.jsonl
  shard_00001.npz
  ...
```

Băng dữ liệu dùng tệp mảng không nén để có thể ánh xạ bộ nhớ và cắt chuỗi nhanh. Các mảnh mẫu chỉ chứa chỉ số và nhãn nên rất nhỏ.

Collection nhiều symbol có thêm `dataset_collection.json` ở thư mục gốc và mỗi
symbol nằm trong `symbols/<symbol>/` với đầy đủ tape/manifest/shard riêng. Symbol có
0 mẫu vẫn được ghi nhận trong summary nhưng không tạo episode huấn luyện.

## Dùng với môi trường hiện tại

```python
from zigzag_miner.five_adapter import iter_project_five_episodes

for item in iter_project_five_episodes(
    "datasets/XAUUSD_impulse",
    project_root="C:/duong_dan/new-chat",
    config_path="configs/he_chuyen_gia_latest_ui_v193.yaml",
):
    env = item["environment"]
    context = item["context_observations"]
    anchor = item["anchor_observation"]
```

`context` có dạng `[T,137]` và **không chứa điểm neo**. Nó dùng để dựng ký ức trước quyết định. `anchor` là quan sát 137 chiều tại điểm quyết định. Từ điểm neo trở đi phải lấy quan sát mới trực tiếp từ môi trường sau mỗi hành động; không được dùng dữ liệu tương lai trung tính làm đầu vào.

Bộ não năm tác nhân của dự án sẽ nhân quan sát chung thành năm bản chạy và thêm nhận dạng tác nhân/hành động trước ở bên trong bộ mã hóa.

## MAE/MFE hậu nghiệm

Công cụ hiện ghi rõ cả hai hướng:

- `long_mfe_atr`, `long_mae_atr`, `long_final_return_atr`;
- `short_mfe_atr`, `short_mae_atr`, `short_final_return_atr`.

Ba tên cũ `mfe_atr`, `mae_atr`, `final_return_atr` vẫn được giữ để đọc tập dữ liệu cũ và mang nghĩa **hướng mua**.

Các giá trị tương lai này chỉ dùng làm nhãn/đánh giá hậu nghiệm. Không được đưa vào quan sát tại điểm neo.

## Chống rò rỉ tương lai

- cực ZigZag có `pivot_index` và `confirm_index`;
- điểm neo luôn là `confirm_index` của cực cuối;
- nối thêm dữ liệu tương lai không được thay đổi các cực đã xác nhận trong phần trước;
- phần ngữ cảnh kết thúc **ngay trước điểm neo**;
- đặc trưng của dự án được tính trên toàn chuỗi trước khi cắt mẫu, nhưng toàn bộ công thức đặc trưng là nhân quả và chỉ dùng thông tin đã đóng tại thời điểm tương ứng;
- không chia ngẫu nhiên các mẫu cửa sổ chồng lấn giữa tập học và tập đánh giá. Hãy chia theo khối thời gian hoặc theo mã giao dịch + khối thời gian.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

Để chạy cả kiểm tra nối trực tiếp với dự án:

```powershell
$env:FIVE_PROJECT_ROOT = "C:\duong_dan\new-chat"
$env:FIVE_PROJECT_CONFIG = "configs/he_chuyen_gia_latest_ui_v193.yaml"
python -m unittest discover -s tests -v
```

### Ghi dữ liệu MT5 vào thư mục mới

`mt5-fetch` tự tạo toàn bộ thư mục cha của `--output` trước khi ghi. Việc đặt
`--bars` rất lớn chỉ là giới hạn trên cần yêu cầu; số thanh thực nhận vẫn phụ
thuộc lịch sử mà MT5/nhà môi giới đang cung cấp. Kết quả lệnh in cả
`requested_rows`, `rows` và `history_exhausted` để phân biệt hai giá trị này. Công cụ đọc thêm
`terminal_info.maxbars`, tự giảm `chunk-size` xuống dưới giới hạn mà MT5 chấp nhận và ghi
`limited_by_terminal_maxbars=true` cùng `history_limit_reason=terminal_maxbars` khi yêu cầu lớn
hơn cấu hình **Max bars in chart**. Muốn lấy nhiều hơn phải tăng giới hạn trong MT5 và khởi động
lại terminal; công cụ không tự sửa cấu hình hay đóng terminal đang đăng nhập. Lệnh nhiều symbol
trả exit code khác 0 nếu còn bất kỳ symbol nào thất bại.

Trường `time` trong NPZ mới luôn là Unix epoch nanosecond (`int64`). Loader tự nhận diện và đọc
tương thích các file cũ dùng epoch second/millisecond/microsecond, tránh biến timestamp thành năm 1970.
Thêm `--no-resume` khi cần tải lại và thay thế atomic các NPZ hiện có theo contract mới.
Tệp được ghi qua tệp tạm `.part` rồi thay thế đích sau khi hoàn tất để tránh
để lại NPZ hỏng nếu quá trình bị dừng giữa lúc nén.
