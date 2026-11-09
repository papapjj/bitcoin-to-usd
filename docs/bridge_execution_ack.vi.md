# Bridge execution acknowledgement — schema v4

Xem đặc tả execution hiện hành tại [contract hình học lệnh học được V15](MT5_EXECUTION_CONTRACT_LEARNED_GEOMETRY_V15.vi.md).

## Quy tắc hiện hành

- EA/Python bridge dùng schema `4`.
- Action vector vẫn có 20 outputs; action 11 quản lý vị thế theo target/mode đã học.
- Ba head độc lập chọn SL, TP theo RR và trailing; các lựa chọn được đóng gói vào
  row 16. Marker 7 dùng codebook TP từ 0,25R đến 5R để học pivot target không ép
  minimum-RR; marker 6 vẫn giữ nguyên năm bin 2R–5R.
- Checkpoint cũ khởi tạo trung tính tại `2.5/5.0 ATR`, trailing tắt; `FixedSlAtr/FixedTpAtr` là fallback cho wire cũ.
- `HOLD` là no-op tuyệt đối.
- EA ghi lifecycle event từ `OnTradeTransaction()` vào `self_growing_execution.csv`.
- `execution_pipe` chỉ aggregate event của intent ngay trước đó, kèm `ack_sequence`.
- Python chỉ nhận execution credit khi `ack_sequence == before.sequence`.
- Snapshot thay đổi một mình không đủ để credit schema v4; điều này chặn attribution sai khi SL/TP hoặc safety close xảy ra trong thời gian chờ bridge.

## Không tương thích

EA schema v1–v3 và policy 19-output không tương thích v4 runtime. Runtime chỉ dùng `SelfGrowing_MT5_Trainer_EA.mq5`.
