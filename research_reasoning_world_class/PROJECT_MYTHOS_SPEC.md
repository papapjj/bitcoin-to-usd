# Đề Án Tối Thượng: Hệ Thống Suy Luận "Mythos của Claude" Cho xLSTM 17.5M

> **Mục tiêu số học cốt lõi (Kiểm định 3 tháng Walk-Forward)**:
> 1. **Tăng trưởng tài khoản**: $\ge \mathbf{50\times}$ (Lợi nhuận ròng $\ge +5,000\%$ vốn).
> 2. **Profit Factor (PF)**: $\ge \mathbf{3.0}$.
> 3. **Tỷ lệ Thắng (Win Rate)**: $\ge \mathbf{85.0\%}$.
> 4. **Tần suất giao dịch**: Tối thiểu $> \mathbf{4,000\text{ lệnh}}$ đóng hợp lệ.
> 5. **Max Drawdown (DD)**: Tuyệt đối $< \mathbf{15.0\%}$ (Mục tiêu kiểm soát nội bộ $< 10\%$).

---

## 1. Yêu Cầu Kỹ Thuật (Architecture Modules)

### R1. Bộ Máy Nhận Thức & Suy Luận Đa Bước (Metacognitive Multi-hop Reasoning)
- Xây dựng chuỗi lập luận nơ-ron (Chain-of-Thought): Mô hình tự sinh biểu diễn logic nhiều bước trước khi phát tín hiệu lệnh.
- Cơ chế phản biện giả thuyết (Adversarial Self-Reflection): Mô phỏng 3-5 kịch bản giá tiếp diễn và đo lường độ suy giảm xác suất theo thời gian thực.

### R2. Giải Mã Vi Cấu Trúc Thanh Khoản & Dòng Tiền Cá Mập (Deep Order Flow & Liquidity Engine)
- Tích hợp các bộ dò bẫy giá định chế (SMC/ICT): Quét thanh khoản (Liquidity Sweeps), Vùng mất cân bằng cung cầu (Fair Value Gap - FVG), và Khối lệnh tổ chức (Order Blocks - OB).
- Triệt tiêu hoàn toàn rủi ro bị bẫy trong các đợt bứt phá giả (False Breakout) và biến động giật quét stop-loss.

### R3. Hệ Thống Suy Luận Tương Quan Vĩ Mô Đa Tài Sản (Cross-Asset Macro Reasoning)
- Lượng hóa đồng thời tác động của: Chỉ số Đô la Mỹ (DXY), Lợi suất trái phiếu chính phủ Mỹ (US10Y), Vàng (XAUUSD) và Chỉ số S&P 500 (SPX) lên động lực giá của Crypto.
- Nhận diện sớm các pha luân chuyển dòng tiền (Risk-On / Risk-Off) toàn cầu.

### R4. Tối Ưu Hóa Độ Trễ Thực Thi Thời Gian Thực Trên MT5 (Real-time Inference $\le$ 50ms)
- Tối ưu hóa tensor bằng JIT Compilation và ONNX Runtime INT8/FP16.
- Đảm bảo toàn bộ pipeline suy luận đa tầng từ lúc nhận tick nến đến khi gửi gói lệnh qua Transport Bridge của MT5 không vượt quá 50ms.

### R5. Quản Lý Vốn Lãi Kép Phân Đoạn Kelly Động & Co Giãn Rủi Ro Tuyến Tính (Fractional Kelly Dynamic Compounding & Continuous Risk Scaling)
- **Cơ chế Tăng trưởng Lãi kép**: Kích hoạt công thức co giãn Lot size theo hàm số mũ của Equity $(1 + r)^N$, áp dụng phân đoạn Kelly thích ứng ($f^* = 15\% - 25\%$ Full Kelly) nhằm hiện thực hóa mục tiêu $\times 50$ tài khoản trong 4,000 lệnh.
- **Kiểm Soát Drawdown Liên Tục (Không ngắt quãng giao dịch 24/7)**:
  - *Co giãn Lot mượt mà (Continuous De-leveraging)*: Quy mô rủi ro mỗi lệnh tự động điều chỉnh theo độ sâu rút vốn: $\text{Risk} \propto \max(0.1, 1 - \frac{\text{Current DD}}{15\%})$. Càng gần mốc $15\%$, lot size càng thu nhỏ về mức an toàn tối đa nhưng hệ thống vẫn vào lệnh liên tục 24/7 để tận dụng các cơ hội phục hồi lợi nhuận ngay lập tức, không dừng hay đóng băng giao dịch.
  - *Chế độ Tuyển Lựa Sniper A+ khi DD gia tăng*: Khi Drawdown vượt $8\%$, nâng ngưỡng xác suất chấp thuận lệnh lên $P(\text{Win}) \ge 90\%$ để dồn vốn vào các lệnh có độ chắc thắng cao nhất.

### R6. Ma Trận Đa Tài Sản & Đa Khung Thời Gian (Multi-Symbol & Multi-Timeframe High-Frequency Routing)
- Mở rộng phân tích từ 1 cặp đơn lẻ sang rổ 6 tài sản có thanh khoản cao nhất: **BTCUSD, ETHUSD, SOLUSD, XAUUSD (Vàng), EURUSD, US30** trên khung thời gian kết hợp M1 (Micro-Scalping) và M5 (Market Structure).
- Phân bổ tìm kiếm cơ hội 24/7 nhằm đạt tần suất $45 - 60$ lệnh chuẩn xác/ngày, đáp ứng đầy đủ điều kiện mẫu thống kê $> 4,000$ lệnh trong 90 ngày.
- **Bộ lọc Triệt tiêu Tương quan Đồng pha**: Tuyệt đối không mở các vị thế cùng chiều rủi ro trên các cặp có hệ số tương quan $\rho > 0.85$.

### R7. Bộ Lọc Vi Cấu Trúc Khớp Lệnh & Bù Trượt Giá (Microstructure Execution & Spread-Decay Shield)
- Bảo vệ lợi nhuận trước sự bào mòn của phí sàn: Chỉ mở lệnh khi tỷ số kỳ vọng $\frac{\text{Expected Alpha}}{\text{Spread} + \text{Commission} + \text{Slippage}} \ge 3.5$.
- Áp dụng kỹ thuật giả lập lệnh Limit / Sniper Entry tại điểm hồi quy thanh khoản nhằm triệt tiêu tối đa trượt giá.

### R8. Cơ Chế Tranh Luận Đối Kháng Đa Tác Nhân (Bull-Bear Adversarial Debate with Process Reward Verifier)
- Tích hợp kiến trúc 2 tác nhân phản biện nội bộ: **Bull Agent** (tìm kiếm luận điểm Mua) và **Bear Agent** (săn lùng bẫy phân phối và rủi ro đảo chiều).
- Trọng tài phán quyết (**Process Reward Model - PRM**) chỉ chấp thuận vào lệnh khi một phe chứng minh được lợi thế áp đảo với độ tin cậy $\ge 85\%$.

---

## 2. Tiêu Chuẩn Nghiệm Thu Đo Lường (Acceptance Criteria)

- [ ] **AC-1 (Siêu Tăng Trưởng Vốn)**: Đạt mức tăng trưởng ròng $\ge \mathbf{5,000\%}$ ($\ge \mathbf{50\times}$ tài khoản) trong chu kỳ 3 tháng.
- [ ] **AC-2 (Chỉ Số Lợi Nhuận)**: Đạt $\text{Profit Factor (PF)} \ge \mathbf{3.0}$.
- [ ] **AC-3 (Độ Chính Xác Siêu Đẳng)**: Tỷ lệ lệnh thắng $\text{Win Rate (WR)} \ge \mathbf{85.0\%}$ trên tổng số lệnh đóng.
- [ ] **AC-4 (Quy Mô Mẫu Thống Kê)**: Tổng số lệnh giao dịch thực hiện $\ge \mathbf{4,000\text{ lệnh}}$ phân bổ trên rổ đa tài sản.
- [ ] **AC-5 (Kỷ Luật Rủi Ro)**: $\text{Maximum Relative Drawdown} < \mathbf{15.0\%}$ trong suốt toàn bộ quá trình giao dịch.
- [ ] **AC-6 (Tốc Độ Thực Thi)**: Độ trễ suy luận $\le \mathbf{50\text{ms}}$/quyết định trên nền tảng MetaTrader 5.
- [ ] **AC-7 (Bảo Toàn Số Học)**: Tuyệt đối $0$ lỗi NaN, $0$ lỗi Inf trong toàn bộ các tensor nơ-ron và trọng số attention.
