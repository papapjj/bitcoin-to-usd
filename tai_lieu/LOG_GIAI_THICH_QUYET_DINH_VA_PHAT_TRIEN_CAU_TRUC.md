# Nhật ký giải thích quyết định và phát triển cấu trúc

## Mục đích

Bổ sung hai nhật ký audit đọc được bằng tiếng Việt, **chỉ đọc** và không được quyền thay đổi policy, trọng số, action, risk governor, MT5 bridge hay quá trình học:

1. **Hồ sơ quyết định**: vì sao context hiện tại được route tới policy/chuyên gia nào, policy đề xuất action gì, neuron/cạnh/feature nào đóng góp mạnh nhất, action có bị kiểm tra an toàn hoặc bridge sửa không.
2. **Nhật ký phát triển cấu trúc**: vì sao một neuron, liên kết hoặc module được tạo/tắt/nối lại; cấu trúc trước/sau; candidate nào được sinh; các điểm module-need, neural logits, causal/replay evidence và trạng thái promotion.

Hai file là bằng chứng vận hành. Chúng không được xem là bằng chứng nhân quả tuyệt đối hoặc bằng chứng policy đã có lợi thế giao dịch. Lợi nhuận/lỗ chỉ được xác nhận sau execution ledger và lifecycle MT5.

---

# 1. Ba luồng log

| File | Vai trò | Tần suất |
|---|---|---|
| `outputs/quan_sat_chuyen_gia_live.jsonl` | Telemetry gọn cho UI live: topology, activation, decision summary, feedback, lifecycle | Theo `decision_sample_every` |
| `outputs/ho_so_quyet_dinh_chi_tiet.jsonl` | Một hồ sơ đầy đủ cho mỗi request đã được ghi telemetry | Theo `decision_sample_every` |
| `outputs/nhat_ky_phat_trien_neuron_lien_ket.jsonl` | Một bản ghi cho mỗi event mọc/tắt/nối lại neuron, edge hoặc module | Khi có growth/lifecycle event |

Khi dùng `--live-events outputs/neural_live.jsonl` mà không chỉ định hai đường dẫn detail, hệ tự tạo:

```text
outputs/neural_live.ho_so_quyet_dinh.jsonl
outputs/neural_live.phat_trien_cau_truc.jsonl
```

---

# 2. Bật log khi huấn luyện hệ một context — một chuyên gia

Dùng profile:

```text
configs/he_chuyen_gia_mot_context_mot_policy.yaml
```

Lệnh ví dụ trên Windows:

```powershell
python -m self_growing_trader.cli --config configs/he_chuyen_gia_mot_context_mot_policy.yaml huan-luyen-he-chuyen-gia-mt5 `
  --output outputs/he_chuyen_gia.pkl `
  --from-date 2025.01.01 `
  --to-date 2025.02.01 `
  --runner-dir mt5_v2_runner `
  --timeout-sec 300 `
  --model 1 `
  --mt5-visual `
  --live-observatory `
  --live-events outputs/quan_sat_chuyen_gia_live.jsonl `
  --live-decision-profiles outputs/ho_so_quyet_dinh_chi_tiet.jsonl `
  --live-structural-events outputs/nhat_ky_phat_trien_neuron_lien_ket.jsonl `
  --live-max-responsible-neurons 40 `
  --live-max-responsible-edges 64
```

Các option mới:

```text
--live-decision-profiles <đường_dẫn>
--live-structural-events <đường_dẫn>
--live-max-responsible-neurons <số_lượng>
--live-max-responsible-edges <số_lượng>
```

Các option chỉ thay đổi lượng telemetry lưu ra file. Chúng không thay đổi action, score, weight, memory, fitness, risk hoặc MT5.

---

# 3. Hồ sơ quyết định: chuỗi ownership bắt buộc

Mỗi dòng trong `ho_so_quyet_dinh_chi_tiet.jsonl` có cấu trúc cốt lõi:

```text
trace_id
  → context.ma_context
  → context.ma_chuyen_gia
  → context.ma_policy
  → quyet_dinh
  → ly_do
  → dau_vao_anh_huong
  → neuron_chiu_trach_nhiem
  → lien_ket_chiu_trach_nhiem
```

## 3.1. Context và chuyên gia

```json
{
  "context": {
    "ma_context": "CTX_...",
    "ma_context_cha": "CTX_...",
    "ma_chuyen_gia": "CG_...",
    "ma_policy": "POLICY_...",
    "phien_ban_policy": "...",
    "trang_thai_chuyen_gia": "HOAT_DONG_HAN_CHE",
    "context_on_dinh": true,
    "do_tin_cay_context": 0.91,
    "ly_do_context": "..."
  }
}
```

Quy tắc đọc:

```text
ma_context + ma_chuyen_gia + ma_policy có đủ
→ có đúng một policy owner đã được gọi.

CHƯA_CÓ_CHUYÊN_GIA_PHÙ_HỢP hoặc CONTEXT_CHƯA_ỔN_ĐỊNH
→ không policy nào được chạy.
→ HOLD trên wire chỉ là no-op transport.
→ không được coi là policy quyết định HOLD và không được học.
```

## 3.2. Action và lý do đặt hoặc từ chối lệnh

`quyet_dinh` giữ ba tầng action:

```text
hanh_dong_policy
→ policy đề xuất.

hanh_dong_hop_le
→ sau action mask / trạng thái position / legality.

hanh_dong_gui_mt5
→ action cuối được bridge gửi sang MT5.
```

`quyet_dinh.tham_so_lenh` cũng giữ đủ 8 head tham số của vector 20 output:

```text
rui_ro_raw
stop_loss_ATR_raw
take_profit_ATR_raw
offset_vao_lenh_ATR_raw
trailing_ATR_raw
chot_mot_phan_raw
che_do_quan_ly_raw
muc_tieu_vi_the_raw
```

Nếu bridge đã giải mã, cùng record sẽ có `tham_so_da_giai_ma_neu_da_co` như `risk_pct`, `sl_atr`, `tp_atr`, `trailing_stop_atr`, `partial_close_pct`, `management_mode` và `target_slot`. Giá trị raw không được coi là lệnh đã khớp.

`ly_do` ghi các reason code quan trọng:

| Mã | Ý nghĩa |
|---|---|
| `CONTEXT_OWNER` | Context đã được route tới đúng chuyên gia owner |
| `CHON_UNG_VIEN_POLICY` | Candidate được policy chọn theo selector/critic score |
| `ACTION_POLICY` | Action do policy đề xuất |
| `ACTION_MASK_HOAC_PHAP_LY` | Action bị đổi bởi mask hoặc legality |
| `DIEU_CHINH_THUC_THI` | Bridge/MT5 phát action khác action hợp lệ |
| `ACTION_MASKED` | Action gốc bị khóa do trạng thái thực thi/risk |
| `PHAN_BIEN_DOI_UNG_VIEN` | Reasoning graph thay candidate sau phản biện |
| `THAY_THE_HANH_DONG` | Learner recovery thay action/parameter nhờ evidence cục bộ |
| `EXPECTANCY_CUC_BO` | Có expectancy cục bộ và số mẫu tương ứng |
| `GIOI_HAN_RUI_RO_CHUYEN_GIA` | Policy thăm dò bị cap rủi ro riêng |

Không được kết luận “neuron X khiến đặt BUY” chỉ từ action score. Phải đọc đồng thời:

```text
context owner
+ action policy
+ action hợp lệ
+ action gửi MT5
+ risk/legality gate
+ execution ledger
+ lifecycle outcome.
```

## 3.3. Feature đầu vào ảnh hưởng

`dau_vao_anh_huong` xếp hạng feature theo tổng độ lớn đóng góp từ các cạnh input → graph đang hoạt động.

Ví dụ tên đã Việt hóa:

```text
thi_truong.độ_dốc_EMA20_theo_ATR
thi_truong.spread_theo_ATR
thi_truong.mức_hồi_về_EMA20_theo_ATR
vi_the.slot_03
tai_khoan.slot_02
thuc_thi.slot_01
```

Mỗi row có:

```text
chi_so
+ ten
+ gia_tri
+ tong_dong_gop_tuyet_doi
+ tong_dong_gop_co_dau
+ canh liên quan.
```

## 3.4. Neuron chịu trách nhiệm

`neuron_chiu_trach_nhiem` xếp hạng theo:

```text
Tổng |đóng góp| của các cạnh hoạt động đi qua neuron
+ activation hiện tại.
```

Mỗi neuron ghi:

```text
id
local_id
graph_id
module
module_id
region
lifecycle
activation
state
utility
plasticity
fade
tong_dong_gop_tuyet_doi
tong_dong_gop_co_dau
so_canh_dang_hoat_dong
canh_lien_quan
huong_tin_hieu
```

`muc_do_quy_trach` luôn ghi rõ đây là:

```text
xấp xỉ theo đóng góp cạnh hoạt động
```

Không được coi nó là causal proof tuyệt đối.

## 3.5. Liên kết chịu trách nhiệm

`lien_ket_chiu_trach_nhiem` xếp hạng theo `|contribution|` trong trace hiện tại.

Mỗi edge ghi:

```text
id
src / dst
src_region / dst_region
module / module_id
kind
lifecycle
contribution / contribution_abs
gate
weight
utility
plasticity
fade
created_by
need_reason
candidate_id
lien_quan_truc_tiep_dau_ra
```

Nhờ `created_by`, `need_reason`, `candidate_id`, một edge hoạt động mạnh có thể nối ngược về event đã tạo nó.

---

# 4. Nhật ký tạo neuron, liên kết và module

Mỗi event trong `nhat_ky_phat_trien_neuron_lien_ket.jsonl` có:

```text
trace_id
+ context
+ ma_su_kien
+ thoi_diem_su_kien
+ hành động cấu trúc
+ lý do kỹ thuật
+ số neuron/edge trước và sau
+ neuron/edge tạo mới có snapshot
+ nguồn kích hoạt
+ giới hạn diễn giải/promotion.
```

## 4.1. Hành động được ghi

```text
Tạo neuron mới.
Tạo liên kết mới.
Tạo module chuyên môn ứng viên.
Tắt/cắt liên kết utility thấp.
Nối lại liên kết.
Củng cố liên kết.
Tách vùng/module.
Hợp nhất vùng/module.
Đánh giá mọc vi mô theo lifecycle.
Không thay đổi cấu trúc.
```

## 4.2. Lý do tạo cấu trúc

`nguon_kich_hoat` ghi:

```text
vai_tro
điểm kích hoạt
module_need_scores
neural_logits
learning_signal
causal_confidence
causal_validated
```

Một event chỉ được hiểu là:

```text
Controller thấy nhu cầu cấu trúc và tạo candidate.
```

Nó **không** có nghĩa:

```text
Neuron/module mới đã được promote.
Neuron/module mới đã chứng minh tạo lợi nhuận.
```

Promotion vẫn cần lifecycle, replay, OOS và risk gate theo policy/context owner.

## 4.3. Snapshot cấu trúc mới

`su_kien.neuron_duoc_tao` giữ snapshot neuron:

```text
id
region
module_id
bias
plasticity
lifecycle
```

`su_kien.lien_ket_duoc_tao` giữ snapshot edge:

```text
id
src
dst
kind
weight
gate
lifecycle
```

Do đó có thể tìm được:

```text
module_need hoặc causal pattern nào tạo edge
→ edge đó có được active trong decision nào
→ outcome MT5 của decision đó là gì
→ edge được giữ, fade, prune hay rewire như thế nào.
```

---

# 5. Cách nối các file với nhau

Khóa nối bắt buộc là `trace_id`.

```text
Hồ sơ quyết định
  trace_id
      ↓
Action metadata sidecar
  trace_id + context/expert/policy provenance
      ↓
Execution ledger MT5
  request/event/trace provenance
      ↓
Lifecycle
  realized R, close reason, duration
      ↓
Nhật ký phát triển cấu trúc
  trace_id + context + event_id
```

Một câu audit đầy đủ phải trả lời được:

```text
Context nào xuất hiện?
→ Chuyên gia nào sở hữu?
→ Policy nào ra action?
→ Action nào được policy đề xuất?
→ Mask/risk/bridge đã thay đổi gì?
→ Neuron, edge, feature nào đóng góp mạnh trong trace?
→ MT5 có thực sự khớp lệnh không?
→ Lifecycle có lời/lỗ ra sao?
→ Policy owner đã sửa memory/graph/candidate nào?
```

---

# 6. Cảnh báo bắt buộc khi đọc log

```text
Neuron/cạnh được xếp hạng theo attribution của trace hiện tại,
không phải bằng chứng nhân quả tuyệt đối.

Policy đề xuất action không đồng nghĩa MT5 đã mở lệnh.

MT5 mở lệnh không đồng nghĩa lệnh đã đóng hoặc có lifecycle outcome.

FLAT_IDLE, bridge timeout, EA từ chối và context chưa ổn định
không được dùng làm reward/loss learning evidence.

Mọc neuron hoặc edge không đồng nghĩa được promote.

Không có context owner hợp lệ thì không có policy được phép quyết định.
```

---

# 7. Kiểm tra nhanh sau khi chạy

```powershell
Get-Content outputs\ho_so_quyet_dinh_chi_tiet.jsonl -Tail 1
Get-Content outputs\nhat_ky_phat_trien_neuron_lien_ket.jsonl -Tail 1
Get-Content outputs\quan_sat_chuyen_gia_live.jsonl -Tail 5
```

Cần thấy tối thiểu với một policy thực sự chạy:

```text
ma_context khác rỗng
ma_chuyen_gia khác rỗng
ma_policy khác rỗng
neuron_chiu_trach_nhiem có dữ liệu
lien_ket_chiu_trach_nhiem có dữ liệu
```

Với context chưa ổn định phải thấy:

```text
ly_do_khong_giao_dich = CONTEXT_CHUA_ON_DINH
khong_co_policy_ra_lenh = true
```

Đây là trạng thái an toàn, không phải HOLD mà hệ phải học.
