# UI V193 · Five Bridge v10

Bản mã nguồn này đã có JAR mới tại
`ui-java/build/SelfGrowingTraderControlV193.jar` và có thể mở bằng
`ui-java/run-ui.cmd`. Trên Windows, chạy `ui-java/build-ui.cmd` để tạo lại EXE
từ đúng mã nguồn hiện tại.

Khi EXE đang mở và app-image bị Windows khóa, có thể chỉ build JAR đã kiểm thử
bằng `powershell -File ui-java/build-ui.ps1 -JarOnly`, rồi mở bản mới qua
`ui-java/run-ui.cmd` sau khi workflow hiện tại kết thúc.

UI giữ nguyên các workflow context-expert: candidate, Fresh/Frozen,
Champion/Shadow, walk-forward OOS, quản trị artifact và telemetry/Live 3D.
Tab Train candidate hiện hiển thị rõ teacher reward đang dùng
`gpt-5.6-luna · xhigh`. Mỗi terminal action có target trace/ticket và outcome
hợp lệ được đưa ngay vào hàng Codex nền; `0 = không giới hạn`, nên không còn chỉ chấm hai case
cuối lượt. Tab receipt đọc cả status live (`submitted/pending/graded/updates`)
và JSONL append-only. Trạng thái vẫn candidate-only, không được hiểu là bằng
chứng OOS hay quyền promote.
Khi config đang là `configs/he_chuyen_gia_codex_teacher_candidate.yaml`, UI bắt
buộc dùng `configs/experiments/codex_teacher_every_close_candidate.yaml` để giữ
`mt5_learning_enabled=true`; profile edge-discovery frozen không được ghép nhầm
với teacher.
Candidate mặc định chọn **Reasoning RL V1 · action + optimizer**; có thể chuyển về
**V193 tìm edge · không reasoning** hoặc **Codex teacher · checkpoint nguồn cố định**
ngay trong cùng tab, không tạo thêm tab workflow. Dropdown này là nguồn quyết định
config candidate; trường `Config nền` ở tab nâng cao không còn có thể vô tình ép
Candidate vào Codex do giá trị cũ còn lưu trong phiên UI.
Reasoning active dùng overlay
`configs/experiments/reasoning_rl_v1_active_candidate.yaml`: `action_scale=0.25`,
`mt5_learning_enabled=true`, encoder credit vẫn bằng 0 và writer duy nhất là
FiveAdam. Config nền `configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml`
vẫn giữ scale 0 để chạy telemetry-only trực tiếp khi cần đối chứng.
Khi dùng Reasoning RL, ô **Bộ nhãn bảy lựa chọn** cho phép chọn đúng một trong ba
hợp đồng đã ghép với dữ liệu: `sl500 · giữ 12 nến`, `sl500 · giữ 24 nến` hoặc
`sl500 · giữ 48 nến`. UI tự truyền đúng tệp cấu hình tương ứng và lưu lựa chọn
trong `.context-expert-v193-ui.properties`; không được ghép checkpoint của một
hợp đồng với thời gian giữ hoặc mức cắt lỗ của hợp đồng khác.
Tùy chọn **MT5 Visual** được truyền vào cả workflow Candidate `mt5-smoke`; khi
bỏ chọn, Strategy Tester tiếp tục chạy nền để giữ tốc độ. Thay đổi tùy chọn chỉ
có hiệu lực với lượt Tester được khởi động sau đó, không thể đổi giữa một lượt
đang chạy.

Tab **Codex teacher** ghi log từng terminal action đã có outcome: action/family,
target trace/ticket, quality rubric, `teacher reward` signed và confidence. Khi
status có update tương ứng, UI nối cùng `receipt_sha256` để hiển thị thêm
`scaled reward`, `advantage` và `PPO ĐÃ CẬP NHẬT`; reward dương được gắn nhãn
**THƯỞNG**, reward âm gắn nhãn **PHẠT**, còn reward bằng 0 hoặc chưa có điểm
được tách riêng. Cuối bảng có tổng `THƯỞNG/PHẠT/TRUNG TÍNH/CHƯA CHẤM` để không
nhầm số receipt đã chấm với số action thật sự đã gửi vào queue; UI hiển thị
riêng `đã gửi/đã chấm/đang chờ`.
Mỗi receipt mới còn có block **CODEX CLI**: `process`, `returncode`, số byte
stdout/stderr, trạng thái `json_parse`, `receipt_validation`, command đã gọi,
`stdout preview` và `stderr tail`. Receipt của các run cũ trước khi bật telemetry
này chỉ có reward/rationale và được đánh dấu là thiếu execution metadata, không
được suy ngược thành một lần chạy CLI thành công.

Trước khi MT5 mở Strategy Tester, candidate có Codex teacher phải qua preflight
giới hạn một probe structured-output: UI/status ghi rõ `executable_path`,
`cli_version`, `model`, `reasoning_effort`, stage và lý do. Resolver ưu tiên
binary Codex Desktop, sau đó mới tới `codex.exe`/npm shim; preflight hỏng sẽ
dừng fail-closed và không đánh dấu `five_codex_teacher_effective`. Nếu file
status chính bị khóa trên Windows, runner retry ngắn rồi ghi
`*.status.latest.json`; UI đọc cả hai. Mỗi lỗi của từng packet được ghi cùng
JSONL receipt với `packet_sha256`, trace/ticket, action, stage, returncode,
stderr và failure type, nên `submitted → execution → parsed → validated →
hypothesis → update` có thể đối chiếu từng case. Bộ đếm **Generic PPO updates**
và **Codex optimizer updates** được hiển thị riêng; chỉ update có receipt
  được verifier cục bộ xác nhận mới được tính là Codex update. Phần tổng hợp
  **ACCOUNT FEEDBACK** hiển thị riêng `records/confirmed/partial`, tổng reward
  account đã đi vào learner, số parameter đã đổi, số probability shift và max
  shift; các counter này được ghi cả trong status JSON và runtime contract sau
  khi run finalize. Với receipt entry, block **UPDATE AUDIT** còn ghi riêng
 `factorized_geometry_update`, danh sách head đang được trace dùng
  (`delay/risk/entry_offset/sl/tp/trailing` hoặc `delay` cho management), hash
  geometry trước/sau, `geometry_changed`, update norm và auxiliary ratio. Hash
  geometry giữ nguyên khi probability guard rollback; vì vậy `updates>0` không
tự động có nghĩa là SL/TP/risk đã thay đổi. Nếu model thỉnh thoảng rút gọn `packet_sha256`, runner chỉ
retry tối đa hai lần với cùng packet bất biến và yêu cầu echo đủ 64 ký tự; hash
sai sau retry cuối vẫn bị reject, không được tự sửa để tạo update.
Trong **ACCOUNT FEEDBACK**, `cashflow confirmed` là bằng chứng tiền của đúng
`deal_ticket` (`profit + commission + swap = net_pnl`) khi lifecycle chỉ có một
final close; `balance/equity mark confirmed` chỉ đếm snapshot balance/equity
duy nhất. Hai số này cố ý tách nhau: nhiều close cùng batch có thể giữ
`account_mark_attribution=ambiguous_same_close_batch` nhưng vẫn có cashflow
riêng theo deal. Receipt chi tiết hiển thị `cashflow Δ`, `deal_ticket` và
`evidence_scope`; không coi cashflow là snapshot balance/equity hay
counterfactual HOLD.
Khi factorized PPO ratio nằm ngoài vùng clip `[0.80,1.20]`, UPDATE AUDIT còn ghi
`clip_correction=true/<value>` và `correction_scale=0.05`. Đây là directional
correction có neo vào log-prob cũ cho các head geometry; nó vẫn bị giới hạn bởi
packet probability guard, entry-mass floor và rollback, không phải nới guard.
Nếu profile bật recovery snapshot, status còn có
`recovery_snapshot_updates`, `recovery_snapshot_path` và
`recovery_snapshot_error`. Snapshot nằm ở `<candidate>.recovery.pkl`, luôn
ghi `promotable=false` và chỉ là điểm cứu hộ research-only; nó không thay thế
candidate cuối hoặc bằng chứng OOS. `-JarOnly` chỉ cập nhật JAR, không thay
EXE/app-image đang mở.
Khi trường output candidate đã trỏ tới một thư mục cụ thể, tab này chỉ tìm
receipt/status trong đúng thư mục đó (kể cả sidecar `.latest.json`); nó không
fallback sang `mt5_v2_runner/Reports` dùng chung, vì nơi đó có thể còn receipt
cũ với executable hoặc model khác. Fallback Reports chỉ dành cho chế độ không
chọn output run cụ thể.
Block `UPDATE AUDIT` cũng hiển thị norm của bước cập nhật action head và delta
logit trên action mục tiêu; receipt cũ không có hai trường này sẽ hiện `—`.
Mỗi case còn có block **ACCOUNT IMPACT**: `status=confirmed` chỉ xuất hiện khi
packet có snapshot balance/equity tại quyết định và sau close; `partial` chỉ có
immutable `realized_r/net_pnl`, còn `inconclusive` không được dùng để tự nhận
account-level confirmation. UI cũng hiển thị `mark attribution` (`unique_final_deal`
hoặc `ambiguous_same_close_batch`) để phân biệt mark theo một deal với snapshot
dùng chung cho nhiều close. Nếu execution ledger có post-exit hold observation,
UI hiển thị thêm `action-minus-hold R`; đây là counterfactual riêng, không được
nhầm với dấu `realized_r`. Scorer đưa tín hiệu account-impact có dấu vào reward
trước optimizer, nhưng vẫn giữ probability-shift và frozen-reference guards.
Credit không dùng chung cho mọi owner: entry owner có thể kết hợp realized R
với counterfactual như context bounded; management/exit owner chỉ dùng delta
`action-minus-hold R` khi đã validated và hiển thị
`signal source=management_counterfactual_delta`. Vì vậy realized P/L dương của
cả lifecycle không tự biến một lệnh đóng kém hơn hold thành reward dương.
Status cũng tách `counterfactual validated/unavailable` và
`deferred/released/pending/forced partial`, để biết case đang chờ
`POST_EXIT_OBSERVATION` hay đã bị buộc dùng final-close evidence ở terminal.
Codex update mặc định dùng scope
`codex_context_action_residual_v3_internal_policy_state_global_bias`:

- `codex_action_residual` và `codex_action_bias` giữ intervention bounded theo
  packet;
- `heads.internal_policy_state_net` cũng nằm trong cùng snapshot/rollback và
  FiveAdam scope, nên receipt account-impact đã xác nhận có thể học latent
  policy-state tạo action logits theo context, thay vì chỉ đổi một bias toàn
  cục;
- receipt phải ghi `rlhf_internal_policy_state_trained` và
  `rlhf_internal_policy_state_parameter_changed`; probability guard vẫn là
  điều kiện chấp nhận cuối cùng.
  Khi guard loại contextual step, UI đọc residual fraction và các bộ đếm
  `shift/direction/nonfinite` để phân biệt backtrack an toàn với fallback
  global-only; residual line-search có thể giảm tới `1/2048` trước khi bỏ
  latent. Chỉ `latent policy-state changed=true` mới là bằng chứng tensor
  latent thật sự đã đổi.

Residual được điều kiện bởi hidden state của packet và khởi tạo bằng zero, đồng
thời có một global action bias bounded để receipt đã xác nhận vẫn tác động lên
context mới. Vì vậy action head không ghi đè trực tiếp `action_query`/
`action_embedding` dùng chung cho mọi context; update audit và tab log vẫn ghi
scope, hash tham số, probability trước/sau cùng trạng thái/scale/max_abs của
global bias để kiểm chứng tác động thật. Candidate dùng
`five_codex_context_similarity_threshold=0.90` với `support_floor=0.50`, tức
prototype residual chỉ được tái sử dụng khi cosine thực tế đạt tối thiểu
`0.95`; mức thấp hơn bị coi là context mới. Smoke `codex_context_reuse_090`
đã quan sát 4/6 update tái sử dụng prototype; run dài
`codex_context_reuse_default_20261001_210000` quan sát 11/20 update reuse,
prototype chỉ tăng `1→9` và packet probability guard vẫn giữ max shift
`0.04987`. Run dài bị wall guard và LCB còn âm, nên đây chỉ là bằng chứng
plumbing/reuse, không phải bằng chứng edge hay promotion.
Smoke non-metrics `codex_context_reuse_persisted_20261001_220000` còn lưu
được checkpoint với `5/5` Codex updates, `3/5` prototype reuse và reload xác
nhận `prototype_count=2`, threshold `0.90`, surface enabled cùng bias/residual
khác zero. Kết quả kinh tế của smoke vẫn âm, nên đây là bằng chứng persistence
nhưng chưa phải bằng chứng edge.

Tab **Codex teacher** cũng đọc `network_development.jsonl` của đúng candidate
output và hiển thị riêng **POLICY LIVE ACTION VETO**: số decision có evidence âm,
số lần veto thực sự áp dụng, tổng action bị mask, mapping `aN/action_name` và
trace gần nhất. Đây là execution-time action mask, không được cộng vào
`Codex optimizer updates` hay `Generic PPO updates`; nếu stream chưa có
record thì UI hiển thị `veto applied=0` thay vì suy đoán từ cấu hình.
Phần **ACTION SURFACE** tách thêm `contextual escape`: đây là entry lease có
giới hạn được giữ lại khi contextual expectancy của checkpoint cũ vừa mask sạch
toàn bộ entry nhưng live-veto chưa đủ mẫu. Nó không phải live probe và không
được tính là Codex receipt hay optimizer update; `mask-only-HOLD` và
`contextual escape` phải được đọc cùng nhau để phát hiện action-surface collapse.
và causal update, không phải bằng chứng edge.
Mỗi case trong receipt còn hiển thị `teacher reward`, `account input`,
`combined learner`, `scaled reward` và `advantage`. Với record cũ chưa có
`combined_reward`, UI chỉ tính fallback từ hai input finite để đọc/audit và
gắn nhãn `legacy fallback`; nó không ghi ngược log và không dùng fallback để
cập nhật learner.
Để xác nhận update đã đi tới quyết định MT5 kế tiếp, status phải có
`codex_causal_links_observed>0` và `codex_causal_updates_linked>0`; mỗi action
được nối sẽ ghi `codex_causal_link=barrier_to_next_action`, sequence/trace/action
kế tiếp và hash policy sau update trong `action_details`. Khi policy facade hỗ
trợ hash runtime, `codex_causal_hash_matches` phải tăng và link phải có
`policy_parameter_hash_matches_update=true`; `hash_unknown` không được coi là
bằng chứng mạnh. `updates>0` nhưng
causal links bằng không chỉ chứng minh optimizer đã chạy, chưa chứng minh
action sau đó đã đọc state mới. Barrier timeout không được tính là causal link.
Grading chạy với số worker bounded lấy từ
`learning.five_codex_teacher_max_workers` (mặc định `2`, cho phép `1..8`),
còn optimizer vẫn drain tuần tự trên bridge thread để không tranh chấp policy.
`runtime_learning_contract.json` được đồng bộ trong lúc run; không dùng
snapshot đầu run (`preflight=pending`, `updates=0`) để kết luận learner không
nhận feedback. Nếu contract chính bị khóa, sidecar
`runtime_learning_contract.json.latest.json` giữ snapshot mới nhất.

Riêng profile `he_chuyen_gia_codex_teacher_candidate.yaml` khóa
`five_rlhf_source_checkpoint_sha256`. Nút **Tạo kho mới** của profile này không
tạo seed ngẫu nhiên; UI dùng checkpoint nguồn `outputs/he_chuyen_gia/kho_chuyen_gia.pkl`
và kiểm tra SHA-256 trước khi chạy. Seed mới không khớp hash sẽ bị chặn ngay ở
UI thay vì để MT5 chạy rồi dừng ở bước source-checkpoint.

Ngày Candidate được kiểm tra nghiêm ngặt theo `YYYY.MM.DD`; ngày không tồn tại
như `2025.02.30` bị báo lỗi ngay tại UI và cũng bị script Python từ chối nếu gọi
trực tiếp. Profile Candidate dành 12 GiB cho checkpoint cùng telemetry và tính
trước kích thước cần thiết từ seed để không chạy hàng giờ rồi mới vỡ quota.

Trong tab Candidate, checkpoint để trống được đổi thành một đường dẫn seed mới;
workflow tự chạy `khoi-tao-he-chuyen-gia` trước `mt5-smoke`. Khi nạp cấu hình UI,
thư mục candidate đã tồn tại cũng được thay bằng tên output mới để không tái sử
dụng hoặc ghi đè một lượt cũ.
Hợp đồng **Five + Bridge v10** được đặt ngay dưới form Candidate thay vì chiếm
một tab đọc-only riêng:

- observation encoder v3 nhận 137 giá trị có cấu trúc và hợp nhất chúng qua Pattern Tree;
- năm replica dùng chung trọng số, giữ memory LSTM 4096 riêng và học đồng bộ `5 × T`;
- target attention cùng các head delay, entry offset, risk và quản lý vị thế;
- bridge schema 10 ghi packet năm action atomically; execution ledger schema 14 ghép receipt theo `trace_id + agent_id`.
- quyền cố định: agent 0 entry mua, 1 entry bán, 2 quản lý vị thế, 3 quản lý lệnh chờ, 4 quản trị danh mục; `CLOSE_ALL` là phủ quyết packet duy nhất;
- reward lấy marked-PnL theo owner trace, trừ chi phí/drawdown/risk/conflict và thêm bonus cạnh tranh zero-sum; terminal xếp hạng lợi nhuận từng agent nhưng vẫn giữ mục tiêu equity/drawdown của cả đội;
- Reasoning RL V1 dùng ba bước giả thuyết, nhóm tám rollout, verifier path 48-M1 và GRPO/PPO với KL + entropy. Candidate active học head reasoning qua FiveAdam (`encoder_credit_scale=0`) và tác động có giới hạn lên bảy action đầu (`action_scale=0.25`); config telemetry-only vẫn có scale 0 để đối chứng.

Phần hợp đồng không được trình bày như bằng chứng MT5/OOS. Tab chi tiết
**Đầu ra tư duy** của ReasoningGraph cũ đã bị xóa vì không đại diện cho GRPO mới;
trace Reasoning RL causal hiện được hiển thị trong tab **Quyết định** cùng trạng
thái action scale và cảnh báo future-label.

## Hợp đồng train mới nhất

Chế độ V193 edge dùng profile cố định
`configs/experiments/edge_discovery_v193_candidate.yaml`. Chế độ Reasoning RL
dùng config cô lập `configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml` và
overlay active `configs/experiments/reasoning_rl_v1_active_candidate.yaml`;
hoặc một trong ba cấu hình nhãn đã ghép:

- `configs/he_chuyen_gia_reasoning_rl_v1_candidate_sl500_h12.yaml` — cắt lỗ 5 ATR, chốt lời 10 ATR, giữ 12 nến;
- `configs/he_chuyen_gia_reasoning_rl_v1_candidate_sl500_h24.yaml` — cắt lỗ 5 ATR, chốt lời 10 ATR, giữ 24 nến;
- `configs/he_chuyen_gia_reasoning_rl_v1_candidate_sl500_h48.yaml` — cắt lỗ 5 ATR, chốt lời 10 ATR, giữ 48 nến.

Chế độ V193 cũ vẫn dùng `configs/he_chuyen_gia_latest_ui_v193.yaml`. Profile:

- tắt timing-first và mở actor entry nhận lifecycle PnL;
- phạt đúng trace LIMIT/STOP nếu policy hủy trước khi khớp;
- giữ entropy Five ở 0,01 đến khi có 3 scope LCB dương, rồi giảm theo số edge;
- để TensorRule ở `shadow`, không sửa live logits;
- dùng Five PPO Adam làm writer của shared policy.

Config `configs/he_chuyen_gia_latest_ui_v193.yaml` kế thừa full base contract
V191 đã kiểm thử và khóa profile tìm edge cho mọi workflow có học. Lượt đang
chạy không hot-reload source/config, nên thay đổi chỉ áp dụng khi bắt đầu lượt
mới.

Nếu checkpoint chưa tồn tại, nút **Tạo kho mới** khởi tạo seed trước rồi mới
train candidate. UI không tạo file checkpoint rỗng và không direct-resume online
vào checkpoint nguồn.

Toolbar **Lưu và dừng** dùng chung cho Imitation và **Train candidate/MT5/Codex**.
Với imitation, UI yêu cầu trainer chốt optimizer batch BPTT rồi ghi
`<candidate>.resume.pkl` cùng sidecar audit. Với Candidate, UI ghi một request
file cạnh output; Python bridge chỉ mirror file này vào portable MT5, EA gọi
`TesterStop()` sau safety/execution boundary kế tiếp, chờ report/ledger chính
thức, drain toàn bộ hàng đợi Codex teacher rồi mới persist checkpoint và
`manifest.json`. Artifact dừng giữa cửa sổ được ghi `save_and_stop.requested`
và `promotable=false`, nhưng không bị mất checkpoint đã serialize. Lần bấm
**Chạy workflow** kế tiếp sẽ xoá request cũ trước khi chạy mới. Nút **Dừng
workflow** màu đỏ vẫn là dừng cưỡng bức và có thể làm mất tiến trình chưa lưu.

Tab **Train candidate** chặn seed có `<seed>.resume.pkl` trước bước bootstrap/MT5
và chỉ dẫn quay lại tab imitation để hoàn tất candidate. Vì vậy một output đang
pause không bị thay bằng checkpoint khởi tạo rỗng cùng tên.

Nguồn imitation mặc định là seed V3 riêng tại
`outputs/he_chuyen_gia/kho_chuyen_gia_transformer_v3.pkl`. Nếu field nguồn trống
hoặc default V1/V2 cũ không còn file, UI tự chuyển sang seed V3 và tạo seed bằng
cấu hình Five khóa cứng trước khi bắt đầu trainer.

Vùng kết quả có tab **Chẩn đoán học** gồm ba phần tách biệt:

- **MT5 online** đọc `mt5_diagnostics.json` được tổng hợp tăng dần từ execution
  ledger, bridge ledger và optimizer observatory. Nó hiển thị close-deal/final-position count, deal WR, PF,
  expectancy, payoff, WR hòa vốn, return, peak-to-trough equity DD, số PPO
  update/rollout, reasoning optimizer update, policy/value/total loss, optimizer entropy, actor/reward
  eligibility, reward density, action entropy/effective-action count,
  HOLD/top-action share, reject và unfilled-timeout rate. `selected_idx` không
  được dùng thay cho lệnh MT5-confirmed.
- **Ablation** đọc `ablation_summary.json`, so ba arm tuần tự cùng checkpoint
  hash/date/model/seed: `frozen_policy`, `full_online`, `reasoning_off`, rồi
  hiển thị delta PF/expectancy/max DD so với full online. Reasoning-off chỉ tắt
  auxiliary reasoning learning vì action scale của full hiện vẫn bằng 0; nó
  không được gọi là Transformer ablation.
- **Imitation** giữ lịch sử train loss/BA online, BA train phát lại bằng trọng số
  cuối epoch, validation loss/BA/precision/recall/MCC, BA gap cùng chế độ phát
  lại, underfit/overfit và tỷ lệ confidence
  ENTRY đúng trên áp lực `false_entry_penalty`. Tỷ lệ này là proxy loss
  imitation, không phải reward RL, PnL hay profit factor.

Thẻ workflow **Ablation test** chạy ba arm nói trên bằng các candidate clone,
không sửa checkpoint nguồn và kiểm tra lại SHA-256 sau mỗi arm. Mặc định các
checkpoint arm rất lớn bị xóa sau khi diagnostics/manifest đã được giữ; bật
`Giữ checkpoint lớn của từng arm` chỉ khi thực sự cần phân tích trọng số.

Build và self-test:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ui-java/build-ui.ps1
java -jar .\ui-java\build\SelfGrowingTraderControlV193.jar --self-test
```

Sau khi chạy `ui-java/build-ui.cmd`, EXE được tạo tại
`ui-java/dist/SelfGrowingTraderControlV193/SelfGrowingTraderControlV193.exe`.
`-JarOnly` chỉ cập nhật JAR trong `ui-java/build` và không cập nhật EXE/app image
đang có; nếu chạy EXE phải dùng build đầy đủ ở trên. Build đầy đủ sẽ kiểm tra
launcher config `app/SelfGrowingTraderControlV193.cfg` và manifest `.jpackage.xml`
trước khi báo thành công.

## Thẻ Công cụ dữ liệu

Giao diện hiện có thêm thẻ **Công cụ dữ liệu**, gắn trực tiếp bộ khai thác
`tools/five_zigzag_dataset_tool` vào dự án. Thẻ này không cập nhật trọng số mô
hình; nó chỉ chuẩn bị và kiểm tra dữ liệu.

Các mục bên trong:

- **Khai thác tập 137**: chọn nguồn giá, mẫu ZigZag, thư mục mới, cấu hình mô
  hình, chu kỳ ATR, ngưỡng ZigZag, số mẫu mỗi mảnh và dữ liệu dự phòng cho
  chênh lệch giá/khối lượng. Mặc định chạy kiểm tra hợp đồng trước khi khai
  thác.
- **Kiểm tra tương thích**: xác nhận 137 đầu vào, 5 tác nhân, phiên bản sơ đồ,
  dấu vân tay đặc trưng, chế độ chuẩn hóa và khả năng dựng đặc trưng từ nguồn.
- **Lấy dữ liệu MT5**: lấy mã giao dịch, khung thời gian và số thanh giá trực
  tiếp từ MT5 cục bộ thành tệp `.npz`; khi hoàn tất giao diện đọc `manifest.json`,
  không tìm nhầm `dataset_summary.json` của bước khai thác.
- **Thử nhanh mẫu**: kiểm tra mẫu ZigZag trên thị trường giả lập trước khi chạy
  dữ liệu lớn.
- **Mẫu cấu trúc & kết quả**: đọc/sửa/lưu tệp mẫu và xem
  `dataset_summary.json` của tập vừa tạo; kèm hợp đồng chống nhìn trước tương
  lai và cấu trúc thư mục đầu ra.

Fetch raw và khai thác tập 137 là hai tác vụ riêng. Với một symbol, người dùng có thể
chọn NPZ và chạy `mine-five`. Với thư mục raw nhiều symbol, UI chạy
`mine-five-selected` và tạo `dataset_summary.json` cùng `dataset_collection.json`.

Trong **Công cụ dữ liệu → Lấy dữ liệu MT5**, tùy chọn mặc định
**Lấy xong tự tạo tập 137** ghép hai lệnh thành một pipeline fail-fast:
`mt5-fetch-selected → mine-five-selected`. Người dùng quét/chọn symbol và bấm
**Lấy dữ liệu và tạo tập 137**; bước khai thác chỉ chạy khi bước fetch trả exit
code 0. Workflow này chỉ tạo dữ liệu và không cập nhật trọng số mô hình.

Trước khi nạp checkpoint Five lớn, train dataset chạy execution preflight trên mọi
điểm neo theo đúng symbol contract và risk/spread gates. Nếu không có mẫu nào có thể
thực thi, pipeline dừng ngay với symbol và `min_spread_cost_r`. Preflight không chỉ
đếm: iterator bỏ qua mọi anchor không đạt đúng các gate này và phân phối vòng tròn
chỉ trên anchor hợp lệ của từng symbol. Telemetry phân biệt `raw_samples` với
`available_samples`; giới hạn `0 = tất cả episode` nghĩa là toàn bộ mẫu hợp lệ, không
phải toàn bộ mẫu raw. Tape M1 còn loại mọi episode có cạnh thời gian khác đúng một
phút; bước mine mới bỏ phần prehistory M1 thưa (ví dụ broker chỉ trả một dòng/ngày)
trước khi tính ATR/feature. Guard tám episode chỉ áp dụng lúc cold-start chưa từng có
giao dịch; HOLD hợp lệ sau khi đã giao dịch không còn làm dừng cả workflow.

Rollout PPO 16 bước được giữ qua ranh giới các episode dataset ngắn; done-mask vẫn
reset ký ức/GAE đúng điểm kết thúc. `allowed_action_mask` dùng lúc gửi quyết định được
đóng băng trong từng trace và truyền lại nguyên vẹn khi tính log-prob PPO, tránh học
trên một phân phối hành động khác phân phối đã thực thi.

Actor credit còn bám execution provenance: proposal bị arbitration chặn không nhận
`conflict_cost` và không vào policy loss; trace vẫn được giữ cho critic/audit. Một gói
toàn HOLD không có execution/counterfactual evidence chỉ cập nhật critic và entropy,
không tự củng cố HOLD bằng reward 0. Sau khi policy đã từng giao dịch, 64 episode liên
tiếp không có giao dịch làm workflow dừng với lỗi policy-collapse thay vì chạy nhiều
ngày trong trạng thái HOLD.

Profile V193 mới bật **Reward V2 expectancy-first**. Marked net-R sau chi phí vẫn là
mục tiêu chính. Feedback đầu tiên của mỗi episode được so với trạng thái kinh tế 0,
không còn nuốt PnL của lệnh đã mở/đóng trước feedback đầu. Khi trade đóng, reward có
thêm penalty cho expectancy shortfall và loss tail. Bonus hướng tới WR 55% rất nhỏ và
chỉ mở cho agent có realized expectancy tối thiểu +0,05R; WR cao nhưng expectancy âm
không nhận bonus. Telemetry xuất riêng closed trades, wins, WR balance, shortfall,
tail loss và tỷ lệ expectancy gate để audit reward hacking.

V193 còn bật **Pattern Discovery Reward** cho chế độ Tập cấu trúc. Outcome
long/short final-return, MFE/MAE của miner chỉ chấm decision tại anchor sau khi toàn
bộ future của sample đã kết thúc. Mỗi reward bị chặn ±0,10R, đã trừ spread và chỉ
dạy BUY/SELL/HOLD; MARKET/LIMIT/STOP vẫn học bằng execution PnL thật. Mười sáu anchor
độc lập được re-forward từ đúng LSTM state riêng bằng cùng PPO writer. Chế độ khoảng
thời gian, frozen/live và MT5 không tạo oracle, và future không đi vào 137 input.

Encoder Five v3 còn có **Observation Pattern Tree**: bốn leaf market/position/
account/execution hợp thành node price/context rồi root pattern. Root đi vào LSTM
qua residual; các head utility, MFE/adverse và no-edge chỉ nhận nhãn sau khi sample
kết thúc. Giao diện không cần thêm bước chuẩn bị dữ liệu: fetch → mine → train hiện
có tự truyền nhãn auxiliary trong cùng nút workflow. Telemetry `pattern_learning`
hiển thị riêng prediction/loss của cây để phân biệt học representation với PnL thật.
HOLD không còn nhận economic actor credit từ lệnh của agent khác. Trên anchor có
edge, cây supervise trực tiếp nhóm LONG/SHORT và áp trần HOLD 20%; trên anchor no-edge
HOLD vẫn là target hợp lệ, tránh biến chống collapse thành ép overtrade.

Để đầu action chính không thể bão hòa số học về HOLD trong lúc train, behavior policy
dành tối thiểu `0,5%` cho từng entry đang hợp lệ của mỗi tác nhân. PPO tính lại đúng
chính phân phối đã trộn này, gồm cùng mask và cùng log-prob; không dùng xác suất softmax
gốc để cập nhật. Sàn này chỉ áp dụng khi lấy mẫu học, không đổi quyết định
`deterministic`/frozen. Telemetry tách `primary_action_entropy` và
`primary_entry_probability_mass` khỏi entropy tổng có nhánh delay, nên không còn che
giấu action collapse bằng entropy của tham số phụ.

Bộ công cụ dùng tệp gọi:
`tools/five_zigzag_dataset_tool/run_tool.py`; không cần cài riêng gói công cụ
nếu chạy từ cây dự án này. Các thư viện Python cần thiết vẫn phải có theo
`tools/five_zigzag_dataset_tool/requirements.txt`.

Mỗi tape trong collection mang `symbol_contract` lấy trực tiếp từ MT5. Đây là
điều kiện bắt buộc để spread/point và feature đầu vào không bị tính theo symbol
mặc định của cấu hình.

JAR đã biên dịch có thể chạy bằng `ui-java/run-ui.cmd`. Trên Windows, chạy
`ui-java/build-ui.cmd` để tạo lại ảnh ứng dụng/EXE từ đúng mã nguồn hiện tại.

## Thẻ Imitation tối đa hiện có

Thẻ **Imitation tối đa hiện có** mặc định dùng toàn bộ chuỗi BTCUSDm M1 liên
tục mà server Exness cung cấp: 2021-07-01 đến hết 2025-04-23. Phần lịch sử
2018-12 đến 2021-06 chỉ có một bar mỗi ngày nên bị loại khỏi dataset M1.
Dataset schema 9 được build từ
`BottomTop_Teacher_EA` tạo trên lịch sử. UI nhận checkpoint gốc chỉ đọc,
dataset `.npz` kèm manifest, checkpoint candidate mới và các tham số
epoch/batch chuỗi/độ dài chuỗi/learning-rate/phạt vào lệnh sai/early-stop.

Trước khi cập nhật trọng số, pipeline bắt buộc kiểm tra hash CSV nguồn, 137
đầu vào, authority agent mua/bán, split theo thời gian, label chỉ được hoàn tất
sau thời điểm quan sát, NaN/Inf và dung lượng đĩa. Chế độ **Chỉ kiểm tra** chạy
toàn bộ preflight nhưng không nạp hay sửa model.

Imitation replay observation theo cửa sổ recurrent 16 bước. Nhãn BUY/SELL không
còn nằm trên chính bar pivot hậu nghiệm: builder dời nhãn tới observation đầu
tiên có sẵn sau ba bar xác nhận, tính lại entry theo Bid/Ask tại observation đó
và chỉ giữ BUY/SELL khi PnL tới pivot đối diện vẫn dương sau spread. Trường hợp
không còn dương trở thành HOLD hard-negative. Dataset giữ SL cấu trúc cách pivot
1,50 ATR và TP tại đúng pivot đối diện; giá TP liên tục được giữ riêng, còn index
20-output là bin RR gần nhất.

Stage action chạy BPTT thật qua observation encoder, toàn bộ LSTM, action-query
và `entry_role_calibrator` của agent LONG/SHORT. Prior tác động tổng entry mass của
MARKET/LIMIT/STOP cùng phía trong residual. Trainer reset prior imitation cũ rồi
học representation recurrent và adapter residual phi tuyến có embedding riêng
cho BUY/SELL. Pairwise ranking
loss học thứ tự ENTRY/HOLD; hai ngưỡng role chỉ được hiệu chỉnh sau khi chọn epoch.
Ngưỡng validation không được phát entry cao hơn 1,10 lần tỷ lệ nhãn từng role.
Trọng số action embedding, value, target và geometry head giữ nguyên hash, nhưng
logits dùng representation chung có thể đổi nên vẫn phải đo lại bằng MT5/OOS.
Geometry delay/SL/TP/trailing đã bị khóa khỏi workflow chính sau khi full geometry
và SL-only đều thua action-only trong paired MT5; warm-up chỉ giữ action prior.
Sau train, script
lưu một checkpoint candidate đầy đủ, nạp lại artifact, đối chiếu các hash và
hash lại checkpoint gốc. Báo cáo `.pkl.imitation.json` luôn ghi
`candidate_only` và `promotable=false`; kết quả này chưa thay thế frozen OOS
hoặc bằng chứng MT5.
Log workflow có `seed_imitation_preflight`; `effective_for_mt5=true` xác nhận
action prior. `geometry_effective_for_mt5` phải là false trong workflow action-only.

Khi cần kiểm tra MT5 bằng checkpoint giữa chừng, bật **Bỏ qua gate imitation
seed (research-only)** trong tab **Train candidate**. UI truyền
`--allow-unverified-imitation-seed`; manifest giữ nguyên lý do gate thất bại,
ghi `gate_bypassed=true`, `promotable=false` và chặn lineage promotion của output.
Khi imitation train thành công, UI tự gắn checkpoint vừa tạo vào ô seed của
**Candidate**, tránh vô tình chạy lại checkpoint gốc hoặc candidate schema 2.

Candidate persisted có thể nhẹ hơn source vì config có thể strip moment `m/v`
của FiveAdam. UI/status phải hiển thị rõ `optimizer_state_resume_mode`:
`continuation` chỉ dành cho source/champion giữ nguyên optimizer; muốn dùng
policy weights đã strip cho một vòng research mới phải bật
`five_rlhf_allow_optimizer_state_reinitialization_for_research=true`, khi đó
mode là `research_reinitialized` và không được hiểu là resume cùng quỹ đạo.

Dataset mặc định là
`outputs/he_chuyen_gia/v193_repair_20260910/teacher_16w_confirmed_schema9.npz`.
Nó giữ 77.295 dòng theo thời gian, gồm 4.775 BUY, 4.865 SELL và 67.655 HOLD;
cả 9.640 entry đều có geometry hợp lệ. Action BPTT recompute action-query theo từng window/epoch;
cache `[N,128]` detach đã bị bỏ vì sẽ cắt gradient và stale ngay sau optimizer
step. Hidden cache `[N,4096]` chỉ còn thuộc stage geometry nghiên cứu frozen.

Khi tải cấu hình UI đã lưu từ bản trước, đường dẫn chính xác của dataset 7 ngày,
16 tuần SL 0,25, SL 0,75 hoặc schema 8 được tự migrate sang dataset tối đa hiện có.
Dataset tùy chỉnh của người dùng không bị thay đổi.

Trainer hiện hành dùng dataset schema 9, recurrent window 16 và nonlinear
entry-role adapter nhận action-query 128 chiều live. UI tách LR recurrent mặc
định `1e-5` khỏi LR adapter `3e-3`, batch BPTT mặc định 2 window. Validation hiệu chỉnh
ngưỡng BUY/SELL riêng tại rate ceiling bảo thủ; report hiển thị precision,
false-positive rate, MCC, label/predicted entry rate, metric threshold theo epoch
và hash riêng cho recurrent trainable/frozen head weights.

Khi bấm chạy Imitation, UI tự chọn vùng **Nhật ký**. Stdout/stderr được đọc theo
thời gian thực; JSON event được rút gọn thành dòng dễ đọc. Initial/final replay
báo khoảng mỗi 10%, từng epoch BPTT báo khoảng mỗi 5%, và cuối epoch hiển thị
train loss, validation balanced accuracy, precision, MCC cùng thời gian chạy.
