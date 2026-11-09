# Checklist hoàn thiện tổng thể — Self-Growing Trader

**Ngày audit:** 23-06-2026  
**Baseline kiểm tra:** archive `new-chat(5).zip` và source Python/EA nằm trong archive.  
**Mục tiêu:** biến prototype R&D thành pipeline có thể tái lập, đánh giá đúng và có đủ guardrail trước khi cân nhắc paper trading.

> **Phạm vi và giới hạn:** Đây là checklist toàn diện đối với lỗi và rủi ro đã quan sát bằng source, test và artifact có trong archive. Nó không thể chứng minh không còn lỗi ẩn, không thay thế compile MT5, backtest tick-level, walk-forward hoặc kiểm thử broker.

## Quy ước

- `[x]` đã được áp dụng trong hardened baseline này.
- `[ ]` chưa hoàn thành; không được đánh dấu xong nếu không có bằng chứng kiểm thử kèm theo.
- **P0** chặn tái lập/độ đúng execution hoặc có thể làm kết quả training sai bản chất.
- **P1** là điều kiện bắt buộc trước paper trading.
- **P2/P3** là hardening, observability hoặc nâng cấp hiệu năng.

## Trạng thái đã kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| `pytest -q` sau patch fitness/neuron v5 | **129 passed** |
| `python -m compileall -q src tests scripts` sau patch fitness/neuron v5 | Thành công |
| `pytest -q` ở root trước baseline | Lỗi collect 4 script legacy trong `work/` (`neat_trader`, `neat`, `MetaTrader5` không có) |
| `pip install -e . --no-deps` trước baseline | Thất bại, thiếu `pyproject.toml`/`setup.py` |
| `mt5_v2_runner/` trong archive | Không có |
| `compute_features.mqh` trong archive | Không có |
| EA bridge trong archive gốc | Schema v2, không ghi `orders_pipe`/`execution_pipe`; patch hiện tại nâng source lên schema v4 nhưng chưa compile MT5 |

---


## Patch status — Fitness MT5 & neuron attribution v5 (23-06-2026)

| Mục | Trạng thái | Bằng chứng / giới hạn |
|---|---|---|
| P0-F1 | **Code + Python regression implemented; MT5 runtime pending** | EA ledger schema 5 thêm `position_final_close`, `initial_risk`, `age_bars`; Python group theo final `position_ticket`, không count partial deal thành sample mới. |
| P0-F2 | **Implemented** | `mt5_fitness_breakdown()` tách eligibility/rank/learning signal; PF=999 sample nhỏ không rank dương; evaluator hard-reject ledger thiếu. |
| P0-N1 | **Implemented in same-process runtime** | Trace giữ module hidden/gate/raw output; final lifecycle credit theo `trace_id`, signed action margin và realized R; restart reconciliation còn mở. |
| P0-N2 | **Implemented with evidence thresholds** | `neuron_samples`, utility/contribution/activation/redundancy EMA; freeze/prune cần sample + LCB. Chưa có MT5 real-life calibration. |

Chi tiết acceptance criteria và phần còn mở: [Fitness MT5 v5 & Neuron Attribution](FITNESS_V5_NEURON_ATTRIBUTION_CHECKLIST.vi.md).

## Patch status — execution ledger v4 (23-06-2026)

| Mục | Trạng thái | Bằng chứng / giới hạn |
|---|---|---|
| P0-A5 | **Code implemented; MT5 runtime pending** | EA/runner schema v4 gồm `orders_pipe`, `execution_pipe`, 20-output action contract. Cần MetaEditor + Strategy Tester smoke test. |
| P0-A6 | **Code implemented; lifecycle matrix pending** | `OnTradeTransaction()`, intent ledger, order/deal/position ticket mapping và append-only `self_growing_execution.csv` đã được thêm. Cần kiểm chứng market/pending/SL/TP/modify/cancel với terminal thật. |
| P0-A7 | **Partially implemented** | Payload phân biệt accepted/placed/filled/cancelled/modified/closed/rejected; v5 backfill final lifecycle về original trace trong cùng Python process. Durable reconciliation qua restart/orphan lifecycle còn mở. |
| True `HOLD` | **Code + regression tests implemented** | `HOLD` early-return; management được tách thành action 11; `OnTick()` không hủy pending order vô điều kiện. Cần runtime test. |

Chi tiết contract và test matrix hiện hành: [MT5 learned geometry V15](MT5_EXECUTION_CONTRACT_LEARNED_GEOMETRY_V15.vi.md).

# A. P0 — Tái lập MT5 và contract Python ↔ EA

## A1. Runtime/compile artifacts

- [ ] **P0-A1** Đưa `mt5_v2_runner/` portable hoặc một manifest bootstrap có version/build/hash cụ thể của terminal MT5, data server, symbol suffix và tester agent.
  - Evidence: CLI/evaluator mặc định dùng `mt5_v2_runner`; archive không chứa thư mục này.
  - Done khi: `scripts/preflight_mt5.py --runner-dir ...` pass trên máy sạch và một smoke test Strategy Tester sinh report.
- [ ] **P0-A2** Version-control hoặc package `compute_features.mqh` cùng EA.
  - Evidence: `SelfGrowing_MT5_Trainer_EA.mq5:11` include file này, nhưng archive không có `.mqh`.
  - Done khi: MetaEditor compile EA từ checkout sạch, không phụ thuộc file local không được quản lý.
- [ ] **P0-A3** Thêm build manifest cho EA: mã nguồn, `.ex5` hash, terminal build, config `.set`, symbol, period, history date range, model tick và broker specification.
  - Done khi: report ghi đủ manifest và CI/runner đối chiếu hash trước mỗi trial.
- [ ] **P0-A4** Tách runtime files theo `run_id`/agent để không xảy ra tranh chấp CSV khi MT5 tester có nhiều Agent hoặc trial chạy song song.
  - Evidence: bridge/request/action filenames hiện tĩnh.
  - Done khi: hai trial song song không đọc action hoặc bridge của nhau.

## A2. Version hóa bridge và execution acknowledgement

- [ ] **P0-A5** Nâng EA từ `BRIDGE_SCHEMA_VERSION 2` thành schema v3 thống nhất với Python.
  - Evidence: EA tại `SelfGrowing_MT5_Trainer_EA.mq5:15`; runner đã có parser `orders_pipe` và `execution_pipe` ở `mt5_runner.py:554-625`.
  - Done khi: request/bridge header chứa `orders_pipe` và `execution_pipe`; test round-trip Python↔EA kiểm tra đầy đủ trường.
- [ ] **P0-A6** Dùng `OnTradeTransaction` để ghi execution event theo `event_id`, `trace_id`, request sequence, retcode, order ticket, deal ticket, action lifecycle và timestamp.
  - Evidence: EA không có `OnTradeTransaction`; hiện chỉ suy luận qua thay đổi position/day trades.
  - Done khi: một lệnh open, pending fill, modify SL/TP, partial close, cancel và forced close đều tạo acknowledgement có ticket/deal liên kết được.
- [ ] **P0-A7** Tách rõ *intent accepted*, *order placed*, *deal filled*, *position modified*, *order cancelled*, *position closed*; không dùng một boolean `execution_confirmed` chung cho mọi mục đích học.
  - Done khi: memory policy chọn loại transition theo action lifecycle chứ không chỉ snapshot chênh lệch.
- [ ] **P0-A8** Ghi retcode và lý do reject; phân loại reject do stops level, volume, margin, market closed, requote, invalid price, timeout.
  - Done khi: dashboard có reject-rate theo retcode và training không cấp reward cho rejected request.

## A3. Contract 20 outputs — phải đồng nhất

- [ ] **P0-A9** Viết một contract versioned duy nhất cho 20 outputs, với tên/index, units, transform, valid range và semantics EA/Python.
- [x] **P0-A10** Đồng bộ output risk percentage với EA position sizing.
  - Evidence: Python action wire ghi `risk_pct`; EA dùng `PolicyVolumeFromRisk(riskPct, entry, sl)` để tính `req.volume` từ equity, stop distance, tick value và broker min/max/step volume. Route `req.volume = Lots` và input `.set` `Lots=` đã bị xoá.
  - Done khi: entry/pending order không dùng lot cố định; sizing bắt nguồn từ output risk của policy và EA chỉ normalize/validate theo broker constraints.
- [ ] **P0-A11** Đồng bộ output `[14]` entry offset với pending order.
  - Evidence: Python decode `outputs[14]` là `entry_offset_atr` (`orders.py:100`); EA hard-code pending offset ±`0.5 * atr` (`SelfGrowing_MT5_Trainer_EA.mq5:576-579`).
  - Done khi: EA dùng output đã decode và report ghi entry price mong đợi/thực tế.
- [ ] **P0-A12** Giải quyết overload output `[17]`.
  - Evidence: Python dùng `[17]` vừa cho time window vừa suy ra management mode (`orders.py:111-127`); EA chỉ dùng nó làm management mode (`SelfGrowing_MT5_Trainer_EA.mq5:595-613`).
  - Done khi: tách thành các output riêng hoặc loại bỏ time filter khỏi Python contract; EA và Python assertion-test cùng mapping.
- [ ] **P0-A13** Đồng bộ output `[18]` target slot/position scale và định nghĩa quy tắc slot ổn định theo ticket.
  - Done khi: action quản lý target đúng ticket trước/sau khi thứ tự MT5 positions thay đổi.
- [ ] **P0-A14** Đưa time filter vào EA hoặc bỏ hẳn khỏi fitness/simulator để tránh policy tối ưu tham số không chạy ở MT5.
- [ ] **P0-A15** Thêm golden tests: 100 vector outputs ngẫu nhiên + boundary vectors phải decode cùng action/risk/SL/TP/entry/trail/partial/mode/target trong Python và EA test harness.

## A4. Execution safety của EA

- [ ] **P0-A16** Xác thực `SYMBOL_VOLUME_MIN`, `SYMBOL_VOLUME_MAX`, `SYMBOL_VOLUME_STEP`, tick size/value, stop level, freeze level và margin trước `OrderSend`.
  - Done khi: request được normalize hoặc reject rõ lý do trước gửi.
- [ ] **P0-A17** Không gọi `CancelAllOrders()` vô điều kiện mỗi bar.
  - Evidence: EA gọi trước mọi action (`SelfGrowing_MT5_Trainer_EA.mq5:122`), kể cả khi policy không yêu cầu cancel.
  - Done khi: pending order có lifecycle/expiry riêng, và cancel được trace/event hóa.
- [ ] **P0-A18** Gắn forced close do `MaxHoldingBars` với transition mở vị thế ban đầu, không credit nhầm cho action bar hiện tại.
  - Evidence: `CloseExpiredPositions()` chạy trước request/action mới (`:107`), nhưng không tạo attribution event.
- [ ] **P0-A19** Thêm emergency kill switch: daily loss, max consecutive loss, max spread, max slippage, broker disconnect, time/news blackout, and manual disable file.
- [ ] **P0-A20** Cấm live mode mặc định. Tách explicit `paper` và `live`, buộc acknowledgement/confirmation với `--live-confirm` hoặc config có secret ngoài source.

---

# B. P0/P1 — Memory, self-organization và self-improvement

## B1. Chất lượng transition và credit assignment

- [ ] **P0-B1** Chỉ ghi episodic/causal memory khi execution event có provenance đầy đủ: request → order/deal → position outcome.
  - Done khi: journal chứa `trace_id`, request/order/deal ticket, event type và timestamp; mọi event không liên kết bị đánh dấu orphan, không dùng học.
- [ ] **P0-B2** Reward phải là equity delta/log-return từ transition trước đến transition sau, tách realized/unrealized PnL, commission, swap, slippage và risk-normalized R.
  - Done khi: reward reconciliation khớp account history/deals theo ticket.
- [ ] **P1-B3** Chuẩn hóa same-representation cho `state_embedding` và `next_embedding`.
  - Evidence: current state là `encoded_state`; `next_embedding` đang `_resize(next_obs)` raw observation (`self_growing_runner.py:2013-2015`).
  - Done khi: cùng encoder/version/normalization được ghi vào transition.
- [ ] **P1-B4** Lưu action embedding đủ 19 chiều hoặc một semantic action schema đầy đủ.
  - Evidence: memory cắt action còn 16 chiều (`self_growing_runner.py:1929`, `1963`, `2014`).
  - Done khi: partial close, mode và target slot được recall/replay chính xác.
- [ ] **P1-B5** Tách capacity budgets: episodic nodes, causal events, causal edges, clusters, strategies, meta strategies.
  - Evidence: causal nodes và graph neurons không còn bị hard-cap bởi `max_neurons`; cần theo dõi growth bằng metric/cadence thay vì trần tổng.
  - Done khi: graph growth không ép quên episodic memory tốt và mọi prune ghi audit reason.
- [ ] **P1-B6** Consolidate causal events theo regime + action signature + lifecycle outcome; không tạo event-level node vô hạn khi prediction error cao.
- [ ] **P1-B7** Thay metric “positive neuron” bằng dashboard utility thực: query hit-rate, nonzero action-bias rate, positive-cluster ratio, strategy/meta count, support distribution và causal growth rate.

## B2. Utility, retrieval và self-improvement loop

- [ ] **P0-B8** Calibrate utility: không trừ cố định `0.1 * raw_prediction_error` khi prediction error chưa normalized.
  - Evidence: `utility = td_delta + reward - cost - 0.1 * pred_error` tại `self_growing_runner.py:2003-2005`.
  - Done khi: prediction error được z-score/percentile-normalize, utility scale có test và cluster dương xuất hiện/dịu đi theo evidence.
- [ ] **P1-B9** Học predictor đầy đủ hoặc giảm trọng số novelty/surprise trong utility.
  - Evidence: update quan sát được chủ yếu là `predictor.w_proj`; error cao có thể biến mọi cluster thành utility âm.
- [ ] **P1-B10** Nối `MetaStrategyMemory.growth_bias/prune_bias` vào developmental controller với clamp, delay, validation gate và rollback.
  - Evidence: meta memory định nghĩa hai signal (`self_growing_runner.py:1638-1646`), nhưng developmental path `develop()` không dùng chúng.
  - Done khi: mỗi structural change có memory evidence, shadow validation và rollback metric.
- [ ] **P1-B11** Memory retrieval phải state/regime-conditioned và có confidence/support threshold; không để global bad outcome drift toàn policy.
- [ ] **P1-B12** Tách inference-only retrieval khỏi learning mode để frozen OOS không update state, memory, critic, predictor, plasticity hay growth.
- [ ] **P1-B13** Bổ sung `reset_episode_state()` reset mọi recurrent state: runtime memory, working memory, predictor prior, selector prior, hypothesis scores, novelty bank; tách với `reset_long_term_memory()`.
- [ ] **P1-B14** Cân bằng replay: reward dương/âm, forced exit, rejected order, partial close, giveback và regimes; theo ưu tiên nhưng không chỉ chọn experience tốt.
- [ ] **P1-B15** Journal replay phải validate schema/version, checksum, policy version, feature version và execution provenance trước khi replay.

## B3. Growth/prune an toàn

- [ ] **P1-B16** Hạn chế structural mutation theo budget mỗi evaluation window; không grow/prune trên một transition.
- [ ] **P1-B17** Đặt shadow module bắt buộc validation trên holdout trước commit; phải rollback weights/lateral edges/state sạch.
- [ ] **P1-B18** Đo complexity-adjusted improvement, chi phí inference và memory growth rate để tránh bloat.
- [ ] **P1-B19** Thêm deterministic seed + state snapshot cho mọi evolutionary/growth decision để rerun được.
- [ ] **P2-B20** Tạo offline memory replay simulator từ journal đã clean để unit-test consolidate/query/prune không cần MT5.

---

# C. P1 — Evaluation, fitness và scientific validity

- [ ] **P1-C1** Tách hoàn toàn `Frozen OOS` và `Adaptive OOS`.
  - Evidence: `evaluate_policy_mt5_oos()` copy policy và tắt journal nhưng evaluator vẫn phục vụ transition; cần explicit no-learning flag trong evaluator/policy.
  - Done khi: frozen OOS có checksum weights/memory bất biến trước/sau; adaptive OOS report riêng tốc độ cập nhật và uplift.
- [ ] **P1-C2** Walk-forward purged/anchored theo timeline với train/validation/test không overlap và report mỗi window.
- [ ] **P1-C3** Báo cáo distribution qua nhiều seed, symbol, regime và time windows; không chọn best run.
- [ ] **P1-C4** Dùng tick model đủ bảo thủ cho execution claims; report spread động, commission, swap, slippage, stop-level rejections và latency assumption.
- [ ] **P1-C5** So sánh baseline mạnh: hold, random constrained, fixed rule strategy, fixed-topology policy và no-memory policy.
- [ ] **P1-C6** Tạo ablation bắt buộc: working memory-only, episodic-only, causal-off, growth-off, online-learning-off, no-starvation, no-bootstrap.
- [ ] **P1-C7** Promotion cần test threshold gồm number of trades, profit factor, expectancy R, max DD, tail metric, stable OOS và complexity penalty.
- [ ] **P1-C8** Không dùng PF cực lớn của sample nhỏ; xử lý zero-loss/PF infinity và confidence interval.
- [ ] **P1-C9** Ghi report machine-readable (JSON) có raw deal list/equity curve, thay vì chỉ parse HTML summary.

---

# D. P1/P2 — Codebase, packaging, quality gate

## D1. Baseline hygiene được áp dụng

- [x] **D1** Thêm `pyproject.toml` với src layout, dependencies, test extra và CLI entry point.
- [x] **D2** Thêm `pytest.ini` để root `pytest` chỉ collect `tests/`.
- [x] **D3** Thêm `.gitignore` cho Python caches, training outputs, bridge files, MT5 runtime và Java builds.
- [x] **D4** Thêm GitHub Actions Python test/compile/CLI gate.
- [x] **D5** Thêm `scripts/preflight_mt5.py` để fail fast khi runtime/EA/include thiếu.
- [x] **D6** Thêm README root, trạng thái R&D và warning về unsafe pickle.

## D2. Việc còn lại

- [ ] **P1-D7** Di chuyển hoặc port toàn bộ script legacy trong `work/` đang import `neat_trader`/`neat`/`MetaTrader5`; không để chúng trông như executable supported.
  - Done khi: mỗi script có owner/status; script supported import package hiện tại và test được, script obsolete đi vào `legacy/`.
- [ ] **P1-D8** Chuyển documentation cũ thành historical hoặc cập nhật nó theo trạng thái source thực.
  - Evidence: `docs/checklist_cap_nhat_va_sua_loi.vi.md` từng khẳng định đã có packaging/pytest config trong khi archive audit ban đầu không có.
- [ ] **P1-D9** Phân rã `self_growing_runner.py` thành modules rõ boundary: policy, memory, causal graph, learning, development, training, persistence.
- [ ] **P1-D10** Bổ sung static lint/type checks (ruff + pyright/mypy), coverage floor và mutation/property tests cho output decoder/bridge parser.
- [ ] **P2-D11** Không commit generated files: `__pycache__`, `.egg-info`, binary Java build/runtime, local output models, massive journal.
- [ ] **P2-D12** Có dependency lock/constraints và SBOM/release manifest.
- [ ] **P2-D13** Chọn và ghi LICENSE, contributor policy, security disclosure, versioning/release notes.

---

# E. P1 — Security, persistence và operations

- [ ] **P1-E1** Thay `pickle` runtime artifact bằng format an toàn/verified (JSON + NPZ/safetensors) hoặc ký hash/chỉ load artifact tin cậy.
- [ ] **P1-E2** Atomic writes + file lock + run-id cho policy artifact, journal và bridge action để chống partial/cross-run overwrite.
- [ ] **P1-E3** Redact credentials/account identifiers khỏi logs; quản lý broker credentials bằng secret store, không source/config plaintext.
- [ ] **P1-E4** Telemetry bắt buộc: run id, policy hash, feature schema, action schema, EA hash, terminal build, symbol, account type, time zone, seed.
- [ ] **P1-E5** Thiết lập alerting: no-action timeout, bridge stale, reject burst, mismatch action ack, risk breach, DD breach, abnormal memory growth.
- [ ] **P2-E6** Data-retention/compaction policy cho journal/artifact; có replayability manifest trước khi prune.

---

# F. Nâng cấp kiến trúc đề xuất

## F1. Bridge v4 tối thiểu

- [ ] Request: `schema_version`, `run_id`, `policy_hash`, `feature_schema_hash`, `event_id`, `trace_id`, `sequence`, snapshot trước action.
- [ ] Action: `action_schema_hash`, 20 outputs, decoded semantic action, action validity mask, write timestamp.
- [ ] Execution event: `request_event_id`, `order_ticket`, `deal_ticket`, event type, retcode, requested/executed volume, requested/fill price, SL/TP, commission, swap, slippage.
- [ ] Snapshot sau: positions/orders by ticket, balance/equity/drawdown, realized/unrealized PnL.
- [ ] Contract test: encode/decode fixture shared by Python and MQL5.

## F2. Memory v2 tối thiểu

- [ ] Single `TransitionV2` schema: encoded state, 20-output semantic vector, next encoded state, lifecycle outcome, normalized reward, provenance.
- [ ] Two stores: bounded episodic exemplars và consolidated causal patterns; separate budgets/pruning.
- [ ] Retrieval returns bias only when similarity, support, utility, confidence và regime match đủ threshold.
- [ ] Memory-to-growth signals pass qua stability filter, validation gate và cooldown.
- [ ] Frozen mode cấm mọi mutation; adaptive mode ghi rõ update budget.

## F3. Release dashboard tối thiểu

- [ ] Execution: ack rate, fill rate, reject rate by retcode, bridge lag, stale action rate.
- [ ] Risk: actual R per trade, open risk, DD, slippage, exposure, consecutive losses.
- [ ] Memory: hit rate, nonzero influence rate, utility distribution, prune/grow count, causal graph size.
- [ ] Research: OOS PF/expectancy/DD distribution, ablation delta, seed variance, complexity/inference cost.

---

# G. Definition of Done theo phase

## Gate 0 — Reproducible offline

- [ ] A1–A4 pass.
- [ ] Build EA từ checkout sạch pass.
- [ ] `pip install -e '.[test]'`, `pytest -q`, compileall và CLI pass trên máy sạch.
- [ ] One fixed-seed MT5 smoke test tái lập được report + bridge traces.

## Gate 1 — Correct execution and risk

- [ ] A5–A20 pass.
- [ ] Contract 20 outputs golden tests pass.
- [ ] 100% execution lifecycle events link được tới trace/order/deal hoặc bị marked orphan.
- [ ] No risk/volume/stops-level violation trong stress test.

## Gate 2 — Memory is actually useful

- [ ] B1–B20 pass.
- [ ] Frozen OOS hash bất biến; adaptive OOS ghi update rõ ràng.
- [ ] Long-term memory có nonzero influence rate và ablation vượt memory-off trên nhiều seed, không chỉ best run.
- [ ] Growth/prune có validation gate + rollback evidence.

## Gate 3 — Research validity

- [ ] C1–C9 pass.
- [ ] Walk-forward gồm multiple regimes/symbols, costs realistic, confidence intervals.
- [ ] Performance được so với baselines và ablations; không chỉ aggregate best metrics.

## Gate 4 — Paper trading readiness

- [ ] D7–E6 pass.
- [ ] Monitoring, kill switches, runbook, incident response, credentials hygiene được test.
- [ ] Ít nhất 4–8 tuần paper trading với acceptance thresholds xác định trước, không manual cherry-pick.

> **Không có Gate 5 “live trading ready” trong baseline này.** Việc tiến tới live cần review risk/compliance riêng, broker-specific testing và chấp thuận vận hành có trách nhiệm.

---

# Thứ tự thực hiện khuyến nghị

1. **A1–A8**: làm pipeline tái lập và có execution facts.
2. **A9–A20**: một action schema, risk sizing thật, safety controls.
3. **B1–B15**: làm transition/memory đáng tin; dừng thêm tầng memory mới.
4. **C1–C9**: frozen/adaptive OOS, walk-forward và ablation.
5. **B16–B20 + F2**: chỉ cho self-growth sau khi memory có evidence.
6. **D/E + Gate 4**: hardening và paper trading.
