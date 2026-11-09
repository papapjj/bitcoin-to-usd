# Checklist nâng cấp học giống con người

Checklist này là scope triển khai chi tiết cho hướng:

```text
episode memory + liên tưởng + dung hợp giả thuyết + mistake replay + sleep consolidation + homeostasis + MT5 validation
```

## 0. Nguyên tắc bắt buộc

- [x] Không ghi đè `winner.pkl` nếu trial hồi quy theo `goal_progress_score`.
- [ ] Không dùng replay nội bộ thay MT5 validation.
- [x] Không copy topology candidate lỗi vào baseline.
- [x] Không mutate global output/gate/recurrent nếu chỉ có evidence action lỗi.
- [x] Không replay journal schema cũ vào memory schema mới.
- [ ] Không đưa graph backend thành default production khi chưa có paired MT5 A/B.
- [x] Mọi schema mới phải có version và migration/load fallback.
- [x] Mọi tài liệu phải cập nhật sau khi đổi behavior.

## 1. MistakeEpisode schema

- [x] Tạo dataclass `MistakeEpisode`.
- [x] Thêm `schema_version`.
- [x] Thêm `episode_id` ổn định.
- [x] Thêm `trace_id`.
- [x] Thêm `source`: `mt5_ledger|journal|replay|rejection`.
- [x] Lưu `raw_obs` hoặc compressed market embedding.
- [x] Lưu `market_key`.
- [x] Lưu `account_key`.
- [x] Lưu `position_key`.
- [x] Lưu `action_index`.
- [x] Lưu `action_name`.
- [x] Lưu `action_details`.
- [x] Lưu `selected_output`.
- [x] Lưu `confidence`.
- [x] Lưu `risk_pct`.
- [x] Lưu SL/TP/trailing/partial/mode/slot params.
- [x] Lưu `module_action_traces` dạng compact.
- [x] Lưu `active_neuron_ids`.
- [x] Lưu `active_edge_ids`.
- [x] Lưu `graph_edge_gates`.
- [x] Lưu `graph_region_activity`.
- [x] Lưu `positions_before`.
- [x] Lưu `positions_after`.
- [x] Lưu `execution_details`.
- [x] Lưu lifecycle fields: MFE, MAE, giveback, bars since MFE, close reason.
- [x] Lưu `reward`.
- [x] Lưu `td_delta`.
- [x] Lưu `prediction_error`.
- [x] Lưu `realized_r`.
- [x] Lưu `net_profit_delta`.
- [x] Lưu `drawdown_delta`.
- [x] Lưu `spread_points`.
- [x] Lưu `mistake_tags`.
- [x] Lưu `suspected_causes`.
- [x] Lưu `proposed_repairs`.
- [x] Lưu `severity`.
- [x] Lưu `confidence_score`.
- [x] Lưu `support_count`.
- [x] Lưu `contradiction_count`.
- [x] Lưu `created_at`.
- [x] Lưu `last_seen_at`.
- [x] Thêm serialization JSONL.
- [x] Thêm load-time migration cho episode schema cũ.
- [x] Thêm test round-trip JSONL.

## 2. MistakeEpisodeStore

- [x] Tạo store trong `ExperienceSuperNetwork` hoặc module riêng.
- [x] Index theo `trace_id`.
- [x] Index theo `action_index`.
- [x] Index theo `mistake_tags`.
- [x] Index theo market/account/position key.
- [x] Deduplicate episode lặp.
- [x] Merge support count khi same pattern lặp lại.
- [x] Giới hạn dung lượng store bằng LRU + utility.
- [x] Không xóa episode có severity cao nếu còn support.
- [x] Query recent bad episodes.
- [x] Query similar bad episodes theo memory key.
- [x] Query bad episodes theo action.
- [x] Query bad episodes theo tag.
- [ ] Query bad episodes theo concept sau khi có `AssociativeConceptGraph`.
- [x] Thêm metrics: count, severity mean, top tags, repeated mistakes.
- [x] Ghi summary vào train history.

## 3. Mistake tagger

- [x] Tạo `MistakeTagger`.
- [x] Tag `wrong_direction` từ adverse R sau entry.
- [x] Tag `late_entry` từ entry sau move quá xa hoặc signal decay.
- [x] Tag `stress_entry` từ drawdown/exposure/floating loss cao.
- [x] Tag `high_spread_entry` từ spread cao có trong bridge/ledger.
- [x] Tag `overtrading` từ day trade pressure cao.
- [x] Tag `bad_risk_pct` từ volume/risk quá lớn so với SL/account.
- [x] Tag `sl_too_tight` từ stop-out nhanh và MAE nhỏ/normal volatility.
- [ ] Tag `tp_too_far` từ MFE tốt nhưng không đạt TP.
- [x] Tag `near_tp_giveback`.
- [x] Tag `missed_partial`.
- [x] Tag `missed_trailing`.
- [x] Tag `bad_target_slot`.
- [x] Tag `held_loser_too_long`.
- [x] Tag `closed_winner_too_early`.
- [x] Tag `execution_reject`.
- [x] Tag `no_trade_starvation`.
- [x] Tag `immediate_exit_loop`.
- [x] Tag `ledger_integrity_bad`.
- [x] Mỗi tag phải có evidence fields.
- [x] Tagger không được dựa vào phỏng đoán không có trace/ledger.
- [ ] Thêm unit tests cho từng tag.

## 4. Liên tưởng: AssociativeConceptGraph

- [ ] Tạo `AssociativeConceptGraph`.
- [ ] Node type: `episode`.
- [ ] Node type: `mistake_tag`.
- [ ] Node type: `condition`.
- [ ] Node type: `action`.
- [ ] Node type: `position_lifecycle`.
- [ ] Node type: `concept`.
- [ ] Node type: `hypothesis`.
- [ ] Edge: `episode_has_tag`.
- [ ] Edge: `episode_used_action`.
- [ ] Edge: `episode_had_condition`.
- [ ] Edge: `tag_supports_concept`.
- [ ] Edge: `condition_supports_concept`.
- [ ] Edge: `concept_supports_hypothesis`.
- [ ] Edge: `hypothesis_repairs_action`.
- [ ] Weight edge bằng support/severity/recency.
- [ ] Decay edge không được support lại.
- [ ] Merge concept gần nhau khi overlap cao.
- [ ] Split concept khi contradiction cao.
- [ ] Query concepts theo current state.
- [ ] Query concepts theo action lỗi.
- [ ] Query concepts theo lifecycle tag.
- [ ] Ghi graph summary vào artifact `.pkl`.
- [ ] Thêm migration cho artifact cũ không có concept graph.
- [ ] Thêm tests: association query, merge, split, decay.

## 5. Dung hợp: ConceptFusionEngine

- [ ] Tạo `ConceptFusionEngine`.
- [ ] Lấy top mistake concepts theo severity/support.
- [ ] Tìm concept đồng xuất hiện cùng action.
- [ ] Tìm concept đồng xuất hiện cùng account stress.
- [ ] Tìm concept đồng xuất hiện cùng lifecycle failure.
- [ ] Tạo fusion candidate khi support đủ.
- [ ] Không fusion nếu contradiction cao.
- [ ] Sinh concept mới có parent ids.
- [ ] Sinh hypothesis từ fused concept.
- [ ] Ghi `fusion_reason`.
- [ ] Ghi `support_episodes`.
- [ ] Ghi `expected_repair`.
- [ ] Không áp repair trực tiếp khi chưa replay.
- [ ] Thêm tests cho `stress_entry + high_spread -> bad_entry_quality`.
- [ ] Thêm tests cho `near_tp_giveback + missed_trailing -> exit_management_failure`.

## 6. HypothesisMemory

- [ ] Tạo dataclass `Hypothesis`.
- [ ] Field `hypothesis_id`.
- [ ] Field `schema_version`.
- [ ] Field `condition`.
- [ ] Field `suspected_cause`.
- [ ] Field `target_action`.
- [ ] Field `target_connections`.
- [ ] Field `repair_type`.
- [ ] Field `repair_strength`.
- [ ] Field `expected_effect`.
- [ ] Field `support_episodes`.
- [ ] Field `contradiction_episodes`.
- [ ] Field `replay_score`.
- [ ] Field `mt5_score_delta`.
- [ ] Field `status`.
- [ ] Status values: `candidate`, `replay_passed`, `mt5_passed`, `rejected`, `retired`.
- [ ] Query hypothesis theo current context.
- [ ] Query hypothesis theo action.
- [ ] Query hypothesis theo concept.
- [ ] Update support sau replay.
- [ ] Update contradiction sau replay.
- [ ] Update support sau MT5 promoted.
- [ ] Retire hypothesis nếu contradiction vượt ngưỡng.
- [ ] Không cho hypothesis sửa global network.
- [ ] Thêm tests cho lifecycle status.

## 7. Repair actions

- [ ] Define repair `reduce_action_bias`.
- [ ] Define repair `increase_hold_bias`.
- [ ] Define repair `increase_close_or_manage_bias`.
- [ ] Define repair `reduce_risk_pct`.
- [ ] Define repair `tighten_entry_gate`.
- [ ] Define repair `increase_trailing_gate`.
- [ ] Define repair `increase_partial_close_ratio`.
- [ ] Define repair `change_target_slot_preference`.
- [ ] Define repair `reduce_connection_weight`.
- [ ] Define repair `increase_connection_weight`.
- [ ] Define repair `lower_graph_edge_gate`.
- [ ] Define repair `raise_graph_edge_gate`.
- [ ] Define repair `grow_specialist_module`.
- [ ] Define repair `add_delay_edge`.
- [ ] Mỗi repair có clamp riêng.
- [ ] Mỗi repair cần evidence target.
- [ ] Repair không được chạm action không liên quan nếu không có hypothesis.
- [ ] Thêm audit event cho từng repair.

## 8. MistakeReplayEngine

- [x] Tạo `MistakeReplayEngine`.
- [x] Input: policy candidate.
- [x] Input: bad episodes.
- [x] Input: hypotheses.
- [x] Reconstruct observation/account/position context.
- [x] Run `policy.act()` hoặc graph activation trên episode state.
- [x] Compare old bad action với new action.
- [x] Score action improvement.
- [x] Score risk improvement.
- [x] Score position-management improvement.
- [x] Score repeated mistake.
- [x] Penalize repeated mistake có attribution.
- [x] Reward safer alternative có evidence.
- [x] Không mutate baseline trực tiếp.
- [x] Replay result lưu `episode_id`, `old_action`, `new_action`, `repeated_mistake`, `safer_alternative`.
- [ ] Aggregate replay score theo hypothesis.
- [ ] Reject candidate trước MT5 nếu replay regression nặng.
- [x] Thêm tests cho replay action tránh lỗi.
- [ ] Thêm tests cho replay không sửa khi no evidence.

## 9. SleepConsolidationEngine

- [ ] Tạo `SleepConsolidationEngine`.
- [ ] Trigger sau N rejected trials.
- [ ] Trigger cuối mỗi train-to-goal round.
- [ ] Trigger khi outcome_starvation cao.
- [ ] Trigger trước khi overwrite `winner.pkl`.
- [ ] Deduplicate episodes.
- [ ] Update concept graph.
- [ ] Fuse concepts.
- [ ] Generate hypotheses.
- [ ] Run mistake replay.
- [ ] Apply homeostasis guard.
- [ ] Create candidate checkpoint.
- [ ] Run MT5 validation nếu candidate replay pass.
- [ ] Promote only if non-regression.
- [ ] Log sleep summary.
- [ ] Persist accepted hypotheses.
- [ ] Persist rejected hypotheses with reasons.
- [ ] Thêm CLI flag `--sleep-consolidation`.
- [ ] Thêm CLI flag `--sleep-max-episodes`.
- [ ] Thêm CLI flag `--sleep-replay-threshold`.
- [ ] Thêm docs command examples.

## 10. HomeostasisGuard

- [ ] Tạo `HomeostasisGuard`.
- [ ] Clamp max delta per action row per round.
- [ ] Clamp max delta per hidden connection per round.
- [ ] Clamp max graph edge weight delta.
- [ ] Clamp max graph edge gate delta.
- [ ] Clamp max output bias shift.
- [ ] Freeze action paths có recent positive support.
- [ ] Freeze graph edges có high utility.
- [ ] Reduce plasticity when drawdown rises.
- [ ] Reduce entry plasticity when account stress high.
- [ ] Require repeated support for large update.
- [ ] Increase learning only when repeated mistake support high.
- [ ] Rollback candidate if replay score below baseline.
- [ ] Rollback candidate if goal_progress_score below baseline.
- [ ] Log homeostasis metrics.
- [ ] Thêm tests cho freeze positive path.
- [ ] Thêm tests cho max delta clamp.

## 11. Multi-scale delay links

- [ ] Extend `ConnectionEdge` với `delay_steps`.
- [ ] Extend `ConnectionEdge` với ring buffer hoặc state buffer.
- [ ] Add migration default `delay_steps=0`.
- [ ] Support delay edge in graph activation.
- [ ] Delay edge chỉ dùng bounded state.
- [ ] Delay edge có decay riêng.
- [ ] Delay edge có budget riêng.
- [ ] Delay edge trace phải ghi active delayed source.
- [ ] Delay options: 1, 5, 20, 60, 240, 1440 bars.
- [ ] Bắt đầu chỉ bật delay cho memory/predictor/critic/risk/management.
- [ ] Không bật delay cho raw input toàn bộ ngay.
- [ ] Add invariant for delay buffer finite values.
- [ ] Add tests for delay activation.
- [ ] Add tests for serialization.
- [ ] Add tests for no unbounded memory growth.

## 12. Basal-ganglia-like action gating

- [ ] Tạo action gate layer riêng.
- [ ] Gate entry actions.
- [ ] Gate close/manage actions.
- [ ] Gate risk_pct.
- [ ] Gate pending order actions.
- [ ] Gate chịu ảnh hưởng account stress.
- [ ] Gate chịu ảnh hưởng concept/hypothesis.
- [ ] Gate không thay EA hard risk authority.
- [ ] Gate có direct attribution khi action xấu.
- [ ] Gate có positive consolidation khi action tốt.
- [ ] Thêm tests cho stress giảm entry.

## 13. Hippocampus/cortex split

- [ ] Tách fast episodic store.
- [ ] Tách slow strategy memory.
- [ ] Episode mới chỉ vào fast store.
- [ ] Sleep consolidation mới chuyển pattern sang slow memory.
- [ ] Slow memory không ghi episode đơn lẻ.
- [ ] Contradiction trong fast store có thể giảm confidence slow memory.
- [ ] Thêm metrics fast/slow memory.
- [ ] Thêm tests cho consolidation threshold.

## 14. MT5 integration

- [x] Ledger/trace path tạo MistakeEpisode từ lifecycle xấu đã có trong `observe_transition()`.
- [ ] Journal replay tạo episode nếu có outcome xấu đủ evidence.
- [x] `train_self_growing_mt5` ghi mistake summary vào history.
- [ ] `improve_self_growing` gọi sleep consolidation khi flag bật.
- [x] `train-to-goal` giữ non-regression guard.
- [x] Rejected trial vẫn sinh lessons nhưng không overwrite seed.
- [ ] Promoted trial consolidate positive hypotheses.
- [ ] Graph trial history ghi concept/hypothesis/replay metrics.
- [ ] Không tăng timeout mặc định nếu replay chạy nội bộ nhanh.
- [ ] Thêm integration tests với fake evaluator.

## 15. UI/CLI

- [ ] Thêm CLI `--sleep-consolidation`.
- [ ] Thêm CLI `--mistake-replay-max`.
- [ ] Thêm CLI `--hypothesis-max`.
- [ ] Thêm CLI `--homeostasis-strength`.
- [ ] Thêm CLI `--delay-links`.
- [ ] Thêm CLI `--delay-steps`.
- [ ] UI Java thêm checkbox Sleep consolidation.
- [ ] UI Java thêm input Mistake replay max.
- [ ] UI Java thêm input Homeostasis strength.
- [ ] UI Java hiển thị goal_progress_score baseline/trial.
- [ ] UI Java hiển thị top mistake tags.
- [ ] UI Java hiển thị promoted/rejected reason.
- [x] CLI/UI `model-viewer` hiển thị 3D active node/edge, endpoint, edge kind, degree và active connection.
- [x] Cập nhật docs UI cho viewer 3D.

## 16. Observability

- [x] Log `mistake_episode_count` qua `mistake_episodes.count`.
- [x] Log top mistake tags qua `mistake_episodes.top_tags`.
- [ ] Log concept count.
- [ ] Log hypothesis count.
- [ ] Log replay episodes used.
- [ ] Log replay repeated mistakes.
- [ ] Log replay improvements.
- [ ] Log homeostasis clamps.
- [ ] Log delay edge count.
- [ ] Log positive consolidated paths.
- [ ] Log rejected hypothesis reasons.
- [ ] Add JSONL compact summary.
- [ ] Add graph trial history fields.
- [x] Export model-viewer summary có edge-kind counts và active edge-kind counts.

## 17. Tests bắt buộc

- [x] Unit test MistakeEpisode serialization.
- [ ] Unit test tagger từng lỗi.
- [ ] Unit test associative graph query.
- [ ] Unit test concept fusion.
- [ ] Unit test hypothesis lifecycle.
- [ ] Unit test repair target clamp.
- [ ] Unit test mistake replay avoids old action.
- [ ] Unit test mistake replay does not mutate baseline.
- [ ] Unit test sleep consolidation rollback.
- [x] Unit test non-regression guard.
- [ ] Unit test delay edge activation.
- [ ] Unit test delay edge serialization.
- [x] Integration test fake MT5 rejected trial creates episode.
- [ ] Integration test fake MT5 promoted trial consolidates hypothesis.
- [ ] Full `python -m pytest -q`.

## 18. Validation thật trên MT5

- [ ] Chạy baseline legacy MT5.
- [ ] Chạy graph hiện tại MT5.
- [ ] Chạy graph + mistake episode.
- [ ] Chạy graph + mistake replay.
- [ ] Chạy graph + sleep consolidation.
- [ ] So sánh paired A/B cùng date/window/runner.
- [ ] Kiểm tra report HTML.
- [ ] Kiểm tra execution ledger reconciliation.
- [ ] Kiểm tra net profit.
- [ ] Kiểm tra profit factor.
- [ ] Kiểm tra expectancy.
- [ ] Kiểm tra drawdown.
- [ ] Kiểm tra immediate exit ratio.
- [ ] Kiểm tra execution failure rate.
- [ ] Kiểm tra goal_progress_score.
- [ ] Không deploy nếu chỉ Python tests pass.

## 19. Tiêu chí hoàn thành

- [x] Trial xấu tạo MistakeEpisode có tag đúng.
- [x] Rejected trial tạo lesson nhưng không overwrite `winner.pkl`.
- [x] Mutation chỉ chạm action/connection có attribution.
- [ ] Hypothesis được tạo từ nhiều episode có support.
- [ ] Mistake replay giảm lặp lại action sai trên episode cũ.
- [ ] Sleep consolidation không promote candidate replay-regressed.
- [x] MT5 non-regression guard giữ baseline khi trial drawdown/lỗ nặng hơn.
- [ ] Graph backend vẫn experimental cho tới paired MT5 A/B thắng legacy.
- [ ] Docs và knowledge graph phản ánh đúng behavior.

## 20. Nâng cấp liên quan và phụ thuộc

### 20.1 MT5 lifecycle/ledger

- [ ] Đảm bảo every intent/action có `trace_id`.
- [ ] Đảm bảo ledger giữ ticket/slot/side/volume/entry/SL/TP.
- [ ] Đảm bảo ledger giữ accepted/rejected/fill status.
- [ ] Đảm bảo ledger giữ spread tại thời điểm action.
- [ ] Đảm bảo ledger giữ account drawdown/exposure tại thời điểm action.
- [ ] Đảm bảo ledger giữ MFE/MAE.
- [ ] Đảm bảo ledger giữ near TP/SL distance.
- [ ] Đảm bảo ledger giữ giveback.
- [ ] Đảm bảo ledger giữ duration bars.
- [ ] Đảm bảo ledger giữ close reason.
- [ ] Đảm bảo ledger giữ realized R.
- [ ] Đảm bảo ledger giữ execution failure reason.
- [ ] Thêm test parse ledger thành lifecycle đầy đủ.
- [ ] Thêm test ledger thiếu `trace_id` không tạo MistakeEpisode sai.

### 20.2 Artifact schema/migration

- [x] Thêm artifact field `mistake_episode_store`.
- [ ] Thêm artifact field `associative_concept_graph`.
- [ ] Thêm artifact field `hypothesis_memory`.
- [ ] Thêm artifact field `sleep_consolidation_state`.
- [ ] Thêm artifact field `homeostasis_state`.
- [ ] Thêm artifact field delay buffers nếu bật delay edge.
- [x] Migration model cũ phải tạo store rỗng.
- [x] Migration model cũ không được làm đổi output count.
- [x] Migration model cũ không được reset ExperienceSuperNetwork.
- [x] Thêm test load artifact cũ.
- [ ] Thêm test save/load artifact mới.

### 20.3 Journal/replay schema

- [ ] Tách raw transition journal và mistake episode journal.
- [ ] Thêm `schema_version` cho mistake journal.
- [ ] Thêm `source_policy_id`.
- [ ] Thêm `source_trial_id`.
- [ ] Thêm replay result journal.
- [ ] Thêm sleep consolidation audit journal.
- [ ] Replay phải skip schema không tương thích.
- [ ] Replay phải ghi số item skipped.
- [ ] Replay positive transition không được trộn với mistake episode.
- [ ] Thêm test schema gate.

### 20.4 Fitness/non-regression

- [ ] Ghi `baseline_progress_score` vào train log.
- [ ] Ghi `trial_progress_score` vào train log.
- [ ] Ghi reason `goal_progress_not_improved`.
- [ ] Thêm repeated mistake rate vào progress score sau khi MistakeReplayEngine có.
- [ ] Thêm replay regression penalty.
- [ ] Thêm hypothesis contradiction penalty.
- [ ] Không dùng fitness bị kẹp `-100` làm tiêu chí duy nhất.
- [ ] Test trial lỗ nặng hơn không được ghi đè baseline.

### 20.5 UI/CLI

- [ ] CLI flag `--sleep-consolidation`.
- [ ] CLI flag `--mistake-episode-journal`.
- [ ] CLI flag `--mistake-replay-max`.
- [ ] CLI flag `--hypothesis-max`.
- [ ] CLI flag `--homeostasis-strength`.
- [ ] CLI flag `--delay-links`.
- [ ] UI hiển thị top mistake tags.
- [ ] UI hiển thị hypothesis đang test.
- [ ] UI hiển thị replay pass/fail.
- [ ] UI hiển thị homeostasis clamps.
- [ ] UI hiển thị `baseline_progress_score`.
- [ ] UI hiển thị `trial_progress_score`.
- [ ] UI hiển thị reject/promote reason rõ ràng.
- [ ] UI không ghi "promotion gate off = always save".

### 20.6 Observability/audit

- [ ] Audit mỗi targeted mutation.
- [ ] Audit mỗi graph edge repair.
- [ ] Audit mỗi hidden connection repair.
- [ ] Audit mỗi hypothesis repair.
- [ ] Audit old/new value.
- [ ] Audit source episode ids.
- [ ] Audit source hypothesis id.
- [ ] Audit homeostasis clamp.
- [ ] Audit replay result.
- [ ] Audit MT5 result.
- [ ] Ghi compact summary vào JSONL để không phình log.

### 20.7 Validation pipeline

- [ ] Unit tests cho schema mới.
- [ ] Unit tests cho tagger.
- [ ] Unit tests cho concept graph.
- [ ] Unit tests cho hypothesis memory.
- [ ] Unit tests cho replay engine.
- [ ] Fake MT5 tests cho reject/promote.
- [ ] Real MT5 smoke test legacy.
- [ ] Real MT5 smoke test graph.
- [ ] Real MT5 smoke test graph + human-like learning.
- [ ] Paired A/B cùng window.
- [ ] Walk-forward sau khi smoke pass.
- [ ] Lưu report và ledger evidence.

### 20.8 Cleanup/deprecation

- [ ] Search lại mọi path random global mutation còn được gọi.
- [ ] Đánh dấu hoặc xóa docs cũ nói train-to-goal lưu mọi best trial.
- [ ] Cập nhật UI text promotion gate.
- [ ] Cập nhật tests cũ kỳ vọng gate off promote luôn.
- [ ] Xóa sample/config không còn đúng schema replay.
- [ ] Cập nhật `.understand-anything/knowledge-graph.json`.
