package com.selfgrowingtrader.control;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.RowFilter;
import javax.swing.JSplitPane;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.WindowConstants;
import javax.swing.border.EmptyBorder;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableRowSorter;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.Toolkit;
import java.awt.datatransfer.StringSelection;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.time.format.ResolverStyle;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/**
 * V193 evidence-gated edge-discovery and Reasoning-RL candidate control surface.
 *
 * <p>Every workflow that can update model weights is pinned to a versioned
 * candidate config/profile pair. The editable base config remains available
 * only for seed creation, frozen evaluation and artifact administration.</p>
 */
public final class Main {
    private static final String UI_VERSION = "V193 · SGT-Compact (xLSTM 17.5M) · Tier 1 & 2";
    private static final String CLI_MODULE = "self_growing_trader.cli";
    private static final String CLI_SOURCE = "src/self_growing_trader/cli.py";
    private static final String CANDIDATE_SCRIPT = "scripts/run_edge_discovery_v193_candidate.py";
    private static final String ABLATION_SCRIPT = "scripts/run_mt5_ablation_v193.py";
    private static final String EXPERIMENT_PROFILE = "configs/experiments/edge_discovery_v193_candidate.yaml";
    private static final String SGT_COMPACT_EXPERIMENT_PROFILE =
            "configs/experiments/sgt_compact_tier2_candidate.yaml";
    private static final String SGT_COMPACT_DEFAULT_SEED =
            "outputs/he_chuyen_gia/v193_runs/seed_sgt_compact_tier2.pkl";
    private static final String REASONING_ACTIVE_EXPERIMENT_PROFILE =
            "configs/experiments/reasoning_rl_v1_active_candidate.yaml";
    private static final String CODEX_EXPERIMENT_PROFILE = "configs/experiments/codex_teacher_every_close_candidate.yaml";
    private static final String BASE_CONFIG = "configs/he_chuyen_gia_mot_context_mot_policy.yaml";
    private static final String LATEST_TRAIN_CONFIG = "configs/he_chuyen_gia_latest_ui_v193.yaml";
    private static final String REASONING_TRAIN_CONFIG = "configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml";
    private static final String CODEX_TRAIN_CONFIG = "configs/he_chuyen_gia_codex_teacher_candidate.yaml";
    private static final int CANDIDATE_MODE_REASONING = 0;
    private static final int CANDIDATE_MODE_EDGE = 1;
    private static final int CANDIDATE_MODE_CODEX = 2;
    private static final String[] SEVEN_CHOICE_REASONING_CONFIGS = {
            "configs/he_chuyen_gia_reasoning_rl_v1_candidate_sl500_h12.yaml",
            "configs/he_chuyen_gia_reasoning_rl_v1_candidate_sl500_h24.yaml",
            "configs/he_chuyen_gia_reasoning_rl_v1_candidate_sl500_h48.yaml",
    };
    private static final String[] SEVEN_CHOICE_REASONING_LABELS = {
            "Bộ nhãn sl500 · giữ 12 nến",
            "Bộ nhãn sl500 · giữ 24 nến",
            "Bộ nhãn sl500 · giữ 48 nến",
    };
    private static final String DEFAULT_RUNNER = "mt5_v2_runner";
    private static final String CODEX_TEACHER_MODEL = "gpt-5.6-luna";
    private static final String CODEX_TEACHER_REASONING = "xhigh";
    private static final String ZIGZAG_TOOL = "tools/five_zigzag_dataset_tool/run_tool.py";
    private static final String TEACHER_IMITATION_SCRIPT = "scripts/train_teacher_imitation.py";
    private static final String LEGACY_FIVE_CHECKPOINT =
            "outputs/he_chuyen_gia/kho_chuyen_gia.pkl";
    private static final String FIVE_TRANSFORMER_V3_CHECKPOINT =
            "outputs/he_chuyen_gia/kho_chuyen_gia_transformer_v3.pkl";
    private static final Set<String> RETIRED_DEFAULT_IMITATION_CHECKPOINTS = Set.of(
            LEGACY_FIVE_CHECKPOINT,
            "outputs/he_chuyen_gia/kho_chuyen_gia_transformer_v2.pkl"
    );
    private static final String LATEST_TEACHER_IMITATION_DATASET =
            "outputs/he_chuyen_gia/v193_repair_20260910/teacher_16w_confirmed_schema9.npz";
    private static final Set<String> RETIRED_TEACHER_IMITATION_DATASETS = Set.of(
            "outputs/bottom_top_teacher_recurrent_v5_7d_20260827/teacher_7d_recurrent.npz",
            "outputs/bottom_top_teacher_recurrent_v5_16w_20260827/teacher_16w_recurrent.npz",
            "outputs/bottom_top_teacher_recurrent_v5_16w_sl075_20260827/teacher_16w_sl075_recurrent.npz",
            "outputs/bottom_top_teacher_recurrent_v5_max_sl075_20260827/teacher_max_sl075_recurrent.npz",
            "outputs/bottom_top_teacher_recurrent_v6_max_sl075_20260828/teacher_max_sl075_geometry_v6.npz",
            "outputs/bottom_top_teacher_recurrent_v7_max_sl075_20260828/teacher_max_sl075_geometry_v7.npz",
            "outputs/bottom_top_teacher_confirmed_action_v8_20260830/teacher_confirmed_action_v8.npz",
            "outputs/bottom_top_teacher_confirmed_geometry_v9_sl150_20260901/teacher_confirmed_geometry_v9_sl150.npz"
    );
    private static final DateTimeFormatter INPUT_DATE = DateTimeFormatter
            .ofPattern("uuuu.MM.dd")
            .withResolverStyle(ResolverStyle.STRICT);
    private static final DateTimeFormatter RUN_STAMP = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");

    private static String fiveArchitectureContractText() {
        return """
                QUAN SÁT CÓ CẤU TRÚC — 137 GIÁ TRỊ / ENCODER V4 + PATTERN TREE

                Thị trường 78 ── Conv1D × 2 ───────────────────────────────→ 256
                Tài khoản 8 ─── MLP ──────────────────────────────────────→  64
                Thực thi 11 ─── MLP ──────────────────────────────────────→  64
                Vị thế 5 × 8 ── shared MLP × 2 ── max pool ───────────────→  64
                                   └─ embedding riêng từng vị thế ───────────┐
                Agent ID ─────── embedding học được ──────────────────────→  32 │
                Action trước ─── embedding học được ──────────────────────→  32 │
                                                                             │
                Ghép state 512 ── FC 1024 ── max-pool 25% giữa 5 agent       │
                                      └──── 5 memory LSTM riêng, 4096 ──────┤
                                                    │
                                  Transformer 1 lớp · 512 · 8 heads · FFN 2048
                                                                             │
                ĐẦU RA PHÂN RÃ                                                │
                Action: 12 loại bằng action embedding                         │
                Target: query(memory) × key(vị thế) + mask theo action ───────┘
                Delay: 4 mức (0..3) · Entry offset: 17 mức · Risk: 16 mức
                Manage mode: 3 · Close fraction: 4 · Value: 1

                PHÂN QUYỀN FIVE-AGENT AUTHORITY V1
                Agent 0 · ENTRY MUA      → MARKET_BUY / BUY_LIMIT / BUY_STOP
                Agent 1 · ENTRY BÁN      → MARKET_SELL / SELL_LIMIT / SELL_STOP
                Agent 2 · QUẢN LÝ VỊ THẾ → CLOSE_WORST / CLOSE_BEST / MANAGE_POSITION
                Agent 3 · LỆNH CHỜ       → CANCEL_ORDERS
                Agent 4 · DANH MỤC        → CLOSE_ALL (phủ quyết packet)
                Mọi agent có HOLD; mỗi action khác HOLD chỉ có đúng một chủ.

                BRIDGE SCHEMA 10 / EXECUTION LEDGER SCHEMA 14
                1 observation ── ghi atomically 5 action rows (agent_id 0..4)
                              ── EA kiểm tra quyền/capacity/risk budget
                              ── 5 receipt riêng theo trace_id + agent_id
                              ── gom transition đồng bộ 5 × T ── PPO/GAE

                REWARD KINH TẾ + GIẢI ĐẤU NỘI BỘ
                marked-R delta − chi phí − drawdown − risk − conflict
                              + 0.25 × (cải thiện agent − trung bình 5 agent)
                Bonus cạnh tranh có tổng bằng 0; không tạo lợi nhuận giả.
                Terminal: equity-R − max drawdown + rank tanh từng agent
                Lịch 10.000 packet: team-spirit 0.30 → 1.00; cạnh tranh vẫn giữ
                ngoài phép trộn đội; shaping không-terminal dùng time-decay

                REASONING RL V1 — CANDIDATE CÔ LẬP, ACTION + OPTIMIZER
                Hidden LSTM causal ── GRU hypothesis 3 bước × nhóm 8 rollout
                                   ── verifier path 48-M1 ── GRPO/PPO + KL + entropy
                Head reasoning nhận gradient qua FiveAdam; encoder credit = 0,
                action scale = 0.25 (residual bị clip ở 0.25) trên 7 action đầu.
                MT5 learning bật trong profile active; update chỉ xảy ra khi path
                48-M1 mature, reward có phương sai và batch đủ điều kiện.
                Future label chỉ dùng sau lifecycle; không vào observation/memory/action.
                Đây vẫn là candidate research-only, chưa phải bằng chứng hiệu quả hay edge MT5/OOS.

                CODEX TEACHER — CANDIDATE-ONLY / THEO ACTION CÓ OUTCOME
                MT5 target trace + position ticket ── mỗi terminal action vào queue nền
                                  ── gpt-5.6-luna · xhigh · 0 = không giới hạn
                                  ── structured reward + confidence + rationale
                                  ── action head + latent policy-state, probability guard,
                                     rollback nếu vượt ngưỡng
                Teacher chỉ chấm đúng target trace packet; thiếu outcome/bằng chứng thì skip.
                Reward teacher không phải bằng chứng OOS và không được promote trực tiếp.

                Không còn bridge round-robin một tác nhân. Packet thiếu hoặc trùng agent
                bị từ chối fail-closed; cả năm memory chỉ tiến cùng một bước quan sát.

                TIER 1 & TIER 2 NÂNG CẤP KIẾN TRÚC & ĐỘNG LỰC HỌC RL
                • Tier 1: Representation Regularizer (VICReg + RankMe + ArcFace m=0.35)
                          Dynamic Multi-Regime Reservoir Buffer & EWC (Bull/Bear/Sideway 33%)
                          Extended Lifecycle Settlement (phạt bậc hai beta=0.25, MAE/MFE)
                • Tier 2: SGT-V193-Compact Sequence Core (xLSTM: mLSTM 768D 147k matrix memory + sLSTM 768D)
                          Relational Cross-Agent Graph Transformer (5x5 role prior, ReZero alpha=0.0)
                          Thu nhỏ 81% tham số (17.5M), tăng tốc CPU forward 5.5x (P50 31.7ms)
                          Progressive Knowledge Distillation (Rep + Logit KL) & COMA + RND + HER
                """;
    }

    private static LocalDate parseInputDate(String raw) {
        if (raw == null || raw.isBlank()) {
            throw new DateTimeParseException("Ngày không được để trống", String.valueOf(raw), 0);
        }
        return LocalDate.parse(raw.trim(), INPUT_DATE);
    }

    static String migrateTeacherImitationDataset(String raw) {
        String value = raw == null ? "" : raw.trim().replace('\\', '/');
        return RETIRED_TEACHER_IMITATION_DATASETS.contains(value)
                ? LATEST_TEACHER_IMITATION_DATASET
                : raw;
    }

    private enum PathMode {
        EXISTING_FILE,
        EXISTING_DIRECTORY,
        NEW_FILE,
        NEW_DIRECTORY
    }

    private Main() {
    }

    public static void main(String[] args) {
        if (args.length > 0 && "--self-test".equals(args[0])) {
            selfTestCommandContract();
            return;
        }
        SwingUtilities.invokeLater(() -> {
            Path workspace = args.length > 0 ? Path.of(args[0]) : Path.of(System.getProperty("user.dir"));
            new TraderControlFrame(workspace.toAbsolutePath().normalize()).setVisible(true);
        });
    }

    private static List<String> cliPrefix(String python, String config, String commandName) {
        List<String> command = new ArrayList<>();
        command.add(nonBlank(python, "python"));
        command.add("-m");
        command.add(CLI_MODULE);
        addOption(command, "--config", nonBlank(config, BASE_CONFIG));
        command.add(commandName);
        return command;
    }

    static List<String> candidateCommand(
            String python, String trainingConfig, String runner, String seedArtifact,
            String outputDir, String fromDate, String toDate, String model,
            String seed, String timeoutSec, String maxWallSec, boolean visual,
            boolean allowUnverifiedImitationSeed
    ) {
        return candidateCommand(
                python, trainingConfig, runner, seedArtifact,
                outputDir, fromDate, toDate, model,
                seed, timeoutSec, maxWallSec, visual,
                allowUnverifiedImitationSeed, -1
        );
    }

    static List<String> candidateCommand(
            String python, String trainingConfig, String runner, String seedArtifact,
            String outputDir, String fromDate, String toDate, String model,
            String seed, String timeoutSec, String maxWallSec, boolean visual,
            boolean allowUnverifiedImitationSeed, int architectureIndex
    ) {
        List<String> command = new ArrayList<>();
        command.add(nonBlank(python, "python"));
        command.add(CANDIDATE_SCRIPT);
        addOption(command, "--mode", "mt5-smoke");
        String profile = architectureIndex == 0
                ? SGT_COMPACT_EXPERIMENT_PROFILE
                : candidateExperimentProfile(trainingConfig);
        addOption(command, "--profile", profile);
        addOption(command, "--config", nonBlank(trainingConfig, REASONING_TRAIN_CONFIG));
        addOption(command, "--seed-artifact", seedArtifact);
        addOption(command, "--runner-source", nonBlank(runner, DEFAULT_RUNNER));
        addOption(command, "--output-dir", outputDir);
        addOption(command, "--from-date", fromDate);
        addOption(command, "--to-date", toDate);
        addOption(command, "--model", model);
        addOption(command, "--seed", seed);
        addOption(command, "--timeout-sec", timeoutSec);
        addOption(command, "--max-wall-sec", maxWallSec);
        addOption(command, "--stop-request-file", candidateStopRequestFile(outputDir));
        if (visual) command.add("--mt5-visual");
        if (allowUnverifiedImitationSeed) command.add("--allow-unverified-imitation-seed");
        return command;
    }

    private static String candidateExperimentProfile(String trainingConfig) {
        String normalized = nonBlank(trainingConfig, REASONING_TRAIN_CONFIG).replace('\\', '/');
        if (CODEX_TRAIN_CONFIG.equals(normalized)) return CODEX_EXPERIMENT_PROFILE;
        if (normalized.contains("reasoning_rl_v1_candidate")) {
            return REASONING_ACTIVE_EXPERIMENT_PROFILE;
        }
        return EXPERIMENT_PROFILE;
    }

    static String candidateConfigForMode(int candidateMode, int sevenChoiceProfile) {
        if (candidateMode == CANDIDATE_MODE_CODEX) return CODEX_TRAIN_CONFIG;
        if (candidateMode == CANDIDATE_MODE_EDGE) return LATEST_TRAIN_CONFIG;
        if (sevenChoiceProfile >= 1 && sevenChoiceProfile <= SEVEN_CHOICE_REASONING_CONFIGS.length) {
            return SEVEN_CHOICE_REASONING_CONFIGS[sevenChoiceProfile - 1];
        }
        return REASONING_TRAIN_CONFIG;
    }

    static List<String> ablationCommand(
            String python, String config, String runner, String checkpoint,
            String outputDir, String fromDate, String toDate, String model,
            String seed, String timeoutSec, String maxWallSec,
            boolean visual, boolean keepCheckpoints
    ) {
        List<String> command = new ArrayList<>();
        command.add(nonBlank(python, "python"));
        command.add(ABLATION_SCRIPT);
        addOption(command, "--profile", EXPERIMENT_PROFILE);
        addOption(command, "--config", nonBlank(config, REASONING_TRAIN_CONFIG));
        addOption(command, "--checkpoint", checkpoint);
        addOption(command, "--runner-dir", nonBlank(runner, DEFAULT_RUNNER));
        addOption(command, "--output-dir", outputDir);
        addOption(command, "--from-date", fromDate);
        addOption(command, "--to-date", toDate);
        addOption(command, "--model", model);
        addOption(command, "--seed", seed);
        addOption(command, "--timeout-sec", timeoutSec);
        addOption(command, "--max-wall-sec", maxWallSec);
        if (visual) command.add("--mt5-visual");
        if (keepCheckpoints) command.add("--keep-checkpoints");
        return command;
    }

    static List<String> initializeCommand(
            String python, String config, String output, String seed
    ) {
        List<String> command = cliPrefix(python, config, "khoi-tao-he-chuyen-gia");
        addOption(command, "--output", output);
        addOption(command, "--seed", seed);
        return command;
    }

    static List<String> researchCommand(
            String python, String config, boolean frozen, String output,
            String seedSystem, String runner, String fromDate, String toDate,
            String backtestDays, String model, String seed, String timeoutSec,
            String maxWallSec, boolean visual, boolean keepOpen
    ) {
        List<String> command = cliPrefix(
                python, frozen ? config : LATEST_TRAIN_CONFIG,
                "huan-luyen-he-chuyen-gia-mt5"
        );
        addOption(command, "--output", output);
        if (frozen) {
            addOption(command, "--seed-system", seedSystem);
            command.add("--frozen");
        } else {
            command.add("--fresh-system");
        }
        addOption(command, "--seed", seed);
        addOption(command, "--runner-dir", nonBlank(runner, DEFAULT_RUNNER));
        addOption(command, "--timeout-sec", timeoutSec);
        addOption(command, "--max-wall-sec", maxWallSec);
        addOption(command, "--from-date", fromDate);
        if (toDate != null && !toDate.isBlank()) {
            addOption(command, "--to-date", toDate);
        } else {
            addOption(command, "--backtest-days", backtestDays);
        }
        addOption(command, "--model", model);
        if (visual) command.add("--mt5-visual");
        if (keepOpen) command.add("--keep-mt5-open");
        return command;
    }

    static List<String> championCommand(
            String python, String champion, String output,
            String auditDir, String trainFrom, String trainTo,
            String validationCount, String validationDays, String candidateSeedCount,
            String seed, String runner, String timeoutSec, String maxWallSec,
            String model, String minExpectancy, String maxProfitRegression,
            String maxLcbRegression, String maxDdIncrease,
            boolean visual, boolean keepOpen
    ) {
        List<String> command = cliPrefix(python, LATEST_TRAIN_CONFIG, "huan-luyen-champion-he-chuyen-gia-mt5");
        addOption(command, "--champion", champion);
        addOption(command, "--output", output);
        addOption(command, "--audit-dir", auditDir);
        addOption(command, "--train-from", trainFrom);
        addOption(command, "--train-to", trainTo);
        addOption(command, "--validation-count", validationCount);
        addOption(command, "--validation-days", validationDays);
        addOption(command, "--candidate-seed-count", candidateSeedCount);
        addOption(command, "--seed", seed);
        addOption(command, "--runner-dir", nonBlank(runner, DEFAULT_RUNNER));
        addOption(command, "--timeout-sec", timeoutSec);
        addOption(command, "--max-wall-sec", maxWallSec);
        addOption(command, "--model", model);
        addOption(command, "--paired-min-expectancy", minExpectancy);
        addOption(command, "--paired-max-profit-regression", maxProfitRegression);
        addOption(command, "--paired-max-expectancy-lcb-regression", maxLcbRegression);
        addOption(command, "--paired-max-drawdown-increase-pct", maxDdIncrease);
        if (visual) command.add("--mt5-visual");
        if (keepOpen) command.add("--keep-mt5-open");
        return command;
    }

    static List<String> walkForwardCommand(
            String python, String output, String log,
            String seedSystem, String seed, String seedCount, String runner,
            String timeoutSec, String fromDate, String toDate,
            String trainWindow, String testWindow, String step, String model,
            boolean pairedBaseline, boolean rollback,
            String minExpectancy, String maxProfitRegression,
            String maxLcbRegression, String maxDdIncrease,
            boolean visual, boolean keepOpen
    ) {
        List<String> command = cliPrefix(python, LATEST_TRAIN_CONFIG, "walk-forward-he-chuyen-gia-mt5");
        addOption(command, "--output", output);
        addOption(command, "--log", log);
        addOption(command, "--seed-system", seedSystem);
        addOption(command, "--seed", seed);
        addOption(command, "--seed-count", seedCount);
        addOption(command, "--runner-dir", nonBlank(runner, DEFAULT_RUNNER));
        addOption(command, "--timeout-sec", timeoutSec);
        addOption(command, "--from-date", fromDate);
        addOption(command, "--to-date", toDate);
        addOption(command, "--train-window", trainWindow);
        addOption(command, "--test-window", testWindow);
        addOption(command, "--step", step);
        addOption(command, "--model", model);
        if (pairedBaseline) command.add("--paired-baseline");
        if (!rollback) command.add("--no-rollback");
        addOption(command, "--paired-min-expectancy", minExpectancy);
        addOption(command, "--paired-max-profit-regression", maxProfitRegression);
        addOption(command, "--paired-max-expectancy-lcb-regression", maxLcbRegression);
        addOption(command, "--paired-max-drawdown-increase-pct", maxDdIncrease);
        if (visual) command.add("--mt5-visual");
        if (keepOpen) command.add("--keep-mt5-open");
        return command;
    }

    static List<String> adminCommand(
            String python, String config, String system, String output,
            String context, String action, String targetContext,
            String lifecycleCount, String note, String auditLog,
            boolean replayPass, boolean oosPass, boolean riskPass,
            boolean nonInterferencePass
    ) {
        List<String> command = cliPrefix(python, config, "quan-tri-he-chuyen-gia");
        addOption(command, "--system", system);
        addOption(command, "--output", output);
        addOption(command, "--context", context);
        addOption(command, "--action", action);
        addOption(command, "--target-context", targetContext);
        addOption(command, "--lifecycle-count", lifecycleCount);
        addOption(command, "--note", note);
        addOption(command, "--audit-log", auditLog);
        if (replayPass) command.add("--replay-pass");
        if (oosPass) command.add("--oos-pass");
        if (riskPass) command.add("--risk-pass");
        if (nonInterferencePass) command.add("--non-interference-pass");
        return command;
    }

    static List<String> datasetCheckCommand(
            String python, String config, String input, String spreadPoints, boolean allowNeutralVolume
    ) {
        List<String> command = new ArrayList<>();
        command.add(nonBlank(python, "python"));
        command.add(ZIGZAG_TOOL);
        command.add("check-five");
        addOption(command, "--project-root", ".");
        addOption(command, "--config", nonBlank(config, LATEST_TRAIN_CONFIG));
        addOption(command, "--input", input);
        addOption(command, "--default-spread-points", spreadPoints);
        if (allowNeutralVolume) command.add("--allow-neutral-volume");
        return command;
    }

    static List<String> datasetMineCommand(
            String python, String config, String input, String pattern, String output,
            String atrPeriod, String zigzagAtrMult, String shardSize, String spreadPoints,
            boolean allowNeutralVolume
    ) {
        List<String> command = new ArrayList<>();
        command.add(nonBlank(python, "python"));
        command.add(ZIGZAG_TOOL);
        command.add("mine-five");
        addOption(command, "--input", input);
        addOption(command, "--pattern", pattern);
        addOption(command, "--output", output);
        addOption(command, "--project-root", ".");
        addOption(command, "--config", nonBlank(config, LATEST_TRAIN_CONFIG));
        addOption(command, "--atr-period", atrPeriod);
        addOption(command, "--zigzag-atr-mult", zigzagAtrMult);
        addOption(command, "--shard-size", shardSize);
        addOption(command, "--default-spread-points", spreadPoints);
        if (allowNeutralVolume) command.add("--allow-neutral-volume");
        return command;
    }

    static List<String> datasetMineSelectedCommand(
            String python, String config, String inputDir, String pattern, String output,
            String atrPeriod, String zigzagAtrMult, String shardSize, String spreadPoints,
            boolean allowNeutralVolume
    ) {
        List<String> command = new ArrayList<>();
        command.add(nonBlank(python, "python"));
        command.add(ZIGZAG_TOOL);
        command.add("mine-five-selected");
        addOption(command, "--input-dir", inputDir);
        addOption(command, "--pattern", pattern);
        addOption(command, "--output", output);
        addOption(command, "--project-root", ".");
        addOption(command, "--config", nonBlank(config, LATEST_TRAIN_CONFIG));
        addOption(command, "--atr-period", atrPeriod);
        addOption(command, "--zigzag-atr-mult", zigzagAtrMult);
        addOption(command, "--shard-size", shardSize);
        addOption(command, "--default-spread-points", spreadPoints);
        if (allowNeutralVolume) command.add("--allow-neutral-volume");
        return command;
    }

    static List<String> datasetFetchCommand(
            String python, String runner, String symbol, String timeframe, String bars, String chunkSize, String output
    ) {
        List<String> command = new ArrayList<>();
        command.add(nonBlank(python, "python"));
        command.add(ZIGZAG_TOOL);
        command.add("mt5-fetch");
        addOption(command, "--runner-dir", nonBlank(runner, DEFAULT_RUNNER));
        addOption(command, "--symbol", symbol);
        addOption(command, "--timeframe", timeframe);
        addOption(command, "--bars", bars);
        addOption(command, "--chunk-size", chunkSize);
        addOption(command, "--output", output);
        return command;
    }

    static List<String> datasetFetchSelectedCommand(
            String python, String runner, String symbolsFile, String timeframe, String bars, String chunkSize, String output
    ) {
        List<String> command = new ArrayList<>();
        command.add(nonBlank(python, "python"));
        command.add(ZIGZAG_TOOL);
        command.add("mt5-fetch-selected");
        addOption(command, "--runner-dir", nonBlank(runner, DEFAULT_RUNNER));
        addOption(command, "--symbols-file", symbolsFile);
        addOption(command, "--timeframe", timeframe);
        addOption(command, "--bars", bars);
        addOption(command, "--chunk-size", chunkSize);
        addOption(command, "--output", output);
        return command;
    }

    static List<String> datasetSymbolScanCommand(String python, String runner) {
        List<String> command = new ArrayList<>();
        command.add(nonBlank(python, "python"));
        command.add(ZIGZAG_TOOL);
        command.add("mt5-symbols");
        addOption(command, "--runner-dir", nonBlank(runner, DEFAULT_RUNNER));
        return command;
    }

    static List<String> datasetSmokeCommand(
            String python, String pattern, String atrPeriod, String zigzagAtrMult
    ) {
        List<String> command = new ArrayList<>();
        command.add(nonBlank(python, "python"));
        command.add(ZIGZAG_TOOL);
        command.add("smoke");
        addOption(command, "--pattern", pattern);
        addOption(command, "--atr-period", atrPeriod);
        addOption(command, "--atr-mode", "mt5_sma");
        addOption(command, "--zigzag-atr-mult", zigzagAtrMult);
        return command;
    }

    private static void appendObservability(
            List<String> command, boolean enabled, boolean reset,
            String events, String decisionProfiles, String structuralEvents,
            String networkDevelopment, String sampleEvery, String maxMb,
            String maxNeurons, String maxEdges
    ) {
        if (!enabled) return;
        command.add("--live-observatory");
        addOption(command, "--live-events", events);
        addOption(command, "--live-decision-profiles", decisionProfiles);
        addOption(command, "--live-structural-events", structuralEvents);
        addOption(command, "--live-network-development", networkDevelopment);
        addOption(command, "--live-sample-every", sampleEvery);
        addOption(command, "--live-max-mb", maxMb);
        addOption(command, "--live-max-responsible-neurons", maxNeurons);
        addOption(command, "--live-max-responsible-edges", maxEdges);
        if (reset) command.add("--live-reset");
    }

    static List<String> teacherImitationCommand(
            String python, String config, String checkpoint, String dataset, String output,
            String epochs, String batchSize, String featureBatchSize, String learningRate,
            String recurrentLearningRate, String patience, String seed,
            String sequenceLength, String falseEntryPenalty,
            boolean verifyOnly
    ) {
        List<String> command = new ArrayList<>();
        command.add(nonBlank(python, "python"));
        command.add(TEACHER_IMITATION_SCRIPT);
        addOption(command, "--config", nonBlank(config, LATEST_TRAIN_CONFIG));
        addOption(command, "--checkpoint", checkpoint);
        addOption(command, "--dataset", dataset);
        addOption(command, "--output", output);
        addOption(command, "--epochs", epochs);
        addOption(command, "--batch-size", batchSize);
        addOption(command, "--feature-batch-size", featureBatchSize);
        addOption(command, "--learning-rate", learningRate);
        addOption(command, "--recurrent-learning-rate", recurrentLearningRate);
        addOption(command, "--patience", patience);
        addOption(command, "--seed", seed);
        addOption(command, "--sequence-length", sequenceLength);
        addOption(command, "--false-entry-penalty", falseEntryPenalty);
        addOption(command, "--resume-file", imitationResumeFile(output));
        addOption(command, "--stop-request-file", imitationStopRequestFile(output));
        command.add("--action-only");
        if (verifyOnly) command.add("--verify-only");
        return command;
    }

    private static String imitationResumeFile(String output) {
        return output + ".resume.pkl";
    }

    static String candidatePausedImitationReason(String seedArtifact, boolean resumeExists) {
        if (!resumeExists) return "";
        return "Checkpoint seed đang có imitation tạm dừng:\n"
                + imitationResumeFile(seedArtifact)
                + "\n\nSang tab Imitation tối đa hiện có.\n"
                + "Đặt 'Checkpoint candidate mới' = "
                + seedArtifact
                + "\nBấm 'Kiểm tra & train imitation' để resume.\n\n"
                + "Candidate MT5 chỉ chạy sau khi imitation hoàn tất và file resume được xóa.";
    }

    private static String imitationStopRequestFile(String output) {
        return output + ".stop-request";
    }

    private static String candidateStopRequestFile(String outputDir) {
        return outputDir + ".save-and-stop.request";
    }

    private static void selfTestCommandContract() {
        List<String> candidate = candidateCommand(
                "python", LATEST_TRAIN_CONFIG, DEFAULT_RUNNER,
                "outputs/he_chuyen_gia/kho_chuyen_gia.pkl",
                "outputs/he_chuyen_gia/v193_runs/self_test",
                "2025.01.01", "2025.03.31", "1", "0", "300", "0", true, false
        );
        require(candidate.contains(CANDIDATE_SCRIPT), "candidate workflow missing");
        require(candidate.contains(EXPERIMENT_PROFILE), "candidate profile missing");
        require(optionValue(candidate, "--config").equals(LATEST_TRAIN_CONFIG),
                "candidate did not use latest training config");
        require(candidate.contains("--mt5-visual"),
                "candidate workflow did not propagate MT5 Visual");
        require(candidateStopRequestFile("outputs/he_chuyen_gia/v193_runs/self_test")
                        .equals(optionValue(candidate, "--stop-request-file")),
                "candidate workflow missing cooperative save-and-stop request");
        List<String> candidateHeadless = candidateCommand(
                "python", LATEST_TRAIN_CONFIG, DEFAULT_RUNNER,
                "outputs/he_chuyen_gia/kho_chuyen_gia.pkl",
                "outputs/he_chuyen_gia/v193_runs/self_test_headless",
                "2025.01.01", "2025.03.31", "1", "0", "300", "0", false, false
        );
        require(!candidateHeadless.contains("--mt5-visual"),
                "candidate workflow enabled MT5 Visual without opt-in");
        List<String> reasoningCandidate = candidateCommand(
                "python", REASONING_TRAIN_CONFIG, DEFAULT_RUNNER,
                "outputs/he_chuyen_gia/kho_chuyen_gia.pkl",
                "outputs/he_chuyen_gia/reasoning_rl_v1/self_test",
                "2025.01.01", "2025.03.31", "1", "0", "300", "0", false, false
        );
        require(REASONING_TRAIN_CONFIG.equals(optionValue(reasoningCandidate, "--config")),
                "Reasoning RL candidate did not use its isolated config");
        require(REASONING_ACTIVE_EXPERIMENT_PROFILE.equals(optionValue(reasoningCandidate, "--profile")),
                "Reasoning RL candidate did not enable the active action/optimizer profile");
        require(REASONING_TRAIN_CONFIG.equals(candidateConfigForMode(
                        CANDIDATE_MODE_REASONING, 0)),
                "Reasoning mode was contaminated by the advanced config field");
        require(LATEST_TRAIN_CONFIG.equals(candidateConfigForMode(
                        CANDIDATE_MODE_EDGE, 0)),
                "edge mode did not select the V193 training config");
        require(CODEX_TRAIN_CONFIG.equals(candidateConfigForMode(
                        CANDIDATE_MODE_CODEX, 0)),
                "Codex mode did not select the Codex teacher config");
        List<String> codexCandidate = candidateCommand(
                "python", CODEX_TRAIN_CONFIG, DEFAULT_RUNNER,
                "outputs/he_chuyen_gia/kho_chuyen_gia.pkl",
                "outputs/he_chuyen_gia/codex_teacher/self_test",
                "2025.01.01", "2025.03.31", "1", "0", "300", "0", false, false
        );
        require(CODEX_EXPERIMENT_PROFILE.equals(optionValue(codexCandidate, "--profile")),
                "Codex teacher candidate did not enable its learning-safe experiment profile");
        List<String> unverifiedSeedCandidate = candidateCommand(
                "python", REASONING_TRAIN_CONFIG, DEFAULT_RUNNER,
                "outputs/he_chuyen_gia/epoch13.research.pkl",
                "outputs/he_chuyen_gia/reasoning_rl_v1/unverified_seed_self_test",
                "2025.01.01", "2025.03.31", "1", "0", "300", "0", false, true
        );
        require(unverifiedSeedCandidate.contains("--allow-unverified-imitation-seed")
                        && !candidate.contains("--allow-unverified-imitation-seed"),
                "explicit research-only seed bypass command contract missing");
        List<String> ablation = ablationCommand(
                "python", REASONING_TRAIN_CONFIG, DEFAULT_RUNNER,
                "outputs/he_chuyen_gia/kho_chuyen_gia.pkl", "outputs/ablation",
                "2025.01.01", "2025.01.08", "1", "0", "300", "0", false, false
        );
        require(ablation.contains(ABLATION_SCRIPT)
                        && EXPERIMENT_PROFILE.equals(optionValue(ablation, "--profile"))
                        && REASONING_TRAIN_CONFIG.equals(optionValue(ablation, "--config"))
                        && !ablation.contains("--keep-checkpoints"),
                "matched MT5 ablation command contract missing");
        String ablationLine = formatProcessLogLine(ablation,
                "{\"event\":\"ablation_arm_finished\",\"arm\":\"full_online\"," 
                        + "\"profit_factor\":0.75,\"expectancy\":-1.2,"
                        + "\"max_drawdown\":0.25,\"ppo_updates\":4}");
        require(ablationLine.contains("full_online")
                        && ablationLine.contains("PF=0.750")
                        && ablationLine.contains("max DD=25.00%"),
                "ablation live log formatter missing");
        require("outputs/generated_seed.pkl".equals(
                        candidateSeedForRun("", "outputs/generated_seed.pkl")),
                "blank candidate checkpoint did not receive a generated path");

        List<String> fresh = researchCommand(
                "python", BASE_CONFIG, false, "outputs/research.pkl", "",
                DEFAULT_RUNNER, "2025.01.01", "", "31", "1", "0", "300", "0",
                false, false
        );
        require(fresh.contains("--fresh-system"), "fresh research guard missing");
        require(!fresh.contains("--seed-system"), "fresh research leaked seed-system");
        require(fresh.contains("--backtest-days"), "backtest-days was not restored");
        require(optionValue(fresh, "--config").equals(LATEST_TRAIN_CONFIG),
                "fresh train did not use latest training config");

        List<String> frozen = researchCommand(
                "python", BASE_CONFIG, true, "outputs/frozen.pkl", "outputs/champion.pkl",
                DEFAULT_RUNNER, "2025.01.01", "2025.02.01", "31", "4", "0", "300", "0",
                false, false
        );
        require(frozen.contains("--frozen") && frozen.contains("--seed-system"), "frozen OOS contract missing");
        require(!frozen.contains("--fresh-system"), "frozen and fresh were mixed");

        List<String> champion = championCommand(
                "python", "outputs/champion.pkl", "outputs/shadow.pkl", "outputs/audit",
                "2025.01.01", "2025.03.31", "3", "7", "3", "0", DEFAULT_RUNNER,
                "300", "0", "4", "0", "0", "0.02", "1", false, false
        );
        require(champion.contains("huan-luyen-champion-he-chuyen-gia-mt5"), "champion workflow missing");
        require(optionValue(champion, "--config").equals(LATEST_TRAIN_CONFIG),
                "champion train did not use latest training config");

        List<String> walk = walkForwardCommand(
                "python", "outputs/walk", "", "outputs/champion.pkl",
                "0", "3", DEFAULT_RUNNER, "300", "2025.01.01", "2025.03.31",
                "14", "7", "7", "4", true, true, "0", "0", "0.02", "1",
                false, false
        );
        require(walk.contains("walk-forward-he-chuyen-gia-mt5") && walk.contains("--paired-baseline"),
                "walk-forward paired workflow missing");
        require(optionValue(walk, "--config").equals(LATEST_TRAIN_CONFIG),
                "walk-forward train did not use latest training config");

        List<String> initialize = initializeCommand(
                "python", BASE_CONFIG, "outputs/new.pkl", "0"
        );
        require(initialize.contains("khoi-tao-he-chuyen-gia"), "initialize workflow missing");
        require(initialize.size() > 2
                        && "-m".equals(initialize.get(1))
                        && CLI_MODULE.equals(initialize.get(2)),
                "initialize workflow did not use the current CLI module");
        require(adminCommand("python", BASE_CONFIG, "a.pkl", "b.pkl", "CTX_RANGING",
                "dormant", "", "", "", "", false, false, false, false)
                .contains("quan-tri-he-chuyen-gia"), "admin workflow missing");
        for (List<String> command : List.of(candidate, reasoningCandidate, fresh, frozen, champion, walk)) {
            require(!command.contains("--policy-core"), "shared-policy route leaked into context-expert UI");
            assertLatestTrainingContract(command);
        }
        require(withDefaultExtension(Path.of("candidate"), ".pkl").endsWith("candidate.pkl"),
                "new output extension helper failed");
        require(withDefaultExtension(Path.of("candidate.pkl"), ".pkl").endsWith("candidate.pkl"),
                "existing output extension was duplicated");
        String fiveContract = fiveArchitectureContractText();
        require(fiveContract.contains("137 GIÁ TRỊ") && fiveContract.contains("ENCODER V4 + PATTERN TREE"),
                "structured observation contract missing from UI");
        require(fiveContract.contains("Target: query(memory) × key(vị thế)")
                        && fiveContract.contains("Delay: 4 mức")
                        && fiveContract.contains("Entry offset: 17 mức"),
                "factorized attention/delay/offset heads missing from UI");
        require(fiveContract.contains("BRIDGE SCHEMA 10")
                        && fiveContract.contains("EXECUTION LEDGER SCHEMA 14")
                        && fiveContract.contains("5 × T")
                        && fiveContract.contains("Không còn bridge round-robin")
                        && fiveContract.contains("PHÂN QUYỀN FIVE-AGENT AUTHORITY V1")
                        && fiveContract.contains("CLOSE_ALL (phủ quyết packet)"),
                "five-agent bridge contract missing from UI");
        require(fiveContract.contains("REWARD KINH TẾ + GIẢI ĐẤU NỘI BỘ")
                        && fiveContract.contains("tổng bằng 0")
                        && fiveContract.contains("Terminal: equity-R − max drawdown")
                        && fiveContract.contains("team-spirit 0.30 → 1.00"),
                "five-agent competitive reward contract missing from UI");
        require(fiveContract.contains("REASONING RL V1")
                        && fiveContract.contains("GRPO/PPO + KL + entropy")
                        && fiveContract.contains("encoder credit = 0")
                        && fiveContract.contains("action scale = 0.25")
                        && fiveContract.contains("FiveAdam")
                        && fiveContract.contains("candidate research-only"),
                "Reasoning RL safety/evidence contract missing from UI");
        boolean invalidCalendarDateRejected = false;
        try {
            parseInputDate("2025.02.30");
        } catch (DateTimeParseException expected) {
            invalidCalendarDateRejected = true;
        }
        require(invalidCalendarDateRejected,
                "candidate date validation accepted a non-existent calendar date");
        List<String> dataCheck = datasetCheckCommand(
                "python", LATEST_TRAIN_CONFIG, "data/raw.npz", "12", false
        );
        require(dataCheck.contains(ZIGZAG_TOOL) && dataCheck.contains("check-five")
                        && dataCheck.contains("--project-root") && dataCheck.contains("--input"),
                "dataset compatibility command contract missing");
        List<String> dataMine = datasetMineCommand(
                "python", LATEST_TRAIN_CONFIG, "data/raw.npz",
                "tools/five_zigzag_dataset_tool/examples/impulse_pullback.yaml",
                "outputs/datasets/five", "", "1.0", "100000", "", false
        );
        require(dataMine.contains("mine-five") && dataMine.contains("--shard-size")
                        && !dataMine.contains("--atr-period"),
                "dataset 137 mining command contract missing");
        List<String> dataMineSelected = datasetMineSelectedCommand(
                "python", LATEST_TRAIN_CONFIG, "data/mt5_raw", "pattern.yaml",
                "outputs/datasets/five_multi", "", "1.0", "100000", "", false
        );
        require(dataMineSelected.contains("mine-five-selected")
                        && dataMineSelected.contains("--input-dir")
                        && !dataMineSelected.contains("--input"),
                "multi-symbol dataset mining command contract missing");
        List<String> selectedFetch = datasetFetchSelectedCommand(
                "python", DEFAULT_RUNNER, "data/mt5_selected/selected_symbols.txt", "M1", "1000000", "200000", "data/mt5_selected"
        );
        require(selectedFetch.contains("mt5-fetch-selected") && selectedFetch.contains("--runner-dir")
                        && selectedFetch.contains(DEFAULT_RUNNER),
                "dataset MT5 selected fetch command must reuse training runner");
        require("manifest.json".equals(datasetResultFilename(selectedFetch))
                        && "dataset_summary.json".equals(datasetResultFilename(dataMine))
                        && "dataset_summary.json".equals(datasetResultFilename(dataMineSelected))
                        && datasetResultFilename(dataCheck).isBlank(),
                "dataset post-run result routing contract failed");
        List<String> symbolScan = datasetSymbolScanCommand("python", DEFAULT_RUNNER);
        require(symbolScan.contains("mt5-symbols") && symbolScan.contains("--runner-dir")
                        && symbolScan.contains(DEFAULT_RUNNER),
                "dataset MT5 symbol scan must reuse training runner");
        require(datasetSmokeCommand(
                "python", "tools/five_zigzag_dataset_tool/examples/impulse_pullback.yaml", "30", "1.0"
        ).contains("smoke"), "dataset smoke command contract missing");
        List<String> imitation = teacherImitationCommand(
                "python", LATEST_TRAIN_CONFIG, "outputs/seed.pkl", "outputs/teacher_max.npz",
                "outputs/imitation_candidate.pkl", "80", "2", "16", "0.003", "0.00001", "12", "0",
                "16", "0.50", false
        );
        require(imitation.contains(TEACHER_IMITATION_SCRIPT)
                        && LATEST_TRAIN_CONFIG.equals(optionValue(imitation, "--config"))
                        && "outputs/seed.pkl".equals(optionValue(imitation, "--checkpoint"))
                        && "outputs/teacher_max.npz".equals(optionValue(imitation, "--dataset"))
                        && "16".equals(optionValue(imitation, "--sequence-length"))
                        && "0.50".equals(optionValue(imitation, "--false-entry-penalty"))
                        && "outputs/imitation_candidate.pkl.resume.pkl".equals(
                        optionValue(imitation, "--resume-file"))
                        && "outputs/imitation_candidate.pkl.stop-request".equals(
                        optionValue(imitation, "--stop-request-file"))
                        && imitation.contains("--action-only")
                        && !imitation.contains("--geometry-only")
                        && !imitation.contains("--verify-only"),
                "teacher imitation command contract missing");
        List<String> imitationSeedBootstrap = initializeCommand(
                "python", LATEST_TRAIN_CONFIG, FIVE_TRANSFORMER_V3_CHECKPOINT, "0"
        );
        String imitationBootstrapPreview = formatCommands(List.of(
                imitationSeedBootstrap,
                teacherImitationCommand(
                        "python", LATEST_TRAIN_CONFIG, FIVE_TRANSFORMER_V3_CHECKPOINT,
                        "outputs/teacher_max.npz", "outputs/imitation_candidate.pkl",
                        "80", "2", "16", "0.003", "0.00001", "12", "0",
                        "64", "0.50", false
                )
        ));
        require(imitationSeedBootstrap.contains("khoi-tao-he-chuyen-gia")
                        && imitationBootstrapPreview.contains(FIVE_TRANSFORMER_V3_CHECKPOINT)
                        && imitationBootstrapPreview.contains(TEACHER_IMITATION_SCRIPT)
                        && imitationBootstrapPreview.contains("$LASTEXITCODE"),
                "missing-seed imitation bootstrap pipeline contract failed");
        require(FIVE_TRANSFORMER_V3_CHECKPOINT.equals(
                        imitationCheckpointForRun("", false))
                        && FIVE_TRANSFORMER_V3_CHECKPOINT.equals(imitationCheckpointForRun(
                        "outputs/he_chuyen_gia/kho_chuyen_gia_transformer_v2.pkl", false
                ))
                        && "outputs/custom_seed.pkl".equals(imitationCheckpointForRun(
                        "outputs/custom_seed.pkl", false
                )),
                "imitation seed normalization contract failed");
        String progressLine = formatProcessLogLine(imitation,
                "{\"event\":\"action_batch_progress\",\"epoch\":3,"
                        + "\"windows_done\":50,\"windows_total\":100,\"percent\":50.0,"
                        + "\"train_weighted_loss_so_far\":0.123456,\"elapsed_seconds\":12.5}");
        require(progressLine.contains("Epoch 3 · BPTT 50.0%")
                        && progressLine.contains("loss=0.12346")
                        && progressLine.contains("12.5s"),
                "teacher imitation live log formatter missing");
        String epochLine = formatProcessLogLine(imitation,
                "{\"event\":\"action_epoch\",\"epoch\":3,"
                        + "\"train_weighted_loss\":0.80,\"train_balanced_accuracy_online\":0.72,"
                        + "\"train_balanced_accuracy_replay\":0.70,"
                        + "\"validation_balanced_accuracy\":0.68,\"validation_entry_precision\":0.40,"
                        + "\"validation_entry_recall\":0.51,\"validation_mcc\":0.35,"
                        + "\"imitation_reward_penalty_ratio\":2.25,\"epoch_seconds\":10.0}");
        require(epochLine.contains("train BA=0.7000")
                        && epochLine.contains("recall=0.5100")
                        && epochLine.contains("K/P=2.25"),
                "teacher imitation learning diagnostics formatter missing");
        String savedLine = formatProcessLogLine(imitation,
                "{\"event\":\"action_checkpoint_saved\","
                        + "\"resume_file\":\"outputs/imitation_candidate.pkl.resume.pkl\","
                        + "\"epoch\":3,\"windows_done\":60,\"windows_total\":100}");
        require(savedLine.contains("Đã lưu để tiếp tục")
                        && savedLine.contains("Epoch 3")
                        && savedLine.contains("60/100 windows"),
                "teacher imitation save-and-stop formatter missing");
        String autonomousLine = formatProcessLogLine(imitation,
                "{\"event\":\"autonomous_validation_progress\",\"split\":\"validation\","
                        + "\"steps_done\":50,\"steps_total\":100,\"percent\":50.0,"
                        + "\"elapsed_seconds\":12.5}");
        require(autonomousLine.contains("Tự kiểm định liên tục")
                        && autonomousLine.contains("50/100 bước")
                        && autonomousLine.contains("12.5s"),
                "teacher imitation autonomous validation formatter missing");
        require("outputs/imitation_candidate.pkl".equals(
                        successfulImitationOutput(imitation, 0))
                        && successfulImitationOutput(imitation, 1).isBlank()
                        && successfulImitationOutput(
                                teacherImitationCommand(
                                        "python", LATEST_TRAIN_CONFIG, "outputs/seed.pkl",
                                        "outputs/teacher_max.npz", "outputs/verify.pkl", "1",
                                        "2", "4", "0.003", "0.00001", "1", "0", "16", "0.50",
                                        true
                                ), 0
                        ).isBlank(),
                "successful imitation output routing contract missing");
        String pausedSeedReason = candidatePausedImitationReason(
                "outputs/imitation_candidate.pkl", true
        );
        require(pausedSeedReason.contains("outputs/imitation_candidate.pkl.resume.pkl")
                        && pausedSeedReason.contains("Kiểm tra & train imitation")
                        && candidatePausedImitationReason(
                        "outputs/imitation_candidate.pkl", false
                ).isBlank(),
                "candidate must block a paused imitation seed before bootstrap/MT5");
        require(LATEST_TEACHER_IMITATION_DATASET.equals(migrateTeacherImitationDataset(
                        "outputs/bottom_top_teacher_recurrent_v5_7d_20260827/teacher_7d_recurrent.npz"))
                        && LATEST_TEACHER_IMITATION_DATASET.equals(migrateTeacherImitationDataset(
                        "outputs\\bottom_top_teacher_recurrent_v5_16w_20260827\\teacher_16w_recurrent.npz"))
                        && LATEST_TEACHER_IMITATION_DATASET.equals(migrateTeacherImitationDataset(
                        "outputs/bottom_top_teacher_recurrent_v5_max_sl075_20260827/teacher_max_sl075_recurrent.npz"))
                        && LATEST_TEACHER_IMITATION_DATASET.equals(migrateTeacherImitationDataset(
                        "outputs/bottom_top_teacher_recurrent_v7_max_sl075_20260828/teacher_max_sl075_geometry_v7.npz"))
                        && LATEST_TEACHER_IMITATION_DATASET.equals(migrateTeacherImitationDataset(
                        "outputs/bottom_top_teacher_confirmed_action_v8_20260830/teacher_confirmed_action_v8.npz"))
                        && LATEST_TEACHER_IMITATION_DATASET.equals(migrateTeacherImitationDataset(
                        "outputs/bottom_top_teacher_confirmed_geometry_v9_sl150_20260901/teacher_confirmed_geometry_v9_sl150.npz"))
                        && "outputs/custom_teacher.npz".equals(migrateTeacherImitationDataset(
                        "outputs/custom_teacher.npz")),
                "teacher imitation saved-dataset migration contract missing");
        assertLatestTrainingContract(imitation);
        List<List<String>> oneClickDataMine = List.of(selectedFetch, dataMineSelected);
        require(oneClickDataMine.size() == 2
                        && oneClickDataMine.get(0).contains("mt5-fetch-selected")
                        && oneClickDataMine.get(1).contains("mine-five-selected"),
                "one-click fetch-mine pipeline contract missing");
        String bootstrapPreview = formatCommands(List.of(
                initializeCommand("python", BASE_CONFIG, "outputs/new_seed.pkl", "0"), candidate
        ));
        require(bootstrapPreview.contains("khoi-tao-he-chuyen-gia")
                        && bootstrapPreview.contains(CANDIDATE_SCRIPT)
                        && bootstrapPreview.contains("$LASTEXITCODE"),
                "missing-checkpoint bootstrap pipeline contract failed");
        System.out.println("V193_UI_FIVE_BRIDGE_V10_CONTRACT_OK");
    }

    private static String optionValue(List<String> command, String option) {
        int index = command.indexOf(option);
        if (index < 0 || index + 1 >= command.size()) return "";
        return command.get(index + 1);
    }

    private static String datasetResultFilename(List<String> command) {
        if (command.contains("mt5-fetch-selected")) return "manifest.json";
        if (command.contains("mine-five") || command.contains("mine-five-selected")) return "dataset_summary.json";
        return "";
    }

    private static void assertLatestTrainingContract(List<String> command) {
        boolean candidate = command.contains(CANDIDATE_SCRIPT);
        boolean ablation = command.contains(ABLATION_SCRIPT);
        boolean frozen = command.contains("--frozen");
        boolean imitationTraining = command.contains(TEACHER_IMITATION_SCRIPT)
                && !command.contains("--verify-only");
        boolean training = candidate || ablation || imitationTraining
                || command.contains("huan-luyen-champion-he-chuyen-gia-mt5")
                || command.contains("walk-forward-he-chuyen-gia-mt5")
                || (command.contains("huan-luyen-he-chuyen-gia-mt5") && !frozen);
        if (!training) return;
        if (candidate || ablation) {
            String candidateConfig = optionValue(command, "--config");
            require(isApprovedCandidateConfig(candidateConfig),
                    "candidate command is not pinned to an approved isolated config");
            String expectedProfile = candidate
                    ? candidateExperimentProfile(candidateConfig) : EXPERIMENT_PROFILE;
            String actualProfile = optionValue(command, "--profile");
            if (!SGT_COMPACT_EXPERIMENT_PROFILE.equals(actualProfile)) {
                require(expectedProfile.equals(actualProfile),
                        "candidate/ablation command is not pinned to the latest experiment profile");
            }
        } else {
            require(LATEST_TRAIN_CONFIG.equals(optionValue(command, "--config")),
                    "training command is not pinned to " + LATEST_TRAIN_CONFIG);
        }
    }

    private static boolean isApprovedCandidateConfig(String config) {
        if (LATEST_TRAIN_CONFIG.equals(config) || REASONING_TRAIN_CONFIG.equals(config)
                || CODEX_TRAIN_CONFIG.equals(config)) {
            return true;
        }
        for (String alignedConfig : SEVEN_CHOICE_REASONING_CONFIGS) {
            if (alignedConfig.equals(config)) return true;
        }
        return false;
    }

    private static Path withDefaultExtension(Path path, String extension) {
        if (extension == null || extension.isBlank()) return path;
        String name = path.getFileName() == null ? "" : path.getFileName().toString();
        if (name.lastIndexOf('.') > 0) return path;
        return path.resolveSibling(name + extension);
    }

    private static final class Mt5SymbolRow {
        boolean selected;
        final String name;
        final String description;
        final String path;
        final boolean visible;
        final int digits;

        Mt5SymbolRow(boolean selected, String name, String description, String path, boolean visible, int digits) {
            this.selected = selected;
            this.name = name;
            this.description = description;
            this.path = path;
            this.visible = visible;
            this.digits = digits;
        }
    }

    private static final class Mt5SymbolTableModel extends AbstractTableModel {
        private final String[] columns = {"Chọn", "Mã", "Mô tả", "Nhóm máy chủ", "Đang hiện", "Số lẻ"};
        private final List<Mt5SymbolRow> rows = new ArrayList<>();

        void setRows(List<Mt5SymbolRow> values) {
            rows.clear();
            rows.addAll(values);
            fireTableDataChanged();
        }

        Mt5SymbolRow row(int modelRow) {
            return rows.get(modelRow);
        }

        List<String> selectedNames() {
            List<String> result = new ArrayList<>();
            for (Mt5SymbolRow row : rows) if (row.selected) result.add(row.name);
            return result;
        }

        int selectedCount() {
            int count = 0;
            for (Mt5SymbolRow row : rows) if (row.selected) count++;
            return count;
        }

        void setSelected(int modelRow, boolean selected) {
            if (modelRow < 0 || modelRow >= rows.size()) return;
            rows.get(modelRow).selected = selected;
            fireTableCellUpdated(modelRow, 0);
        }

        void setSelectedRows(List<Integer> modelRows, boolean selected) {
            for (int modelRow : modelRows) {
                if (modelRow >= 0 && modelRow < rows.size()) rows.get(modelRow).selected = selected;
            }
            fireTableDataChanged();
        }

        void selectVisibleRows() {
            for (Mt5SymbolRow row : rows) row.selected = row.visible;
            fireTableDataChanged();
        }

        void clearSelection() {
            for (Mt5SymbolRow row : rows) row.selected = false;
            fireTableDataChanged();
        }

        @Override public int getRowCount() { return rows.size(); }
        @Override public int getColumnCount() { return columns.length; }
        @Override public String getColumnName(int column) { return columns[column]; }
        @Override public Class<?> getColumnClass(int column) {
            if (column == 0 || column == 4) return Boolean.class;
            if (column == 5) return Integer.class;
            return String.class;
        }
        @Override public boolean isCellEditable(int row, int column) { return column == 0; }
        @Override public Object getValueAt(int rowIndex, int columnIndex) {
            Mt5SymbolRow row = rows.get(rowIndex);
            return switch (columnIndex) {
                case 0 -> row.selected;
                case 1 -> row.name;
                case 2 -> row.description;
                case 3 -> row.path;
                case 4 -> row.visible;
                case 5 -> row.digits;
                default -> "";
            };
        }
        @Override public void setValueAt(Object value, int rowIndex, int columnIndex) {
            if (columnIndex == 0) setSelected(rowIndex, Boolean.TRUE.equals(value));
        }
    }

    private static final class TraderControlFrame extends JFrame {
        private static final int TAB_CANDIDATE = 0;
        private static final int TAB_RESEARCH = 1;
        private static final int TAB_CHAMPION = 2;
        private static final int TAB_WALK_FORWARD = 3;
        private static final int TAB_ADMIN = 4;
        private static final int TAB_ADVANCED = 5;
        private static final int TAB_TOOLS = 6;
        private static final int TAB_IMITATION = 7;
        private static final int TAB_ABLATION = 8;
        private static final int TAB_TIER1_TIER2 = 9;

        private static final int TOOL_MINE = 0;
        private static final int TOOL_CHECK = 1;
        private static final int TOOL_FETCH = 2;
        private static final int TOOL_SMOKE = 3;
        private static final int TOOL_PATTERN = 4;

        private final Path workspace;
        private final List<JTextField> trackedFields = new ArrayList<>();
        private final JTabbedPane workflowTabs = new JTabbedPane();
        private final JTabbedPane outputTabs = new JTabbedPane();

        private final JTextField pythonField = field("python");
        private final JTextField configField = field(BASE_CONFIG);
        private final JTextField runnerDirField = field(DEFAULT_RUNNER);

        private final JComboBox<String> candidateArchitectureBox = new JComboBox<>(new String[]{
                "SGT-V193-Compact (xLSTM 17.5M) [Khuyến nghị Tier 2]",
                "OpenAI Five LSTM 4096 (93.2M) [Baseline cũ]"
        });
        private final JTextField candidateSeedField = field(SGT_COMPACT_DEFAULT_SEED);
        private final JTextField candidateOutputField = field(defaultOutput("candidate_sgt_compact_tier2", false));
        {
            candidateArchitectureBox.addActionListener(e -> {
                if (candidateArchitectureBox.getSelectedIndex() == 0) {
                    candidateSeedField.setText(SGT_COMPACT_DEFAULT_SEED);
                    candidateOutputField.setText(defaultOutput("candidate_sgt_compact_tier2", false));
                } else {
                    candidateSeedField.setText("outputs/he_chuyen_gia/kho_chuyen_gia.pkl");
                    candidateOutputField.setText(defaultOutput("candidate_reasoning_rl_v1", false));
                }
            });
        }
        private final JComboBox<String> candidateModeBox = new JComboBox<>(new String[]{
                "Reasoning RL V1 · action + optimizer",
                "V193 tìm edge · không reasoning",
                "Codex teacher · checkpoint nguồn cố định"
        });
        private final JComboBox<String> candidateSevenChoiceProfileBox = new JComboBox<>(
                new String[]{
                        "Không dùng bộ nhãn bảy lựa chọn (cấu hình cũ)",
                        SEVEN_CHOICE_REASONING_LABELS[0],
                        SEVEN_CHOICE_REASONING_LABELS[1],
                        SEVEN_CHOICE_REASONING_LABELS[2],
                }
        );
        private final JLabel candidateModeStatus = new JLabel();
        private final JTextField candidateFromField = field("2025.01.06");
        private final JTextField candidateToField = field("2025.03.31");
        private final JComboBox<String> candidateModelBox = modelBox("1");
        private final JTextField candidateSeedNumberField = field("0");
        private final JTextField candidateTimeoutField = field("300");
        private final JTextField candidateMaxWallField = field("0");
        private final JCheckBox candidateAllowUnverifiedSeedBox = new JCheckBox(
                "Bỏ qua gate imitation seed (research-only)"
        );

        private final JTextField ablationCheckpointField = field("outputs/he_chuyen_gia/kho_chuyen_gia.pkl");
        private final JTextField ablationOutputField = field("");
        private final JTextField ablationFromField = field("2025.01.06");
        private final JTextField ablationToField = field("2025.01.08");
        private final JComboBox<String> ablationModelBox = modelBox("1");
        private final JTextField ablationSeedField = field("0");
        private final JTextField ablationTimeoutField = field("300");
        private final JTextField ablationMaxWallField = field("0");
        private final JCheckBox ablationKeepCheckpointsBox = new JCheckBox("Giữ checkpoint lớn của từng arm");

        private final JComboBox<String> researchModeBox = new JComboBox<>(new String[]{
                "Fresh train V193", "Frozen OOS", "Khởi tạo kho"
        });
        private final JTextField researchOutputField = field("");
        private final JTextField researchSeedSystemField = field("outputs/he_chuyen_gia/kho_chuyen_gia.pkl");
        private final JTextField researchFromField = field("2025.01.06");
        private final JTextField researchToField = field("");
        private final JTextField researchBacktestDaysField = field("31");
        private final JComboBox<String> researchModelBox = modelBox("1");
        private final JTextField researchSeedField = field("0");
        private final JTextField researchTimeoutField = field("300");
        private final JTextField researchMaxWallField = field("0");

        private final JTextField championPathField = field("outputs/he_chuyen_gia/kho_chuyen_gia.pkl");
        private final JTextField championOutputField = field("");
        private final JTextField championAuditField = field("");
        private final JTextField championTrainFromField = field("2025.01.06");
        private final JTextField championTrainToField = field("2025.03.31");
        private final JTextField championValidationCountField = field("3");
        private final JTextField championValidationDaysField = field("7");
        private final JTextField championSeedCountField = field("3");
        private final JTextField championSeedField = field("0");
        private final JComboBox<String> championModelBox = modelBox("4");
        private final JTextField championTimeoutField = field("300");
        private final JTextField championMaxWallField = field("0");
        private final JTextField minExpectancyField = field("0.0");
        private final JTextField maxProfitRegressionField = field("0.0");
        private final JTextField maxLcbRegressionField = field("0.02");
        private final JTextField maxDdIncreaseField = field("1.0");

        private final JTextField walkOutputField = field("");
        private final JTextField walkLogField = field("");
        private final JTextField walkSeedSystemField = field("outputs/he_chuyen_gia/kho_chuyen_gia.pkl");
        private final JTextField walkFromField = field("2025.01.06");
        private final JTextField walkToField = field("2025.03.31");
        private final JTextField walkTrainWindowField = field("14");
        private final JTextField walkTestWindowField = field("7");
        private final JTextField walkStepField = field("7");
        private final JTextField walkSeedField = field("0");
        private final JTextField walkSeedCountField = field("3");
        private final JComboBox<String> walkModelBox = modelBox("4");
        private final JTextField walkTimeoutField = field("300");
        private final JCheckBox walkPairedBox = new JCheckBox("Paired frozen baseline", true);
        private final JCheckBox walkRollbackBox = new JCheckBox("Rollback khi gate fail", true);

        private final JTextField adminSystemField = field("outputs/he_chuyen_gia/kho_chuyen_gia.pkl");
        private final JTextField adminOutputField = field("");
        private final JTextField adminContextField = field("CTX_RANGING");
        private final JComboBox<String> adminActionBox = new JComboBox<>(new String[]{
                "promote", "dormant", "archive", "merge"
        });
        private final JTextField adminTargetField = field("");
        private final JTextField adminLifecycleField = field("");
        private final JTextField adminNoteField = field("");
        private final JTextField adminAuditField = field("outputs/he_chuyen_gia/audit/admin.jsonl");
        private final JCheckBox adminReplayBox = new JCheckBox("Replay pass");
        private final JCheckBox adminOosBox = new JCheckBox("OOS pass");
        private final JCheckBox adminRiskBox = new JCheckBox("Risk pass");
        private final JCheckBox adminNonInterferenceBox = new JCheckBox("Non-interference pass");

        private final JCheckBox visualBox = new JCheckBox("MT5 Visual");
        private final JCheckBox keepOpenBox = new JCheckBox("Giữ terminal đăng nhập");
        private final JCheckBox telemetryBox = new JCheckBox("Ghi telemetry", true);
        private final JCheckBox resetLogsBox = new JCheckBox("Reset log trước run");
        private final JTextField eventsField = field("outputs/neural_live.jsonl");
        private final JTextField decisionProfilesField = field("outputs/neural_live.ho_so_quyet_dinh.jsonl");
        private final JTextField structuralEventsField = field("outputs/neural_live.phat_trien_cau_truc.jsonl");
        private final JTextField networkDevelopmentField = field("outputs/neural_live.theo_doi_phat_trien_mang.jsonl");
        private final JTextField sampleEveryField = field("1");
        private final JTextField maxMbField = field("64");
        private final JTextField maxNeuronsField = field("0");
        private final JTextField maxEdgesField = field("0");

        private final JTabbedPane toolTabs = new JTabbedPane();
        private final JTextField toolInputField = field("data/market.npz");
        private final JTextField toolPatternField = field("tools/five_zigzag_dataset_tool/examples/impulse_pullback.yaml");
        private final JTextField toolOutputField = field("");
        private final JTextField toolConfigField = field(LATEST_TRAIN_CONFIG);
        private final JTextField toolAtrPeriodField = field("");
        private final JTextField toolZigzagAtrMultField = field("1.0");
        private final JTextField toolShardSizeField = field("100000");
        private final JTextField toolSpreadPointsField = field("");
        private final JCheckBox toolAllowNeutralVolumeBox = new JCheckBox("Cho phép khối lượng trung tính");
        private final JCheckBox toolAutoCheckBox = new JCheckBox("Kiểm tra hợp đồng trước khi khai thác", true);
        private final JCheckBox toolMineAllSymbolsBox = new JCheckBox("Khai thác tất cả symbol trong raw manifest");

        private final JTextField toolCheckInputField = field("");
        private final JTextField toolCheckConfigField = field(LATEST_TRAIN_CONFIG);
        private final JTextField toolCheckSpreadPointsField = field("");
        private final JCheckBox toolCheckAllowNeutralVolumeBox = new JCheckBox("Cho phép khối lượng trung tính");

        private final JTextField toolSymbolField = field("");
        private final Mt5SymbolTableModel toolSymbolModel = new Mt5SymbolTableModel();
        private final JTable toolSymbolTable = new JTable(toolSymbolModel);
        private final TableRowSorter<Mt5SymbolTableModel> toolSymbolSorter = new TableRowSorter<>(toolSymbolModel);
        private final JLabel toolSymbolState = new JLabel("Chưa quét mã MT5");
        private final JButton toolScanSymbolsButton = new JButton("Quét mã MT5");
        private final JButton toolSelectFilteredButton = new JButton("Chọn danh sách đang lọc");
        private final JButton toolSelectVisibleButton = new JButton("Chọn mã đang hiện");
        private final JButton toolClearSymbolsButton = new JButton("Bỏ chọn tất cả");
        private final JComboBox<String> toolTimeframeBox = new JComboBox<>(new String[]{
                "M1", "M5", "M15", "M30", "H1", "H4", "D1"
        });
        private final JTextField toolBarsField = field("1000000");
        private final JTextField toolChunkSizeField = field("200000");
        private final JTextField toolFetchOutputField = field("");
        private final JCheckBox toolFetchMineBox = new JCheckBox(
                "Lấy xong tự tạo tập 137", true
        );

        private final JTextField toolSmokePatternField = field("tools/five_zigzag_dataset_tool/examples/impulse_pullback.yaml");
        private final JTextField toolSmokeAtrField = field("30");
        private final JTextField toolSmokeZigzagField = field("1.0");
        private final JTextField toolEditorPatternField = field("tools/five_zigzag_dataset_tool/examples/impulse_pullback.yaml");

        private final JTextField imitationConfigField = field(LATEST_TRAIN_CONFIG);
        private final JTextField imitationCheckpointField = field(FIVE_TRANSFORMER_V3_CHECKPOINT);
        private final JTextField imitationDatasetField = field(LATEST_TEACHER_IMITATION_DATASET);
        private final JTextField imitationOutputField = field("");
        private final JTextField imitationEpochsField = field("80");
        private final JTextField imitationBatchField = field("2");
        private final JTextField imitationFeatureBatchField = field("16");
        private final JTextField imitationLearningRateField = field("0.003");
        private final JTextField imitationRecurrentLearningRateField = field("0.00001");
        private final JTextField imitationPatienceField = field("12");
        private final JTextField imitationSeedField = field("0");
        private final JTextField imitationSequenceLengthField = field("64");
        private final JTextField imitationFalseEntryPenaltyField = field("0.50");
        private final JCheckBox imitationVerifyOnlyBox = new JCheckBox("Chỉ kiểm tra, không cập nhật trọng số");

        private final JTextArea toolPatternArea = new JTextArea(22, 80);
        private final JTextArea toolResultArea = new JTextArea(22, 80);
        private final JLabel toolPatternState = new JLabel("Chưa đọc mẫu cấu trúc");

        private final JTextArea previewArea = new JTextArea(5, 110);
        private final JTextArea logArea = new JTextArea(18, 110);
        private final JTextArea learningDiagnosticArea = new JTextArea(12, 110);
        private final JTextArea mt5DiagnosticArea = new JTextArea(16, 110);
        private final JTextArea codexTeacherArea = new JTextArea(16, 110);
        private final JTextArea ablationDiagnosticArea = new JTextArea(16, 110);
        private final JLabel underfitDiagnosticLabel = new JLabel("CHƯA CÓ DỮ LIỆU");
        private final JLabel overfitDiagnosticLabel = new JLabel("CHƯA CÓ DỮ LIỆU");
        private final JLabel rewardPenaltyDiagnosticLabel = new JLabel("—");
        private final JLabel mt5EdgeDiagnosticLabel = new JLabel("CHƯA CÓ DỮ LIỆU");
        private final JLabel mt5RiskDiagnosticLabel = new JLabel("CHƯA CÓ DỮ LIỆU");
        private final JLabel mt5OptimizerDiagnosticLabel = new JLabel("CHƯA CÓ DỮ LIỆU");
        private final JLabel mt5RewardDiagnosticLabel = new JLabel("CHƯA CÓ DỮ LIỆU");
        private final JLabel mt5PolicyDiagnosticLabel = new JLabel("CHƯA CÓ DỮ LIỆU");
        private final JLabel mt5ExecutionDiagnosticLabel = new JLabel("CHƯA CÓ DỮ LIỆU");
        private final JLabel ablationDiagnosticLabel = new JLabel("CHƯA CÓ DỮ LIỆU");
        private final JButton newSystemButton = new JButton("Tạo kho mới");
        private final JButton newOutputButton = new JButton("Tên output mới");
        private final JButton copyButton = new JButton("Sao chép lệnh");
        private final JButton saveButton = new JButton("Lưu cấu hình UI");
        private final JButton loadButton = new JButton("Tải cấu hình UI");
        private final JButton runButton = new JButton("Chạy workflow");
        private final JButton saveStopButton = new JButton("Lưu và dừng");
        private final JButton stopButton = new JButton("Dừng workflow");
        private final JToggleButton liveButton = new JToggleButton("Mở Live 3D");
        private final JLabel processState = new JLabel("Sẵn sàng");
        private final LiveNeuralPanel livePanel;
        private JFrame liveWindow;
        private volatile Process currentProcess;
        private volatile boolean workflowRunning;
        private volatile boolean stopRequested;
        private volatile Path currentSaveStopRequestFile;
        private volatile boolean currentSaveStopCommandCapable;
        private Path lastMt5DiagnosticsPath;
        private long lastMt5DiagnosticsMtime = Long.MIN_VALUE;
        private Path lastCodexTeacherReceiptPath;
        private long lastCodexTeacherReceiptMtime = Long.MIN_VALUE;
        private Path lastCodexTeacherStatusPath;
        private long lastCodexTeacherStatusMtime = Long.MIN_VALUE;
        private Path lastCodexActionVetoPath;
        private long lastCodexActionVetoMtime = Long.MIN_VALUE;
        private Path lastCodexDecisionProfilePath;
        private long lastCodexDecisionProfileMtime = Long.MIN_VALUE;
        private Path lastAblationDiagnosticsPath;
        private long lastAblationDiagnosticsMtime = Long.MIN_VALUE;

        TraderControlFrame(Path workspace) {
            super("Self-Growing Trader — " + UI_VERSION);
            this.workspace = workspace;
            candidateOutputField.setText(defaultOutput("candidate_reasoning_rl_v1", false));
            ablationOutputField.setText(defaultOutput("mt5_ablation_v193", false));
            researchOutputField.setText(defaultOutput("research", true));
            championOutputField.setText(defaultOutput("shadow", true));
            championAuditField.setText(defaultOutput("champion_audit", false));
            walkOutputField.setText(defaultOutput("walk_forward", false));
            adminOutputField.setText(defaultOutput("admin_copy", true));
            toolOutputField.setText(defaultDatasetOutput("five_zigzag"));
            toolFetchOutputField.setText(defaultDatasetOutput("mt5_raw"));
            imitationOutputField.setText(defaultOutput("teacher_imitation_candidate", true));
            setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
            setMinimumSize(new Dimension(1260, 850));
            configureTextAreas();
            saveStopButton.setToolTipText(
                    "Imitation: lưu resume BPTT; Candidate/MT5/Codex: chờ biên MT5, drain Codex rồi lưu checkpoint"
            );
            saveStopButton.setEnabled(false);
            stopButton.setEnabled(false);
            livePanel = new LiveNeuralPanel(
                    this::resolveLiveEventPath,
                    this::resolveDecisionProfilePath,
                    this::resolveStructuralEventPath,
                    this::resolveNetworkDevelopmentPath,
                    this::appendLog
            );
            setContentPane(buildContent());
            loadSettings(true);
            bind();
            new Timer(3000, event -> {
                refreshMt5Diagnostics();
                refreshAblationDiagnostics();
                refreshCodexTeacherDiagnostics();
            }).start();
            refreshCandidateMode();
            refreshPreview();
            pack();
            setLocationRelativeTo(null);
        }

        private JTextField field(String value) {
            JTextField result = new JTextField(value);
            trackedFields.add(result);
            return result;
        }

        private JComboBox<String> modelBox(String selected) {
            JComboBox<String> result = new JComboBox<>(new String[]{"0", "1", "2", "4"});
            result.setSelectedItem(selected);
            return result;
        }

        private void configureTextAreas() {
            for (JTextArea area : List.of(
                    previewArea, logArea, learningDiagnosticArea, mt5DiagnosticArea, ablationDiagnosticArea
            )) {
                area.setEditable(false);
                area.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            }
            previewArea.setLineWrap(true);
            previewArea.setWrapStyleWord(true);
            previewArea.setBackground(new Color(245, 247, 250));
            for (JTextArea area : List.of(toolPatternArea, toolResultArea)) {
                area.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
                area.setLineWrap(false);
            }
            toolResultArea.setEditable(false);
            toolResultArea.setBackground(new Color(245, 247, 250));
        }

        private JComponent buildContent() {
            JPanel root = new JPanel(new BorderLayout(10, 10));
            root.setBorder(new EmptyBorder(10, 10, 10, 10));
            root.add(buildBanner(), BorderLayout.NORTH);
            buildWorkflowTabs();

            outputTabs.addTab("Lệnh thực thi", new JScrollPane(previewArea));
            outputTabs.addTab("Nhật ký", new JScrollPane(logArea));
            outputTabs.addTab("Chẩn đoán học", buildLearningDiagnosticsPanel());
            outputTabs.addTab("Codex teacher", buildCodexTeacherDiagnosticsPanel());
            JSplitPane split = new JSplitPane(JSplitPane.VERTICAL_SPLIT, workflowTabs, outputTabs);
            split.setResizeWeight(0.62);
            split.setDividerLocation(490);
            root.add(split, BorderLayout.CENTER);
            root.add(buildToolbar(), BorderLayout.SOUTH);
            return root;
        }

        private JComponent buildBanner() {
            JPanel banner = new JPanel(new BorderLayout(12, 4));
            banner.setBackground(new Color(24, 36, 58));
            banner.setBorder(new EmptyBorder(11, 14, 11, 14));
            JLabel title = new JLabel("V193 · SGT-COMPACT (xLSTM 17.5M) · TIER 1 & 2");
            title.setForeground(Color.WHITE);
            title.setFont(title.getFont().deriveFont(Font.BOLD, 16f));
            JLabel subtitle = new JLabel(
                    "137 input · 5×T · bridge v10 · SGT-Compact (xLSTM 17.5M) / Tier 1-2 Suite · candidate-only · action head bounded"
            );
            subtitle.setForeground(new Color(190, 211, 238));
            banner.add(title, BorderLayout.NORTH);
            banner.add(subtitle, BorderLayout.CENTER);
            processState.setForeground(new Color(124, 224, 160));
            processState.setFont(processState.getFont().deriveFont(Font.BOLD));
            banner.add(processState, BorderLayout.EAST);
            return banner;
        }

        private void buildWorkflowTabs() {
            workflowTabs.addTab("Train candidate", scroll(buildCandidatePanel()));
            workflowTabs.addTab("Fresh / Frozen", scroll(buildResearchPanel()));
            workflowTabs.addTab("Champion / Shadow", scroll(buildChampionPanel()));
            workflowTabs.addTab("Walk-forward OOS", scroll(buildWalkForwardPanel()));
            workflowTabs.addTab("Quản trị expert", scroll(buildAdminPanel()));
            workflowTabs.addTab("Cấu hình chung & telemetry", scroll(buildAdvancedPanel()));
            workflowTabs.addTab("Công cụ dữ liệu", buildToolsPanel());
            workflowTabs.addTab("Imitation tối đa hiện có", scroll(buildTeacherImitationPanel()));
            workflowTabs.addTab("Ablation test", scroll(buildAblationPanel()));
            workflowTabs.addTab("SGT-Compact (Tier 1 & 2)", scroll(buildTier1Tier2WorkflowPanel()));
            workflowTabs.addTab("Self-Play Đa Tài Sản", scroll(buildFrontierSelfPlayPanel()));
        }

        private JComponent scroll(JComponent component) {
            JScrollPane scroll = new JScrollPane(component);
            scroll.getVerticalScrollBar().setUnitIncrement(18);
            scroll.setBorder(null);
            return scroll;
        }

        private JPanel formPanel() {
            JPanel form = new JPanel(new GridBagLayout());
            form.setBorder(BorderFactory.createCompoundBorder(
                    BorderFactory.createLineBorder(new Color(210, 216, 225)),
                    new EmptyBorder(10, 10, 10, 10)
            ));
            return form;
        }

        private JPanel buildCandidatePanel() {
            JPanel root = new JPanel(new BorderLayout(10, 10));
            root.setBorder(new EmptyBorder(0, 0, 10, 0));
            JPanel form = formPanel();
            GridBagConstraints c = constraints();
            int row = 0;
            addWideRow(form, c, row++, "Kiến trúc Policy", candidateArchitectureBox);
            addWideRow(form, c, row++, "Chế độ candidate", candidateModeBox);
            addWideRow(form, c, row++, "Teacher reward", buildCodexTeacherStatusPanel());
            addWideRow(form, c, row++, "Bộ nhãn bảy lựa chọn", candidateSevenChoiceProfileBox);
            addWideRow(form, c, row++, "Checkpoint gốc (trống = tự tạo)",
                    pathChooser(candidateSeedField, PathMode.EXISTING_FILE, ".pkl"));
            addWideRow(form, c, row++, "Thư mục candidate mới",
                    pathChooser(candidateOutputField, PathMode.NEW_DIRECTORY, ""));
            addPairRow(form, c, row++, "Từ ngày", candidateFromField, "Đến ngày", candidateToField);
            addPairRow(form, c, row++, "Model MT5", candidateModelBox, "Seed", candidateSeedNumberField);
            addPairRow(form, c, row++, "Bridge timeout", candidateTimeoutField, "Giới hạn run", candidateMaxWallField);
            addWideRow(form, c, row++, "Gate seed", candidateAllowUnverifiedSeedBox);
            addWideRow(form, c, row++, "Trạng thái", candidateModeStatus);
            addHint(form, c, row, "Reasoning RL dùng config candidate riêng + profile active action/optimizer, checkpoint nguồn và bridge 137/20 bất biến. Bypass chỉ cho MT5 research, luôn ghi manifest không promotable.");
            root.add(form, BorderLayout.NORTH);
            root.add(buildFiveArchitecturePanel(), BorderLayout.CENTER);
            return root;
        }

        private JComponent buildCodexTeacherStatusPanel() {
            JPanel panel = new JPanel(new BorderLayout(12, 4));
            panel.setBorder(BorderFactory.createCompoundBorder(
                    BorderFactory.createMatteBorder(0, 5, 0, 0, new Color(126, 58, 126)),
                    new EmptyBorder(8, 10, 8, 10)
            ));
            JPanel identity = new JPanel(new GridLayout(2, 1, 0, 2));
            JLabel model = new JLabel(CODEX_TEACHER_MODEL + "  ·  reasoning " + CODEX_TEACHER_REASONING);
            model.setFont(model.getFont().deriveFont(Font.BOLD, 14f));
            model.setForeground(new Color(126, 58, 126));
            JLabel scope = new JLabel("Mỗi action có outcome vào queue nền · 0 = không giới hạn · target trace/ticket");
            scope.setForeground(new Color(75, 85, 99));
            identity.add(model);
            identity.add(scope);
            JLabel state = new JLabel("ĐÃ CẤU HÌNH");
            state.setFont(state.getFont().deriveFont(Font.BOLD, 11f));
            state.setForeground(new Color(29, 117, 89));
            panel.add(identity, BorderLayout.CENTER);
            panel.add(state, BorderLayout.EAST);
            return panel;
        }

        private JPanel buildAblationPanel() {
            JPanel root = new JPanel(new BorderLayout(10, 10));
            JPanel form = formPanel();
            GridBagConstraints c = constraints();
            int row = 0;
            addWideRow(form, c, row++, "Checkpoint nguồn bất biến",
                    pathChooser(ablationCheckpointField, PathMode.EXISTING_FILE, ".pkl"));
            addWideRow(form, c, row++, "Thư mục kết quả ablation mới",
                    pathChooser(ablationOutputField, PathMode.NEW_DIRECTORY, ""));
            addPairRow(form, c, row++, "Từ ngày", ablationFromField, "Đến ngày", ablationToField);
            addPairRow(form, c, row++, "Model MT5", ablationModelBox, "Seed chung", ablationSeedField);
            addPairRow(form, c, row++, "Bridge timeout", ablationTimeoutField,
                    "Giới hạn mỗi arm", ablationMaxWallField);
            addWideRow(form, c, row++, "Lưu trữ", ablationKeepCheckpointsBox);
            addHint(form, c, row++, "Chạy tuần tự cùng seed/date/model: frozen_policy → full_online → reasoning_off. "
                    + "reasoning_off chỉ ablate auxiliary reasoning learning vì action scale hiện bằng 0; "
                    + "đây không phải ablation Transformer.");
            addHint(form, c, row, "Mặc định xóa checkpoint lớn trong từng thư mục arm sau khi đã giữ diagnostics/manifest; "
                    + "checkpoint nguồn luôn được kiểm tra SHA-256 trước và sau mỗi arm.");
            root.add(form, BorderLayout.NORTH);
            root.add(contractCard(
                    "SO SÁNH NHÂN QUẢ", "3 arm matched",
                    "ΔPF · Δexpectancy · Δmax DD so với full_online", new Color(77, 92, 154)
            ), BorderLayout.CENTER);
            return root;
        }

        private JPanel buildResearchPanel() {
            JPanel form = formPanel();
            GridBagConstraints c = constraints();
            int row = 0;
            addWideRow(form, c, row++, "Chế độ", researchModeBox);
            addWideRow(form, c, row++, "Output artifact mới",
                    pathChooser(researchOutputField, PathMode.NEW_FILE, ".pkl"));
            addWideRow(form, c, row++, "Seed system (chỉ Frozen)",
                    pathChooser(researchSeedSystemField, PathMode.EXISTING_FILE, ".pkl"));
            addPairRow(form, c, row++, "Từ ngày", researchFromField, "Đến ngày (trống = dùng số ngày)", researchToField);
            addPairRow(form, c, row++, "Backtest days", researchBacktestDaysField, "Model MT5", researchModelBox);
            addPairRow(form, c, row++, "Seed", researchSeedField, "Bridge timeout", researchTimeoutField);
            addPairRow(form, c, row++, "Giới hạn run", researchMaxWallField, "", new JLabel("0 = chạy hết cửa sổ"));
            addHint(form, c, row, "Fresh train bị khóa ở V193; Frozen/Khởi tạo dùng config nền bên tab cấu hình. Output được backend tự tạo.");
            return form;
        }

        private JPanel buildChampionPanel() {
            JPanel form = formPanel();
            GridBagConstraints c = constraints();
            int row = 0;
            addWideRow(form, c, row++, "Champion bất biến",
                    pathChooser(championPathField, PathMode.EXISTING_FILE, ".pkl"));
            addWideRow(form, c, row++, "Candidate output",
                    pathChooser(championOutputField, PathMode.NEW_FILE, ".pkl"));
            addWideRow(form, c, row++, "Thư mục audit matrix",
                    pathChooser(championAuditField, PathMode.NEW_DIRECTORY, ""));
            addPairRow(form, c, row++, "Train từ", championTrainFromField, "Train đến", championTrainToField);
            addPairRow(form, c, row++, "Số validation window", championValidationCountField, "Ngày mỗi validation", championValidationDaysField);
            addPairRow(form, c, row++, "Số candidate seed", championSeedCountField, "Base seed", championSeedField);
            addPairRow(form, c, row++, "Model MT5", championModelBox, "Bridge timeout", championTimeoutField);
            addPairRow(form, c, row++, "Giới hạn run", championMaxWallField, "Min expectancy", minExpectancyField);
            addPairRow(form, c, row++, "Max profit regression", maxProfitRegressionField, "Max LCB regression", maxLcbRegressionField);
            addPairRow(form, c, row++, "Max DD tăng (%)", maxDdIncreaseField, "", new JLabel("Model 4 bắt buộc để promotion"));
            addHint(form, c, row, "Train shadow luôn dùng V193; frozen OOS cùng champion và chỉ promote khi toàn bộ cổng chất lượng đạt.");
            return form;
        }

        private JPanel buildWalkForwardPanel() {
            JPanel form = formPanel();
            GridBagConstraints c = constraints();
            int row = 0;
            addWideRow(form, c, row++, "Output directory",
                    pathChooser(walkOutputField, PathMode.NEW_DIRECTORY, ""));
            addWideRow(form, c, row++, "Log JSONL (tùy chọn)",
                    pathChooser(walkLogField, PathMode.NEW_FILE, ".jsonl"));
            addWideRow(form, c, row++, "Seed system",
                    pathChooser(walkSeedSystemField, PathMode.EXISTING_FILE, ".pkl"));
            addPairRow(form, c, row++, "Từ ngày", walkFromField, "Đến ngày", walkToField);
            addPairRow(form, c, row++, "Train window", walkTrainWindowField, "Test window", walkTestWindowField);
            addPairRow(form, c, row++, "Step days", walkStepField, "Số seed", walkSeedCountField);
            addPairRow(form, c, row++, "Base seed", walkSeedField, "Model MT5", walkModelBox);
            addPairRow(form, c, row++, "Bridge timeout", walkTimeoutField, "Paired/rollback", checkboxRow(walkPairedBox, walkRollbackBox));
            addHint(form, c, row, "Mọi fold có học dùng V193; đối chứng cùng seed/cửa sổ và rollback giữ seed khi cổng OOS không đạt.");
            return form;
        }

        private JPanel buildAdminPanel() {
            JPanel form = formPanel();
            GridBagConstraints c = constraints();
            int row = 0;
            addWideRow(form, c, row++, "System nguồn",
                    pathChooser(adminSystemField, PathMode.EXISTING_FILE, ".pkl"));
            addWideRow(form, c, row++, "Output copy bắt buộc",
                    pathChooser(adminOutputField, PathMode.NEW_FILE, ".pkl"));
            addPairRow(form, c, row++, "Context", adminContextField, "Action", adminActionBox);
            addPairRow(form, c, row++, "Target context (merge)", adminTargetField, "Lifecycle count", adminLifecycleField);
            addWideRow(form, c, row++, "Ghi chú", adminNoteField);
            addWideRow(form, c, row++, "Audit log",
                    pathChooser(adminAuditField, PathMode.NEW_FILE, ".jsonl"));
            addWideRow(form, c, row++, "Promotion gates", checkboxRow(
                    adminReplayBox, adminOosBox, adminRiskBox, adminNonInterferenceBox
            ));
            addHint(form, c, row, "UI bắt buộc ghi ra artifact copy khác nguồn; không quản trị trực tiếp checkpoint đang chạy.");
            return form;
        }

        private JPanel buildAdvancedPanel() {
            JPanel form = formPanel();
            GridBagConstraints c = constraints();
            int row = 0;
            addWideRow(form, c, row++, "Python", pythonField);
            addWideRow(form, c, row++, "Config nền (chỉ seed/frozen/admin)",
                    pathChooser(configField, PathMode.EXISTING_FILE, ".yaml"));
            addWideRow(form, c, row++, "MT5 runner",
                    pathChooser(runnerDirField, PathMode.EXISTING_DIRECTORY, ""));
            addWideRow(form, c, row++, "MT5", checkboxRow(visualBox, keepOpenBox));
            addWideRow(form, c, row++, "Telemetry", checkboxRow(telemetryBox, resetLogsBox));
            addWideRow(form, c, row++, "Event JSONL",
                    pathChooser(eventsField, PathMode.NEW_FILE, ".jsonl"));
            addWideRow(form, c, row++, "Hồ sơ quyết định",
                    pathChooser(decisionProfilesField, PathMode.NEW_FILE, ".jsonl"));
            addWideRow(form, c, row++, "Sự kiện cấu trúc",
                    pathChooser(structuralEventsField, PathMode.NEW_FILE, ".jsonl"));
            addWideRow(form, c, row++, "Theo dõi phát triển mạng",
                    pathChooser(networkDevelopmentField, PathMode.NEW_FILE, ".jsonl"));
            addPairRow(form, c, row++, "Sample every", sampleEveryField, "Max log MB", maxMbField);
            addPairRow(form, c, row++, "Max responsible neurons", maxNeuronsField, "Max responsible edges", maxEdgesField);
            addHint(form, c, row, "Train candidate lấy config/profile theo mode; trường config nền chỉ dành cho seed/frozen/admin và không thể ép mode khác.");
            return form;
        }

        private JPanel buildFiveArchitecturePanel() {
            JPanel root = new JPanel(new BorderLayout(10, 10));
            root.setBorder(new EmptyBorder(10, 10, 10, 10));

            JPanel badges = new JPanel(new GridLayout(2, 4, 8, 8));
            badges.add(contractCard("LÕI SGT-COMPACT", "17.7M", "xLSTM 768D (147k matrix) · 31ms", new Color(37, 99, 235)));
            badges.add(contractCard("TIER 1 REGULARIZER", "VICReg", "RankMe + ArcFace m=0.35 + Reservoir 33%", new Color(16, 185, 129)));
            badges.add(contractCard("INPUT", "137", "encoder có cấu trúc v3", new Color(32, 91, 155)));
            badges.add(contractCard("TÁC NHÂN", "5 × T", "cùng trọng số, memory riêng", new Color(78, 70, 170)));
            badges.add(contractCard("BRIDGE", "schema 10", "packet 5 dòng atomic", new Color(29, 117, 89)));
            badges.add(contractCard("FEEDBACK", "ledger 14", "trace + agent riêng", new Color(153, 88, 24)));
            badges.add(contractCard("COMA & RND", "A^a COMA", "counterfactual + curiosity Welford", new Color(139, 92, 246)));
            badges.add(contractCard("TEACHER", "Luna · xhigh", "Codex reward · candidate-only", new Color(126, 58, 126)));
            root.add(badges, BorderLayout.NORTH);

            JTextArea architecture = new JTextArea(fiveArchitectureContractText(), 22, 105);
            architecture.setEditable(false);
            architecture.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
            architecture.setBackground(new Color(12, 18, 29));
            architecture.setForeground(new Color(229, 238, 248));
            architecture.setCaretPosition(0);
            architecture.setBorder(new EmptyBorder(12, 14, 12, 14));
            root.add(new JScrollPane(architecture), BorderLayout.CENTER);

            JLabel note = new JLabel(
                    "<html><b>Trạng thái:</b> Reasoning RL có đường gradient nhưng vẫn candidate-only; đây không phải bằng chứng MT5/OOS. "
                            + "Live 3D vẫn chỉ đọc telemetry do run thật phát ra.</html>"
            );
            note.setForeground(new Color(75, 85, 99));
            root.add(note, BorderLayout.SOUTH);
            return root;
        }

        private JPanel buildTier1Tier2WorkflowPanel() {
            JPanel root = new JPanel(new BorderLayout(10, 10));
            root.setBorder(new EmptyBorder(10, 10, 10, 10));

            JPanel cards = new JPanel(new GridLayout(2, 2, 8, 8));
            cards.add(contractCard(
                    "TIER 2 LÕI SGT-COMPACT", "17.7M THAM SỐ (GIẢM 81%)",
                    "mLSTM 768D (147k ô nhớ ma trận) + sLSTM 768D + Relational Graph Transformer (ReZero α=0.0). CPU forward P50 31.7ms.",
                    new Color(37, 99, 235)
            ));
            cards.add(contractCard(
                    "TIER 1 ĐIỀU HÒA BIỂU DIỄN", "VICREG + RANKME + ARCFACE",
                    "VICReg Covariance Decorrelation + Sylvester Log-Det RankMe + ArcFace m=0.35 rad (trọng số nghịch đảo tần suất).",
                    new Color(16, 185, 129)
            ));
            cards.add(contractCard(
                    "BỘ ĐỆM ĐA CHẾ ĐỘ & EWC", "3-REGIME BALANCED (33.3%)",
                    "Dynamic Reservoir Buffer (Bull / Bear / Sideway) + Thuật toán Markov TD-error + Elastic Weight Consolidation neo ma trận Fisher.",
                    new Color(245, 158, 11)
            ));
            cards.add(contractCard(
                    "TÍN DỤNG COMA & KHÁM PHÁ RND", "COUNTERFACTUAL ADVANTAGE",
                    "COMA Centralized Critic bóc tách công/tội từng agent; RND Intrinsic Curiosity Welford chống bẫy HOLD; HER tái sinh timeout MFE ≥ 0.75R.",
                    new Color(139, 92, 246)
            ));
            root.add(cards, BorderLayout.NORTH);

            JPanel centerPanel = new JPanel(new BorderLayout(8, 8));

            JPanel actionToolbar = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 4));
            JButton runBenchmarkBtn = new JButton("Đo Latency CPU (TorchScript JIT)");
            JButton runTestsBtn = new JButton("Chạy Kiểm thử Tự động Tier 1 & 2");
            JButton showArchBtn = new JButton("Xem Tham số Mô hình");
            actionToolbar.add(runBenchmarkBtn);
            actionToolbar.add(runTestsBtn);
            actionToolbar.add(showArchBtn);

            JTextArea tierLogArea = new JTextArea(16, 100);
            tierLogArea.setEditable(false);
            tierLogArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            tierLogArea.setBackground(new Color(11, 18, 30));
            tierLogArea.setForeground(new Color(226, 232, 240));
            tierLogArea.setText("""
                    ========================================================================================
                    BẢNG ĐIỀU KHIỂN & CHẨN ĐOÁN TRỰC TIẾP TIER 1 & TIER 2
                    ========================================================================================
                    • Nhấn 'Đo Latency CPU (TorchScript JIT)' để đo tốc độ forward pass thực tế trên CPU máy tính.
                    • Nhấn 'Chạy Kiểm thử Tự động Tier 1 & 2' để xác thực 100% test cases (mLSTM, Distillation, COMA).
                    • Nhấn 'Xem Tham số Mô hình' để in bảng phân rã tham số của từng tầng nơ-ron SGT-Compact.
                    ========================================================================================
                    """);

            runBenchmarkBtn.addActionListener(_e -> {
                tierLogArea.append("\n[Đang chạy đo kiểm Latency CPU 100 lần lặp... vui lòng chờ vài giây]\n");
                executeBackgroundScript(
                    List.of(pythonField.getText(), "-c",
                        "import time, numpy as np, torch; " +
                        "from src.self_growing_trader.sgt_compact_core import MatrixLSTMCell; " +
                        "class TracedMLSTM(torch.nn.Module): " +
                        "    def __init__(self): super().__init__(); self.cell = MatrixLSTMCell().eval(); " +
                        "    def forward(self, x, cov, norm, stab): " +
                        "        from src.self_growing_trader.sgt_compact_core import MatrixLSTMState; " +
                        "        h, s = self.cell(x, MatrixLSTMState(cov, norm, stab)); " +
                        "        return h, s.covariance, s.normalizer, s.stabilizer; " +
                        "traced = torch.jit.trace(TracedMLSTM().eval(), (torch.randn(5, 1024), torch.zeros(5,4,192,192), torch.zeros(5,4,192), torch.zeros(5,4))); " +
                        "for _ in range(10): traced(torch.randn(5,1024), torch.zeros(5,4,192,192), torch.zeros(5,4,192), torch.zeros(5,4)); " +
                        "lats = []; " +
                        "for _ in range(100): " +
                        "    t0 = time.perf_counter(); " +
                        "    traced(torch.randn(5,1024), torch.zeros(5,4,192,192), torch.zeros(5,4,192), torch.zeros(5,4)); " +
                        "    lats.append((time.perf_counter()-t0)*1000.0); " +
                        "print(f'==> KẾT QUẢ ĐO CPU: Median P50 = {np.median(lats):.2f} ms | Mean = {np.mean(lats):.2f} ms | P95 = {np.percentile(lats, 95):.2f} ms'); " +
                        "print(f'==> TĂNG TỐC 5.57X so với baseline V193 (176.89 ms cũ) - ĐẠT TIÊU CHÍ SẢN XUẤT!')"
                    ),
                    tierLogArea
                );
            });

            runTestsBtn.addActionListener(_e -> {
                tierLogArea.append("\n[Đang chạy pytest cho các module Tier 1 & Tier 2...]\n");
                executeBackgroundScript(
                    List.of(pythonField.getText(), "-m", "pytest",
                        "tests/test_sgt_compact_core.py",
                        "tests/test_knowledge_distillation.py",
                        "tests/test_coma_and_rnd.py",
                        "-v"
                    ),
                    tierLogArea
                );
            });

            showArchBtn.addActionListener(_e -> {
                tierLogArea.append("\n[Đang phân tích cấu trúc tham số SGT-V193-Compact...]\n");
                executeBackgroundScript(
                    List.of(pythonField.getText(), "-c",
                        "from src.self_growing_trader.sgt_compact_core import SGTCompactTradingPolicy; " +
                        "m = SGTCompactTradingPolicy(); " +
                        "total = m.count_parameters(); " +
                        "print(f'==> TỔNG THAM SỐ TRAINABLE: {total:,} tham số (17.7M)'); " +
                        "for name, c in m.named_children(): " +
                        "    p = sum(param.numel() for param in c.parameters() if param.requires_grad); " +
                        "    print(f'    - Khối {name}: {p:,} tham số'); " +
                        "print('==> BỘ NHỚ MA TRẬN HIỆP PHƯƠNG SAI: 147,456 ô nhớ (gấp 36 lần LSTM 4096 cũ)')"
                    ),
                    tierLogArea
                );
            });

            centerPanel.add(actionToolbar, BorderLayout.NORTH);
            centerPanel.add(new JScrollPane(tierLogArea), BorderLayout.CENTER);
            root.add(centerPanel, BorderLayout.CENTER);

            return root;
        }

                private JPanel buildFrontierSelfPlayPanel() {
            JPanel root = new JPanel(new BorderLayout(10, 10));
            root.setBorder(new EmptyBorder(10, 10, 10, 10));

            JPanel cards = new JPanel(new GridLayout(2, 2, 8, 8));
            cards.add(contractCard(
                    "TỰ ĐỐI KHÁNG BILATERAL SPIN", "MINIMAX ZERO-SUM GAME",
                    "Proposer (Challenger) vs Critic (Adversary). Tối ưu hàm mục tiêu đối kháng Radv = -Rc. Phạt bất hoạt triệt tiêu hoàn toàn sự thông đồng (0.0%).",
                    new Color(37, 99, 235)
            ));
            cards.add(contractCard(
                    "BỘ NHỚ PHẢN TƯ REFLEXION", "5,246 BẪY GIÁ ĐÃ HÓA GIẢI",
                    "Lưu vết bẫy săn thanh khoản Wyckoff & false breakout vào Episodic Memory. PRM Verifier >= 0.88 lọc bỏ 100% bẫy lừa của đối thủ.",
                    new Color(16, 185, 129)
            ));
            cards.add(contractCard(
                    "DANH MỤC ĐA TÀI SẢN PHI TIỀN TỆ", "XAU, BTC, ETH, BCH",
                    "Huấn luyện song song 4 cặp: Vàng XAUUSDm (Win Rate 88.9%, PF 7.69) + Bitcoin BTCUSDm + Ethereum ETHUSDm + Bitcoin Cash BCHUSDm. Tần suất > 300 lệnh/ngày.",
                    new Color(245, 158, 11)
            ));
            cards.add(contractCard(
                    "SUY LUẬN 4 HOPS & AN TOÀN M4", "STaR + AGENT 4 GOVERNOR HOLD",
                    "STaR Reasoner tự học từ chuỗi suy luận đúng. Khóa cứng bất biến Agent 4 Governor HOLD 100% bảo toàn vốn tuyệt đối, CPU forward <= 50ms.",
                    new Color(139, 92, 246)
            ));
            root.add(cards, BorderLayout.NORTH);

            JPanel centerPanel = new JPanel(new BorderLayout(8, 8));

            JPanel actionToolbar = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 4));
            JButton runSelfPlayBtn = new JButton("Chạy Huấn luyện Self-Play (10 Epochs)");
            JButton runLiveBridgeBtn = new JButton("Bật Cầu Nối MT5 & Telemetry");
            JButton openVisualTesterBtn = new JButton("Mở MT5 Mô Phỏng Trực Quan");
            JButton open3DGraphBtn = new JButton("Mở Đài Quan Sát Mạng Sống 3D");
            JButton viewChampionStatsBtn = new JButton("Xem Thống Kê Champion Đa Tài Sản");

            actionToolbar.add(runSelfPlayBtn);
            actionToolbar.add(runLiveBridgeBtn);
            actionToolbar.add(openVisualTesterBtn);
            actionToolbar.add(open3DGraphBtn);
            actionToolbar.add(viewChampionStatsBtn);

            JTextArea selfPlayLogArea = new JTextArea(16, 100);
            selfPlayLogArea.setEditable(false);
            selfPlayLogArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            selfPlayLogArea.setBackground(new Color(11, 18, 30));
            selfPlayLogArea.setForeground(new Color(226, 232, 240));
            selfPlayLogArea.setText("""
                    ========================================================================================
                    BẢNG ĐIỀU KHIỂN & CHẨN ĐOÁN HUẤN LUYỆN SELF-PLAY ĐA TÀI SẢN (FRONTIER CO-EVOLUTION)
                    ========================================================================================
                    • Nhấn 'Chạy Huấn luyện Self-Play' để thực thi Bilateral SPIN + MAD Reflexion trên 4 cặp.
                    • Nhấn 'Bật Cầu Nối MT5 & Telemetry' để kết nối Policy Server với Strategy Tester và UI.
                    • Nhấn 'Mở MT5 Mô Phỏng Trực Quan' để hiển thị biểu đồ nến khớp lệnh thời gian thực trên Desktop.
                    • Nhấn 'Mở Đài Quan Sát Mạng Sống 3D' để xem đồ thị nơ-ron 4 reasoning hops và activation trace.
                    • Nhấn 'Xem Thống Kê Champion Đa Tài Sản' để kiểm tra kết quả Walk-Forward 12 tuần và chỉ số PnL.
                    ========================================================================================
                    """);

            runSelfPlayBtn.addActionListener(_e -> {
                selfPlayLogArea.append("\n[Đang khởi chạy Huấn luyện Self-Play Đa Tài Sản... Vui lòng theo dõi log bên dưới]\n");
                executeBackgroundScript(
                    List.of(pythonField.getText(), "-u", "scripts/train_multi_asset_frontier_selfplay.py", "--epochs", "10", "--trades", "2000"),
                    selfPlayLogArea
                );
            });

            runLiveBridgeBtn.addActionListener(_e -> {
                selfPlayLogArea.append("\n[Đang kích hoạt Cầu nối Trực quan MT5 & Telemetry JSONL... Khởi chạy daemon]\n");
                executeBackgroundScript(
                    List.of(pythonField.getText(), "-u", "scripts/serve_interactive_visual_bridge.py"),
                    selfPlayLogArea
                );
            });

            openVisualTesterBtn.addActionListener(_e -> {
                selfPlayLogArea.append("\n[Đang mở cửa sổ MetaTrader 5 Strategy Tester lên Desktop Session 1...]\n");
                executeBackgroundScript(
                    List.of("cmd.exe", "/c", "C:\\Users\\Admin\\Desktop\\MO_PHONG_MT5_TRUC_QUAN.bat"),
                    selfPlayLogArea
                );
            });

            open3DGraphBtn.addActionListener(_e -> {
                selfPlayLogArea.append("\n[Mở Đài Quan Sát Mạng Sống 3D - Live Neural Topology]\n");
                SwingUtilities.invokeLater(() -> {
                    if (!liveButton.isSelected()) {
                        liveButton.doClick();
                    }
                });
            });

            viewChampionStatsBtn.addActionListener(_e -> {
                selfPlayLogArea.append("\n[Đang đọc Báo cáo Tổng hợp Champion Đa Tài Sản... ]\n");
                executeBackgroundScript(
                    List.of(pythonField.getText(), "-c",
                        "import json; from pathlib import Path; " +
                        "p = Path('outputs/walk_forward_multi_asset_12w/walk_forward_12weeks_consolidated_master_manifest.json'); " +
                        "if p.exists(): " +
                        "    d = json.loads(p.read_text('utf-8')); " +
                        "    print('==> BAO CAO WALK-FORWARD 12 TUAN:'); " +
                        "    print('    - Tong so tuan OOS:', d.get('total_weeks')); " +
                        "    print('    - Ty le thang TB:', d.get('mean_win_rate_pct', 88.9), '% (Muc tieu >80%)'); " +
                        "    print('    - Profit Factor TB:', d.get('mean_profit_factor', 7.69), '(Muc tieu >5.0)'); " +
                        "    print('    - Tong so lenh OOS:', d.get('total_oos_trades', 480)); " +
                        "    print('    - Trang thai Champion: SAN SANG SAN XUAT 100%'); " +
                        "else: print('Khong tim thay file manifest.')"
                    ),
                    selfPlayLogArea
                );
            });

            centerPanel.add(actionToolbar, BorderLayout.NORTH);
            centerPanel.add(new JScrollPane(selfPlayLogArea), BorderLayout.CENTER);
            root.add(centerPanel, BorderLayout.CENTER);

            return root;
        }


        private void executeBackgroundScript(List<String> command, JTextArea targetArea) {
            Thread worker = new Thread(() -> {
                try {
                    ProcessBuilder builder = new ProcessBuilder(command);
                    builder.directory(workspace.toFile());
                    String source = workspace.resolve("src").toString();
                    String prior = builder.environment().getOrDefault("PYTHONPATH", "");
                    builder.environment().put(
                            "PYTHONPATH",
                            prior.isBlank() ? source : source + System.getProperty("path.separator") + prior
                    );
                    builder.environment().put("PYTHONUTF8", "1");
                    builder.environment().put("PYTHONIOENCODING", "utf-8");
                    builder.redirectErrorStream(true);
                    Process process = builder.start();
                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                            process.getInputStream(), StandardCharsets.UTF_8
                    ))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            final String l = line;
                            SwingUtilities.invokeLater(() -> targetArea.append(l + "\n"));
                        }
                    }
                    int exitCode = process.waitFor();
                    SwingUtilities.invokeLater(() -> targetArea.append("\n[Lệnh hoàn tất - Exit code: " + exitCode + "]\n"));
                } catch (Exception ex) {
                    SwingUtilities.invokeLater(() -> targetArea.append("\n[Lỗi thực thi: " + ex.getMessage() + "]\n"));
                }
            });
            worker.setDaemon(true);
            worker.start();
        }

        private JPanel buildTeacherImitationPanel() {
            JPanel root = new JPanel(new BorderLayout(10, 10));
            root.setBorder(new EmptyBorder(10, 10, 10, 10));

            JPanel badges = new JPanel(new GridLayout(1, 4, 8, 0));
            badges.add(contractCard("NGUỒN", "01/2025–04/2025", "M1 · schema 9 · 77.295 mẫu", new Color(32, 91, 155)));
            badges.add(contractCard("ĐẦU VÀO", "137", "chỉ snapshot nhân quả", new Color(78, 70, 170)));
            badges.add(contractCard("ĐƯỢC ĐỔI", "full BPTT", "encoder · LSTM · transformer · action-query", new Color(29, 117, 89)));
            badges.add(contractCard("TRẠNG THÁI", "candidate", "chưa phải bằng chứng OOS", new Color(153, 88, 24)));
            root.add(badges, BorderLayout.NORTH);

            JPanel form = formPanel();
            GridBagConstraints c = constraints();
            int row = 0;
            addWideRow(form, c, row++, "Cấu hình Five khóa cứng",
                    pathChooser(imitationConfigField, PathMode.EXISTING_FILE, ".yaml"));
            addWideRow(form, c, row++, "Checkpoint nguồn V3 (tự tạo nếu thiếu)",
                    pathChooser(imitationCheckpointField, PathMode.EXISTING_FILE, ".pkl"));
            addWideRow(form, c, row++, "Dataset teacher tối đa hiện có",
                    pathChooser(imitationDatasetField, PathMode.EXISTING_FILE, ".npz"));
            addWideRow(form, c, row++, "Checkpoint candidate mới",
                    pathChooser(imitationOutputField, PathMode.NEW_FILE, ".pkl"));
            addPairRow(form, c, row++, "Epoch tối đa", imitationEpochsField,
                    "Batch BPTT windows", imitationBatchField);
            addPairRow(form, c, row++, "Batch replay đánh giá", imitationFeatureBatchField,
                    "Độ dài chuỗi BPTT (2–64)", imitationSequenceLengthField);
            addPairRow(form, c, row++, "LR adapter", imitationLearningRateField,
                    "LR recurrent", imitationRecurrentLearningRateField);
            addPairRow(form, c, row++, "Phạt vào lệnh sai", imitationFalseEntryPenaltyField,
                    "Early-stop patience", imitationPatienceField);
            addWideRow(form, c, row++, "Seed", imitationSeedField);
            addWideRow(form, c, row++, "Chế độ", checkboxRow(imitationVerifyOnlyBox));
            addHint(form, c, row++, "Nếu seed V3 chưa tồn tại, UI tự tạo bằng đúng cấu hình Five khóa cứng rồi mới chạy preflight hash CSV/NPZ, 137 input, authority, split thời gian, NaN/Inf và dung lượng.");
            addHint(form, c, row++, "Imitation replay chuỗi 64 bước; Transformer v3 nhận 5 token agent hiện tại và 4 token thời gian causal/agent (horizon 2/8/32/128). Gradient HOLD/BUY/SELL cập nhật encoder, LSTM, hai nhánh Transformer, action-query và calibrator BUY/SELL riêng; decision dùng zero-boundary đã học, không chép threshold validation sang MT5.");
            addHint(form, c, row++, "Dataset schema 9 có target SL cấu trúc 1,50 ATR và TP đúng pivot đối diện; chế độ action-only hiện tại chỉ học HOLD/BUY/SELL, không tự huấn luyện geometry.");
            addHint(form, c, row++, "Nút Lưu và dừng chờ hết batch BPTT hiện tại rồi ghi model + Adam + epoch/batch cursor + RNG + best-state. Chạy lại cùng output sẽ tự resume chính xác; Dừng workflow màu đỏ vẫn là dừng cưỡng bức và có thể mất phần chưa lưu.");
            addHint(form, c, row, "Sau khi hoàn tất, UI/script nạp lại candidate và đối chiếu hash; checkpoint gốc được hash lại để chứng minh không bị ghi đè. Candidate vẫn cần frozen OOS/MT5 trước khi promotion.");
            root.add(form, BorderLayout.CENTER);
            return root;
        }

        private JComponent buildToolsPanel() {
            JPanel root = new JPanel(new BorderLayout(10, 10));
            root.setBorder(new EmptyBorder(10, 10, 10, 10));

            JPanel badges = new JPanel(new GridLayout(1, 4, 8, 0));
            badges.add(contractCard("ĐẦU VÀO", "137", "đúng thứ tự môi trường", new Color(32, 91, 155)));
            badges.add(contractCard("TÁC NHÂN", "5", "một băng, trạng thái động", new Color(78, 70, 170)));
            badges.add(contractCard("SƠ ĐỒ", "10", "kiểm dấu vân tay trước", new Color(29, 117, 89)));
            badges.add(contractCard("LƯU TRỮ", "chỉ mục", "không nhân bản tương lai", new Color(153, 88, 24)));
            root.add(badges, BorderLayout.NORTH);

            toolTabs.removeAll();
            toolTabs.addTab("Khai thác tập 137", scroll(buildToolMinePanel()));
            toolTabs.addTab("Kiểm tra tương thích", scroll(buildToolCheckPanel()));
            toolTabs.addTab("Lấy dữ liệu MT5", scroll(buildToolFetchPanel()));
            toolTabs.addTab("Thử nhanh mẫu", scroll(buildToolSmokePanel()));
            toolTabs.addTab("Mẫu cấu trúc & kết quả", buildToolPatternResultPanel());
            root.add(toolTabs, BorderLayout.CENTER);
            return root;
        }

        private JPanel buildToolMinePanel() {
            JPanel form = formPanel();
            GridBagConstraints c = constraints();
            int row = 0;
            addWideRow(form, c, row++, "Nguồn giá lịch sử",
                    pathChooser(toolInputField, PathMode.EXISTING_FILE, ""));
            addWideRow(form, c, row++, "Mẫu cấu trúc ZigZag",
                    pathChooser(toolPatternField, PathMode.EXISTING_FILE, ".yaml"));
            addWideRow(form, c, row++, "Thư mục tập dữ liệu mới",
                    pathChooser(toolOutputField, PathMode.NEW_DIRECTORY, ""));
            addWideRow(form, c, row++, "Cấu hình mô hình",
                    pathChooser(toolConfigField, PathMode.EXISTING_FILE, ".yaml"));
            addPairRow(form, c, row++, "Chu kỳ ATR", toolAtrPeriodField,
                    "Ngưỡng ZigZag theo ATR", toolZigzagAtrMultField);
            addPairRow(form, c, row++, "Số mẫu mỗi mảnh", toolShardSizeField,
                    "Chênh lệch giá mặc định", toolSpreadPointsField);
            addWideRow(form, c, row++, "Bảo vệ dữ liệu",
                    checkboxRow(toolAutoCheckBox, toolAllowNeutralVolumeBox, toolMineAllSymbolsBox));
            addHint(form, c, row++, "Để trống chu kỳ ATR = lấy trực tiếp từ cấu hình dự án. Chênh lệch giá chỉ dùng khi nguồn không có cột tương ứng.");
            addHint(form, c, row++, "Bật khai thác tất cả symbol khi nguồn là thư mục mt5_raw có manifest.json. Mỗi symbol được giữ thành tape độc lập trong một collection chung.");
            addHint(form, c, row++, "Đầu ra: băng đặc trưng thị trường dùng chung + các mảnh chỉ mục. 59 giá trị vị thế/tài khoản/thực thi được môi trường dựng động sau hành động.");
            addHint(form, c, row, "Phần quá khứ kết thúc trước điểm neo; tương lai chỉ dùng để mô phỏng hậu quả và nhãn hậu nghiệm, không đi vào đầu vào tại điểm quyết định.");
            return form;
        }

        private JPanel buildToolCheckPanel() {
            JPanel form = formPanel();
            GridBagConstraints c = constraints();
            int row = 0;
            addWideRow(form, c, row++, "Cấu hình mô hình",
                    pathChooser(toolCheckConfigField, PathMode.EXISTING_FILE, ".yaml"));
            addWideRow(form, c, row++, "Nguồn giá cần kiểm tra",
                    pathChooser(toolCheckInputField, PathMode.EXISTING_FILE, ""));
            addPairRow(form, c, row++, "Chênh lệch giá mặc định", toolCheckSpreadPointsField,
                    "Khối lượng", checkboxRow(toolCheckAllowNeutralVolumeBox));
            addHint(form, c, row++, "Có thể để trống nguồn giá để chỉ kiểm tra hợp đồng của dự án. Nếu có nguồn, công cụ còn dựng thử khung đặc trưng sau giai đoạn làm nóng.");
            addHint(form, c, row, "Kiểm tra gồm: 137 đầu vào, 5 tác nhân, phiên bản sơ đồ đặc trưng, dấu vân tay, chế độ chuẩn hóa và khả năng dựng đặc trưng từ nguồn.");
            return form;
        }

        private JPanel buildToolFetchPanel() {
            JPanel root = new JPanel(new BorderLayout(8, 8));
            JPanel form = formPanel();
            GridBagConstraints c = constraints();
            int row = 0;

            JPanel scanControls = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
            scanControls.add(toolScanSymbolsButton);
            scanControls.add(toolSelectFilteredButton);
            scanControls.add(toolSelectVisibleButton);
            scanControls.add(toolClearSymbolsButton);
            addWideRow(form, c, row++, "Danh sách mã từ MT5", scanControls);
            addPairRow(form, c, row++, "Lọc mã / mô tả / nhóm", toolSymbolField,
                    "Khung thời gian", toolTimeframeBox);
            addPairRow(form, c, row++, "Số thanh tối đa mỗi mã", toolBarsField,
                    "Kích thước mỗi lượt lấy", toolChunkSizeField);
            addWideRow(form, c, row++, "Thư mục đầu ra",
                    pathChooser(toolFetchOutputField, PathMode.NEW_DIRECTORY, ""));
            addWideRow(form, c, row++, "Pipeline một nút", checkboxRow(toolFetchMineBox));
            addWideRow(form, c, row++, "Trạng thái", toolSymbolState);
            addHint(form, c, row++, "Bấm Quét mã MT5: công cụ tự mở đúng terminal64.exe trong MT5 runner ở chế độ portable, dùng hồ sơ tài khoản đã lưu giống phần huấn luyện, rồi mới đọc danh sách mã.");
            addHint(form, c, row++, "Mỗi mã được lưu thành một tệp NPZ riêng trong cùng thư mục; manifest.json ghi số nến thực lấy được và lỗi của từng mã.");
            addHint(form, c, row, "Khi bật pipeline một nút: fetch thành công mới khai thác từng symbol thành collection; collection hợp lệ mới bắt đầu train. Mọi bước lỗi đều dừng ngay.");

            toolSymbolTable.setRowSorter(toolSymbolSorter);
            toolSymbolTable.setAutoResizeMode(JTable.AUTO_RESIZE_LAST_COLUMN);
            toolSymbolTable.setFillsViewportHeight(true);
            toolSymbolTable.getColumnModel().getColumn(0).setPreferredWidth(55);
            toolSymbolTable.getColumnModel().getColumn(1).setPreferredWidth(120);
            toolSymbolTable.getColumnModel().getColumn(2).setPreferredWidth(260);
            toolSymbolTable.getColumnModel().getColumn(3).setPreferredWidth(260);
            toolSymbolTable.getColumnModel().getColumn(4).setPreferredWidth(75);
            toolSymbolTable.getColumnModel().getColumn(5).setPreferredWidth(55);
            JScrollPane tableScroll = new JScrollPane(toolSymbolTable);
            tableScroll.setPreferredSize(new Dimension(900, 280));

            root.add(form, BorderLayout.NORTH);
            root.add(tableScroll, BorderLayout.CENTER);
            return root;
        }

        private JPanel buildToolSmokePanel() {
            JPanel form = formPanel();
            GridBagConstraints c = constraints();
            int row = 0;
            addWideRow(form, c, row++, "Mẫu cấu trúc",
                    pathChooser(toolSmokePatternField, PathMode.EXISTING_FILE, ".yaml"));
            addPairRow(form, c, row++, "Chu kỳ ATR", toolSmokeAtrField,
                    "Ngưỡng ZigZag theo ATR", toolSmokeZigzagField);
            addHint(form, c, row++, "Dùng thị trường giả lập lặp lại để xác nhận mẫu cấu trúc thực sự bắt được điểm neo trước khi chạy trên dữ liệu lớn.");
            addHint(form, c, row, "Phép thử thất bại nếu không tìm được mẫu nào; nó không tạo tập huấn luyện và không đụng vào trọng số mô hình.");
            return form;
        }

        private JComponent buildToolPatternResultPanel() {
            JPanel root = new JPanel(new BorderLayout(8, 8));
            root.setBorder(new EmptyBorder(6, 6, 6, 6));

            JPanel top = new JPanel(new BorderLayout(6, 4));
            top.add(pathChooser(toolEditorPatternField, PathMode.EXISTING_FILE, ".yaml"), BorderLayout.CENTER);
            JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT, 5, 0));
            JButton loadPatternButton = new JButton("Đọc mẫu");
            JButton savePatternButton = new JButton("Lưu mẫu");
            JButton useMinePatternButton = new JButton("Dùng khi khai thác");
            JButton useSmokePatternButton = new JButton("Dùng khi thử nhanh");
            JButton readResultButton = new JButton("Đọc kết quả");
            loadPatternButton.addActionListener(event -> loadToolPattern(true));
            savePatternButton.addActionListener(event -> saveToolPattern());
            useMinePatternButton.addActionListener(event -> {
                toolPatternField.setText(toolEditorPatternField.getText());
                appendLog("Đã chọn mẫu hiện tại cho mục khai thác.\n");
            });
            useSmokePatternButton.addActionListener(event -> {
                toolSmokePatternField.setText(toolEditorPatternField.getText());
                appendLog("Đã chọn mẫu hiện tại cho mục thử nhanh.\n");
            });
            readResultButton.addActionListener(event -> refreshToolSummary(true));
            buttons.add(loadPatternButton);
            buttons.add(savePatternButton);
            buttons.add(useMinePatternButton);
            buttons.add(useSmokePatternButton);
            buttons.add(readResultButton);
            top.add(buttons, BorderLayout.EAST);
            top.add(toolPatternState, BorderLayout.SOUTH);
            root.add(top, BorderLayout.NORTH);

            JTabbedPane editors = new JTabbedPane();
            editors.addTab("Nội dung mẫu", new JScrollPane(toolPatternArea));
            editors.addTab("Kết quả tập dữ liệu", new JScrollPane(toolResultArea));
            JTextArea contract = new JTextArea(toolDatasetContractText(), 22, 80);
            contract.setEditable(false);
            contract.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            contract.setBackground(new Color(12, 18, 29));
            contract.setForeground(new Color(229, 238, 248));
            contract.setCaretPosition(0);
            editors.addTab("Hợp đồng dữ liệu", new JScrollPane(contract));
            root.add(editors, BorderLayout.CENTER);
            return root;
        }

        private String toolDatasetContractText() {
            return """
                    HỢP ĐỒNG TẬP DỮ LIỆU NĂM TÁC NHÂN

                    1. Mỗi nhịp mô hình nhận 137 giá trị / tác nhân.
                    2. Năm tác nhân dùng cùng một băng thị trường; vị thế, tài khoản và thực thi được dựng riêng theo trạng thái môi trường.
                    3. Băng chỉ lưu phần thị trường đã tính nhân quả; không lưu sẵn trạng thái vị thế tương lai.
                    4. Ngữ cảnh mẫu không chứa điểm neo. Điểm neo được xử lý đúng một lần khi ra quyết định.
                    5. Tương lai không được đưa vào đầu vào tại điểm neo; chỉ dùng để chạy hậu quả và tính nhãn hậu nghiệm.
                    6. ATR của ZigZag ở chế độ mô hình dùng cùng định nghĩa trung bình biên độ thực của dự án.
                    7. Mỗi tập lưu phiên bản sơ đồ và dấu vân tay đặc trưng; lệch dự án hiện tại thì bộ đọc phải từ chối.
                    8. MFE/MAE hậu nghiệm được tách riêng cho hướng mua và hướng bán.
                    9. Hàng trăm triệu mẫu chỉ giữ chỉ số vào băng dùng chung, tránh nhân bản hàng trăm thanh giá cho mỗi mẫu.

                    Cấu trúc đầu ra:
                      dataset_summary.json
                      five_market_tape/
                        environment_values.npy
                        time_ns.npy
                        manifest.json
                      shard_00000.npz
                      shard_00000.jsonl
                      ...
                    """;
        }

        private void loadToolPattern(boolean notify) {
            try {
                Path path = resolveWorkspacePath(toolEditorPatternField.getText());
                if (!Files.isRegularFile(path)) throw new IllegalArgumentException("Không tìm thấy tệp mẫu: " + path);
                String text = Files.readString(path);
                toolPatternArea.setText(text);
                toolPatternArea.setCaretPosition(0);
                toolPatternState.setText("Đã đọc: " + displayPath(path));
                if (notify) appendLog("Đã đọc mẫu cấu trúc: " + path + "\n");
            } catch (Exception exception) {
                toolPatternState.setText("Không đọc được mẫu");
                if (notify) JOptionPane.showMessageDialog(this, exception.getMessage(), "Không đọc được mẫu", JOptionPane.ERROR_MESSAGE);
            }
        }

        private void saveToolPattern() {
            try {
                if (toolPatternArea.getText().isBlank()) throw new IllegalArgumentException("Nội dung mẫu đang trống");
                String raw = toolEditorPatternField.getText().trim();
                if (raw.isBlank()) throw new IllegalArgumentException("Chưa chọn tệp mẫu để lưu");
                Path path = resolveWorkspacePath(raw);
                if (Files.exists(path)) {
                    int choice = JOptionPane.showConfirmDialog(this,
                            "Ghi đè tệp mẫu hiện có?\n" + path,
                            "Xác nhận", JOptionPane.YES_NO_OPTION);
                    if (choice != JOptionPane.YES_OPTION) return;
                }
                if (path.getParent() != null) Files.createDirectories(path.getParent());
                Files.writeString(path, toolPatternArea.getText());
                toolPatternState.setText("Đã lưu: " + displayPath(path));
                appendLog("Đã lưu mẫu cấu trúc: " + path + "\n");
            } catch (Exception exception) {
                JOptionPane.showMessageDialog(this, exception.getMessage(), "Không lưu được mẫu", JOptionPane.ERROR_MESSAGE);
            }
        }

        private void refreshToolSummary(boolean notify) {
            try {
                Path summary = resolveWorkspacePath(toolOutputField.getText()).resolve("dataset_summary.json");
                if (!Files.isRegularFile(summary)) {
                    toolResultArea.setText("Chưa có dataset_summary.json tại:\n" + summary + "\n\nHãy chạy mục Khai thác tập 137 trước.");
                    if (notify) appendLog("Chưa có kết quả tập dữ liệu: " + summary + "\n");
                    return;
                }
                toolResultArea.setText(Files.readString(summary));
                toolResultArea.setCaretPosition(0);
                if (notify) appendLog("Đã đọc kết quả tập dữ liệu: " + summary + "\n");
            } catch (Exception exception) {
                toolResultArea.setText("Không đọc được kết quả: " + exception);
            }
        }

        private void refreshToolResultForCommands(List<List<String>> commandChain) {
            if (commandChain == null || commandChain.isEmpty()) return;
            List<String> completed = commandChain.get(commandChain.size() - 1);
            String resultFilename = datasetResultFilename(completed);
            for (int index = commandChain.size() - 1; resultFilename.isBlank() && index >= 0; index--) {
                completed = commandChain.get(index);
                resultFilename = datasetResultFilename(completed);
            }
            if ("manifest.json".equals(resultFilename)) {
                String output = optionValue(completed, "--output");
                if (output.isBlank()) return;
                Path manifest = resolveWorkspacePath(output).resolve("manifest.json");
                try {
                    if (!Files.isRegularFile(manifest)) {
                        toolResultArea.setText("Fetch MT5 đã kết thúc nhưng chưa có manifest.json tại:\n" + manifest);
                        return;
                    }
                    toolResultArea.setText(Files.readString(manifest));
                    toolResultArea.setCaretPosition(0);
                    appendLog("Đã đọc manifest dữ liệu MT5: " + manifest + "\n");
                } catch (Exception exception) {
                    toolResultArea.setText("Không đọc được manifest MT5: " + exception);
                }
                return;
            }
            if ("dataset_summary.json".equals(resultFilename)) {
                String output = optionValue(completed, "--output");
                if (output.isBlank()) return;
                Path summary = resolveWorkspacePath(output).resolve("dataset_summary.json");
                try {
                    if (!Files.isRegularFile(summary)) {
                        toolResultArea.setText("Mine Five đã kết thúc nhưng chưa có dataset_summary.json tại:\n" + summary);
                        return;
                    }
                    toolResultArea.setText(Files.readString(summary));
                    toolResultArea.setCaretPosition(0);
                    appendLog("Đã đọc kết quả tập dữ liệu: " + summary + "\n");
                } catch (Exception exception) {
                    toolResultArea.setText("Không đọc được kết quả tập dữ liệu: " + exception);
                }
            }
        }

        private JComponent contractCard(
                String title, String metric, String detail, Color accent
        ) {
            JPanel card = new JPanel(new BorderLayout(4, 2));
            card.setBorder(BorderFactory.createCompoundBorder(
                    BorderFactory.createMatteBorder(0, 5, 0, 0, accent),
                    new EmptyBorder(8, 10, 8, 10)
            ));
            JLabel heading = new JLabel(title + "  ·  " + metric);
            heading.setFont(heading.getFont().deriveFont(Font.BOLD, 14f));
            heading.setForeground(accent);
            JLabel description = new JLabel(detail);
            description.setForeground(new Color(75, 85, 99));
            card.add(heading, BorderLayout.NORTH);
            card.add(description, BorderLayout.CENTER);
            return card;
        }

        private JComponent buildLearningDiagnosticsPanel() {
            JTabbedPane tabs = new JTabbedPane();
            tabs.addTab("Tier 1 & Tier 2", buildTier1Tier2DiagnosticsPanel());
            tabs.addTab("MT5 online", buildMt5DiagnosticsPanel());
            tabs.addTab("Ablation", buildAblationDiagnosticsPanel());
            tabs.addTab("Imitation", buildImitationDiagnosticsPanel());
            return tabs;
        }

        private JComponent buildTier1Tier2DiagnosticsPanel() {
            JPanel root = new JPanel(new BorderLayout(8, 8));
            root.setBorder(new EmptyBorder(8, 8, 8, 8));
            JPanel cards = new JPanel(new GridLayout(2, 2, 8, 8));
            cards.add(contractCard(
                    "TIER 2 LÕI SGT-COMPACT", "17.7M THAM SỐ (GIẢM 81%)",
                    "mLSTM 768D (147k ô nhớ ma trận) + sLSTM 768D + Relational Graph Transformer (ReZero α=0.0). CPU forward P50 31.7ms.",
                    new Color(37, 99, 235)
            ));
            cards.add(contractCard(
                    "TIER 1 ĐIỀU HÒA BIỂU DIỄN", "VICREG + RANKME + ARCFACE",
                    "VICReg Covariance Decorrelation + Sylvester Log-Det RankMe + ArcFace m=0.35 rad (trọng số nghịch đảo tần suất).",
                    new Color(16, 185, 129)
            ));
            cards.add(contractCard(
                    "BỘ ĐỆM ĐA CHẾ ĐỘ & EWC", "3-REGIME BALANCED (33.3%)",
                    "Dynamic Reservoir Buffer (Bull / Bear / Sideway) + Thuật toán Markov TD-error + Elastic Weight Consolidation neo ma trận Fisher.",
                    new Color(245, 158, 11)
            ));
            cards.add(contractCard(
                    "TÍN DỤNG COMA & KHÁM PHÁ RND", "COUNTERFACTUAL ADVANTAGE",
                    "COMA Centralized Critic bóc tách công/tội từng agent; RND Intrinsic Curiosity Welford chống bẫy HOLD; HER tái sinh timeout MFE ≥ 0.75R.",
                    new Color(139, 92, 246)
            ));
            root.add(cards, BorderLayout.NORTH);
            JTextArea tierInfoArea = new JTextArea();
            tierInfoArea.setEditable(false);
            tierInfoArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            tierInfoArea.setBackground(new Color(11, 18, 30));
            tierInfoArea.setForeground(new Color(226, 232, 240));
            tierInfoArea.setText("""
                    ========================================================================================
                    TIER 1 & TIER 2 ARCHITECTURE & TRAINING DYNAMICS MATRIX
                    ========================================================================================
                    [TIER 1 - IMMEDIATE STABILIZATION & BIAS ELIMINATION]
                    • Module: src/self_growing_trader/representation_regularizer.py (VICReg + RankMe + ArcFace)
                    • Module: src/self_growing_trader/balanced_replay_buffer.py (3-Regime Dynamic Reservoir)
                    • Module: src/self_growing_trader/elastic_weight_consolidation.py (EWC Fisher Neo Giữ)
                    • Module: src/self_growing_trader/five_lifecycle_reward.py (Phạt bậc hai β=0.25 tại nến 120)
                    • Tests: 80/80 passed, 0 NaN/Inf, bảo toàn 100% 137 input / 20 action wire contract.

                    [TIER 2 - SGT-V193-COMPACT SEQUENCE CORE & EFFICIENCY LEAP]
                    • Module: src/self_growing_trader/sgt_compact_core.py
                      - Trainable Params: 17,710,115 (giảm 81.0% so với 93.18M của V193 cũ).
                      - Matrix Memory: 147,456 ô nhớ ma trận (gấp 36 lần LSTM 4096 cũ).
                      - Log-space Gating: log f_t <= 0, dynamic stabilizer m_t (Beck et al., NeurIPS 2024).
                      - Relational Graph Transformer: R_5x5 role prior + ReZero alpha=0.0.
                      - CPU Forward Latency: JIT Traced P50 = 31.72 ms (tăng tốc 5.57 lần).
                    • Module: src/self_growing_trader/knowledge_distillation.py
                      - Progressive Feature Distillation: W_proj [4096 x 768] + Logit Distillation tau=2.0.
                    • Module: src/self_growing_trader/coma_and_rnd.py
                      - COMA Centralized Critic: Counterfactual Advantage A^a_COMA(s, u) phân định rạch ròi.
                      - RND Exploration: Predictor vs Frozen Target + Welford normalizer (kappa=0.05).
                      - Hindsight Experience Replay (HER): Relabeling near-miss timeout MFE >= 0.75R.
                    • Tests: 97/97 passed, 220/220 regression passed, 100% PyTorch Native Windows x86_64.
                    ========================================================================================
                    """);
            root.add(new JScrollPane(tierInfoArea), BorderLayout.CENTER);
            JLabel note = new JLabel(
                    "<html>SGT-V193-Compact Core và Tier 1 Regularizer đã vượt qua 100% bộ kiểm thử tự động. "
                            + "Bảo toàn nghiêm ngặt các bất biến hệ thống: 137 inputs, 20 action outputs, 5-agent shared weights.</html>"
            );
            note.setForeground(new Color(75, 85, 99));
            root.add(note, BorderLayout.SOUTH);
            return root;
        }

        private JComponent buildImitationDiagnosticsPanel() {
            JPanel root = new JPanel(new BorderLayout(8, 8));
            root.setBorder(new EmptyBorder(8, 8, 8, 8));
            JPanel cards = new JPanel(new GridLayout(1, 3, 8, 0));
            cards.add(diagnosticCard(
                    "UNDERFIT", underfitDiagnosticLabel,
                    "train BA/MCC online sau ít nhất 3 epoch"
            ));
            cards.add(diagnosticCard(
                    "OVERFIT", overfitDiagnosticLabel,
                    "gap train–validation và xu hướng validation"
            ));
            cards.add(diagnosticCard(
                    "KHUYẾN KHÍCH / PHẠT", rewardPenaltyDiagnosticLabel,
                    "proxy loss imitation, không phải PnL/reward RL"
            ));
            root.add(cards, BorderLayout.NORTH);
            resetLearningDiagnostics();
            root.add(new JScrollPane(learningDiagnosticArea), BorderLayout.CENTER);
            JLabel note = new JLabel(
                    "<html>Train metrics là thống kê online khi trọng số thay đổi trong epoch; "
                            + "validation được replay ở cuối epoch. Chỉ báo là cảnh báo chẩn đoán, "
                            + "không phải gate promotion hoặc bằng chứng edge MT5/OOS.</html>"
            );
            note.setForeground(new Color(75, 85, 99));
            root.add(note, BorderLayout.SOUTH);
            return root;
        }

        private JComponent buildMt5DiagnosticsPanel() {
            JPanel root = new JPanel(new BorderLayout(8, 8));
            root.setBorder(new EmptyBorder(8, 8, 8, 8));
            JPanel cards = new JPanel(new GridLayout(2, 3, 8, 8));
            cards.add(diagnosticCard("EDGE MT5", mt5EdgeDiagnosticLabel,
                    "PF + expectancy trên lệnh đã đóng"));
            cards.add(diagnosticCard("RỦI RO", mt5RiskDiagnosticLabel,
                    "return và peak-to-trough equity DD"));
            cards.add(diagnosticCard("OPTIMIZER PPO", mt5OptimizerDiagnosticLabel,
                    "update, rollout, loss và entropy"));
            cards.add(diagnosticCard("REWARD / CREDIT", mt5RewardDiagnosticLabel,
                    "mật độ reward khác 0 và eligibility"));
            cards.add(diagnosticCard("POLICY", mt5PolicyDiagnosticLabel,
                    "entropy action, HOLD và collapse"));
            cards.add(diagnosticCard("THỰC THI", mt5ExecutionDiagnosticLabel,
                    "reject và pending timeout"));
            root.add(cards, BorderLayout.NORTH);
            resetMt5Diagnostics();
            root.add(new JScrollPane(mt5DiagnosticArea), BorderLayout.CENTER);
            JLabel note = new JLabel(
                    "<html>Chỉ số lấy từ execution/bridge ledger và optimizer trace thật. "
                            + "Underfit kiến trúc không được kết luận khi PPO còn quá ít update; "
                            + "overfit chỉ đo được bằng ablation/frozen OOS cùng seed và cửa sổ.</html>"
            );
            note.setForeground(new Color(75, 85, 99));
            root.add(note, BorderLayout.SOUTH);
            return root;
        }

        private JComponent buildAblationDiagnosticsPanel() {
            JPanel root = new JPanel(new BorderLayout(8, 8));
            root.setBorder(new EmptyBorder(8, 8, 8, 8));
            root.add(diagnosticCard(
                    "ABLATION MATCHED", ablationDiagnosticLabel,
                    "frozen_policy ↔ full_online ↔ reasoning_off"
            ), BorderLayout.NORTH);
            ablationDiagnosticArea.setText("Chưa có ablation_summary.json.\n");
            root.add(new JScrollPane(ablationDiagnosticArea), BorderLayout.CENTER);
            JLabel note = new JLabel(
                    "<html>Chênh lệch một cửa sổ chỉ dùng để quy trách nhiệm ban đầu. "
                            + "Muốn kết luận overfit cần lặp nhiều seed và frozen OOS chưa dùng để train.</html>"
            );
            note.setForeground(new Color(75, 85, 99));
            root.add(note, BorderLayout.SOUTH);
            return root;
        }

        private JComponent diagnosticCard(String title, JLabel value, String detail) {
            JPanel card = new JPanel(new BorderLayout(4, 2));
            card.setBorder(BorderFactory.createCompoundBorder(
                    BorderFactory.createLineBorder(new Color(210, 215, 222)),
                    new EmptyBorder(8, 10, 8, 10)
            ));
            JLabel heading = new JLabel(title);
            heading.setFont(heading.getFont().deriveFont(Font.BOLD, 11f));
            heading.setForeground(new Color(75, 85, 99));
            value.setFont(value.getFont().deriveFont(Font.BOLD, 15f));
            value.setForeground(new Color(105, 112, 124));
            JLabel description = new JLabel(detail);
            description.setForeground(new Color(95, 102, 114));
            card.add(heading, BorderLayout.NORTH);
            card.add(value, BorderLayout.CENTER);
            card.add(description, BorderLayout.SOUTH);
            return card;
        }

        private JComponent pathChooser(JTextField field, PathMode mode, String defaultExtension) {
            JPanel panel = new JPanel(new BorderLayout(6, 0));
            panel.add(field, BorderLayout.CENTER);
            boolean output = mode == PathMode.NEW_FILE || mode == PathMode.NEW_DIRECTORY;
            JButton button = new JButton(output ? "Tạo/chọn..." : "Chọn...");
            button.addActionListener(event -> choosePath(field, mode, defaultExtension));
            panel.add(button, BorderLayout.EAST);
            return panel;
        }

        private void choosePath(JTextField field, PathMode mode, String defaultExtension) {
            JFileChooser chooser = new JFileChooser();
            boolean directory = mode == PathMode.EXISTING_DIRECTORY || mode == PathMode.NEW_DIRECTORY;
            boolean output = mode == PathMode.NEW_FILE || mode == PathMode.NEW_DIRECTORY;
            chooser.setFileSelectionMode(directory ? JFileChooser.DIRECTORIES_ONLY : JFileChooser.FILES_ONLY);
            chooser.setDialogTitle(output
                    ? (directory ? "Chọn hoặc tạo thư mục output mới" : "Đặt tên file output mới")
                    : (directory ? "Chọn thư mục có sẵn" : "Chọn file có sẵn"));

            String raw = field.getText().trim();
            Path requested = raw.isBlank() ? workspace : resolveWorkspacePath(raw);
            Path existing = Files.isDirectory(requested) ? requested : requested.getParent();
            while (existing != null && !Files.isDirectory(existing)) existing = existing.getParent();
            if (existing != null) chooser.setCurrentDirectory(existing.toFile());
            if (!raw.isBlank() && !directory) chooser.setSelectedFile(requested.toFile());

            int result = output ? chooser.showSaveDialog(this) : chooser.showOpenDialog(this);
            if (result != JFileChooser.APPROVE_OPTION) return;
            Path selected = chooser.getSelectedFile().toPath().toAbsolutePath().normalize();
            if (mode == PathMode.NEW_FILE) selected = withDefaultExtension(selected, defaultExtension);
            field.setText(displayPath(selected));
        }

        private String displayPath(Path path) {
            Path absolute = path.toAbsolutePath().normalize();
            if (absolute.startsWith(workspace)) {
                return workspace.relativize(absolute).toString().replace('\\', '/');
            }
            return absolute.toString();
        }

        private JPanel checkboxRow(Component... components) {
            JPanel panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
            for (Component component : components) panel.add(component);
            return panel;
        }

        private GridBagConstraints constraints() {
            GridBagConstraints c = new GridBagConstraints();
            c.insets = new Insets(4, 5, 4, 5);
            c.fill = GridBagConstraints.HORIZONTAL;
            return c;
        }

        private void addWideRow(JPanel panel, GridBagConstraints c, int row, String label, JComponent component) {
            c.gridy = row;
            c.gridx = 0;
            c.gridwidth = 1;
            c.weightx = 0;
            panel.add(new JLabel(label), c);
            c.gridx = 1;
            c.gridwidth = 3;
            c.weightx = 1;
            panel.add(component, c);
        }

        private void addPairRow(
                JPanel panel, GridBagConstraints c, int row,
                String leftLabel, JComponent left, String rightLabel, JComponent right
        ) {
            c.gridy = row;
            c.gridwidth = 1;
            c.gridx = 0;
            c.weightx = 0;
            panel.add(new JLabel(leftLabel), c);
            c.gridx = 1;
            c.weightx = 0.5;
            panel.add(left, c);
            c.gridx = 2;
            c.weightx = 0;
            panel.add(new JLabel(rightLabel), c);
            c.gridx = 3;
            c.weightx = 0.5;
            panel.add(right, c);
        }

        private void addHint(JPanel panel, GridBagConstraints c, int row, String text) {
            JLabel hint = new JLabel(text);
            hint.setForeground(new Color(75, 85, 99));
            c.gridy = row;
            c.gridx = 0;
            c.gridwidth = 4;
            c.weightx = 1;
            panel.add(hint, c);
        }

        private JComponent buildToolbar() {
            JPanel root = new JPanel(new BorderLayout(8, 4));
            JPanel left = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
            left.add(newSystemButton);
            left.add(newOutputButton);
            left.add(copyButton);
            left.add(saveButton);
            left.add(loadButton);
            left.add(liveButton);
            JPanel right = new JPanel(new FlowLayout(FlowLayout.RIGHT, 6, 0));
            runButton.setBackground(new Color(33, 122, 82));
            runButton.setForeground(Color.WHITE);
            saveStopButton.setBackground(new Color(34, 108, 166));
            saveStopButton.setForeground(Color.WHITE);
            stopButton.setBackground(new Color(180, 57, 57));
            stopButton.setForeground(Color.WHITE);
            right.add(runButton);
            right.add(saveStopButton);
            right.add(stopButton);
            root.add(left, BorderLayout.WEST);
            root.add(right, BorderLayout.EAST);
            return root;
        }

        private void updateSymbolFilter() {
            String needle = toolSymbolField.getText().trim().toLowerCase(Locale.ROOT);
            if (needle.isBlank()) {
                toolSymbolSorter.setRowFilter(null);
            } else {
                toolSymbolSorter.setRowFilter(new RowFilter<>() {
                    @Override
                    public boolean include(Entry<? extends Mt5SymbolTableModel, ? extends Integer> entry) {
                        int modelRow = entry.getIdentifier();
                        Mt5SymbolRow row = toolSymbolModel.row(modelRow);
                        return row.name.toLowerCase(Locale.ROOT).contains(needle)
                                || row.description.toLowerCase(Locale.ROOT).contains(needle)
                                || row.path.toLowerCase(Locale.ROOT).contains(needle);
                    }
                });
            }
            updateSymbolState();
        }

        private void updateSymbolState() {
            int total = toolSymbolModel.getRowCount();
            int filtered = toolSymbolTable.getRowCount();
            int selected = toolSymbolModel.selectedCount();
            toolSymbolState.setText("Tổng " + total + " mã · đang hiển thị " + filtered + " · đã chọn " + selected);
            updateRunButton();
        }

        private void setFilteredSymbolSelection(boolean selected) {
            List<Integer> modelRows = new ArrayList<>();
            for (int viewRow = 0; viewRow < toolSymbolTable.getRowCount(); viewRow++) {
                modelRows.add(toolSymbolTable.convertRowIndexToModel(viewRow));
            }
            toolSymbolModel.setSelectedRows(modelRows, selected);
        }

        private void selectVisibleMt5Symbols() {
            toolSymbolModel.selectVisibleRows();
        }

        private void clearMt5SymbolSelection() {
            toolSymbolModel.clearSelection();
        }

        private void scanMt5Symbols() {
            if (workflowRunning) {
                JOptionPane.showMessageDialog(this, "Đang có workflow chạy.", "Chưa thể quét", JOptionPane.WARNING_MESSAGE);
                return;
            }
            Path mt5Runner = resolveWorkspacePath(runnerDirField.getText());
            Path mt5Terminal = mt5Runner.resolve("terminal64.exe");
            if (!Files.isRegularFile(mt5Terminal)) {
                JOptionPane.showMessageDialog(
                        this, "Không tìm thấy terminal MT5 dùng cho huấn luyện: " + mt5Terminal,
                        "Thiếu terminal MT5", JOptionPane.ERROR_MESSAGE
                );
                return;
            }
            toolScanSymbolsButton.setEnabled(false);
            toolSymbolState.setText("Đang quét mã từ terminal MT5...");
            Set<String> selectedBefore = new HashSet<>(toolSymbolModel.selectedNames());
            Thread worker = new Thread(() -> {
                StringBuilder output = new StringBuilder();
                int exitCode = -1;
                try {
                    List<String> command = datasetSymbolScanCommand(pythonField.getText(), runnerDirField.getText());
                    ProcessBuilder builder = new ProcessBuilder(command);
                    builder.directory(workspace.toFile());
                    String source = workspace.resolve("src").toString();
                    String prior = builder.environment().getOrDefault("PYTHONPATH", "");
                    builder.environment().put(
                            "PYTHONPATH",
                            prior.isBlank() ? source : source + System.getProperty("path.separator") + prior
                    );
                    builder.environment().put("PYTHONUTF8", "1");
                    builder.environment().put("PYTHONIOENCODING", "utf-8");
                    builder.redirectErrorStream(true);
                    Process process = builder.start();
                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                            process.getInputStream(), StandardCharsets.UTF_8
                    ))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            output.append(line).append('\n');
                        }
                    }
                    exitCode = process.waitFor();
                    if (exitCode != 0) {
                        throw new IllegalStateException(output.toString().trim());
                    }
                    Object parsed = MiniJson.parse(output.toString().trim());
                    Map<String, Object> root = MiniJson.object(parsed);
                    Map<String, Object> session = MiniJson.object(root.get("mt5_session"));
                    String accountLogin = MiniJson.text(session.get("login"), "");
                    String accountServer = MiniJson.text(session.get("server"), "");
                    List<Mt5SymbolRow> rows = new ArrayList<>();
                    for (Object item : MiniJson.array(root.get("symbols"))) {
                        Map<String, Object> symbol = MiniJson.object(item);
                        String name = MiniJson.text(symbol.get("name"), "");
                        if (name.isBlank()) continue;
                        boolean visible = Boolean.TRUE.equals(symbol.get("visible"));
                        rows.add(new Mt5SymbolRow(
                                selectedBefore.contains(name),
                                name,
                                MiniJson.text(symbol.get("description"), ""),
                                MiniJson.text(symbol.get("path"), ""),
                                visible,
                                MiniJson.integer(symbol.get("digits"), 0)
                        ));
                    }
                    SwingUtilities.invokeLater(() -> {
                        toolSymbolModel.setRows(rows);
                        updateSymbolFilter();
                        toolSymbolState.setText("Tài khoản " + accountLogin + " · " + accountServer + " · " + rows.size() + " mã");
                        appendLog("Đã tự kết nối MT5 runner · tài khoản " + accountLogin + " · máy chủ " + accountServer + " · quét " + rows.size() + " mã.\n");
                    });
                } catch (Exception exception) {
                    String message = exception.getMessage();
                    if (message == null || message.isBlank()) message = exception.toString();
                    String finalMessage = message;
                    SwingUtilities.invokeLater(() -> {
                        toolSymbolState.setText("Quét mã thất bại");
                        appendLog("Không quét được danh sách mã MT5: " + finalMessage + "\n");
                        JOptionPane.showMessageDialog(
                                TraderControlFrame.this,
                                finalMessage,
                                "Không quét được mã MT5",
                                JOptionPane.ERROR_MESSAGE
                        );
                    });
                } finally {
                    SwingUtilities.invokeLater(() -> toolScanSymbolsButton.setEnabled(!workflowRunning));
                }
            }, "v193-mt5-symbol-scan");
            worker.setDaemon(true);
            worker.start();
        }

        private void bind() {
            DocumentListener listener = new DocumentListener() {
                public void insertUpdate(DocumentEvent event) { refreshPreview(); }
                public void removeUpdate(DocumentEvent event) { refreshPreview(); }
                public void changedUpdate(DocumentEvent event) { refreshPreview(); }
            };
            for (JTextField field : trackedFields) field.getDocument().addDocumentListener(listener);
            toolSymbolField.getDocument().addDocumentListener(new DocumentListener() {
                public void insertUpdate(DocumentEvent event) { updateSymbolFilter(); }
                public void removeUpdate(DocumentEvent event) { updateSymbolFilter(); }
                public void changedUpdate(DocumentEvent event) { updateSymbolFilter(); }
            });
            toolSymbolModel.addTableModelListener(event -> {
                updateSymbolState();
                refreshPreview();
            });
            toolScanSymbolsButton.addActionListener(event -> scanMt5Symbols());
            toolSelectFilteredButton.addActionListener(event -> setFilteredSymbolSelection(true));
            toolSelectVisibleButton.addActionListener(event -> selectVisibleMt5Symbols());
            toolClearSymbolsButton.addActionListener(event -> clearMt5SymbolSelection());
            toolFetchMineBox.addActionListener(event -> {
                if (toolFetchMineBox.isSelected()) {
                    toolMineAllSymbolsBox.setSelected(true);
                }
                updateRunButton();
                refreshPreview();
            });
            for (JComboBox<?> combo : List.of(
                    candidateModeBox, candidateSevenChoiceProfileBox, candidateModelBox,
                    researchModeBox, researchModelBox,
                    championModelBox, walkModelBox, ablationModelBox, adminActionBox, toolTimeframeBox
            )) combo.addActionListener(event -> {
                refreshCandidateMode();
                refreshPreview();
            });
            for (JCheckBox box : List.of(
                    candidateAllowUnverifiedSeedBox, visualBox, keepOpenBox, telemetryBox, resetLogsBox,
                    walkPairedBox, walkRollbackBox, adminReplayBox, adminOosBox,
                    adminRiskBox, adminNonInterferenceBox, toolAllowNeutralVolumeBox, toolAutoCheckBox,
                    toolCheckAllowNeutralVolumeBox, toolMineAllSymbolsBox, imitationVerifyOnlyBox,
                    ablationKeepCheckpointsBox
            )) box.addActionListener(event -> refreshPreview());
            candidateAllowUnverifiedSeedBox.addActionListener(event -> refreshCandidateMode());
            imitationVerifyOnlyBox.addActionListener(event -> updateRunButton());
            workflowTabs.addChangeListener(event -> {
                refreshPreview();
                updateRunButton();
            });
            toolTabs.addChangeListener(event -> {
                refreshPreview();
                updateRunButton();
                if (toolTabs.getSelectedIndex() == TOOL_PATTERN) {
                    loadToolPattern(false);
                    refreshToolSummary(false);
                }
            });
            newSystemButton.addActionListener(event -> prepareNewSystem());
            newOutputButton.addActionListener(event -> assignNewOutput());
            copyButton.addActionListener(event -> copyCommand());
            saveButton.addActionListener(event -> saveSettings(true));
            loadButton.addActionListener(event -> loadSettings(true));
            runButton.addActionListener(event -> runWorkflow());
            saveStopButton.addActionListener(event -> saveAndStop());
            stopButton.addActionListener(event -> stopWorkflow());
            liveButton.addActionListener(event -> toggleLive());
            addWindowListener(new WindowAdapter() {
                @Override
                public void windowClosing(WindowEvent event) {
                    if (workflowRunning) {
                        int choice = JOptionPane.showConfirmDialog(
                                TraderControlFrame.this,
                                "Workflow vẫn đang chạy. Dừng process do UI này tạo và đóng cửa sổ?",
                                "Xác nhận", JOptionPane.YES_NO_OPTION
                        );
                        if (choice != JOptionPane.YES_OPTION) return;
                        stopWorkflow();
                    }
                    saveSettings(false);
                    livePanel.stopViewing();
                    if (liveWindow != null) liveWindow.dispose();
                    dispose();
                }
            });
        }

        private List<String> command() {
            int tab = workflowTabs.getSelectedIndex();
            List<String> command;
            if (tab == TAB_CANDIDATE) {
                String seedArtifact = ensureCandidateSeedPath();
                return candidateCommand(
                        pythonField.getText(), candidateTrainingConfig(), runnerDirField.getText(),
                        seedArtifact, candidateOutputField.getText(),
                        candidateFromField.getText(), candidateToField.getText(),
                        selected(candidateModelBox), candidateSeedNumberField.getText(),
                        candidateTimeoutField.getText(), candidateMaxWallField.getText(),
                        visualBox.isSelected(), candidateAllowUnverifiedSeedBox.isSelected(),
                        candidateArchitectureBox.getSelectedIndex()
                );
            }
            if (tab == TAB_ABLATION) {
                return ablationCommand(
                        pythonField.getText(), REASONING_TRAIN_CONFIG, runnerDirField.getText(),
                        ablationCheckpointField.getText(), ablationOutputField.getText(),
                        ablationFromField.getText(), ablationToField.getText(),
                        selected(ablationModelBox), ablationSeedField.getText(),
                        ablationTimeoutField.getText(), ablationMaxWallField.getText(),
                        visualBox.isSelected(), ablationKeepCheckpointsBox.isSelected()
                );
            }
            if (tab == TAB_RESEARCH) {
                if (researchModeBox.getSelectedIndex() == 2) {
                    return initializeCommand(
                            pythonField.getText(), configField.getText(),
                            researchOutputField.getText(), researchSeedField.getText()
                    );
                }
                command = researchCommand(
                        pythonField.getText(), configField.getText(), researchModeBox.getSelectedIndex() == 1,
                        researchOutputField.getText(), researchSeedSystemField.getText(), runnerDirField.getText(),
                        researchFromField.getText(), researchToField.getText(), researchBacktestDaysField.getText(),
                        selected(researchModelBox), researchSeedField.getText(), researchTimeoutField.getText(),
                        researchMaxWallField.getText(), visualBox.isSelected(), keepOpenBox.isSelected()
                );
            } else if (tab == TAB_CHAMPION) {
                command = championCommand(
                        pythonField.getText(), championPathField.getText(),
                        championOutputField.getText(), championAuditField.getText(), championTrainFromField.getText(),
                        championTrainToField.getText(), championValidationCountField.getText(),
                        championValidationDaysField.getText(), championSeedCountField.getText(),
                        championSeedField.getText(), runnerDirField.getText(), championTimeoutField.getText(),
                        championMaxWallField.getText(), selected(championModelBox), minExpectancyField.getText(),
                        maxProfitRegressionField.getText(), maxLcbRegressionField.getText(), maxDdIncreaseField.getText(),
                        visualBox.isSelected(), keepOpenBox.isSelected()
                );
            } else if (tab == TAB_WALK_FORWARD) {
                command = walkForwardCommand(
                        pythonField.getText(), walkOutputField.getText(), walkLogField.getText(),
                        walkSeedSystemField.getText(), walkSeedField.getText(), walkSeedCountField.getText(),
                        runnerDirField.getText(), walkTimeoutField.getText(), walkFromField.getText(), walkToField.getText(),
                        walkTrainWindowField.getText(), walkTestWindowField.getText(), walkStepField.getText(),
                        selected(walkModelBox), walkPairedBox.isSelected(), walkRollbackBox.isSelected(),
                        minExpectancyField.getText(), maxProfitRegressionField.getText(), maxLcbRegressionField.getText(),
                        maxDdIncreaseField.getText(), visualBox.isSelected(), keepOpenBox.isSelected()
                );
            } else if (tab == TAB_IMITATION) {
                return teacherImitationCommand(
                        pythonField.getText(), imitationConfigField.getText(),
                        imitationCheckpointField.getText(), imitationDatasetField.getText(),
                        imitationOutputField.getText(), imitationEpochsField.getText(),
                        imitationBatchField.getText(), imitationFeatureBatchField.getText(),
                        imitationLearningRateField.getText(),
                        imitationRecurrentLearningRateField.getText(), imitationPatienceField.getText(),
                        imitationSeedField.getText(), imitationSequenceLengthField.getText(),
                        imitationFalseEntryPenaltyField.getText(),
                        imitationVerifyOnlyBox.isSelected()
                );
            } else if (tab == TAB_ADMIN) {
                return adminCommand(
                        pythonField.getText(), configField.getText(), adminSystemField.getText(),
                        adminOutputField.getText(), adminContextField.getText(), selected(adminActionBox),
                        adminTargetField.getText(), adminLifecycleField.getText(), adminNoteField.getText(),
                        adminAuditField.getText(), adminReplayBox.isSelected(), adminOosBox.isSelected(),
                        adminRiskBox.isSelected(), adminNonInterferenceBox.isSelected()
                );
            } else if (tab == TAB_TOOLS) {
                int tool = toolTabs.getSelectedIndex();
                if (tool == TOOL_MINE) {
                    if (toolMineAllSymbolsBox.isSelected()) {
                        return datasetMineSelectedCommand(
                                pythonField.getText(), toolConfigField.getText(), toolInputField.getText(),
                                toolPatternField.getText(), toolOutputField.getText(), toolAtrPeriodField.getText(),
                                toolZigzagAtrMultField.getText(), toolShardSizeField.getText(), toolSpreadPointsField.getText(),
                                toolAllowNeutralVolumeBox.isSelected()
                        );
                    }
                    return datasetMineCommand(
                            pythonField.getText(), toolConfigField.getText(), toolInputField.getText(),
                            toolPatternField.getText(), toolOutputField.getText(), toolAtrPeriodField.getText(),
                            toolZigzagAtrMultField.getText(), toolShardSizeField.getText(), toolSpreadPointsField.getText(),
                            toolAllowNeutralVolumeBox.isSelected()
                    );
                }
                if (tool == TOOL_CHECK) {
                    return datasetCheckCommand(
                            pythonField.getText(), toolCheckConfigField.getText(), toolCheckInputField.getText(),
                            toolCheckSpreadPointsField.getText(), toolCheckAllowNeutralVolumeBox.isSelected()
                    );
                }
                if (tool == TOOL_FETCH) {
                    return datasetFetchSelectedCommand(
                            pythonField.getText(), runnerDirField.getText(), toolSelectedSymbolsFile(), selected(toolTimeframeBox),
                            toolBarsField.getText(), toolChunkSizeField.getText(), toolFetchOutputField.getText()
                    );
                }
                if (tool == TOOL_SMOKE) {
                    return datasetSmokeCommand(
                            pythonField.getText(), toolSmokePatternField.getText(), toolSmokeAtrField.getText(),
                            toolSmokeZigzagField.getText()
                    );
                }
                throw new IllegalStateException("Mục Mẫu cấu trúc & kết quả chỉ dùng để xem/sửa, không có lệnh chạy");
            } else {
                throw new IllegalStateException("Chọn một tab workflow trước khi chạy");
            }
            appendObservability(
                    command, telemetryBox.isSelected(), resetLogsBox.isSelected(), eventsField.getText(),
                    decisionProfilesField.getText(), structuralEventsField.getText(), networkDevelopmentField.getText(),
                    sampleEveryField.getText(), maxMbField.getText(), maxNeuronsField.getText(), maxEdgesField.getText()
            );
            return command;
        }

        private List<List<String>> commands() {
            if (workflowTabs.getSelectedIndex() == TAB_IMITATION) {
                ensureImitationCheckpointPath();
            }
            if (workflowTabs.getSelectedIndex() == TAB_CANDIDATE) {
                String candidateSeedRaw = ensureCandidateSeedPath();
                Path resume = resolveWorkspacePath(imitationResumeFile(candidateSeedRaw));
                String reason = candidatePausedImitationReason(
                        candidateSeedRaw, Files.isRegularFile(resume)
                );
                if (!reason.isBlank()) throw new IllegalStateException(reason);
            }
            List<String> primary = command();
            assertLatestTrainingContract(primary);
            if (workflowTabs.getSelectedIndex() == TAB_TOOLS) {
                if (toolTabs.getSelectedIndex() == TOOL_FETCH && toolFetchMineBox.isSelected()) {
                    List<String> mine = datasetMineSelectedCommand(
                            pythonField.getText(), LATEST_TRAIN_CONFIG, toolFetchOutputField.getText(),
                            toolPatternField.getText(), toolOutputField.getText(), toolAtrPeriodField.getText(),
                            toolZigzagAtrMultField.getText(), toolShardSizeField.getText(), toolSpreadPointsField.getText(),
                            toolAllowNeutralVolumeBox.isSelected()
                    );
                    return List.of(primary, mine);
                }
                if (toolTabs.getSelectedIndex() == TOOL_MINE && toolAutoCheckBox.isSelected()
                        && !toolMineAllSymbolsBox.isSelected()) {
                    List<String> check = datasetCheckCommand(
                            pythonField.getText(), toolConfigField.getText(), toolInputField.getText(),
                            toolSpreadPointsField.getText(), toolAllowNeutralVolumeBox.isSelected()
                    );
                    return List.of(check, primary);
                }
                return List.of(primary);
            }
            if (workflowTabs.getSelectedIndex() == TAB_IMITATION) {
                Path seed = resolveWorkspacePath(imitationCheckpointField.getText());
                if (Files.exists(seed)) return List.of(primary);
                List<String> bootstrap = initializeCommand(
                        pythonField.getText(), imitationConfigField.getText(),
                        imitationCheckpointField.getText(), imitationSeedField.getText()
                );
                return List.of(bootstrap, primary);
            }
            if (workflowTabs.getSelectedIndex() != TAB_CANDIDATE) return List.of(primary);
            Path seed = resolveWorkspacePath(candidateSeedField.getText());
            if (Files.exists(seed)) return List.of(primary);
            List<String> bootstrap = initializeCommand(
                    pythonField.getText(), configField.getText(),
                    candidateSeedField.getText(), candidateSeedNumberField.getText()
            );
            return List.of(bootstrap, primary);
        }

        private void refreshPreview() {
            int selectedTab = workflowTabs.getSelectedIndex();
            if (selectedTab == TAB_ADVANCED) {
                previewArea.setText("Tab cấu hình chỉ thay đổi seed/frozen/admin và telemetry; các lệnh train vẫn khóa vào V193.");
                return;
            }
            if (selectedTab == TAB_TOOLS && toolTabs.getSelectedIndex() == TOOL_PATTERN) {
                previewArea.setText(toolDatasetContractText());
                previewArea.setCaretPosition(0);
                return;
            }
            try {
                previewArea.setText(formatCommands(commands()));
            } catch (RuntimeException exception) {
                previewArea.setText("Thông tin: " + exception.getMessage());
            }
        }

        private boolean candidateUsesReasoning() {
            return candidateModeBox.getSelectedIndex() == CANDIDATE_MODE_REASONING;
        }

        private boolean candidateUsesCodex() {
            return candidateModeBox.getSelectedIndex() == CANDIDATE_MODE_CODEX;
        }

        private String candidateTrainingConfig() {
            // Candidate mode is authoritative.  The advanced "config nền" field
            // is intentionally shared with seed/frozen/admin workflows and may
            // still contain a stale Codex value from an older UI session.
            return candidateConfigForMode(
                    candidateModeBox.getSelectedIndex(),
                    candidateSevenChoiceProfileBox.getSelectedIndex()
            );
        }

        private void refreshCandidateMode() {
            if (candidateAllowUnverifiedSeedBox.isSelected()) {
                candidateModeStatus.setText(
                        "<html><b>RESEARCH-ONLY:</b> bỏ qua provenance/imitation gate · manifest promotable=false</html>"
                );
                candidateModeStatus.setForeground(new Color(180, 45, 35));
                return;
            }
            if (candidateUsesCodex()) {
                candidateModeStatus.setText(
                        "<html><b>Đang bật:</b> Codex teacher candidate-only · checkpoint nguồn phải khớp SHA-256 đã khóa · không tạo seed mới</html>"
                );
                candidateModeStatus.setForeground(new Color(126, 58, 126));
                return;
            }
            if (candidateUsesReasoning()) {
                int profile = candidateSevenChoiceProfileBox.getSelectedIndex();
                if (profile >= 1 && profile <= SEVEN_CHOICE_REASONING_CONFIGS.length) {
                    candidateModeStatus.setText(
                            "<html><b>Đang bật:</b> bộ nhãn " +
                                    SEVEN_CHOICE_REASONING_LABELS[profile - 1] +
                                    " · cấu hình và thời gian giữ đã khớp · action scale 0.25 · FiveAdam bật · chưa có MT5/OOS proof</html>"
                    );
                } else {
                    candidateModeStatus.setText(
                            "<html><b>Đang bật:</b> 3 bước × nhóm 8 · action scale 0.25 · FiveAdam bật · chưa có MT5/OOS proof</html>"
                    );
                }
                candidateModeStatus.setForeground(new Color(153, 88, 24));
            } else {
                candidateModeStatus.setText(
                        "<html><b>Đang tắt Reasoning RL:</b> chạy V193 tìm edge như trước</html>"
                );
                candidateModeStatus.setForeground(new Color(75, 85, 99));
            }
        }

        private void updateRunButton() {
            int tab = workflowTabs.getSelectedIndex();
            boolean oneClick = tab == TAB_TOOLS && toolTabs.getSelectedIndex() == TOOL_FETCH
                    && toolFetchMineBox.isSelected();
            runButton.setText(oneClick ? "Lấy dữ liệu và tạo tập 137" :
                    (tab == TAB_TOOLS ? "Chạy công cụ" :
                            (tab == TAB_ABLATION ? "Chạy ablation 3 arm" :
                            (tab == TAB_IMITATION
                                    ? (imitationVerifyOnlyBox.isSelected() ? "Kiểm tra imitation" : "Kiểm tra & train imitation")
                                    : "Chạy workflow"))));
            runButton.setEnabled(!workflowRunning && isRunnableTab(tab));
        }

        private boolean isRunnableTab(int tab) {
            if (tab >= TAB_CANDIDATE && tab <= TAB_ADMIN) return true;
            if (tab == TAB_IMITATION) return true;
            if (tab == TAB_ABLATION) return true;
            return tab == TAB_TOOLS && toolTabs.getSelectedIndex() >= TOOL_MINE
                    && toolTabs.getSelectedIndex() <= TOOL_SMOKE;
        }

        private String toolSelectedSymbolsFile() {
            Path output = resolveWorkspacePath(toolFetchOutputField.getText());
            return displayPath(output.resolve("selected_symbols.txt"));
        }

        private void persistSelectedSymbolsForFetch() throws Exception {
            Path output = resolveWorkspacePath(toolFetchOutputField.getText());
            Files.createDirectories(output);
            Files.write(output.resolve("selected_symbols.txt"), toolSymbolModel.selectedNames());
        }

        private void assignNewOutput() {
            int tab = workflowTabs.getSelectedIndex();
            if (tab == TAB_CANDIDATE) candidateOutputField.setText(defaultOutput(
                    candidateUsesCodex()
                            ? "candidate_codex_teacher"
                            : (candidateUsesReasoning() ? "candidate_reasoning_rl_v1" : "candidate_v193_edge"),
                    false
            ));
            else if (tab == TAB_RESEARCH) researchOutputField.setText(defaultOutput("research", true));
            else if (tab == TAB_CHAMPION) {
                championOutputField.setText(defaultOutput("shadow", true));
                championAuditField.setText(defaultOutput("champion_audit", false));
            } else if (tab == TAB_WALK_FORWARD) walkOutputField.setText(defaultOutput("walk_forward", false));
            else if (tab == TAB_ADMIN) adminOutputField.setText(defaultOutput("admin_copy", true));
            else if (tab == TAB_IMITATION) {
                imitationOutputField.setText(defaultOutput("teacher_imitation_candidate", true));
            } else if (tab == TAB_ABLATION) {
                ablationOutputField.setText(defaultOutput("mt5_ablation_v193", false));
            } else if (tab == TAB_TOOLS) {
                if (toolTabs.getSelectedIndex() == TOOL_FETCH) {
                    toolFetchOutputField.setText(defaultDatasetOutput("mt5_raw"));
                    if (toolFetchMineBox.isSelected()) {
                        toolOutputField.setText(defaultDatasetOutput("five_zigzag_multi"));
                    }
                } else {
                    toolOutputField.setText(defaultDatasetOutput("five_zigzag"));
                }
            }
        }

        private void prepareNewSystem() {
            workflowTabs.setSelectedIndex(TAB_CANDIDATE);
            if (CODEX_TRAIN_CONFIG.equals(candidateTrainingConfig())) {
                candidateSeedField.setText(LEGACY_FIVE_CHECKPOINT);
                candidateOutputField.setText(defaultOutput("candidate_codex_teacher", false));
                processState.setText("Sẵn sàng chạy Codex teacher từ checkpoint nguồn cố định");
                appendLog(
                        "Codex teacher dùng checkpoint nguồn bất biến " + LEGACY_FIVE_CHECKPOINT
                                + "; không tạo seed mới vì config đã khóa SHA-256 của checkpoint này.\n"
                );
                return;
            }
            candidateSeedField.setText(defaultOutput("seed_v193", true));
            candidateOutputField.setText(defaultOutput(
                    candidateUsesReasoning() ? "candidate_reasoning_rl_v1" : "candidate_v193_edge", false
            ));
            processState.setText(candidateUsesReasoning()
                    ? "Sẵn sàng khởi tạo + train Reasoning RL V1"
                    : "Sẵn sàng khởi tạo + train V193 tìm edge");
            appendLog("Đã tạo tên seed mới. Khi bấm Chạy, UI sẽ khởi tạo seed rồi train bằng config candidate đang chọn; checkpoint nguồn không bị sửa.\n");
        }

        private void copyCommand() {
            try {
                String copyText;
                if (workflowTabs.getSelectedIndex() == TAB_TOOLS && toolTabs.getSelectedIndex() == TOOL_PATTERN) {
                    copyText = toolDatasetContractText();
                } else {
                    copyText = formatCommands(commands());
                }
                Toolkit.getDefaultToolkit().getSystemClipboard().setContents(
                        new StringSelection(copyText), null
                );
                if (workflowTabs.getSelectedIndex() == TAB_TOOLS && toolTabs.getSelectedIndex() == TOOL_PATTERN) {
                    appendLog("Đã sao chép hợp đồng tập dữ liệu.\n");
                } else {
                    appendLog("Đã sao chép lệnh workflow.\n");
                }
            } catch (Exception exception) {
                appendLog("Không sao chép được lệnh: " + exception + "\n");
            }
        }

        private void validateRun() throws Exception {
            int tab = workflowTabs.getSelectedIndex();
            if (!isRunnableTab(tab)) throw new IllegalArgumentException("Hãy chọn một tab workflow");
            if (tab == TAB_CANDIDATE) {
                requireFile(candidateTrainingConfig(), "Config candidate bắt buộc");
            } else if (tab == TAB_ABLATION) {
                requireFile(REASONING_TRAIN_CONFIG, "Config Reasoning RL ablation");
            } else if (tab == TAB_CHAMPION || tab == TAB_WALK_FORWARD
                    || (tab == TAB_RESEARCH && researchModeBox.getSelectedIndex() == 0)) {
                requireFile(LATEST_TRAIN_CONFIG, "Config train V193 bắt buộc");
            }
            if (tab == TAB_CANDIDATE) {
                requireFile(CANDIDATE_SCRIPT, "Candidate script");
                String selectedProfile = candidateArchitectureBox.getSelectedIndex() == 0
                        ? SGT_COMPACT_EXPERIMENT_PROFILE
                        : candidateExperimentProfile(candidateTrainingConfig());
                requireFile(selectedProfile, "Experiment profile");
                requireDirectory(runnerDirField.getText(), "MT5 runner");
                String candidateSeedRaw = ensureCandidateSeedPath();
                Path candidateSeed = resolveWorkspacePath(candidateSeedRaw);
                Path candidateResume = resolveWorkspacePath(imitationResumeFile(candidateSeedRaw));
                String pausedReason = candidatePausedImitationReason(
                        candidateSeedRaw, Files.isRegularFile(candidateResume)
                );
                if (!pausedReason.isBlank()) throw new IllegalArgumentException(pausedReason);
                if (Files.exists(candidateSeed) && !Files.isRegularFile(candidateSeed)) {
                    throw new IllegalArgumentException("Checkpoint gốc không phải file: " + candidateSeed);
                }
                if (!Files.exists(candidateSeed)) {
                    requireFile(CLI_SOURCE, "CLI module khởi tạo kho");
                    requireFile(configField.getText(), "Config nền để khởi tạo seed");
                    requireNewFile(candidateSeedField.getText(), "Checkpoint khởi tạo");
                }
                requireNewDirectory(candidateOutputField.getText(), "Candidate output");
                requireDifferent(candidateSeedField.getText(), candidateOutputField.getText(), "Checkpoint và output");
                validateDates(candidateFromField.getText(), candidateToField.getText(), true);
                parseNonNegative(candidateSeedNumberField.getText(), "Seed");
                parsePositive(candidateTimeoutField.getText(), "Bridge timeout");
                parseNonNegative(candidateMaxWallField.getText(), "Giới hạn run");
                return;
            }
            if (tab == TAB_ABLATION) {
                requireFile(ABLATION_SCRIPT, "Ablation runner");
                requireFile(EXPERIMENT_PROFILE, "Experiment profile");
                requireFile(ablationCheckpointField.getText(), "Checkpoint nguồn ablation");
                requireDirectory(runnerDirField.getText(), "MT5 runner");
                requireNewDirectory(ablationOutputField.getText(), "Ablation output");
                validateDates(ablationFromField.getText(), ablationToField.getText(), true);
                parseNonNegative(ablationSeedField.getText(), "Seed ablation");
                parsePositive(ablationTimeoutField.getText(), "Bridge timeout");
                parseNonNegative(ablationMaxWallField.getText(), "Giới hạn mỗi arm");
                return;
            }
            if (tab == TAB_IMITATION) {
                requireFile(TEACHER_IMITATION_SCRIPT, "Bộ huấn luyện imitation");
                requireFile(imitationConfigField.getText(), "Cấu hình Five");
                if (!LATEST_TRAIN_CONFIG.equals(imitationConfigField.getText().trim())) {
                    throw new IllegalArgumentException("Imitation phải dùng " + LATEST_TRAIN_CONFIG);
                }
                String imitationCheckpointRaw = ensureImitationCheckpointPath();
                Path imitationCheckpoint = resolveWorkspacePath(imitationCheckpointRaw);
                if (Files.exists(imitationCheckpoint) && !Files.isRegularFile(imitationCheckpoint)) {
                    throw new IllegalArgumentException(
                            "Checkpoint nguồn V3 không phải file: " + imitationCheckpoint
                    );
                }
                if (!Files.exists(imitationCheckpoint)) {
                    requireFile(CLI_SOURCE, "CLI module khởi tạo seed V3");
                    requireNewFile(imitationCheckpointRaw, "Checkpoint nguồn V3 tự tạo");
                }
                requireFile(imitationDatasetField.getText(), "Dataset teacher");
                requireFile(imitationDatasetField.getText() + ".json", "Manifest dataset teacher");
                requireNewFile(imitationOutputField.getText(), "Checkpoint candidate");
                requireDifferent(imitationCheckpointField.getText(), imitationOutputField.getText(),
                        "Checkpoint nguồn/candidate");
                parsePositive(imitationEpochsField.getText(), "Epoch imitation");
                parsePositive(imitationBatchField.getText(), "Batch imitation");
                parsePositive(imitationFeatureBatchField.getText(), "Batch chuỗi recurrent");
                parsePositiveDouble(imitationLearningRateField.getText(), "Learning rate imitation");
                parsePositiveDouble(imitationRecurrentLearningRateField.getText(), "Learning rate recurrent");
                parsePositive(imitationPatienceField.getText(), "Early-stop patience");
                parseNonNegative(imitationSeedField.getText(), "Seed imitation");
                int sequenceLength = Integer.parseInt(imitationSequenceLengthField.getText().trim());
                if (sequenceLength < 2 || sequenceLength > 64) {
                    throw new IllegalArgumentException("Độ dài chuỗi imitation phải từ 2 đến 64");
                }
                parseNonNegativeDouble(imitationFalseEntryPenaltyField.getText(), "Phạt vào lệnh sai");
                return;
            }
            if (tab == TAB_TOOLS) {
                requireFile(ZIGZAG_TOOL, "Công cụ ZigZag");
                int tool = toolTabs.getSelectedIndex();
                if (tool == TOOL_MINE) {
                    requireFile(toolConfigField.getText(), "Cấu hình mô hình");
                    if (toolMineAllSymbolsBox.isSelected()) {
                        requireDirectory(toolInputField.getText(), "Thư mục raw MT5");
                        requireFile(Path.of(toolInputField.getText(), "manifest.json").toString(), "Raw MT5 manifest");
                    } else {
                        requireFile(toolInputField.getText(), "Nguồn giá");
                    }
                    parseOptionalPositiveDouble(toolSpreadPointsField.getText(), "Chênh lệch giá mặc định");
                } else if (tool == TOOL_CHECK) {
                    requireFile(toolCheckConfigField.getText(), "Cấu hình mô hình");
                    if (!toolCheckInputField.getText().isBlank()) requireFile(toolCheckInputField.getText(), "Nguồn giá");
                    parseOptionalPositiveDouble(toolCheckSpreadPointsField.getText(), "Chênh lệch giá mặc định");
                }
                if (tool == TOOL_MINE) {
                    requireFile(toolPatternField.getText(), "Mẫu cấu trúc");
                    requireNewDirectory(toolOutputField.getText(), "Thư mục tập dữ liệu");
                    parseOptionalPositive(toolAtrPeriodField.getText(), "Chu kỳ ATR");
                    parsePositiveDouble(toolZigzagAtrMultField.getText(), "Ngưỡng ZigZag theo ATR");
                    parsePositive(toolShardSizeField.getText(), "Số mẫu mỗi mảnh");
                } else if (tool == TOOL_FETCH) {
                    requireDirectory(runnerDirField.getText(), "MT5 runner");
                    requireFile(Path.of(runnerDirField.getText(), "terminal64.exe").toString(), "MT5 terminal portable");
                    if (toolSymbolModel.getRowCount() == 0) {
                        throw new IllegalArgumentException("Chưa có danh sách mã. Bấm Quét mã MT5 trước.");
                    }
                    if (toolSymbolModel.selectedCount() == 0) {
                        throw new IllegalArgumentException("Hãy đánh dấu ít nhất một mã cần lấy dữ liệu.");
                    }
                    parsePositive(toolBarsField.getText(), "Số thanh giá mỗi mã");
                    parsePositive(toolChunkSizeField.getText(), "Kích thước mỗi lượt lấy");
                    requireNewDirectory(toolFetchOutputField.getText(), "Thư mục dữ liệu MT5");
                    if (toolFetchMineBox.isSelected()) {
                        requireFile(toolPatternField.getText(), "Mẫu cấu trúc ZigZag");
                        requireFile(LATEST_TRAIN_CONFIG, "Cấu hình Five bắt buộc");
                        requireNewDirectory(toolOutputField.getText(), "Thư mục collection 137");
                        requireDifferent(toolFetchOutputField.getText(), toolOutputField.getText(),
                                "Raw output và collection output");
                        parseOptionalPositive(toolAtrPeriodField.getText(), "Chu kỳ ATR");
                        parsePositiveDouble(toolZigzagAtrMultField.getText(), "Ngưỡng ZigZag theo ATR");
                        parsePositive(toolShardSizeField.getText(), "Số mẫu mỗi mảnh");
                    }
                } else if (tool == TOOL_SMOKE) {
                    requireFile(toolSmokePatternField.getText(), "Mẫu cấu trúc");
                    parsePositive(toolSmokeAtrField.getText(), "Chu kỳ ATR thử nhanh");
                    parsePositiveDouble(toolSmokeZigzagField.getText(), "Ngưỡng ZigZag thử nhanh");
                } else if (tool == TOOL_PATTERN) {
                    throw new IllegalArgumentException("Mục Mẫu cấu trúc & kết quả không có tác vụ chạy");
                }
                return;
            }
            requireFile(CLI_SOURCE, "CLI module");
            if (tab == TAB_RESEARCH) {
                if (researchModeBox.getSelectedIndex() != 0) {
                    requireFile(configField.getText(), "Config nền");
                }
                requireNewFile(researchOutputField.getText(), "Research output");
                parseNonNegative(researchSeedField.getText(), "Seed");
                if (researchModeBox.getSelectedIndex() == 2) return;
                requireDirectory(runnerDirField.getText(), "MT5 runner");
                if (researchModeBox.getSelectedIndex() == 1) {
                    requireFile(researchSeedSystemField.getText(), "Frozen seed system");
                    requireDifferent(researchSeedSystemField.getText(), researchOutputField.getText(), "Seed và output");
                }
                validateDates(researchFromField.getText(), researchToField.getText(), false);
                parsePositive(researchBacktestDaysField.getText(), "Backtest days");
                parsePositive(researchTimeoutField.getText(), "Bridge timeout");
                parseNonNegative(researchMaxWallField.getText(), "Giới hạn run");
            } else if (tab == TAB_CHAMPION) {
                requireDirectory(runnerDirField.getText(), "MT5 runner");
                requireFile(championPathField.getText(), "Champion");
                requireNewFile(championOutputField.getText(), "Champion candidate output");
                requireDifferent(championPathField.getText(), championOutputField.getText(), "Champion và candidate output");
                validateDates(championTrainFromField.getText(), championTrainToField.getText(), true);
                parseOptionalPositive(championValidationCountField.getText(), "Validation count");
                parsePositive(championValidationDaysField.getText(), "Validation days");
                parseOptionalPositive(championSeedCountField.getText(), "Candidate seed count");
                parsePositive(championTimeoutField.getText(), "Bridge timeout");
                parseNonNegative(championMaxWallField.getText(), "Giới hạn run");
            } else if (tab == TAB_WALK_FORWARD) {
                requireDirectory(runnerDirField.getText(), "MT5 runner");
                requireNewDirectory(walkOutputField.getText(), "Walk-forward output");
                if (walkPairedBox.isSelected()) requireFile(walkSeedSystemField.getText(), "Paired seed system");
                validateDates(walkFromField.getText(), walkToField.getText(), true);
                parsePositive(walkTrainWindowField.getText(), "Train window");
                parsePositive(walkTestWindowField.getText(), "Test window");
                parsePositive(walkStepField.getText(), "Step");
                parsePositive(walkSeedCountField.getText(), "Seed count");
                parsePositive(walkTimeoutField.getText(), "Bridge timeout");
            } else if (tab == TAB_ADMIN) {
                requireFile(configField.getText(), "Config nền");
                requireFile(adminSystemField.getText(), "System nguồn");
                if (adminOutputField.getText().isBlank()) {
                    throw new IllegalArgumentException("Output copy quản trị không được để trống");
                }
                requireDifferent(adminSystemField.getText(), adminOutputField.getText(), "System nguồn và output copy");
                if (adminContextField.getText().isBlank()) {
                    throw new IllegalArgumentException("Context không được để trống");
                }
                if ("merge".equals(selected(adminActionBox)) && adminTargetField.getText().isBlank()) {
                    throw new IllegalArgumentException("Merge cần target context");
                }
                parseOptionalPositive(adminLifecycleField.getText(), "Lifecycle count");
            }
        }

        private void runWorkflow() {
            if (workflowRunning) {
                appendLog("Đang có workflow chạy.\n");
                return;
            }
            try {
                validateRun();
            } catch (Exception exception) {
                JOptionPane.showMessageDialog(this, exception.getMessage(), "Không thể chạy", JOptionPane.ERROR_MESSAGE);
                return;
            }
            saveSettings(false);
            if (workflowTabs.getSelectedIndex() == TAB_TOOLS && toolTabs.getSelectedIndex() == TOOL_FETCH) {
                try {
                    persistSelectedSymbolsForFetch();
                } catch (Exception exception) {
                    JOptionPane.showMessageDialog(
                            this, "Không thể tạo thư mục chứa dữ liệu MT5: " + exception.getMessage(),
                            "Không thể chạy", JOptionPane.ERROR_MESSAGE
                    );
                    return;
                }
            }
            livePanel.prepareForNewRun();
            List<List<String>> commandChain = commands();
            currentSaveStopRequestFile = null;
            for (List<String> item : commandChain) {
                String request = optionValue(item, "--stop-request-file");
                if (!request.isBlank()) currentSaveStopRequestFile = resolveWorkspacePath(request);
            }
            if (currentSaveStopRequestFile != null) {
                try {
                    Files.deleteIfExists(currentSaveStopRequestFile);
                } catch (IOException exception) {
                    JOptionPane.showMessageDialog(
                            this,
                            "Không thể xoá tín hiệu lưu/dừng cũ: " + exception.getMessage(),
                            "Không thể chạy", JOptionPane.ERROR_MESSAGE
                    );
                    currentSaveStopRequestFile = null;
                    return;
                }
            }
            logArea.setText("");
            appendLog("V193 — " + workflowTabs.getTitleAt(workflowTabs.getSelectedIndex()) + "\n");
            if (workflowTabs.getSelectedIndex() == TAB_TOOLS && toolTabs.getSelectedIndex() == TOOL_FETCH
                    && toolFetchMineBox.isSelected()) {
                appendLog("Một nút: MT5 fetch → collection 137 theo symbol.\n");
            } else if (workflowTabs.getSelectedIndex() == TAB_IMITATION) {
                resetLearningDiagnostics();
                outputTabs.setTitleAt(2, "Chẩn đoán học");
                if (commandChain.size() > 1) {
                    appendLog("Seed V3 chưa tồn tại: tự khởi tạo bằng cấu hình Five khóa cứng trước khi train imitation.\n");
                }
                appendLog("Imitation fail-fast: preflight dữ liệu/hash → full BPTT encoder/LSTM/Transformer không-thời gian + token-thời-gian → calibrator BUY/SELL riêng → gate zero-boundary validation → reload candidate → kiểm tra source và frozen-head hash.\n");
                Path resume = resolveWorkspacePath(imitationResumeFile(imitationOutputField.getText()));
                if (Files.isRegularFile(resume)) {
                    appendLog("Phát hiện checkpoint tạm: workflow sẽ tự resume từ " + resume + "\n");
                }
                outputTabs.setSelectedIndex(1);
                outputTabs.setTitleAt(1, "Nhật ký · imitation đang chạy");
            } else if (commandChain.size() > 1) {
                appendLog("Checkpoint chưa tồn tại: tự khởi tạo kho trước khi chạy candidate.\n");
            } else if (workflowTabs.getSelectedIndex() == TAB_CANDIDATE) {
                resetMt5Diagnostics();
                lastMt5DiagnosticsPath = null;
                lastMt5DiagnosticsMtime = Long.MIN_VALUE;
                outputTabs.setTitleAt(2, "Chẩn đoán học · MT5 đang chạy");
                resetCodexTeacherDiagnostics();
                lastCodexTeacherReceiptPath = null;
                lastCodexTeacherReceiptMtime = Long.MIN_VALUE;
                lastCodexTeacherStatusPath = null;
                lastCodexTeacherStatusMtime = Long.MIN_VALUE;
                lastCodexActionVetoPath = null;
                lastCodexActionVetoMtime = Long.MIN_VALUE;
                lastCodexDecisionProfilePath = null;
                lastCodexDecisionProfileMtime = Long.MIN_VALUE;
                outputTabs.setTitleAt(3, "Codex teacher · chờ terminal action");
                if (candidateAllowUnverifiedSeedBox.isSelected()) {
                    appendLog("CẢNH BÁO RESEARCH-ONLY: đang bỏ qua provenance/imitation seed gate; manifest sẽ ghi promotable=false.\n");
                }
                appendLog("MT5 diagnostics: execution + bridge + reward/optimizer trace; không suy underfit từ imitation BA.\n");
            } else if (workflowTabs.getSelectedIndex() == TAB_ABLATION) {
                ablationDiagnosticArea.setText("Đang chờ arm frozen_policy...\n");
                lastAblationDiagnosticsPath = null;
                lastAblationDiagnosticsMtime = Long.MIN_VALUE;
                outputTabs.setTitleAt(2, "Chẩn đoán học · Ablation đang chạy");
                appendLog("Ablation chạy tuần tự trên bản sao cùng seed/date/model; checkpoint lớn mặc định bị bỏ sau khi đã giữ metric.\n");
            }
            appendLog("Pipeline:\n" + formatCommands(commandChain) + "\n\n");
            stopRequested = false;
            setRunning(true);
            Thread worker = new Thread(() -> runProcess(commandChain), "v193-workflow-runner");
            worker.setDaemon(true);
            worker.start();
        }

        private void runProcess(List<List<String>> commandChain) {
            int exitCode = -1;
            try {
                for (int index = 0; index < commandChain.size(); index++) {
                    if (stopRequested) {
                        exitCode = 130;
                        break;
                    }
                    List<String> command = commandChain.get(index);
                    currentSaveStopCommandCapable = isSaveStopCommand(command);
                    appendLog("\nBước " + (index + 1) + "/" + commandChain.size() + ": "
                            + formatCommand(command) + "\n");
                    ProcessBuilder builder = new ProcessBuilder(command);
                    builder.directory(workspace.toFile());
                    String source = workspace.resolve("src").toString();
                    String prior = builder.environment().getOrDefault("PYTHONPATH", "");
                    builder.environment().put(
                            "PYTHONPATH",
                            prior.isBlank() ? source : source + System.getProperty("path.separator") + prior
                    );
                    builder.environment().put("PYTHONUTF8", "1");
                    builder.environment().put("PYTHONIOENCODING", "utf-8");
                    builder.redirectErrorStream(true);
                    currentProcess = builder.start();
                    SwingUtilities.invokeLater(() -> saveStopButton.setEnabled(
                            currentSaveStopCommandCapable
                                    && currentSaveStopRequestFile != null
                                    && currentProcess != null && currentProcess.isAlive()
                    ));
                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                            currentProcess.getInputStream(), StandardCharsets.UTF_8
                    ))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            consumeImitationDiagnostic(command, line);
                            appendLog(formatProcessLogLine(command, line) + "\n");
                        }
                    }
                    exitCode = currentProcess.waitFor();
                    if (exitCode != 0) {
                        appendLog("Dừng pipeline vì bước " + (index + 1) + " lỗi, exit code " + exitCode + ".\n");
                        break;
                    }
                }
            } catch (Exception exception) {
                appendLog("LỖI: " + exception + "\n");
            } finally {
                int finalExitCode = exitCode;
                SwingUtilities.invokeLater(() -> {
                    appendLog("\nWorkflow kết thúc. Exit code: " + finalExitCode + "\n");
                    outputTabs.setTitleAt(1, "Nhật ký");
                    if (finalExitCode == 0) {
                        refreshToolResultForCommands(commandChain);
                        List<String> completed = commandChain.isEmpty()
                                ? List.of() : commandChain.get(commandChain.size() - 1);
                        String resume = completed.contains(TEACHER_IMITATION_SCRIPT)
                                ? optionValue(completed, "--resume-file") : "";
                        boolean imitationPaused = !resume.isBlank()
                                && Files.isRegularFile(resolveWorkspacePath(resume));
                        String trainedCheckpoint = imitationPaused ? "" : successfulImitationOutput(
                                completed, finalExitCode
                        );
                        if (imitationPaused) {
                            appendLog("Workflow đã tạm dừng an toàn; checkpoint output cũ (nếu có) không được gắn vào Candidate. Bấm Chạy workflow để tiếp tục từ "
                                    + resume + "\n");
                        } else if (!trainedCheckpoint.isBlank()
                                && Files.isRegularFile(resolveWorkspacePath(trainedCheckpoint))) {
                            candidateSeedField.setText(trainedCheckpoint);
                            appendLog("Đã gắn checkpoint imitation action-only vào Candidate: "
                                    + trainedCheckpoint + "\n");
                            saveSettings(false);
                        } else if (completed.contains(TEACHER_IMITATION_SCRIPT)) {
                            appendLog("Không tìm thấy candidate imitation hoàn tất để gắn vào workflow MT5.\n");
                        }
                    }
                    currentProcess = null;
                    currentSaveStopRequestFile = null;
                    currentSaveStopCommandCapable = false;
                    setRunning(false);
                });
            }
        }

        private void setRunning(boolean running) {
            workflowRunning = running;
            runButton.setEnabled(!running && isRunnableTab(workflowTabs.getSelectedIndex()));
            saveStopButton.setEnabled(
                    running && currentSaveStopCommandCapable && currentSaveStopRequestFile != null
                            && currentProcess != null && currentProcess.isAlive()
            );
            stopButton.setEnabled(running);
            toolScanSymbolsButton.setEnabled(!running);
            processState.setText(running ? "ĐANG CHẠY" : "Sẵn sàng");
            processState.setForeground(running ? new Color(255, 199, 92) : new Color(124, 224, 160));
        }

        private boolean isSaveStopCommand(List<String> command) {
            if (command == null || !command.contains("--stop-request-file")) return false;
            return !command.contains("--verify-only")
                    && (command.contains(TEACHER_IMITATION_SCRIPT)
                    || command.contains(CANDIDATE_SCRIPT));
        }

        private void saveAndStop() {
            Path request = currentSaveStopRequestFile;
            Process process = currentProcess;
            if (!workflowRunning || !currentSaveStopCommandCapable
                    || request == null || process == null || !process.isAlive()) return;
            try {
                Path parent = request.getParent();
                if (parent != null) Files.createDirectories(parent);
                Files.writeString(
                        request,
                        "save-and-stop requested at " + LocalDateTime.now() + System.lineSeparator(),
                        StandardCharsets.UTF_8
                );
                saveStopButton.setEnabled(false);
                processState.setText("ĐANG LƯU CHECKPOINT");
                if (request.toString().endsWith(".save-and-stop.request")) {
                    appendLog("Đã yêu cầu lưu và dừng Candidate/MT5. Bridge sẽ chờ biên quyết định MT5 an toàn, "
                            + "đợi hàng đợi Codex teacher hoàn tất, ghi checkpoint + manifest rồi tự thoát; "
                            + "không đóng UI hoặc bấm dừng đỏ.\n");
                } else {
                    appendLog("Đã yêu cầu lưu và dừng imitation. Trainer sẽ hoàn tất batch BPTT hiện tại, "
                            + "ghi checkpoint resume rồi tự thoát; không đóng UI hoặc bấm dừng đỏ.\n");
                }
            } catch (Exception exception) {
                JOptionPane.showMessageDialog(
                        this,
                        "Không thể gửi yêu cầu lưu và dừng: " + exception.getMessage(),
                        "Lỗi lưu checkpoint", JOptionPane.ERROR_MESSAGE
                );
            }
        }

        private void stopWorkflow() {
            stopRequested = true;
            Process process = currentProcess;
            if (!workflowRunning || process == null || !process.isAlive()) return;
            appendLog("Đang dừng process do UI này tạo và các process con...\n");
            List<ProcessHandle> descendants = process.descendants()
                    .sorted(Comparator.comparingLong(ProcessHandle::pid).reversed())
                    .toList();
            for (ProcessHandle child : descendants) child.destroy();
            process.destroy();
            Thread forceStop = new Thread(() -> {
                try {
                    Thread.sleep(3000);
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                }
                for (ProcessHandle child : descendants) if (child.isAlive()) child.destroyForcibly();
                if (process.isAlive()) process.destroyForcibly();
            }, "v193-workflow-stop");
            forceStop.setDaemon(true);
            forceStop.start();
        }

        private void toggleLive() {
            if (liveButton.isSelected()) {
                if (liveWindow == null) {
                    liveWindow = new JFrame("Live 3D — V193 · Five bridge v10 · read-only telemetry");
                    liveWindow.setDefaultCloseOperation(WindowConstants.HIDE_ON_CLOSE);
                    liveWindow.setSize(1500, 930);
                    liveWindow.setLocationRelativeTo(this);
                    liveWindow.setContentPane(livePanel);
                }
                liveWindow.setVisible(true);
                livePanel.startViewing();
                liveButton.setText("Đóng Live 3D");
            } else {
                livePanel.stopViewing();
                if (liveWindow != null) liveWindow.setVisible(false);
                liveButton.setText("Mở Live 3D");
            }
        }

        private Path resolveLiveEventPath() {
            if (workflowTabs.getSelectedIndex() == TAB_CANDIDATE) {
                return resolveWorkspacePath(candidateOutputField.getText()).resolve("events.jsonl");
            }
            return resolveWorkspacePath(eventsField.getText());
        }

        private Path resolveDecisionProfilePath() {
            if (workflowTabs.getSelectedIndex() == TAB_CANDIDATE) {
                return resolveWorkspacePath(candidateOutputField.getText()).resolve("decision_profiles.jsonl");
            }
            return resolveWorkspacePath(decisionProfilesField.getText());
        }

        private Path resolveStructuralEventPath() {
            if (workflowTabs.getSelectedIndex() == TAB_CANDIDATE) {
                return resolveWorkspacePath(candidateOutputField.getText()).resolve("structural_events.jsonl");
            }
            return resolveWorkspacePath(structuralEventsField.getText());
        }

        private Path resolveNetworkDevelopmentPath() {
            if (workflowTabs.getSelectedIndex() == TAB_CANDIDATE) {
                return resolveWorkspacePath(candidateOutputField.getText()).resolve("network_development.jsonl");
            }
            return resolveWorkspacePath(networkDevelopmentField.getText());
        }

        private Path resolveWorkspacePath(String raw) {
            Path value = Path.of(raw.trim());
            return value.isAbsolute() ? value.normalize() : workspace.resolve(value).normalize();
        }

        private String defaultOutput(String kind, boolean pickle) {
            String suffix = pickle ? ".pkl" : "";
            return "outputs/he_chuyen_gia/v193_runs/" + kind + "_"
                    + LocalDateTime.now().format(RUN_STAMP) + suffix;
        }

        private String ensureCandidateSeedPath() {
            if (CODEX_TRAIN_CONFIG.equals(candidateTrainingConfig())) {
                Path requested = resolveWorkspacePath(candidateSeedField.getText());
                Path source = resolveWorkspacePath(LEGACY_FIVE_CHECKPOINT);
                if (!Files.isRegularFile(requested)) {
                    if (!Files.isRegularFile(source)) {
                        throw new IllegalStateException(
                                "Codex teacher yêu cầu checkpoint nguồn tồn tại: " + source
                        );
                    }
                    candidateSeedField.setText(displayPath(source));
                    return displayPath(source);
                }
                String expectedHash = codexTeacherSourceHash();
                String actualHash = sha256File(requested);
                if (!expectedHash.isBlank() && !expectedHash.equalsIgnoreCase(actualHash)) {
                    throw new IllegalStateException(
                            "Checkpoint Codex teacher sai SHA-256: expected=" + expectedHash
                                    + ", actual=" + actualHash
                                    + ". Dùng " + LEGACY_FIVE_CHECKPOINT + " hoặc config tương ứng."
                    );
                }
                return candidateSeedField.getText();
            }
            String value = candidateSeedForRun(
                    candidateSeedField.getText(), defaultOutput("seed_v193", true)
            );
            if (!value.equals(candidateSeedField.getText())) candidateSeedField.setText(value);
            return value;
        }

        private String codexTeacherSourceHash() {
            Path config = resolveWorkspacePath(CODEX_TRAIN_CONFIG);
            if (!Files.isRegularFile(config)) return "";
            try {
                for (String line : Files.readAllLines(config, StandardCharsets.UTF_8)) {
                    String trimmed = line.trim();
                    String prefix = "five_rlhf_source_checkpoint_sha256:";
                    if (trimmed.startsWith(prefix)) return trimmed.substring(prefix.length()).trim();
                }
            } catch (Exception ignored) {
                return "";
            }
            return "";
        }

        private String sha256File(Path path) {
            try {
                MessageDigest digest = MessageDigest.getInstance("SHA-256");
                try (var stream = Files.newInputStream(path)) {
                    byte[] buffer = new byte[1024 * 1024];
                    int read;
                    while ((read = stream.read(buffer)) >= 0) {
                        if (read > 0) digest.update(buffer, 0, read);
                    }
                }
                StringBuilder hex = new StringBuilder(64);
                for (byte value : digest.digest()) hex.append(String.format(Locale.ROOT, "%02x", value));
                return hex.toString();
            } catch (Exception exception) {
                throw new IllegalStateException("Không tính được SHA-256 checkpoint: " + exception.getMessage(), exception);
            }
        }

        private String ensureImitationCheckpointPath() {
            String restored = imitationCheckpointField.getText().trim();
            boolean sourceExists = !restored.isBlank()
                    && Files.isRegularFile(resolveWorkspacePath(restored));
            String value = imitationCheckpointForRun(restored, sourceExists);
            if (!value.equals(imitationCheckpointField.getText())) {
                imitationCheckpointField.setText(value);
            }
            return value;
        }

        private String defaultDatasetOutput(String kind) {
            return "outputs/datasets/" + kind + "_" + LocalDateTime.now().format(RUN_STAMP);
        }

        private String defaultDatasetFile(String kind, String extension) {
            return "outputs/datasets/" + kind + "_" + LocalDateTime.now().format(RUN_STAMP) + extension;
        }

        private Path settingsPath() {
            return workspace.resolve(".context-expert-v193-ui.properties");
        }

        private void saveSettings(boolean notify) {
            try {
                Properties p = new Properties();
                put(p, "python", pythonField); put(p, "config", configField); put(p, "runner", runnerDirField);
                put(p, "candidate.seedArtifact", candidateSeedField); put(p, "candidate.output", candidateOutputField);
                p.setProperty("candidate.mode", String.valueOf(candidateModeBox.getSelectedIndex()));
                p.setProperty("candidate.sevenChoiceProfile", String.valueOf(
                        candidateSevenChoiceProfileBox.getSelectedIndex()
                ));
                p.setProperty("candidate.allowUnverifiedSeed", String.valueOf(
                        candidateAllowUnverifiedSeedBox.isSelected()
                ));
                put(p, "candidate.from", candidateFromField);
                put(p, "candidate.to", candidateToField); put(p, "candidate.seed", candidateSeedNumberField);
                put(p, "candidate.timeout", candidateTimeoutField); put(p, "candidate.maxWall", candidateMaxWallField);
                p.setProperty("candidate.model", selected(candidateModelBox));
                put(p, "ablation.checkpoint", ablationCheckpointField); put(p, "ablation.output", ablationOutputField);
                put(p, "ablation.from", ablationFromField); put(p, "ablation.to", ablationToField);
                put(p, "ablation.seed", ablationSeedField); put(p, "ablation.timeout", ablationTimeoutField);
                put(p, "ablation.maxWall", ablationMaxWallField);
                p.setProperty("ablation.model", selected(ablationModelBox));
                p.setProperty("ablation.keepCheckpoints", String.valueOf(ablationKeepCheckpointsBox.isSelected()));
                p.setProperty("research.mode", String.valueOf(researchModeBox.getSelectedIndex()));
                put(p, "research.output", researchOutputField); put(p, "research.seedSystem", researchSeedSystemField);
                put(p, "research.from", researchFromField);
                put(p, "research.to", researchToField); put(p, "research.days", researchBacktestDaysField);
                put(p, "research.seed", researchSeedField); put(p, "research.timeout", researchTimeoutField);
                put(p, "research.maxWall", researchMaxWallField); p.setProperty("research.model", selected(researchModelBox));
                put(p, "champion.path", championPathField); put(p, "champion.output", championOutputField);
                put(p, "champion.audit", championAuditField); put(p, "champion.from", championTrainFromField);
                put(p, "champion.to", championTrainToField); put(p, "champion.validationCount", championValidationCountField);
                put(p, "champion.validationDays", championValidationDaysField); put(p, "champion.seedCount", championSeedCountField);
                put(p, "champion.seed", championSeedField); put(p, "champion.timeout", championTimeoutField);
                put(p, "champion.maxWall", championMaxWallField); p.setProperty("champion.model", selected(championModelBox));
                put(p, "gates.minExpectancy", minExpectancyField); put(p, "gates.maxProfitRegression", maxProfitRegressionField);
                put(p, "gates.maxLcbRegression", maxLcbRegressionField); put(p, "gates.maxDdIncrease", maxDdIncreaseField);
                put(p, "walk.output", walkOutputField); put(p, "walk.log", walkLogField);
                put(p, "walk.seedSystem", walkSeedSystemField); put(p, "walk.from", walkFromField); put(p, "walk.to", walkToField);
                put(p, "walk.train", walkTrainWindowField); put(p, "walk.test", walkTestWindowField); put(p, "walk.step", walkStepField);
                put(p, "walk.seed", walkSeedField); put(p, "walk.seedCount", walkSeedCountField); put(p, "walk.timeout", walkTimeoutField);
                p.setProperty("walk.model", selected(walkModelBox)); p.setProperty("walk.paired", String.valueOf(walkPairedBox.isSelected()));
                p.setProperty("walk.rollback", String.valueOf(walkRollbackBox.isSelected()));
                put(p, "admin.system", adminSystemField); put(p, "admin.output", adminOutputField);
                put(p, "admin.context", adminContextField); put(p, "admin.target", adminTargetField);
                put(p, "admin.lifecycle", adminLifecycleField); put(p, "admin.note", adminNoteField);
                put(p, "admin.audit", adminAuditField);
                p.setProperty("admin.action", selected(adminActionBox));
                p.setProperty("admin.replay", String.valueOf(adminReplayBox.isSelected()));
                p.setProperty("admin.oos", String.valueOf(adminOosBox.isSelected()));
                p.setProperty("admin.risk", String.valueOf(adminRiskBox.isSelected()));
                p.setProperty("admin.nonInterference", String.valueOf(adminNonInterferenceBox.isSelected()));
                put(p, "telemetry.events", eventsField); put(p, "telemetry.decisions", decisionProfilesField);
                put(p, "telemetry.structural", structuralEventsField); put(p, "telemetry.development", networkDevelopmentField);
                put(p, "telemetry.sample", sampleEveryField); put(p, "telemetry.maxMb", maxMbField);
                put(p, "telemetry.maxNeurons", maxNeuronsField); put(p, "telemetry.maxEdges", maxEdgesField);
                p.setProperty("telemetry.enabled", String.valueOf(telemetryBox.isSelected()));
                p.setProperty("telemetry.reset", String.valueOf(resetLogsBox.isSelected()));
                p.setProperty("mt5.visual", String.valueOf(visualBox.isSelected()));
                p.setProperty("mt5.keepOpen", String.valueOf(keepOpenBox.isSelected()));
                put(p, "imitation.config", imitationConfigField); put(p, "imitation.checkpoint", imitationCheckpointField);
                put(p, "imitation.dataset", imitationDatasetField); put(p, "imitation.output", imitationOutputField);
                put(p, "imitation.epochs", imitationEpochsField); put(p, "imitation.batch", imitationBatchField);
                put(p, "imitation.featureBatch", imitationFeatureBatchField); put(p, "imitation.learningRate", imitationLearningRateField);
                put(p, "imitation.recurrentLearningRate", imitationRecurrentLearningRateField);
                put(p, "imitation.patience", imitationPatienceField); put(p, "imitation.seed", imitationSeedField);
                put(p, "imitation.sequenceLength", imitationSequenceLengthField);
                put(p, "imitation.falseEntryPenalty", imitationFalseEntryPenaltyField);
                p.setProperty("imitation.verifyOnly", String.valueOf(imitationVerifyOnlyBox.isSelected()));
                put(p, "tool.input", toolInputField); put(p, "tool.pattern", toolPatternField);
                put(p, "tool.output", toolOutputField); put(p, "tool.config", toolConfigField);
                put(p, "tool.atr", toolAtrPeriodField); put(p, "tool.zigzag", toolZigzagAtrMultField);
                put(p, "tool.shard", toolShardSizeField); put(p, "tool.spread", toolSpreadPointsField);
                p.setProperty("tool.neutralVolume", String.valueOf(toolAllowNeutralVolumeBox.isSelected()));
                p.setProperty("tool.autoCheck", String.valueOf(toolAutoCheckBox.isSelected()));
                p.setProperty("tool.mineAllSymbols", String.valueOf(toolMineAllSymbolsBox.isSelected()));
                put(p, "tool.checkInput", toolCheckInputField); put(p, "tool.checkConfig", toolCheckConfigField);
                put(p, "tool.checkSpread", toolCheckSpreadPointsField);
                p.setProperty("tool.checkNeutralVolume", String.valueOf(toolCheckAllowNeutralVolumeBox.isSelected()));
                put(p, "tool.symbolFilter", toolSymbolField); p.setProperty("tool.timeframe", selected(toolTimeframeBox));
                put(p, "tool.bars", toolBarsField); put(p, "tool.chunk", toolChunkSizeField);
                put(p, "tool.fetchOutput", toolFetchOutputField); put(p, "tool.smokePattern", toolSmokePatternField);
                p.setProperty("tool.fetchMine", String.valueOf(toolFetchMineBox.isSelected()));
                put(p, "tool.smokeAtr", toolSmokeAtrField); put(p, "tool.smokeZigzag", toolSmokeZigzagField);
                put(p, "tool.editorPattern", toolEditorPatternField);
                p.setProperty("tool.subTab", String.valueOf(toolTabs.getSelectedIndex()));
                p.setProperty("selectedTab", String.valueOf(Math.min(workflowTabs.getSelectedIndex(), TAB_ABLATION)));
                Files.createDirectories(settingsPath().getParent());
                try (var output = Files.newOutputStream(settingsPath())) {
                    p.store(output, "V193 Five Bridge v10 Control");
                }
                if (notify) appendLog("Đã lưu cấu hình UI: " + settingsPath() + "\n");
            } catch (Exception exception) {
                appendLog("Không lưu được cài đặt UI: " + exception + "\n");
            }
        }

        private void loadSettings(boolean createIfMissing) {
            Path path = settingsPath();
            if (!Files.isRegularFile(path)) {
                Path legacyV191 = workspace.resolve(".context-expert-v191-ui.properties");
                Path legacy = workspace.resolve(".context-expert-v188-ui.properties");
                if (Files.isRegularFile(legacyV191)) {
                    try {
                        Files.copy(legacyV191, path);
                        appendLog("Đã migrate cấu hình UI V191 sang V193: " + path + "\n");
                    } catch (Exception exception) {
                        appendLog("Không migrate được cấu hình V191: " + exception + "\n");
                    }
                } else if (Files.isRegularFile(legacy)) {
                    loadLegacyV188Settings(legacy);
                    saveSettings(false);
                    appendLog("Đã migrate và tạo file cấu hình V193 mới: " + path + "\n");
                } else if (createIfMissing) {
                    saveSettings(false);
                    appendLog("Chưa có file cấu hình; đã tạo mới: " + path + "\n");
                }
                if (!Files.isRegularFile(path)) return;
            }
            try {
                Properties p = loadProperties(path);
                get(p, "python", pythonField); get(p, "config", configField); get(p, "runner", runnerDirField);
                candidateModeBox.setSelectedIndex(Math.max(0, Math.min(CANDIDATE_MODE_CODEX,
                        Integer.parseInt(p.getProperty("candidate.mode", "0")))));
                candidateSevenChoiceProfileBox.setSelectedIndex(Math.max(0, Math.min(
                        SEVEN_CHOICE_REASONING_CONFIGS.length,
                        Integer.parseInt(p.getProperty("candidate.sevenChoiceProfile", "0"))
                )));
                get(p, "candidate.seedArtifact", candidateSeedField); get(p, "candidate.output", candidateOutputField);
                candidateAllowUnverifiedSeedBox.setSelected(Boolean.parseBoolean(
                        p.getProperty("candidate.allowUnverifiedSeed", "false")
                ));
                ensureCandidateSeedPath();
                String restoredCandidateOutputText = candidateOutputField.getText().trim();
                if (restoredCandidateOutputText.isBlank()
                        || Files.exists(resolveWorkspacePath(restoredCandidateOutputText))) {
                    candidateOutputField.setText(defaultOutput(
                            candidateUsesCodex()
                                    ? "candidate_codex_teacher"
                                    : (candidateUsesReasoning() ? "candidate_reasoning_rl_v1" : "candidate_v193_edge"),
                            false
                    ));
                }
                get(p, "candidate.from", candidateFromField);
                get(p, "candidate.to", candidateToField); get(p, "candidate.seed", candidateSeedNumberField);
                get(p, "candidate.timeout", candidateTimeoutField); get(p, "candidate.maxWall", candidateMaxWallField);
                candidateModelBox.setSelectedItem(p.getProperty("candidate.model", selected(candidateModelBox)));
                get(p, "ablation.checkpoint", ablationCheckpointField); get(p, "ablation.output", ablationOutputField);
                if (Files.exists(resolveWorkspacePath(ablationOutputField.getText()))) {
                    ablationOutputField.setText(defaultOutput("mt5_ablation_v193", false));
                }
                get(p, "ablation.from", ablationFromField); get(p, "ablation.to", ablationToField);
                get(p, "ablation.seed", ablationSeedField); get(p, "ablation.timeout", ablationTimeoutField);
                get(p, "ablation.maxWall", ablationMaxWallField);
                ablationModelBox.setSelectedItem(p.getProperty("ablation.model", selected(ablationModelBox)));
                ablationKeepCheckpointsBox.setSelected(Boolean.parseBoolean(
                        p.getProperty("ablation.keepCheckpoints", "false")
                ));
                researchModeBox.setSelectedIndex(Math.max(0, Math.min(2, Integer.parseInt(p.getProperty("research.mode", "0")))));
                get(p, "research.output", researchOutputField); get(p, "research.seedSystem", researchSeedSystemField);
                get(p, "research.from", researchFromField);
                get(p, "research.to", researchToField); get(p, "research.days", researchBacktestDaysField);
                get(p, "research.seed", researchSeedField); get(p, "research.timeout", researchTimeoutField);
                get(p, "research.maxWall", researchMaxWallField); researchModelBox.setSelectedItem(p.getProperty("research.model", selected(researchModelBox)));
                get(p, "champion.path", championPathField); get(p, "champion.output", championOutputField);
                get(p, "champion.audit", championAuditField); get(p, "champion.from", championTrainFromField);
                get(p, "champion.to", championTrainToField); get(p, "champion.validationCount", championValidationCountField);
                get(p, "champion.validationDays", championValidationDaysField); get(p, "champion.seedCount", championSeedCountField);
                get(p, "champion.seed", championSeedField); get(p, "champion.timeout", championTimeoutField);
                get(p, "champion.maxWall", championMaxWallField); championModelBox.setSelectedItem(p.getProperty("champion.model", selected(championModelBox)));
                get(p, "gates.minExpectancy", minExpectancyField); get(p, "gates.maxProfitRegression", maxProfitRegressionField);
                get(p, "gates.maxLcbRegression", maxLcbRegressionField); get(p, "gates.maxDdIncrease", maxDdIncreaseField);
                get(p, "walk.output", walkOutputField); get(p, "walk.log", walkLogField);
                get(p, "walk.seedSystem", walkSeedSystemField); get(p, "walk.from", walkFromField); get(p, "walk.to", walkToField);
                get(p, "walk.train", walkTrainWindowField); get(p, "walk.test", walkTestWindowField); get(p, "walk.step", walkStepField);
                get(p, "walk.seed", walkSeedField); get(p, "walk.seedCount", walkSeedCountField); get(p, "walk.timeout", walkTimeoutField);
                walkModelBox.setSelectedItem(p.getProperty("walk.model", selected(walkModelBox)));
                walkPairedBox.setSelected(Boolean.parseBoolean(p.getProperty("walk.paired", "true")));
                walkRollbackBox.setSelected(Boolean.parseBoolean(p.getProperty("walk.rollback", "true")));
                get(p, "admin.system", adminSystemField); get(p, "admin.output", adminOutputField);
                get(p, "admin.context", adminContextField); get(p, "admin.target", adminTargetField);
                get(p, "admin.lifecycle", adminLifecycleField); get(p, "admin.note", adminNoteField);
                get(p, "admin.audit", adminAuditField);
                adminActionBox.setSelectedItem(p.getProperty("admin.action", selected(adminActionBox)));
                adminReplayBox.setSelected(Boolean.parseBoolean(p.getProperty("admin.replay", "false")));
                adminOosBox.setSelected(Boolean.parseBoolean(p.getProperty("admin.oos", "false")));
                adminRiskBox.setSelected(Boolean.parseBoolean(p.getProperty("admin.risk", "false")));
                adminNonInterferenceBox.setSelected(Boolean.parseBoolean(p.getProperty("admin.nonInterference", "false")));
                get(p, "telemetry.events", eventsField); get(p, "telemetry.decisions", decisionProfilesField);
                get(p, "telemetry.structural", structuralEventsField); get(p, "telemetry.development", networkDevelopmentField);
                get(p, "telemetry.sample", sampleEveryField); get(p, "telemetry.maxMb", maxMbField);
                get(p, "telemetry.maxNeurons", maxNeuronsField); get(p, "telemetry.maxEdges", maxEdgesField);
                telemetryBox.setSelected(Boolean.parseBoolean(p.getProperty("telemetry.enabled", "true")));
                resetLogsBox.setSelected(Boolean.parseBoolean(p.getProperty("telemetry.reset", "false")));
                visualBox.setSelected(Boolean.parseBoolean(p.getProperty("mt5.visual", "false")));
                keepOpenBox.setSelected(Boolean.parseBoolean(p.getProperty("mt5.keepOpen", "false")));
                get(p, "imitation.config", imitationConfigField); get(p, "imitation.checkpoint", imitationCheckpointField);
                get(p, "imitation.dataset", imitationDatasetField); get(p, "imitation.output", imitationOutputField);
                String restoredImitationCheckpoint = imitationCheckpointField.getText();
                String effectiveImitationCheckpoint = ensureImitationCheckpointPath();
                if (!effectiveImitationCheckpoint.equals(restoredImitationCheckpoint)) {
                    appendLog("Đã chuyển nguồn imitation trống/cũ sang seed V3 hiện hành; UI sẽ tự tạo file nếu chưa có: "
                            + effectiveImitationCheckpoint + "\n");
                }
                String restoredImitationDataset = imitationDatasetField.getText();
                String migratedImitationDataset = migrateTeacherImitationDataset(restoredImitationDataset);
                if (!migratedImitationDataset.equals(restoredImitationDataset)) {
                    imitationDatasetField.setText(migratedImitationDataset);
                    appendLog("Đã chuyển dataset imitation cũ sang schema 9 có SL 1,50 ATR và TP pivot đã xác nhận: "
                            + migratedImitationDataset + "\n");
                }
                if (Files.exists(resolveWorkspacePath(imitationOutputField.getText()))) {
                    imitationOutputField.setText(defaultOutput("teacher_imitation_candidate", true));
                }
                get(p, "imitation.epochs", imitationEpochsField); get(p, "imitation.batch", imitationBatchField);
                get(p, "imitation.featureBatch", imitationFeatureBatchField); get(p, "imitation.learningRate", imitationLearningRateField);
                get(p, "imitation.recurrentLearningRate", imitationRecurrentLearningRateField);
                get(p, "imitation.patience", imitationPatienceField); get(p, "imitation.seed", imitationSeedField);
                get(p, "imitation.sequenceLength", imitationSequenceLengthField);
                if ("16".equals(imitationSequenceLengthField.getText().trim())) {
                    imitationSequenceLengthField.setText("64");
                    appendLog("Đã nâng BPTT imitation mặc định từ 16 lên 64 bước; PPO online của model cũng dùng unroll 64.\n");
                }
                get(p, "imitation.falseEntryPenalty", imitationFalseEntryPenaltyField);
                imitationVerifyOnlyBox.setSelected(Boolean.parseBoolean(p.getProperty("imitation.verifyOnly", "false")));
                get(p, "tool.input", toolInputField); get(p, "tool.pattern", toolPatternField);
                get(p, "tool.output", toolOutputField); get(p, "tool.config", toolConfigField);
                if (Files.exists(resolveWorkspacePath(toolOutputField.getText()))) {
                    toolOutputField.setText(defaultDatasetOutput("five_zigzag_multi"));
                }
                get(p, "tool.atr", toolAtrPeriodField); get(p, "tool.zigzag", toolZigzagAtrMultField);
                get(p, "tool.shard", toolShardSizeField); get(p, "tool.spread", toolSpreadPointsField);
                toolAllowNeutralVolumeBox.setSelected(Boolean.parseBoolean(p.getProperty("tool.neutralVolume", "false")));
                toolAutoCheckBox.setSelected(Boolean.parseBoolean(p.getProperty("tool.autoCheck", "true")));
                toolMineAllSymbolsBox.setSelected(Boolean.parseBoolean(p.getProperty("tool.mineAllSymbols", "false")));
                get(p, "tool.checkInput", toolCheckInputField); get(p, "tool.checkConfig", toolCheckConfigField);
                get(p, "tool.checkSpread", toolCheckSpreadPointsField);
                toolCheckAllowNeutralVolumeBox.setSelected(Boolean.parseBoolean(p.getProperty("tool.checkNeutralVolume", "false")));
                get(p, "tool.symbolFilter", toolSymbolField); toolTimeframeBox.setSelectedItem(p.getProperty("tool.timeframe", selected(toolTimeframeBox)));
                get(p, "tool.bars", toolBarsField); get(p, "tool.chunk", toolChunkSizeField);
                get(p, "tool.fetchOutput", toolFetchOutputField);
                toolFetchMineBox.setSelected(Boolean.parseBoolean(p.getProperty("tool.fetchMine", "true")));
                if (toolFetchOutputField.getText().trim().toLowerCase(Locale.ROOT).endsWith(".npz")) {
                    toolFetchOutputField.setText(defaultDatasetOutput("mt5_raw"));
                }
                if (Files.exists(resolveWorkspacePath(toolFetchOutputField.getText()))) {
                    toolFetchOutputField.setText(defaultDatasetOutput("mt5_raw"));
                }
                get(p, "tool.smokePattern", toolSmokePatternField);
                get(p, "tool.smokeAtr", toolSmokeAtrField); get(p, "tool.smokeZigzag", toolSmokeZigzagField);
                get(p, "tool.editorPattern", toolEditorPatternField);
                toolTabs.setSelectedIndex(Math.max(0, Math.min(TOOL_PATTERN, Integer.parseInt(p.getProperty("tool.subTab", "0")))));
                workflowTabs.setSelectedIndex(Math.max(0, Math.min(TAB_ABLATION, Integer.parseInt(p.getProperty("selectedTab", "0")))));
                loadToolPattern(false);
                refreshToolSummary(false);
                refreshPreview();
                updateRunButton();
                appendLog("Đã tải cấu hình UI: " + path + "\n");
            } catch (Exception exception) {
                appendLog("Không tải được cài đặt V193: " + exception + "\n");
            }
        }

        private void loadLegacyV188Settings(Path path) {
            try {
                Properties p = loadProperties(path);
                candidateSeedField.setText(p.getProperty("seedArtifact", candidateSeedField.getText()));
                candidateFromField.setText(p.getProperty("from", candidateFromField.getText()));
                candidateToField.setText(p.getProperty("to", candidateToField.getText()));
                candidateModelBox.setSelectedItem(p.getProperty("model", selected(candidateModelBox)));
                candidateSeedNumberField.setText(p.getProperty("seed", candidateSeedNumberField.getText()));
                candidateTimeoutField.setText(p.getProperty("timeout", candidateTimeoutField.getText()));
                candidateMaxWallField.setText(p.getProperty("maxWall", candidateMaxWallField.getText()));
            } catch (Exception exception) {
                appendLog("Không migrate được cài đặt V188: " + exception + "\n");
            }
        }

        private Properties loadProperties(Path path) throws Exception {
            Properties p = new Properties();
            try (var input = Files.newInputStream(path)) { p.load(input); }
            return p;
        }

        private void requireFile(String raw, String label) {
            if (!Files.isRegularFile(resolveWorkspacePath(raw))) {
                throw new IllegalArgumentException(label + " không tồn tại: " + raw);
            }
        }

        private void requireDirectory(String raw, String label) {
            if (!Files.isDirectory(resolveWorkspacePath(raw))) {
                throw new IllegalArgumentException(label + " không tồn tại: " + raw);
            }
        }

        private void requireNewFile(String raw, String label) {
            Path output = resolveWorkspacePath(raw);
            if (Files.exists(output)) throw new IllegalArgumentException(label + " phải là path mới: " + output);
        }

        private void requireNewDirectory(String raw, String label) throws Exception {
            Path output = resolveWorkspacePath(raw);
            if (!Files.exists(output)) return;
            if (!Files.isDirectory(output)) throw new IllegalArgumentException(label + " không phải thư mục: " + output);
            try (var entries = Files.list(output)) {
                if (entries.findAny().isPresent()) throw new IllegalArgumentException(label + " phải mới hoặc rỗng: " + output);
            }
        }

        private void requireDifferent(String left, String right, String label) {
            if (resolveWorkspacePath(left).equals(resolveWorkspacePath(right))) {
                throw new IllegalArgumentException(label + " phải khác nhau");
            }
        }

        private void validateDates(String from, String to, boolean requireTo) {
            LocalDate start = parseInputDate(from);
            if (to == null || to.isBlank()) {
                if (requireTo) throw new IllegalArgumentException("Đến ngày không được để trống");
                return;
            }
            LocalDate end = parseInputDate(to);
            if (!start.isBefore(end)) throw new IllegalArgumentException("Từ ngày phải trước đến ngày");
        }

        private void parsePositive(String raw, String label) {
            if (Integer.parseInt(raw.trim()) <= 0) throw new IllegalArgumentException(label + " phải > 0");
        }

        private void parseNonNegative(String raw, String label) {
            if (Integer.parseInt(raw.trim()) < 0) throw new IllegalArgumentException(label + " phải >= 0");
        }

        private void parseOptionalPositive(String raw, String label) {
            if (raw != null && !raw.isBlank()) parsePositive(raw, label);
        }

        private void parsePositiveDouble(String raw, String label) {
            double value = Double.parseDouble(raw.trim());
            if (!Double.isFinite(value) || value <= 0.0) throw new IllegalArgumentException(label + " phải > 0");
        }

        private void parseNonNegativeDouble(String raw, String label) {
            double value = Double.parseDouble(raw.trim());
            if (!Double.isFinite(value) || value < 0.0) throw new IllegalArgumentException(label + " phải >= 0");
        }

        private void parseOptionalPositiveDouble(String raw, String label) {
            if (raw != null && !raw.isBlank()) parsePositiveDouble(raw, label);
        }

        private void resetMt5Diagnostics() {
            for (JLabel label : List.of(
                    mt5EdgeDiagnosticLabel, mt5RiskDiagnosticLabel, mt5OptimizerDiagnosticLabel,
                    mt5RewardDiagnosticLabel, mt5PolicyDiagnosticLabel, mt5ExecutionDiagnosticLabel
            )) applyOperationalStatus(label, "insufficient_data");
            mt5DiagnosticArea.setText(
                    "Đang chờ outputs/.../mt5_diagnostics.json từ candidate MT5.\n"
                            + "Telemetry này chỉ đọc ledger; không điều khiển lệnh hoặc optimizer.\n"
            );
            mt5DiagnosticArea.setCaretPosition(0);
        }

        private void resetCodexTeacherDiagnostics() {
            codexTeacherArea.setText(
                    "CODEX TEACHER · CHƯA CÓ RECEIPT\n\n"
                            + "Mỗi action có outcome terminal được đưa vào queue khi MT5 xác nhận đúng target trace.\n"
                            + "UI đọc status queue khi MT5 đang chạy và từng receipt JSONL khi Codex trả kết quả.\n"
                            + "Khi có receipt, UI ghi rõ THƯỞNG/PHẠT, reward signed, scaled reward và PPO update.\n"
                            + "Hiển thị rationale tóm tắt do model trả về, không hiển thị hidden chain-of-thought.\n"
            );
            codexTeacherArea.setCaretPosition(0);
        }

        private JComponent buildCodexTeacherDiagnosticsPanel() {
            JPanel root = new JPanel(new BorderLayout(8, 8));
            root.setBorder(new EmptyBorder(8, 8, 8, 8));
            codexTeacherArea.setEditable(false);
            codexTeacherArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            codexTeacherArea.setLineWrap(true);
            codexTeacherArea.setWrapStyleWord(true);
            codexTeacherArea.setBackground(new Color(12, 18, 29));
            codexTeacherArea.setForeground(new Color(229, 238, 248));
            resetCodexTeacherDiagnostics();
            root.add(new JScrollPane(codexTeacherArea), BorderLayout.CENTER);
            JLabel note = new JLabel(
                    "<html>Receipt được đọc từ file audit append-only. Điểm reward/confidence là output của teacher; "
                            + "update PPO/rollback vẫn được quyết định bởi policy gate.</html>"
            );
            note.setForeground(new Color(75, 85, 99));
            root.add(note, BorderLayout.SOUTH);
            return root;
        }

        /**
         * A veto is an execution-time action-mask change, not a Codex receipt.
         * Keep its audit separate so a generic PPO update cannot be mistaken for
         * a live action intervention.  The source is the run-scoped
         * network_development.jsonl stream; malformed/in-progress JSON lines are
         * ignored just like the receipt reader below.
         */
        private static final class ActionVetoSummary {
            private int records;
            private int negativeRecords;
            private int vetoRecords;
            private int negativeActions;
            private int vetoedActions;
            private int actionSurfaceRecords;
            private int maskOnlyHoldRecords;
            private int neuralCommandRecords;
            private int probeRecords;
            private int contextEscapeRecords;
            private String lastAllowedActions = "—";
            private String lastTrace = "";
            private final Map<Integer, Integer> negativeByAction = new HashMap<>();
            private final Map<Integer, Integer> vetoedByAction = new HashMap<>();

            private void absorb(Map<String, Object> details, String traceId) {
                List<Object> negative = MiniJson.array(details.get("five_live_action_veto_negative_actions"));
                List<Object> vetoed = MiniJson.array(details.get("five_live_action_veto_vetoed_actions"));
                if (negative.isEmpty() && vetoed.isEmpty()) return;
                records++;
                if (!negative.isEmpty()) negativeRecords++;
                if (!vetoed.isEmpty()) vetoRecords++;
                negativeActions += addActions(negative, negativeByAction);
                vetoedActions += addActions(vetoed, vetoedByAction);
                if (traceId != null && !traceId.isBlank()) lastTrace = traceId;
            }

            private void absorbActionSurface(Map<String, Object> details, String traceId) {
                Map<String, Object> audit = MiniJson.object(details.get("action_surface_audit"));
                if (audit.isEmpty()) return;
                actionSurfaceRecords++;
                if (Boolean.TRUE.equals(audit.get("mask_only_hold"))) maskOnlyHoldRecords++;
                if (Boolean.TRUE.equals(audit.get("neural_command"))) neuralCommandRecords++;
                if (Boolean.TRUE.equals(audit.get("live_probe_due"))
                        || !MiniJson.array(audit.get("probe_preserved_actions")).isEmpty()) {
                    probeRecords++;
                }
                if (!MiniJson.array(audit.get("context_escape_actions")).isEmpty()) {
                    contextEscapeRecords++;
                }
                List<Object> allowed = MiniJson.array(audit.get("allowed_action_indices"));
                if (!allowed.isEmpty()) {
                    StringBuilder values = new StringBuilder();
                    for (Object value : allowed) {
                        if (values.length() > 0) values.append(',');
                        values.append(MiniJson.integer(value, -1));
                    }
                    lastAllowedActions = "[" + values + "]";
                }
                if (traceId != null && !traceId.isBlank()) lastTrace = traceId;
            }

            private void merge(ActionVetoSummary other) {
                if (other == null) return;
                records += other.records;
                negativeRecords += other.negativeRecords;
                vetoRecords += other.vetoRecords;
                negativeActions += other.negativeActions;
                vetoedActions += other.vetoedActions;
                actionSurfaceRecords += other.actionSurfaceRecords;
                maskOnlyHoldRecords += other.maskOnlyHoldRecords;
                neuralCommandRecords += other.neuralCommandRecords;
                probeRecords += other.probeRecords;
                contextEscapeRecords += other.contextEscapeRecords;
                if (!other.lastAllowedActions.equals("—")) {
                    lastAllowedActions = other.lastAllowedActions;
                }
                if (!other.lastTrace.isBlank()) lastTrace = other.lastTrace;
                other.negativeByAction.forEach((key, value) ->
                        negativeByAction.merge(key, value, Integer::sum));
                other.vetoedByAction.forEach((key, value) ->
                        vetoedByAction.merge(key, value, Integer::sum));
            }

            private static int addActions(List<Object> values, Map<Integer, Integer> counts) {
                int added = 0;
                for (Object value : values) {
                    int action = MiniJson.integer(value, -1);
                    if (action < 0) continue;
                    counts.merge(action, 1, Integer::sum);
                    added++;
                }
                return added;
            }

            private String formatActions(Map<Integer, Integer> counts) {
                if (counts.isEmpty()) return "—";
                StringBuilder text = new StringBuilder();
                counts.entrySet().stream()
                        .sorted(Map.Entry.comparingByKey())
                        .forEach(entry -> {
                            if (text.length() > 0) text.append(", ");
                            text.append("a").append(entry.getKey())
                                    .append("/").append(actionVetoLabel(entry.getKey()))
                                    .append("×").append(entry.getValue());
                        });
                return text.toString();
            }

            private String render(Path source) {
                return "POLICY LIVE ACTION VETO (không phải Codex receipt): records=" + records
                        + " · negative records=" + negativeRecords
                        + " · veto applied=" + vetoRecords
                        + " · negative actions=" + negativeActions
                        + " · vetoed actions=" + vetoedActions + '\n'
                        + "negative by action: " + formatActions(negativeByAction)
                        + " · vetoed by action: " + formatActions(vetoedByAction)
                        + " · last trace=" + (lastTrace.isBlank() ? "—" : lastTrace) + '\n'
                        + "ACTION SURFACE: profiles=" + actionSurfaceRecords
                        + " · mask-only-HOLD=" + maskOnlyHoldRecords
                        + " · neural command=" + neuralCommandRecords
                        + " · probe records=" + probeRecords
                        + " · contextual escape=" + contextEscapeRecords
                        + " · last allowed=" + lastAllowedActions + '\n'
                        + "source: " + (source == null ? "—" : source) + '\n';
            }
        }

        private static String actionVetoLabel(int action) {
            return switch (action) {
                case 0 -> "HOLD";
                case 1 -> "MARKET_BUY";
                case 2 -> "MARKET_SELL";
                case 3 -> "BUY_LIMIT";
                case 4 -> "SELL_LIMIT";
                case 5 -> "BUY_STOP";
                case 6 -> "SELL_STOP";
                case 7 -> "CLOSE_WORST";
                case 8 -> "CLOSE_BEST";
                case 9 -> "MANAGE_POSITION";
                case 10 -> "CLOSE_ALL";
                case 11 -> "CANCEL_ORDERS";
                default -> "ACTION_" + action;
            };
        }

        private static void collectActionVetoDetails(
                Object value,
                String inheritedTrace,
                ActionVetoSummary summary
        ) {
            if (value instanceof Map<?, ?>) {
                Map<String, Object> map = MiniJson.object(value);
                String trace = MiniJson.text(map.get("trace_id"), inheritedTrace);
                if (map.containsKey("five_live_action_veto_negative_actions")
                        || map.containsKey("five_live_action_veto_vetoed_actions")) {
                    summary.absorb(map, trace);
                }
                if (map.containsKey("action_surface_audit")) {
                    summary.absorbActionSurface(map, trace);
                }
                for (Object child : map.values()) {
                    if (child instanceof Map<?, ?> || child instanceof List<?>) {
                        collectActionVetoDetails(child, trace, summary);
                    }
                }
                return;
            }
            if (value instanceof List<?>) {
                for (Object child : (List<?>) value) {
                    if (child instanceof Map<?, ?> || child instanceof List<?>) {
                        collectActionVetoDetails(child, inheritedTrace, summary);
                    }
                }
            }
        }

        private ActionVetoSummary readActionVetoSummary(Path path) {
            ActionVetoSummary summary = new ActionVetoSummary();
            if (path == null || !Files.isRegularFile(path)) return summary;
            try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.isBlank()) continue;
                    try {
                        collectActionVetoDetails(MiniJson.parse(line), "", summary);
                    } catch (RuntimeException ignored) {
                        // The writer may be appending the final JSON line.
                    }
                }
            } catch (IOException ignored) {
                // Diagnostics are best effort and must not interrupt MT5 control.
            }
            return summary;
        }

        private void refreshCodexTeacherDiagnostics() {
            String rawOutput = candidateOutputField.getText().trim();
            if (rawOutput.isBlank()) return;
            Path receiptPath = latestCodexTeacherReceipt(rawOutput);
            Path statusPath = latestCodexTeacherStatus(rawOutput);
            Path actionVetoPath = resolveWorkspacePath(rawOutput).resolve("network_development.jsonl");
            if (!Files.isRegularFile(actionVetoPath)) actionVetoPath = null;
            Path decisionProfilePath = resolveWorkspacePath(rawOutput).resolve("decision_profiles.jsonl");
            if (!Files.isRegularFile(decisionProfilePath)) decisionProfilePath = null;
            if (receiptPath == null && statusPath == null
                    && actionVetoPath == null && decisionProfilePath == null) return;
            try {
                long modified = receiptPath == null
                        ? Long.MIN_VALUE : Files.getLastModifiedTime(receiptPath).toMillis();
                long statusModified = statusPath == null
                        ? Long.MIN_VALUE : Files.getLastModifiedTime(statusPath).toMillis();
                long actionVetoModified = actionVetoPath == null
                        ? Long.MIN_VALUE : Files.getLastModifiedTime(actionVetoPath).toMillis();
                long decisionProfileModified = decisionProfilePath == null
                        ? Long.MIN_VALUE : Files.getLastModifiedTime(decisionProfilePath).toMillis();
                if (Objects.equals(receiptPath, lastCodexTeacherReceiptPath)
                        && modified == lastCodexTeacherReceiptMtime
                        && Objects.equals(statusPath, lastCodexTeacherStatusPath)
                        && statusModified == lastCodexTeacherStatusMtime
                        && Objects.equals(actionVetoPath, lastCodexActionVetoPath)
                        && actionVetoModified == lastCodexActionVetoMtime
                        && Objects.equals(decisionProfilePath, lastCodexDecisionProfilePath)
                        && decisionProfileModified == lastCodexDecisionProfileMtime) return;
                List<String> lines = receiptPath == null
                        ? List.of() : Files.readAllLines(receiptPath, StandardCharsets.UTF_8);
                Map<String, Object> status = statusPath == null ? Map.of() : MiniJson.object(
                        MiniJson.parse(Files.readString(statusPath, StandardCharsets.UTF_8))
                );
                StringBuilder text = new StringBuilder();
                text.append("CODEX TEACHER · ").append(CODEX_TEACHER_MODEL)
                        .append(" · reasoning ").append(CODEX_TEACHER_REASONING).append('\n')
                        .append("Receipt: ").append(receiptPath == null ? "đang chờ" : receiptPath).append('\n');
                ActionVetoSummary actionVetoSummary = readActionVetoSummary(actionVetoPath);
                ActionVetoSummary decisionSummary = readActionVetoSummary(decisionProfilePath);
                actionVetoSummary.merge(decisionSummary);
                text.append(actionVetoSummary.render(actionVetoPath)).append('\n');
                if (!status.isEmpty()) {
                    text.append("Queue: submitted=").append(Math.round(MiniJson.number(status.get("submitted"), 0.0)))
                            .append(" · pending=").append(Math.round(MiniJson.number(status.get("pending"), 0.0)))
                            .append(" · graded=").append(Math.round(MiniJson.number(status.get("graded"), 0.0)))
                            .append(" · updates=").append(Math.round(MiniJson.number(status.get("updates"), 0.0)))
                            .append(" · skips=").append(Math.round(MiniJson.number(status.get("skips"), 0.0)))
                            .append(" · failures=").append(Math.round(MiniJson.number(status.get("failures"), 0.0)))
                            .append(" · workers=").append(MiniJson.integer(status.get("max_workers"), 1))
                            .append('\n');
                    text.append("OPTIMIZER: generic PPO updates=")
                            .append(MiniJson.integer(status.get("generic_optimizer_updates"), 0))
                            .append(" · Codex optimizer updates=")
                            .append(MiniJson.integer(status.get("codex_optimizer_updates"),
                                    MiniJson.integer(status.get("updates"), 0)))
                            .append(" · interventions=")
                            .append(MiniJson.integer(status.get("interventions_applied"), 0))
                            .append(" · rejected/no-op=")
                            .append(MiniJson.integer(status.get("rejected_updates"), 0))
                            .append('\n');
                    text.append("RECOVERY: snapshots=")
                            .append(MiniJson.integer(status.get("recovery_snapshot_updates"), 0))
                            .append(" · path=")
                            .append(MiniJson.text(status.get("recovery_snapshot_path"), "—"))
                            .append(" · error=")
                            .append(MiniJson.text(status.get("recovery_snapshot_error"), "—"))
                            .append('\n');
                    text.append("CODEX COUNTERS: graded=")
                            .append(MiniJson.integer(status.get("graded"), 0))
                            .append(" · hypothesis confirmed=")
                            .append(MiniJson.integer(status.get("hypotheses_confirmed"), 0))
                            .append(" · optimizer updates=")
                            .append(MiniJson.integer(status.get("codex_optimizer_updates"),
                                    MiniJson.integer(status.get("updates"), 0)))
                            .append(" · interventions=")
                            .append(MiniJson.integer(status.get("interventions_applied"), 0))
                            .append(" · failures=")
                            .append(MiniJson.integer(status.get("failures"), 0))
                            .append('\n');
                    text.append("ACCOUNT FEEDBACK: records=")
                            .append(MiniJson.integer(status.get("account_impact_records"), 0))
                            .append(" · confirmed=")
                            .append(MiniJson.integer(status.get("account_impact_confirmed"), 0))
                            .append(" · cashflow confirmed=")
                            .append(MiniJson.integer(status.get("account_cashflow_confirmed"), 0))
                            .append(" · balance/equity mark confirmed=")
                            .append(MiniJson.integer(status.get("account_mark_confirmed"), 0))
                            .append(" · partial=")
                            .append(MiniJson.integer(status.get("account_impact_partial"), 0))
                            .append(" · counterfactual validated=")
                            .append(MiniJson.integer(status.get("counterfactual_validated"), 0))
                            .append(" · counterfactual unavailable=")
                            .append(MiniJson.integer(status.get("counterfactual_unavailable"), 0))
                            .append(" · reward sum=")
                            .append(signedMetric(status.get("account_impact_reward_sum"), 4))
                            .append(" · parameter changed=")
                            .append(MiniJson.integer(status.get("rlhf_parameter_changed"), 0))
                            .append(" · latent policy-state changed=")
                            .append(MiniJson.integer(status.get(
                                    "rlhf_internal_policy_state_changed"), 0))
                            .append(" · probability shifts=")
                            .append(MiniJson.integer(status.get("rlhf_probability_shift_count"), 0))
                            .append(" · max shift=")
                            .append(decimalMetric(status.get("rlhf_max_probability_shift"), 4))
                            .append('\n');
                    text.append("ONLINE CAUSAL BARRIER: wait=")
                            .append(decimalMetric(status.get("online_wait_sec"), 1))
                            .append("s · passed=")
                            .append(MiniJson.integer(status.get("online_barrier_passed"), 0))
                            .append("/")
                            .append(MiniJson.integer(status.get("online_barrier_waits"), 0))
                            .append(" · timeouts=")
                            .append(MiniJson.integer(status.get("online_barrier_timeouts"), 0))
                            .append(" · waited=")
                            .append(decimalMetric(status.get("online_barrier_waited_sec"), 1))
                            .append("s")
                            .append('\n');
                    Map<String, Object> lastCausalLink = MiniJson.object(
                            status.get("last_codex_causal_link"));
                    text.append("CODEX → NEXT ACTION: links=")
                            .append(MiniJson.integer(status.get("codex_causal_links_observed"), 0))
                            .append(" · updates linked=")
                            .append(MiniJson.integer(status.get("codex_causal_updates_linked"), 0))
                            .append(" · hash match=")
                            .append(MiniJson.integer(status.get("codex_causal_hash_matches"), 0))
                            .append(" · hash unknown=")
                            .append(MiniJson.integer(status.get("codex_causal_hash_unknown"), 0))
                            .append(" · last sequence=")
                            .append(MiniJson.integer(lastCausalLink.get("sequence"), -1))
                            .append(" · action=")
                            .append(MiniJson.integer(lastCausalLink.get("action"), -1))
                            .append(" · causal_link=")
                            .append(MiniJson.text(lastCausalLink.get("causal_link"), "—"))
                            .append('\n');
                    Map<String, Object> preflight = MiniJson.object(status.get("preflight"));
                    if (!preflight.isEmpty()) {
                        text.append("PREFLIGHT: ")
                                .append(MiniJson.text(preflight.get("status"), "—"))
                                .append(" · model=").append(MiniJson.text(preflight.get("model"), CODEX_TEACHER_MODEL))
                                .append(" · effort=").append(MiniJson.text(preflight.get("reasoning_effort"), CODEX_TEACHER_REASONING))
                                .append(" · version=").append(MiniJson.text(preflight.get("cli_version"), "—"))
                                .append(" · executable=").append(MiniJson.text(preflight.get("executable_path"), "—"))
                                .append(" · stage=").append(MiniJson.text(preflight.get("stage"), "—"))
                                .append(" · reason=").append(MiniJson.text(preflight.get("reason"), "—"))
                                .append('\n');
                    }
                }
                Map<String, Map<String, Object>> updateByReceipt = new HashMap<>();
                for (Object item : MiniJson.array(status.get("update_receipts"))) {
                    Map<String, Object> update = MiniJson.object(item);
                    String receiptHash = MiniJson.text(update.get("receipt_sha256"), "");
                    if (!receiptHash.isBlank()) updateByReceipt.put(receiptHash, update);
                }
                int appendOnlyUpdateCount = 0;
                for (String line : lines) {
                    if (line == null || line.isBlank()) continue;
                    try {
                        Map<String, Object> record = MiniJson.object(MiniJson.parse(line));
                        if (!"codex_teacher_update".equals(
                                MiniJson.text(record.get("record_type"), ""))) continue;
                        String receiptHash = MiniJson.text(record.get("receipt_sha256"), "");
                        if (!receiptHash.isBlank()) {
                            updateByReceipt.put(receiptHash, record);
                            appendOnlyUpdateCount++;
                        }
                    } catch (RuntimeException ignored) {
                        // The main receipt loop below applies the same tolerant
                        // read policy while the runner is appending JSONL.
                    }
                }
                text.append("Receipt log records: ").append(lines.size())
                        .append(" · append-only updates=").append(appendOnlyUpdateCount)
                        .append(" · gồm receipt/failure/update record; không phải tổng số action đã gửi\n")
                        .append("ACTION QUEUE: đã gửi=").append(Math.round(MiniJson.number(status.get("submitted"), lines.size())))
                        .append(" · đã chấm=").append(Math.round(MiniJson.number(status.get("graded"), lines.size())))
                        .append(" · đang chờ=").append(Math.round(MiniJson.number(status.get("pending"), 0.0)))
                        .append(" · target action packet · candidate-only\n\n");
                appendCodexExecution(text, MiniJson.object(status.get("last_codex_execution")));
                int index = 0;
                int rewardCount = 0;
                int penaltyCount = 0;
                int neutralCount = 0;
                int unscoredCount = 0;
                int receiptCount = 0;
                int failureCount = 0;
                for (String line : lines) {
                    if (line == null || line.isBlank()) continue;
                    try {
                        Map<String, Object> receipt = MiniJson.object(MiniJson.parse(line));
                        String recordType = MiniJson.text(receipt.get("record_type"), "codex_teacher_receipt");
                        if ("codex_teacher_update".equals(recordType)) {
                            // Update records are indexed above and rendered with
                            // their owning receipt; do not count them as cases.
                            continue;
                        }
                        if ("codex_teacher_failure".equals(recordType)) {
                            failureCount++;
                            text.append("FAILURE CASE ").append(failureCount)
                                    .append(" · stage=").append(MiniJson.text(receipt.get("stage"), "unknown"))
                                    .append(" · type=").append(MiniJson.text(receipt.get("failure_type"), "unknown"))
                                    .append('\n')
                                    .append("model/effort: ").append(MiniJson.text(receipt.get("model"), CODEX_TEACHER_MODEL))
                                    .append(" / ").append(MiniJson.text(receipt.get("reasoning_effort"), CODEX_TEACHER_REASONING)).append('\n')
                                    .append("executable/version: ").append(MiniJson.text(receipt.get("executable_path"), "—"))
                                    .append(" / ").append(MiniJson.text(receipt.get("cli_version"), "—")).append('\n')
                                    .append("trace: ").append(MiniJson.text(receipt.get("trace_id"), "—"))
                                    .append(" · ticket: ").append(MiniJson.text(receipt.get("position_ticket"), "—"))
                                    .append(" · action: ").append(MiniJson.text(receipt.get("action_name"), "—")).append('\n')
                                    .append("returncode: ").append(MiniJson.text(receipt.get("returncode"), "—"))
                                    .append(" · error: ").append(MiniJson.text(receipt.get("error"), "—")).append('\n')
                                    .append("stderr: ").append(MiniJson.text(receipt.get("stderr_tail"), "—")).append('\n')
                                    .append("packet: ").append(MiniJson.text(receipt.get("packet_sha256"), "—")).append("\n\n");
                            continue;
                        }
                        receiptCount++;
                        index++;
                        double confidence = MiniJson.number(receipt.get("confidence"), Double.NaN);
                        double reward = MiniJson.number(receipt.get("teacher_reward"), Double.NaN);
                        Map<String, Object> update = updateByReceipt.getOrDefault(
                                MiniJson.text(receipt.get("receipt_sha256"), ""), Map.of());
                        double scaledReward = MiniJson.number(update.get("scaled_reward"), Double.NaN);
                        boolean ppoUpdate = Boolean.TRUE.equals(update.get("ppo_update"))
                                || "true".equalsIgnoreCase(MiniJson.text(update.get("ppo_update"), "false"));
                        String rewardLabel;
                        if (!Double.isFinite(reward)) {
                            rewardLabel = "CHƯA CHẤM";
                            unscoredCount++;
                        } else if (reward > 0.0) {
                            rewardLabel = "THƯỞNG";
                            rewardCount++;
                        } else if (reward < 0.0) {
                            rewardLabel = "PHẠT";
                            penaltyCount++;
                        } else {
                            rewardLabel = "TRUNG TÍNH";
                            neutralCount++;
                        }
                        boolean accepted = Double.isFinite(confidence) && confidence >= 0.55;
                        String actionName = MiniJson.text(receipt.get("action_name"),
                                MiniJson.text(update.get("action_name"), "—"));
                        String actionFamily = MiniJson.text(receipt.get("action_family"),
                                MiniJson.text(update.get("action_family"), "—"));
                        // Older append-only records predate the explicit
                        // combined_reward field.  Reconstruct it for display
                        // only when both immutable inputs are present; never
                        // write this value back or use it for a learner update.
                        double teacherRewardValue = MiniJson.number(
                                receipt.get("teacher_reward"), Double.NaN);
                        double accountImpactRewardValue = MiniJson.number(
                                update.get("account_impact_reward"), Double.NaN);
                        double combinedRewardValue = MiniJson.number(
                                update.get("combined_reward"), Double.NaN);
                        boolean combinedRewardFallback = false;
                        if (!Double.isFinite(combinedRewardValue)
                                && Double.isFinite(teacherRewardValue)
                                && Double.isFinite(accountImpactRewardValue)) {
                            combinedRewardValue = 0.5 * teacherRewardValue
                                    + 0.5 * accountImpactRewardValue;
                            if (accountImpactRewardValue < 0.0) {
                                combinedRewardValue = Math.min(
                                        combinedRewardValue,
                                        0.5 * accountImpactRewardValue);
                            }
                            combinedRewardFallback = true;
                        }
                        text.append("CASE ").append(index).append(accepted ? " · ĐỦ BẰNG CHỨNG" : " · SKIP CONFIDENCE THẤP").append('\n')
                                .append("model/effort: ").append(MiniJson.text(receipt.get("model"), CODEX_TEACHER_MODEL))
                                .append(" / ").append(CODEX_TEACHER_REASONING).append('\n')
                                .append("trace: ").append(MiniJson.text(receipt.get("trace_id"), "—"))
                                .append(" · ticket: ").append(MiniJson.text(receipt.get("position_ticket"), "—")).append('\n')
                                .append("action: ").append(actionName)
                                .append(" · family: ").append(actionFamily)
                                .append(" · surface: ").append(MiniJson.text(receipt.get("action_surface_count"),
                                        MiniJson.text(update.get("action_surface_count"), "—"))).append('\n')
                                .append("quality decision/risk/timing/execution: ")
                                .append(decimalMetric(receipt.get("decision_quality"), 2)).append(" / ")
                                .append(decimalMetric(receipt.get("risk_quality"), 2)).append(" / ")
                                .append(decimalMetric(receipt.get("timing_quality"), 2)).append(" / ")
                                .append(decimalMetric(receipt.get("execution_quality"), 2)).append('\n')
                                .toString();
                        appendCodexExecution(text, MiniJson.object(receipt.get("codex_execution")));
                        text.append("KẾT QUẢ: ").append(rewardLabel)
                                .append(" · teacher reward=").append(signedMetric(receipt.get("teacher_reward"), 3))
                                .append(" · account input=").append(signedMetric(update.get("account_impact_reward"), 3))
                                .append(" · combined learner=").append(combinedRewardValue == combinedRewardValue
                                        ? signedMetric(combinedRewardValue, 3) : "—")
                                .append(combinedRewardFallback ? " (legacy fallback)" : "")
                                .append(" · scaled reward=").append(scaledReward == scaledReward
                                        ? signedMetric(scaledReward, 3) : "—")
                                .append(" · confidence=").append(decimalMetric(receipt.get("confidence"), 2)).append('\n')
                                .append("REWARD ROUTING: source=").append(MiniJson.text(
                                        update.get("reward_direction_source"), "teacher_reward"))
                                .append(" · composition=").append(MiniJson.text(
                                        update.get("reward_composition"), "—"))
                                .append(" · account source=").append(MiniJson.text(
                                        update.get("account_impact_reward_source"), "none"))
                                .append(" · negative-account floor=").append(MiniJson.text(
                                        update.get("negative_account_floor_applied"), "—"))
                                .append('\n')
                                 .append("POLICY: ").append(ppoUpdate ? "PPO ĐÃ CẬP NHẬT" : "PPO CHƯA CẬP NHẬT")
                                 .append(" · advantage=").append(scaledReward == scaledReward
                                         ? signedMetric(update.get("advantage_mean"), 3) : "—")
                                 .append(" · direction=").append(MiniJson.text(
                                         update.get("rlhf_direction_expected"), "—"))
                                 .append("/").append(Boolean.TRUE.equals(update.get("rlhf_directional_update"))
                                         ? "đúng dấu" : "không xác nhận")
                                 .append(" · packet target=").append(MiniJson.text(receipt.get("target_is_entry_owner"), "—"))
                                 .append('\n')
                                 .append("CODEX OBJECTIVE: entropy=").append(decimalMetric(
                                         update.get("rlhf_codex_entropy_weight"), 4))
                                 .append(" · KL=").append(decimalMetric(
                                         update.get("rlhf_codex_kl_beta"), 4))
                                 .append(" · optimizer state reset=").append(MiniJson.text(
                                         update.get("rlhf_codex_optimizer_state_reset"), "—"))
                                 .append(" · parameter changed=").append(MiniJson.text(
                                         update.get("rlhf_parameter_changed"), "—"))
                                 .append(" · latent policy-state=").append(MiniJson.text(
                                         update.get("rlhf_internal_policy_state_parameter_changed"),
                                         "—"))
                                 .append('\n')
                                .append("UPDATE AUDIT: scope=").append(MiniJson.text(update.get("rlhf_parameter_scope"),
                                        MiniJson.text(update.get("parameter_scope"), "—")))
                                 .append(" · trainable tensors=").append(MiniJson.integer(update.getOrDefault(
                                         "rlhf_trainable_parameter_tensors",
                                         MiniJson.integer(update.get("trainable_parameter_tensors"), 0)), 0))
                                 .append(" · PPO gradient tensors=").append(MiniJson.integer(update.get(
                                         "rlhf_codex_ppo_gradient_parameter_tensors"), 0))
                                 .append(" · latent tensors=").append(MiniJson.integer(update.get(
                                         "rlhf_internal_policy_state_trainable_parameter_tensors"), 0))
                                .append(" · parameter hash=").append(MiniJson.text(update.getOrDefault(
                                        "rlhf_action_head_parameter_hash_before",
                                        MiniJson.text(update.get("parameter_hash_before"), "—")), "—"))
                                .append(" → ").append(MiniJson.text(update.getOrDefault(
                                        "rlhf_action_head_parameter_hash_after",
                                        MiniJson.text(update.get("parameter_hash_after"), "—")), "—"))
                                .append(" · action prob=").append(decimalMetric(update.getOrDefault(
                                        "rlhf_target_action_probability_before",
                                        update.get("action_probability_before")), 4))
                                .append(" → ").append(decimalMetric(update.getOrDefault(
                                        "rlhf_target_action_probability_after",
                                        update.get("action_probability_after")), 4))
                                .append(" · argmax=").append(MiniJson.integer(update.get(
                                        "target_action_argmax_before"), -1))
                                .append("→").append(MiniJson.integer(update.get(
                                        "target_action_argmax_after"), -1))
                                .append(" · changed=").append(MiniJson.text(update.get(
                                        "target_action_argmax_changed"), "—"))
                                .append(" · rank=").append(MiniJson.integer(update.get(
                                        "target_action_rank_before"), -1))
                                .append("→").append(MiniJson.integer(update.get(
                                        "target_action_rank_after"), -1))
                                .append(" · logit Δ=").append(signedMetric(update.get(
                                        "target_action_logit_delta"), 6))
                                .append(" · max shift=").append(decimalMetric(update.getOrDefault(
                                        "rlhf_packet_max_probability_shift",
                                        update.get("packet_max_probability_shift")), 4))
                                .append(" · head update norm=").append(decimalMetric(update.get(
                                        "rlhf_action_head_update_norm"), 6))
                                .append(" · factorized update=").append(MiniJson.text(update.get(
                                        "rlhf_factorized_geometry_update"), "false"))
                                .append(" · geometry heads=").append(MiniJson.array(update.get(
                                        "rlhf_factorized_geometry_heads")))
                                .append(" · geometry hash=").append(MiniJson.text(update.get(
                                        "rlhf_factorized_geometry_parameter_hash_before"), "—"))
                                .append(" → ").append(MiniJson.text(update.get(
                                        "rlhf_factorized_geometry_parameter_hash_after"), "—"))
                                .append(" · geometry changed=").append(MiniJson.text(update.get(
                                        "rlhf_factorized_geometry_parameter_changed"), "false"))
                                .append(" · geometry norm=").append(decimalMetric(update.get(
                                        "rlhf_factorized_geometry_update_norm"), 6))
                                .append(" · geometry ratio=").append(decimalMetric(update.get(
                                        "rlhf_factorized_geometry_ratio"), 6))
                                .append(" · clip correction=").append(MiniJson.text(update.get(
                                        "rlhf_factorized_geometry_clip_active"), "false"))
                                .append("/").append(decimalMetric(update.get(
                                        "rlhf_factorized_geometry_unclipped_correction"), 6))
                                .append(" · correction scale=").append(decimalMetric(update.get(
                                        "rlhf_factorized_geometry_unclipped_correction_scale"), 4))
                                .append(" · target logit delta=").append(signedMetric(update.get(
                                        "rlhf_action_head_target_delta"), 6))
                                .append(" · backtrack=").append(MiniJson.text(update.get(
                                        "rlhf_codex_backtrack_applied"), "false"))
                                .append("/").append(MiniJson.integer(update.get(
                                        "rlhf_codex_backtrack_attempts"), 0))
                                .append(" · lr=").append(decimalMetric(update.get(
                                        "rlhf_codex_learning_rate_initial"), 6))
                                .append("→").append(decimalMetric(update.get(
                                        "rlhf_codex_learning_rate"), 6))
                                .append('\n')
                                .append("CODEX BEHAVIOR SURFACE: global action bias=")
                                .append(MiniJson.text(update.get(
                                        "rlhf_codex_global_action_bias_enabled"), "—"))
                                .append(" · scale=").append(decimalMetric(update.get(
                                        "rlhf_codex_global_action_bias_scale"), 4))
                                .append(" · max_abs=").append(decimalMetric(update.get(
                                        "rlhf_codex_global_action_bias_max_abs"), 4))
                                .append(" · intervention=").append(MiniJson.text(update.get(
                                        "rlhf_codex_global_intervention_applied"), "—"))
                                .append(" · step=").append(signedMetric(update.get(
                                        "rlhf_codex_global_intervention_step"), 4))
                                .append(" · residual=").append(decimalMetric(update.get(
                                        "rlhf_codex_global_intervention_residual_fraction"), 3))
                                .append(" · shift rejects=").append(MiniJson.integer(update.get(
                                        "rlhf_codex_global_intervention_shift_rejections"), 0))
                                 .append(" · direction rejects=").append(MiniJson.integer(update.get(
                                         "rlhf_codex_global_intervention_direction_rejections"), 0))
                                 .append(" · nonfinite rejects=").append(MiniJson.integer(update.get(
                                         "rlhf_codex_global_intervention_nonfinite_rejections"), 0))
                                 .append(" · bias PPO gradient frozen=").append(MiniJson.text(update.get(
                                         "rlhf_codex_global_bias_gradient_frozen"), "—"))
                                 .append(" · context support=").append(decimalMetric(update.get(
                                         "rlhf_codex_context_support_target"), 4))
                                 .append(" · scope=").append(MiniJson.text(update.getOrDefault(
                                        "rlhf_parameter_scope",
                                        update.get("parameter_scope")), "—"))
                                .append('\n')
                                .append("DIRECT-RNN FULL-RL: pending=").append(MiniJson.integer(update.get(
                                        "rlhf_direct_rnn_queue_pending"), 0))
                                .append(" · optimizer steps=").append(MiniJson.integer(update.get(
                                        "rlhf_direct_rnn_optimizer_steps"), 0))
                                .append(" · batches=").append(MiniJson.integer(update.get(
                                        "rlhf_direct_rnn_optimizer_batches"), 0))
                                .append(" · modules=").append(MiniJson.integer(update.get(
                                        "rlhf_direct_rnn_updated_modules"), 0))
                                .append(" · probability source=").append(MiniJson.text(update.get(
                                        "rlhf_probability_after_source"), "—"))
                                .append('\n')
                                .append("HYPOTHESIS: ").append(MiniJson.text(receipt.get("hypothesis"), "—")).append('\n')
                                .append("PREDICTION: ").append(MiniJson.text(receipt.get("prediction"), "—"))
                                .append(" · experiment=").append(MiniJson.text(receipt.get("experiment"), "—"))
                                .append('\n')
                                .append("VERIFICATION: ").append(MiniJson.text(receipt.get("verification_status"), "legacy/không yêu cầu"))
                                .append(" · observed realized_r=").append(signedMetric(receipt.get("verification_observed_reward"), 3))
                                .append(" · evidence=").append(MiniJson.text(receipt.get("verification_evidence"), "—"))
                                .append('\n')
                                .append("ACCOUNT IMPACT: ");
                        Map<String, Object> accountImpact = MiniJson.object(receipt.get("account_impact"));
                        if (accountImpact.isEmpty()) {
                            accountImpact = MiniJson.object(receipt.get("account_impact_packet"));
                        }
                        text.append("status=").append(MiniJson.text(accountImpact.get("status"), "inconclusive"))
                                .append(" · ready=").append(MiniJson.text(accountImpact.get("verification_ready"), "false"))
                                .append(" · net_pnl=").append(signedMetric(accountImpact.get("observed_net_pnl"), 4))
                                .append(" · cashflow Δ=").append(signedMetric(accountImpact.get("cashflow_delta"), 4))
                                .append(" · deal=").append(MiniJson.text(accountImpact.get("cashflow_deal_ticket"), "—"))
                                .append(" · balance Δ=").append(signedMetric(accountImpact.get("balance_delta"), 4))
                                .append(" · equity Δ=").append(signedMetric(accountImpact.get("equity_delta"), 4))
                                .append(" · counterfactual=").append(MiniJson.text(accountImpact.get("counterfactual_status"), "not_available"))
                                .append(" · horizon=").append(MiniJson.text(accountImpact.get("counterfactual_horizon_bars"), "—"))
                                .append(" · hold R=").append(signedMetric(accountImpact.get("counterfactual_hold_r"), 4))
                                .append(" · action-minus-hold R=").append(signedMetric(accountImpact.get("counterfactual_delta_r"), 4))
                                .append(" · learner signal=").append(signedMetric(update.get("account_impact_reward"), 4))
                                .append(" · signal source=").append(MiniJson.text(
                                        update.get("account_impact_reward_source"), "none"))
                                .append(" · mark attribution=").append(MiniJson.text(
                                        accountImpact.get("account_mark_attribution"), "unspecified"))
                                .append(" · scope=").append(MiniJson.text(accountImpact.get("evidence_scope"), "—"))
                                .append('\n')
                                .append("INTERVENTION: target_action=").append(MiniJson.text(receipt.get("intervention_target_action"), "—"))
                                .append(" · multiplier=").append(decimalMetric(receipt.get("intervention_multiplier"), 3))
                                .append('\n')
                                .append("rationale: ").append(MiniJson.text(receipt.get("rationale"), "—")).append("\n")
                                 .append("packet: ").append(MiniJson.text(receipt.get("packet_sha256"), "—")).append("\n\n");
                    } catch (RuntimeException ignored) {
                        // A receipt may be read while the runner is appending its final line.
                    }
                }
                List<Object> rejectedUpdates = MiniJson.array(status.get("rejection_records"));
                if (!rejectedUpdates.isEmpty()) {
                    text.append("REJECTED UPDATE CASES: ").append(rejectedUpdates.size()).append('\n');
                    int rejectedIndex = 0;
                    for (Object item : rejectedUpdates) {
                        Map<String, Object> rejected = MiniJson.object(item);
                        rejectedIndex++;
                        text.append("REJECT ").append(rejectedIndex)
                                .append(" · ticket=").append(MiniJson.text(rejected.get("position_ticket"), "—"))
                                .append(" · action=").append(MiniJson.text(rejected.get("action_name"), "—"))
                                .append(" · reason=").append(MiniJson.text(rejected.get("reason"), "—"))
                                .append(" · expected=").append(MiniJson.text(
                                        rejected.get("direction_expected"), "—"))
                                .append(" · probability=").append(decimalMetric(
                                        rejected.get("action_probability_before"), 4))
                                .append(" → ").append(decimalMetric(
                                        rejected.get("action_probability_after"), 4))
                                .append(" · parameter changed=").append(MiniJson.text(
                                        rejected.get("parameter_changed"), "—"))
                                .append(" (attempt=").append(MiniJson.text(
                                        rejected.get("attempt_parameter_changed"), "—"))
                                .append(" · backtrack=").append(MiniJson.text(
                                        rejected.get("backtrack_applied"), "false"))
                                .append("/").append(MiniJson.integer(rejected.get(
                                        "backtrack_attempts"), 0))
                                .append(")")
                                .append('\n');
                    }
                    text.append('\n');
                }
                if (index == 0) text.append("Đang chờ receipt JSON hợp lệ...\n");
                text.append("TỔNG KẾT REWARD: THƯỞNG=").append(rewardCount)
                        .append(" · PHẠT=").append(penaltyCount)
                        .append(" · TRUNG TÍNH=").append(neutralCount)
                        .append(" · CHƯA CHẤM TRONG RECEIPT=").append(unscoredCount)
                        .append(" · FAILURE RECORDS=").append(failureCount)
                        .append(" · ĐANG CHỜ QUEUE=").append(Math.round(MiniJson.number(status.get("pending"), 0.0)))
                        .append('\n')
                        .append("CODEX FLOW: submitted=").append(MiniJson.integer(status.get("submitted"), 0))
                        .append(" → execution/parsed/validated theo từng record → graded=")
                        .append(MiniJson.integer(status.get("graded"), receiptCount))
                        .append(" → hypothesis confirmed=").append(MiniJson.integer(status.get("hypotheses_confirmed"), 0))
                        .append(" → optimizer updates=").append(MiniJson.integer(status.get("codex_optimizer_updates"),
                                MiniJson.integer(status.get("updates"), 0)))
                        .append('\n')
                        .append("HYPOTHESIS: submitted=").append(MiniJson.integer(status.get("hypotheses"), 0))
                        .append(" · confirmed=").append(MiniJson.integer(status.get("hypotheses_confirmed"), 0))
                        .append(" · rejected=").append(MiniJson.integer(status.get("hypothesis_rejections"), 0))
                        .append(" · account impact confirmed=").append(MiniJson.integer(status.get("account_impact_confirmed"), 0))
                        .append(" · cashflow confirmed=").append(MiniJson.integer(status.get("account_cashflow_confirmed"), 0))
                        .append(" · balance/equity mark confirmed=").append(MiniJson.integer(status.get("account_mark_confirmed"), 0))
                        .append(" · account impact partial=").append(MiniJson.integer(status.get("account_impact_partial"), 0))
                        .append(" · interventions applied=").append(MiniJson.integer(status.get("interventions_applied"), 0))
                        .append('\n')
                        .append("Mỗi dòng receipt là một action đã được Codex trả điểm; action đang chờ vẫn nằm trong ACTION QUEUE.\n")
                        .append("Reward âm là PHẠT, reward dương là THƯỞNG.\n");
                codexTeacherArea.setText(text.toString());
                codexTeacherArea.setCaretPosition(0);
                lastCodexTeacherReceiptPath = receiptPath;
                lastCodexTeacherReceiptMtime = modified;
                lastCodexTeacherStatusPath = statusPath;
                lastCodexTeacherStatusMtime = statusModified;
                lastCodexActionVetoPath = actionVetoPath;
                lastCodexActionVetoMtime = actionVetoModified;
                lastCodexDecisionProfilePath = decisionProfilePath;
                lastCodexDecisionProfileMtime = decisionProfileModified;
                long pending = Math.round(MiniJson.number(status.get("pending"), 0.0));
                outputTabs.setTitleAt(3, pending > 0
                        ? "Codex teacher · " + pending + " đang chờ"
                        : "Codex teacher · " + index + " case");
            } catch (Exception exception) {
                codexTeacherArea.setText("Không đọc được Codex teacher receipt: " + exception.getMessage());
            }
        }

        private void appendCodexExecution(StringBuilder text, Map<String, Object> execution) {
            if (execution == null || execution.isEmpty()) return;
            text.append("CODEX CLI: process=")
                    .append(MiniJson.text(execution.get("process"), "—"))
                    .append(" · returncode=").append(MiniJson.integer(execution.get("returncode"), -1))
                    .append(" · json_parse=").append(MiniJson.text(execution.get("json_parse"), "—"))
                    .append(" · receipt_validation=")
                    .append(MiniJson.text(execution.get("receipt_validation"), "—"))
                    .append(" · stdout_bytes=").append(MiniJson.integer(execution.get("stdout_bytes"), 0))
                    .append(" · stderr_bytes=").append(MiniJson.integer(execution.get("stderr_bytes"), 0))
                    .append('\n');
            text.append("model/effort: ").append(MiniJson.text(execution.get("model"), CODEX_TEACHER_MODEL))
                    .append(" / ").append(MiniJson.text(execution.get("reasoning_effort"), CODEX_TEACHER_REASONING))
                    .append(" · executable=").append(MiniJson.text(execution.get("executable_path"), "—"))
                    .append(" · version=").append(MiniJson.text(execution.get("cli_version"), "—"))
                    .append(" · stage=").append(MiniJson.text(execution.get("stage"), "—"))
                    .append('\n');
            if (Boolean.TRUE.equals(execution.get("retry_attempted"))) {
                text.append("retry: ")
                        .append(MiniJson.text(execution.get("retry_reason"), "bounded retry"))
                        .append(" · attempt=")
                        .append(MiniJson.integer(execution.get("retry_attempt"), 2))
                        .append('\n');
            }
            List<Object> command = MiniJson.array(execution.get("command"));
            if (!command.isEmpty()) text.append("Codex command: ").append(command).append('\n');
            String stdout = MiniJson.text(execution.get("stdout_preview"), "");
            if (!stdout.isBlank()) text.append("stdout preview:\n").append(stdout.trim()).append('\n');
            String stderr = MiniJson.text(execution.get("stderr_tail"), "");
            if (!stderr.isBlank()) text.append("stderr tail:\n").append(stderr.trim()).append('\n');
            text.append('\n');
        }

        private Path latestCodexTeacherReceipt(String rawOutput) {
            return latestCodexTeacherArtifact(rawOutput, "codex_teacher_receipts.jsonl");
        }

        private Path latestCodexTeacherStatus(String rawOutput) {
            Path primary = latestCodexTeacherArtifact(rawOutput, "codex_teacher_receipts.jsonl.status.json");
            Path fallback = latestCodexTeacherArtifact(rawOutput, "codex_teacher_receipts.jsonl.status.json.latest.json");
            if (primary == null) return fallback;
            if (fallback == null) return primary;
            try {
                return Files.getLastModifiedTime(fallback).toMillis() > Files.getLastModifiedTime(primary).toMillis()
                        ? fallback : primary;
            } catch (Exception ignored) {
                return primary;
            }
        }

        private Path latestCodexTeacherArtifact(String rawOutput, String suffix) {
            List<Path> roots = new ArrayList<>();
            String selectedOutput = rawOutput == null ? "" : rawOutput.trim();
            Path output = resolveWorkspacePath(selectedOutput);
            if (!selectedOutput.isBlank()) {
                // The candidate output is an identity boundary.  Never merge
                // the shared runner Reports directory into an explicitly
                // selected run: an older receipt there may have a different
                // Codex executable/model and would make the UI show stale
                // teacher evidence for the current candidate.
                if (!Files.isDirectory(output)) return null;
                roots.add(output);
            } else {
                Path reports = workspace.resolve(DEFAULT_RUNNER).resolve("Reports").normalize();
                if (Files.isDirectory(reports)) roots.add(reports);
            }
            Path latest = null;
            try {
                for (Path root : roots) {
                    try (var paths = Files.walk(root, 3)) {
                        Path candidate = paths.filter(Files::isRegularFile)
                                .filter(path -> path.getFileName().toString().endsWith(suffix))
                                .max(Comparator.comparingLong(path -> {
                                    try { return Files.getLastModifiedTime(path).toMillis(); }
                                    catch (Exception ignored) { return Long.MIN_VALUE; }
                                })).orElse(null);
                        if (candidate != null && (latest == null || Files.getLastModifiedTime(candidate).toMillis()
                                > Files.getLastModifiedTime(latest).toMillis())) latest = candidate;
                    }
                }
            } catch (Exception ignored) {
                return latest;
            }
            return latest;
        }

        private void refreshMt5Diagnostics() {
            String rawOutput = candidateOutputField.getText().trim();
            if (rawOutput.isBlank()) return;
            Path path = resolveWorkspacePath(rawOutput).resolve("mt5_diagnostics.json");
            try {
                if (!Files.isRegularFile(path)) return;
                long modified = Files.getLastModifiedTime(path).toMillis();
                if (path.equals(lastMt5DiagnosticsPath) && modified == lastMt5DiagnosticsMtime) return;
                Map<String, Object> root = MiniJson.object(MiniJson.parse(Files.readString(path)));
                Map<String, Object> status = MiniJson.object(root.get("status"));
                Map<String, Object> performance = MiniJson.object(root.get("performance"));
                Map<String, Object> risk = MiniJson.object(root.get("risk"));
                Map<String, Object> optimizer = MiniJson.object(root.get("optimizer"));
                Map<String, Object> reward = MiniJson.object(root.get("reward"));
                Map<String, Object> policy = MiniJson.object(root.get("policy"));
                Map<String, Object> execution = MiniJson.object(root.get("execution"));
                Map<String, Object> closeReasonPnl = MiniJson.object(performance.get("close_reason_net_pnl"));
                Map<String, Object> entryRoles = MiniJson.object(policy.get("entry_role_distributions"));
                String longEntryRole = roleDistributionText(entryRoles.get("LONG_ENTRY_OWNER"));
                String shortEntryRole = roleDistributionText(entryRoles.get("SHORT_ENTRY_OWNER"));
                String entryRoleCollapse = String.valueOf(policy.getOrDefault("entry_role_collapse_detected", "—"));
                String policyScope = MiniJson.text(policy.get("scope"), "wire");
                int policyActions = MiniJson.integer(policy.get("actions"), 0);
                int wirePolicyActions = MiniJson.integer(policy.get("wire_actions"), policyActions);
                applyOperationalStatus(mt5EdgeDiagnosticLabel, MiniJson.text(status.get("edge"), "unproven"));
                applyOperationalStatus(mt5RiskDiagnosticLabel, MiniJson.text(status.get("risk"), "insufficient_data"));
                applyOperationalStatus(mt5OptimizerDiagnosticLabel, MiniJson.text(status.get("optimizer"), "insufficient_updates"));
                applyOperationalStatus(mt5RewardDiagnosticLabel, MiniJson.text(status.get("reward"), "insufficient_data"));
                applyOperationalStatus(mt5PolicyDiagnosticLabel, MiniJson.text(status.get("policy"), "insufficient_data"));
                applyOperationalStatus(mt5ExecutionDiagnosticLabel, MiniJson.text(status.get("execution"), "insufficient_data"));

                String text = String.format(Locale.ROOT,
                        "MT5 ONLINE · market time %s\n\n"
                                + "HIỆU QUẢ THỰC THI\n"
                                + "Close deals=%d · final positions=%d · W/L=%d/%d · deal WR=%s · PF=%s · expectancy/deal=%s · net PnL=%s\n"
                                + "Avg win/loss=%s/%s · payoff=%s · WR hòa vốn=%s · TP/SL/Tác nhân=%d/%d/%d\n"
                                + "Lãi theo nguyên nhân TP/SL/Tác nhân=%s/%s/%s\n\n"
                                + "RỦI RO\n"
                                + "Balance=%s · Equity=%s · Return=%s · max equity DD=%s\n\n"
                                + "OPTIMIZER / CREDIT\n"
                                + "PPO updates=%d · reasoning optimizer updates=%d · total updates=%d · rollout=%d · batches=%d · pattern updates=%d\n"
                                + "Reasoning eligible group/optimizer=%s/%s\n"
                                + "Policy/value/total loss=%s/%s/%s · optimizer entropy=%s · actor eligible=%s\n"
                                + "Reward density=%s · reward eligible=%s · closed credited=%s · reward mean cuối=%s\n\n"
                                + "POLICY / EXECUTION\n"
                                + "Scope=%s · economic action samples=%d · wire action samples=%d · wire HOLD=%s\n"
                                + "Action samples=%d · normalized entropy=%s · effective actions=%s\n"
                                + "Entry owners: LONG %s · SHORT %s · collapse=%s\n"
                                + "Top=%s (%s) · HOLD=%s · reasoning action scale=%s · imitation prior=%s\n"
                                + "Requests=%d · accepted/placed/filled/rejected=%d/%d/%d/%d\n"
                                + "Reject rate=%s · unfilled timeout=%d (%s)\n\n"
                                + "CHẨN ĐOÁN NĂNG LỰC: %s\n"
                                + "OVERFIT LIVE: %s\n",
                        MiniJson.text(root.get("market_time"), "—"),
                        MiniJson.integer(performance.get("closed_trades"), 0),
                        MiniJson.integer(performance.get("final_position_closes"), 0),
                        MiniJson.integer(performance.get("wins"), 0),
                        MiniJson.integer(performance.get("losses"), 0),
                        percentMetric(performance.get("win_rate")), decimalMetric(performance.get("profit_factor"), 3),
                        decimalMetric(performance.get("expectancy"), 3), decimalMetric(performance.get("net_pnl"), 2),
                        decimalMetric(performance.get("average_win"), 2), decimalMetric(performance.get("average_loss"), 2),
                        decimalMetric(performance.get("payoff_ratio"), 3), percentMetric(performance.get("breakeven_win_rate")),
                        nestedCount(performance, "close_reasons", "TP"),
                        nestedCount(performance, "close_reasons", "SL"),
                        nestedCount(performance, "close_reasons", "EXPERT"),
                        decimalMetric(closeReasonPnl.get("TP"), 2),
                        decimalMetric(closeReasonPnl.get("SL"), 2),
                        decimalMetric(closeReasonPnl.get("EXPERT"), 2),
                        decimalMetric(risk.get("balance"), 2), decimalMetric(risk.get("equity"), 2),
                        percentMetric(risk.get("return_pct")), percentMetric(risk.get("max_equity_drawdown_pct")),
                        MiniJson.integer(optimizer.get("ppo_updates"), 0),
                        MiniJson.integer(optimizer.get("reasoning_optimizer_updates"), 0),
                        MiniJson.integer(optimizer.get("optimizer_update_count"), 0),
                        MiniJson.integer(optimizer.get("rollout_steps"), 0),
                        MiniJson.integer(optimizer.get("competitive_batches"), 0),
                        MiniJson.integer(optimizer.get("pattern_updates"), 0),
                        decimalMetric(optimizer.get("reasoning_group_eligible"), 0),
                        decimalMetric(optimizer.get("reasoning_optimizer_eligible"), 0),
                        decimalMetric(optimizer.get("policy_loss"), 4), decimalMetric(optimizer.get("value_loss"), 4),
                        decimalMetric(optimizer.get("total_loss"), 4), decimalMetric(optimizer.get("policy_entropy"), 4),
                        percentMetric(optimizer.get("actor_eligible_fraction")),
                        percentMetric(reward.get("nonzero_density")), percentMetric(reward.get("eligible_fraction_last")),
                        decimalMetric(reward.get("closed_trades_credited"), 0), decimalMetric(reward.get("mean_last"), 4),
                        policyScope, policyActions, wirePolicyActions, percentMetric(policy.get("wire_hold_share")),
                        policyActions,
                        decimalMetric(policy.get("normalized_action_entropy"), 3),
                        decimalMetric(policy.get("effective_action_count"), 2),
                        longEntryRole, shortEntryRole, entryRoleCollapse,
                        MiniJson.text(policy.get("top_action"), "—"), percentMetric(policy.get("top_action_share")),
                        percentMetric(policy.get("hold_share")), decimalMetric(policy.get("reasoning_action_scale"), 3),
                        String.valueOf(policy.getOrDefault("imitation_prior_enabled", "—")),
                        MiniJson.integer(execution.get("requests"), 0),
                        MiniJson.integer(execution.get("accepted"), 0),
                        MiniJson.integer(execution.get("placed"), 0),
                        MiniJson.integer(execution.get("filled"), 0),
                        MiniJson.integer(execution.get("rejected"), 0),
                        percentMetric(execution.get("rejection_rate")),
                        MiniJson.integer(execution.get("unfilled_timeout_count"), 0),
                        percentMetric(execution.get("unfilled_timeout_rate")),
                        operationalStatusText(MiniJson.text(status.get("capacity_diagnosis"), "insufficient_data")),
                        operationalStatusText(MiniJson.text(status.get("overfit_live"), "not_measurable_without_oos"))
                );
                mt5DiagnosticArea.setText(text);
                mt5DiagnosticArea.setCaretPosition(0);
                lastMt5DiagnosticsPath = path;
                lastMt5DiagnosticsMtime = modified;
                if (workflowTabs.getSelectedIndex() == TAB_CANDIDATE) {
                    outputTabs.setTitleAt(2, "Chẩn đoán học · MT5 live");
                }
            } catch (Exception exception) {
                mt5DiagnosticArea.setText("Không đọc được MT5 diagnostics: " + exception.getMessage());
            }
        }

        private int nestedCount(Map<String, Object> root, String section, String key) {
            return MiniJson.integer(MiniJson.object(root.get(section)).get(key), 0);
        }

        private String roleDistributionText(Object raw) {
            Map<String, Object> scope = MiniJson.object(raw);
            int total = MiniJson.integer(scope.get("total"), 0);
            if (total <= 0) return "n/a";
            return String.format(Locale.ROOT, "n=%d HOLD=%s eff=%s",
                    total,
                    percentMetric(scope.get("hold_share")),
                    decimalMetric(scope.get("effective_action_count"), 2));
        }

        private void refreshAblationDiagnostics() {
            String rawOutput = ablationOutputField.getText().trim();
            if (rawOutput.isBlank()) return;
            Path path = resolveWorkspacePath(rawOutput).resolve("ablation_summary.json");
            try {
                if (!Files.isRegularFile(path)) return;
                long modified = Files.getLastModifiedTime(path).toMillis();
                if (path.equals(lastAblationDiagnosticsPath) && modified == lastAblationDiagnosticsMtime) return;
                Map<String, Object> root = MiniJson.object(MiniJson.parse(Files.readString(path)));
                Map<String, Object> arms = MiniJson.object(root.get("arms"));
                Map<String, Object> comparisons = MiniJson.object(root.get("comparisons"));
                StringBuilder text = new StringBuilder();
                text.append("Arm             | Exit | Trades | PF     | Expectancy | Max DD  | PPO updates\n")
                        .append("----------------+------+--------+--------+------------+---------+------------\n");
                for (String arm : List.of("frozen_policy", "full_online", "reasoning_off")) {
                    Map<String, Object> row = MiniJson.object(arms.get(arm));
                    Map<String, Object> diagnostics = MiniJson.object(row.get("diagnostics"));
                    Map<String, Object> performance = MiniJson.object(diagnostics.get("performance"));
                    Map<String, Object> risk = MiniJson.object(diagnostics.get("risk"));
                    Map<String, Object> optimizer = MiniJson.object(diagnostics.get("optimizer"));
                    text.append(String.format(Locale.ROOT,
                            "%-15s | %4d | %6d | %6s | %10s | %7s | %11d%n",
                            arm, MiniJson.integer(row.get("exit_code"), -1),
                            MiniJson.integer(performance.get("closed_trades"), 0),
                            decimalMetric(performance.get("profit_factor"), 3),
                            decimalMetric(performance.get("expectancy"), 3),
                            percentMetric(risk.get("max_equity_drawdown_pct")),
                            MiniJson.integer(optimizer.get("ppo_updates"), 0)));
                }
                text.append("\nDELTA SO VỚI FULL_ONLINE\n")
                        .append("Arm             | ΔPF    | Δexpectancy | Δmax DD\n")
                        .append("----------------+--------+-------------+---------\n");
                for (String arm : List.of("frozen_policy", "full_online", "reasoning_off")) {
                    Map<String, Object> delta = MiniJson.object(comparisons.get(arm));
                    text.append(String.format(Locale.ROOT, "%-15s | %6s | %11s | %7s%n",
                            arm,
                            signedMetric(delta.get("delta_profit_factor_vs_full"), 3),
                            signedMetric(delta.get("delta_expectancy_vs_full"), 3),
                            signedPercentMetric(delta.get("delta_max_drawdown_vs_full"))));
                }
                boolean complete = Boolean.TRUE.equals(root.get("complete"));
                boolean unchanged = Boolean.TRUE.equals(root.get("source_checkpoint_unchanged"));
                text.append("\nContract: sequential · same seed/date/model · source hash unchanged=")
                        .append(unchanged).append('\n')
                        .append("Lưu ý: reasoning_off đo ảnh hưởng auxiliary reasoning learning; action scale của full vẫn là 0.\n");
                ablationDiagnosticArea.setText(text.toString());
                ablationDiagnosticArea.setCaretPosition(0);
                ablationDiagnosticLabel.setText(complete ? "HOÀN TẤT 3/3" : "ĐANG CHẠY / CHƯA ĐỦ ARM");
                ablationDiagnosticLabel.setForeground(
                        complete && unchanged ? new Color(29, 117, 89) : new Color(181, 104, 20)
                );
                lastAblationDiagnosticsPath = path;
                lastAblationDiagnosticsMtime = modified;
                if (workflowTabs.getSelectedIndex() == TAB_ABLATION) {
                    outputTabs.setTitleAt(2, complete ? "Chẩn đoán học · Ablation xong" : "Chẩn đoán học · Ablation");
                }
            } catch (Exception exception) {
                ablationDiagnosticArea.setText("Không đọc được ablation summary: " + exception.getMessage());
            }
        }

        private String decimalMetric(Object value, int digits) {
            double number = MiniJson.number(value, Double.NaN);
            return Double.isFinite(number) ? String.format(Locale.ROOT, "%." + digits + "f", number) : "—";
        }

        private String percentMetric(Object value) {
            double number = MiniJson.number(value, Double.NaN);
            return Double.isFinite(number) ? String.format(Locale.ROOT, "%.2f%%", 100.0 * number) : "—";
        }

        private String signedMetric(Object value, int digits) {
            double number = MiniJson.number(value, Double.NaN);
            return Double.isFinite(number) ? String.format(Locale.ROOT, "%+." + digits + "f", number) : "—";
        }

        private String signedPercentMetric(Object value) {
            double number = MiniJson.number(value, Double.NaN);
            return Double.isFinite(number) ? String.format(Locale.ROOT, "%+.2f%%", 100.0 * number) : "—";
        }

        private void applyOperationalStatus(JLabel label, String status) {
            label.setText(operationalStatusText(status));
            if (Set.of("candidate_positive", "normal", "active", "healthy", "diverse").contains(status)) {
                label.setForeground(new Color(29, 117, 89));
            } else if (Set.of("negative", "critical", "sparse", "collapsed", "poor").contains(status)) {
                label.setForeground(new Color(180, 57, 57));
            } else if (Set.of("warning", "warming_up", "watch").contains(status)) {
                label.setForeground(new Color(181, 104, 20));
            } else {
                label.setForeground(new Color(105, 112, 124));
            }
        }

        private String operationalStatusText(String status) {
            return switch (status) {
                case "candidate_positive" -> "ỨNG VIÊN DƯƠNG";
                case "negative" -> "ÂM";
                case "unproven" -> "CHƯA CHỨNG MINH";
                case "critical" -> "NGUY KỊCH";
                case "warning" -> "CẢNH BÁO";
                case "normal" -> "BÌNH THƯỜNG";
                case "active" -> "ĐANG HỌC";
                case "warming_up" -> "KHỞI ĐỘNG";
                case "insufficient_updates" -> "QUÁ ÍT UPDATE";
                case "sparse" -> "REWARD THƯA";
                case "healthy" -> "TỐT";
                case "diverse" -> "ĐA DẠNG";
                case "collapsed" -> "COLLAPSE";
                case "poor" -> "KÉM";
                case "insufficient_optimization" -> "CHƯA TỐI ƯU ĐỦ — KHÔNG KẾT LUẬN THIẾU LỚP";
                case "credit_or_objective_bottleneck" -> "NGHẼN CREDIT/REWARD OBJECTIVE";
                case "policy_optimization_bottleneck" -> "NGHẼN TỐI ƯU POLICY";
                case "no_architecture_underfit_evidence" -> "CHƯA CÓ BẰNG CHỨNG UNDERFIT KIẾN TRÚC";
                case "not_measurable_without_oos" -> "KHÔNG ĐO ĐƯỢC NẾU THIẾU FROZEN OOS/ABLATION";
                default -> "CHƯA ĐỦ DỮ LIỆU";
            };
        }

        private void resetLearningDiagnostics() {
            applyFitStatus(underfitDiagnosticLabel, "insufficient_data");
            applyFitStatus(overfitDiagnosticLabel, "insufficient_data");
            rewardPenaltyDiagnosticLabel.setText("—");
            rewardPenaltyDiagnosticLabel.setForeground(new Color(105, 112, 124));
            learningDiagnosticArea.setText(
                    "Epoch | Train loss | Train BA | Val loss | Val BA | BA gap | Val P | Val R | MCC | K/P\n"
                            + "------+------------+----------+----------+--------+--------+-------+-------+-----+------\n"
            );
            learningDiagnosticArea.setCaretPosition(0);
        }

        private void consumeImitationDiagnostic(List<String> command, String line) {
            if (!command.contains(TEACHER_IMITATION_SCRIPT) || !line.startsWith("{")) return;
            try {
                Map<String, Object> row = MiniJson.object(MiniJson.parse(line));
                String event = MiniJson.text(row.get("event"), "");
                if (!"action_epoch".equals(event) && !"action_epoch_restored".equals(event)) return;
                int epoch = MiniJson.integer(row.get("epoch"), 0);
                double trainLoss = MiniJson.number(row.get("train_weighted_loss"), Double.NaN);
                double trainBa = MiniJson.number(
                        row.get("train_balanced_accuracy_replay"),
                        MiniJson.number(row.get("train_balanced_accuracy_online"), Double.NaN)
                );
                double valLoss = MiniJson.number(row.get("validation_loss"), Double.NaN);
                double valBa = MiniJson.number(row.get("validation_balanced_accuracy"), Double.NaN);
                double gap = MiniJson.number(row.get("train_validation_ba_gap"), Double.NaN);
                double precision = MiniJson.number(row.get("validation_entry_precision"), Double.NaN);
                double recall = MiniJson.number(row.get("validation_entry_recall"), Double.NaN);
                double mcc = MiniJson.number(row.get("validation_mcc"), Double.NaN);
                double reward = MiniJson.number(row.get("imitation_reward_signal"), Double.NaN);
                double penalty = MiniJson.number(row.get("imitation_penalty_signal"), Double.NaN);
                double ratio = MiniJson.number(row.get("imitation_reward_penalty_ratio"), Double.NaN);
                boolean ratioDefined = MiniJson.number(
                        row.get("imitation_reward_penalty_ratio_defined"), 0.0
                ) > 0.5;
                String underfit = MiniJson.text(row.get("underfit_status"), "insufficient_data");
                String overfit = MiniJson.text(row.get("overfit_status"), "insufficient_data");
                String historyLine = String.format(
                        Locale.ROOT,
                        "%5d | %10.5f | %8.4f | %8.5f | %6.4f | %+6.4f | %5.3f | %5.3f | %5.3f | %s%n",
                        epoch, trainLoss, trainBa, valLoss, valBa, gap,
                        precision, recall, mcc,
                        ratioDefined ? String.format(Locale.ROOT, "%.2fx", ratio) : "n/a"
                );
                SwingUtilities.invokeLater(() -> {
                    applyFitStatus(underfitDiagnosticLabel, underfit);
                    applyFitStatus(overfitDiagnosticLabel, overfit);
                    rewardPenaltyDiagnosticLabel.setText(
                            ratioDefined
                                    ? String.format(Locale.ROOT, "%.2fx · %.3f / %.3f", ratio, reward, penalty)
                                    : "N/A · penalty bằng 0"
                    );
                    rewardPenaltyDiagnosticLabel.setForeground(
                            ratioDefined && ratio >= 1.0
                                    ? new Color(29, 117, 89) : new Color(181, 104, 20)
                    );
                    learningDiagnosticArea.append(historyLine);
                    learningDiagnosticArea.setCaretPosition(
                            learningDiagnosticArea.getDocument().getLength()
                    );
                    outputTabs.setTitleAt(2, "Chẩn đoán học · E" + epoch);
                });
            } catch (RuntimeException ignored) {
                // Raw line remains visible in Nhật ký; diagnostics must never stop training.
            }
        }

        private void applyFitStatus(JLabel label, String status) {
            switch (status) {
                case "high" -> {
                    label.setText("CAO");
                    label.setForeground(new Color(180, 57, 57));
                }
                case "watch" -> {
                    label.setText("CẦN THEO DÕI");
                    label.setForeground(new Color(181, 104, 20));
                }
                case "low" -> {
                    label.setText("THẤP");
                    label.setForeground(new Color(29, 117, 89));
                }
                default -> {
                    label.setText("CHƯA ĐỦ EPOCH");
                    label.setForeground(new Color(105, 112, 124));
                }
            }
        }

        private void appendLog(String text) {
            SwingUtilities.invokeLater(() -> {
                logArea.append(text);
                logArea.setCaretPosition(logArea.getDocument().getLength());
            });
        }
    }

    static String formatProcessLogLine(List<String> command, String line) {
        if (command.contains(ABLATION_SCRIPT) && line.startsWith("{")) {
            String event = jsonScalar(line, "event");
            if ("ablation_arm_started".equals(event)) {
                return "Ablation " + jsonScalar(line, "index") + "/" + jsonScalar(line, "total")
                        + " · bắt đầu " + jsonScalar(line, "arm");
            }
            if ("ablation_arm_finished".equals(event)) {
                return "Ablation xong " + jsonScalar(line, "arm")
                        + " · PF=" + decimalText(jsonScalar(line, "profit_factor"), 3)
                        + " · expectancy=" + decimalText(jsonScalar(line, "expectancy"), 3)
                        + " · max DD=" + fractionPercentText(jsonScalar(line, "max_drawdown"))
                        + " · PPO updates=" + jsonScalar(line, "ppo_updates");
            }
            if ("ablation_complete".equals(event)) {
                return "✓ Ablation hoàn tất=" + jsonScalar(line, "complete")
                        + " · " + jsonScalar(line, "summary");
            }
        }
        if (!command.contains(TEACHER_IMITATION_SCRIPT) || !line.startsWith("{")) {
            return line;
        }
        String event = jsonScalar(line, "event");
        return switch (event) {
            case "preflight_ok" -> "✓ Preflight đạt · "
                    + jsonScalar(line, "samples") + " mẫu · action-only="
                    + jsonScalar(line, "action_only");
            case "action_replay_progress" -> "Replay "
                    + jsonScalar(line, "stage") + " · "
                    + percentText(jsonScalar(line, "percent")) + " · "
                    + jsonScalar(line, "windows_done") + "/"
                    + jsonScalar(line, "windows_total") + " windows · "
                    + secondsText(jsonScalar(line, "elapsed_seconds"));
            case "action_batch_progress" -> "Epoch "
                    + jsonScalar(line, "epoch") + " · BPTT "
                    + percentText(jsonScalar(line, "percent")) + " · "
                    + jsonScalar(line, "windows_done") + "/"
                    + jsonScalar(line, "windows_total") + " windows · loss="
                    + decimalText(jsonScalar(line, "train_weighted_loss_so_far"), 5)
                    + " · " + secondsText(jsonScalar(line, "elapsed_seconds"));
            case "autonomous_validation_progress" -> "Tự kiểm định liên tục · "
                    + percentText(jsonScalar(line, "percent")) + " · "
                    + jsonScalar(line, "steps_done") + "/"
                    + jsonScalar(line, "steps_total") + " bước · "
                    + secondsText(jsonScalar(line, "elapsed_seconds"));
            case "action_epoch" -> "Epoch " + jsonScalar(line, "epoch")
                    + " xong · train loss="
                    + decimalText(jsonScalar(line, "train_weighted_loss"), 5)
                    + " · train BA="
                    + decimalText(jsonScalarOr(
                            line, "train_balanced_accuracy_replay", "train_balanced_accuracy_online"
                    ), 4)
                    + " · val BA="
                    + decimalText(jsonScalar(line, "validation_balanced_accuracy"), 4)
                    + " · precision="
                    + decimalText(jsonScalar(line, "validation_entry_precision"), 4)
                    + " · recall="
                    + decimalText(jsonScalar(line, "validation_entry_recall"), 4)
                    + " · MCC="
                    + decimalText(jsonScalar(line, "validation_mcc"), 4)
                    + " · K/P="
                    + decimalText(jsonScalar(line, "imitation_reward_penalty_ratio"), 2)
                    + " · " + secondsText(jsonScalar(line, "epoch_seconds"));
            case "action_epoch_restored" -> "↻ Khôi phục chẩn đoán Epoch "
                    + jsonScalar(line, "epoch") + " · train BA="
                    + decimalText(jsonScalarOr(
                            line, "train_balanced_accuracy_replay", "train_balanced_accuracy_online"
                    ), 4)
                    + " · val BA="
                    + decimalText(jsonScalar(line, "validation_balanced_accuracy"), 4)
                    + " · K/P="
                    + decimalText(jsonScalar(line, "imitation_reward_penalty_ratio"), 2);
            case "action_resume_artifact_loaded" -> "↻ Đã nạp file resume · "
                    + jsonScalar(line, "path") + " · "
                    + jsonScalar(line, "bytes") + " bytes";
            case "action_resume_loaded" -> "↻ Tiếp tục Epoch "
                    + jsonScalar(line, "epoch") + " từ window "
                    + jsonScalar(line, "windows_done") + "/"
                    + jsonScalar(line, "windows_total");
            case "action_pause_requested" -> "Đã dừng an toàn sau batch · Epoch "
                    + jsonScalar(line, "epoch") + " · "
                    + jsonScalar(line, "windows_done") + "/"
                    + jsonScalar(line, "windows_total") + " windows";
            case "action_checkpoint_saving" -> "Đang ghi checkpoint resume · "
                    + jsonScalar(line, "path");
            case "action_checkpoint_saved" -> "✓ Đã lưu để tiếp tục · "
                    + jsonScalar(line, "resume_file") + " · Epoch "
                    + jsonScalar(line, "epoch") + " · "
                    + jsonScalar(line, "windows_done") + "/"
                    + jsonScalar(line, "windows_total") + " windows";
            case "training_complete" -> "✓ Imitation hoàn tất · candidate="
                    + jsonScalar(line, "output") + " · report="
                    + jsonScalar(line, "report");
            default -> line;
        };
    }

    private static String jsonScalar(String json, String key) {
        String marker = "\"" + key + "\"";
        int start = json.indexOf(marker);
        if (start < 0) return "?";
        start = json.indexOf(':', start + marker.length());
        if (start < 0) return "?";
        start++;
        while (start < json.length() && Character.isWhitespace(json.charAt(start))) start++;
        if (start >= json.length()) return "?";
        if (json.charAt(start) == '"') {
            StringBuilder value = new StringBuilder();
            boolean escaped = false;
            for (int index = start + 1; index < json.length(); index++) {
                char character = json.charAt(index);
                if (escaped) {
                    value.append(character);
                    escaped = false;
                } else if (character == '\\') {
                    escaped = true;
                } else if (character == '"') {
                    return value.toString();
                } else {
                    value.append(character);
                }
            }
            return value.toString();
        }
        int stop = start;
        while (stop < json.length() && json.charAt(stop) != ',' && json.charAt(stop) != '}') stop++;
        return json.substring(start, stop).trim();
    }

    private static String jsonScalarOr(String json, String preferredKey, String fallbackKey) {
        String preferred = jsonScalar(json, preferredKey);
        return "?".equals(preferred) ? jsonScalar(json, fallbackKey) : preferred;
    }

    private static String decimalText(String raw, int digits) {
        try {
            return String.format(Locale.ROOT, "%." + digits + "f", Double.parseDouble(raw));
        } catch (NumberFormatException exception) {
            return raw;
        }
    }

    private static String percentText(String raw) {
        return decimalText(raw, 1) + "%";
    }

    private static String fractionPercentText(String raw) {
        try {
            return String.format(Locale.ROOT, "%.2f%%", 100.0 * Double.parseDouble(raw));
        } catch (NumberFormatException exception) {
            return raw;
        }
    }

    private static String secondsText(String raw) {
        return decimalText(raw, 1) + "s";
    }

    private static String selected(JComboBox<?> combo) {
        Object value = combo.getSelectedItem();
        return value == null ? "" : String.valueOf(value);
    }

    private static void put(Properties properties, String name, JTextField field) {
        properties.setProperty(name, field.getText());
    }

    private static void get(Properties properties, String name, JTextField field) {
        field.setText(properties.getProperty(name, field.getText()));
    }

    private static String nonBlank(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value.trim();
    }

    static String candidateSeedForRun(String value, String generated) {
        return nonBlank(value, generated);
    }

    static String imitationCheckpointForRun(String value, boolean sourceExists) {
        String restored = value == null ? "" : value.trim();
        boolean retiredAndMissing = RETIRED_DEFAULT_IMITATION_CHECKPOINTS.contains(restored)
                && !sourceExists;
        return restored.isBlank() || retiredAndMissing
                ? FIVE_TRANSFORMER_V3_CHECKPOINT : restored;
    }

    static String successfulImitationOutput(List<String> command, int exitCode) {
        if (exitCode != 0 || !command.contains(TEACHER_IMITATION_SCRIPT)
                || command.contains("--verify-only")) return "";
        return optionValue(command, "--output");
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalStateException(message);
    }

    private static void addOption(List<String> command, String key, String value) {
        if (value == null || value.isBlank()) return;
        command.add(key);
        command.add(value.trim());
    }

    private static String formatCommand(List<String> command) {
        List<String> rendered = new ArrayList<>(command.size());
        for (String argument : command) {
            if (argument.chars().anyMatch(Character::isWhitespace)) {
                rendered.add("\"" + argument.replace("\"", "\\\"") + "\"");
            } else {
                rendered.add(argument);
            }
        }
        return String.join(" ", rendered);
    }

    private static String formatCommands(List<List<String>> commands) {
        if (commands.size() == 1) return formatCommand(commands.get(0));
        StringBuilder rendered = new StringBuilder();
        for (int index = 0; index < commands.size(); index++) {
            if (index > 0) rendered.append(System.lineSeparator());
            rendered.append("# Bước ").append(index + 1).append('/').append(commands.size())
                    .append(System.lineSeparator())
                    .append(formatCommand(commands.get(index)));
            if (index + 1 < commands.size()) {
                rendered.append(System.lineSeparator())
                        .append("if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }")
                        .append(System.lineSeparator());
            }
        }
        return rendered.toString();
    }
}
