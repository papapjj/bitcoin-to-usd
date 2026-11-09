"""Desktop UI for training and observing the self-growing neural graph.

The live network view is embedded directly in the Tk application.  It tails the
same append-only JSONL telemetry stream used by the trainer; it never sends
commands or state back to the policy, MT5 bridge, plasticity, or growth code.
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Iterable, Mapping

from .config import AppConfig, load_config
from .data import load_market_data
from .live_observatory import DEFAULT_EVENT_PATH, LiveEventReader
from .self_growing_runner import (
    audit_policy,
    constant_hold_policy,
    evaluate_policy,
    load_policy_callable,
    train_self_growing_mt5,
)


REGION_ORDER: dict[str, int] = {
    "input": 0,
    "encoder": 1,
    "memory": 2,
    "hypothesis": 3,
    "curiosity": 3,
    "development": 3,
    "predictor": 4,
    "risk": 4,
    "critic": 5,
    "action_head": 5,
    "selector": 6,
    "utility": 6,
    "meta": 7,
    "output": 8,
}

REGION_COLORS: dict[str, str] = {
    "input": "#94a3b8",
    "encoder": "#60a5fa",
    "memory": "#a78bfa",
    "hypothesis": "#f472b6",
    "curiosity": "#f472b6",
    "development": "#eab308",
    "predictor": "#22d3ee",
    "risk": "#f59e0b",
    "critic": "#ffbd59",
    "action_head": "#fb7185",
    "selector": "#fb7185",
    "utility": "#fb7185",
    "meta": "#cbd5e1",
    "output": "#f8fafc",
}


# ---------------------------------------------------------------------------
# Pure projection helpers.  They deliberately have no Tk dependency so their
# contracts can be regression-tested without opening a window.
# ---------------------------------------------------------------------------

def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _latest_activity(state: Mapping[str, Any]) -> dict[str, Any]:
    latest = state.get("latest", {}) if isinstance(state, Mapping) else {}
    for kind in ("decision", "feedback", "lifecycle"):
        event = latest.get(kind, {}) if isinstance(latest, Mapping) else {}
        payload = event.get("payload", {}) if isinstance(event, Mapping) else {}
        network = payload.get("network", {}) if isinstance(payload, Mapping) else {}
        activity = network.get("activity") if isinstance(network, Mapping) else None
        if not isinstance(activity, Mapping):
            activity = payload.get("activity") if isinstance(payload, Mapping) else None
        if isinstance(activity, Mapping):
            return dict(activity)
    return {"nodes": [], "edges": [], "region_activity": {}}


def project_live_graph(
    topology: Mapping[str, Any] | None,
    activity: Mapping[str, Any] | None,
    *,
    max_nodes: int = 700,
    max_edges: int = 1400,
    active_only: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return a bounded, active-first graph suitable for a Tk Canvas.

    Full graphs can have thousands of edges.  The UI retains every active node
    and edge first, then fills remaining capacity by utility/weight.  The
    observer therefore shows the actual current trace even under a display cap.
    """
    top = topology or {}
    act = activity or {}
    all_nodes = [dict(item) for item in list(top.get("nodes", []) or []) if isinstance(item, Mapping)]
    all_edges = [dict(item) for item in list(top.get("edges", []) or []) if isinstance(item, Mapping)]
    active_nodes = {
        str(item.get("id", "")): dict(item)
        for item in list(act.get("nodes", []) or [])
        if isinstance(item, Mapping) and str(item.get("id", ""))
    }
    active_edges = {
        str(item.get("id", "")): dict(item)
        for item in list(act.get("edges", []) or [])
        if isinstance(item, Mapping) and str(item.get("id", ""))
    }

    def node_rank(item: Mapping[str, Any]) -> tuple[int, float, float, str]:
        node_id = str(item.get("id", ""))
        live = active_nodes.get(node_id, {})
        return (
            0 if node_id in active_nodes else 1,
            -abs(_finite(live.get("activation", 0.0))),
            -abs(_finite(item.get("utility", 0.0))),
            node_id,
        )

    if active_only:
        nodes = [item for item in all_nodes if str(item.get("id", "")) in active_nodes]
    else:
        # A cap limits passive topology only.  The current trace is never
        # dropped merely because it has more active nodes than the display cap.
        active_node_list = [item for item in sorted(all_nodes, key=node_rank) if str(item.get("id", "")) in active_nodes]
        inactive_node_list = [item for item in sorted(all_nodes, key=node_rank) if str(item.get("id", "")) not in active_nodes]
        passive_budget = max(0, int(max_nodes) - len(active_node_list))
        nodes = active_node_list + inactive_node_list[:passive_budget]
    node_ids = {str(item.get("id", "")) for item in nodes}

    def edge_rank(item: Mapping[str, Any]) -> tuple[int, float, float, str]:
        edge_id = str(item.get("id", ""))
        live = active_edges.get(edge_id, {})
        return (
            0 if edge_id in active_edges else 1,
            -abs(_finite(live.get("contribution", 0.0))),
            -abs(_finite(item.get("weight", 0.0))),
            edge_id,
        )

    usable_edges = [
        item for item in all_edges
        if str(item.get("src", "")) in node_ids and str(item.get("dst", "")) in node_ids
    ]
    if active_only:
        edges = [item for item in usable_edges if str(item.get("id", "")) in active_edges]
    else:
        active_edge_list = [item for item in sorted(usable_edges, key=edge_rank) if str(item.get("id", "")) in active_edges]
        inactive_edge_list = [item for item in sorted(usable_edges, key=edge_rank) if str(item.get("id", "")) not in active_edges]
        passive_budget = max(0, int(max_edges) - len(active_edge_list))
        edges = active_edge_list + inactive_edge_list[:passive_budget]
    return nodes, edges, active_nodes, active_edges


def summarize_live_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract stable dashboard metrics from the newest telemetry events."""
    state = state or {}
    topology = state.get("topology", {}) if isinstance(state.get("topology", {}), Mapping) else {}
    latest = state.get("latest", {}) if isinstance(state.get("latest", {}), Mapping) else {}
    decision_event = latest.get("decision", {}) if isinstance(latest, Mapping) else {}
    payload = decision_event.get("payload", {}) if isinstance(decision_event, Mapping) else {}
    action = payload.get("action", {}) if isinstance(payload.get("action", {}), Mapping) else {}
    expectancy = payload.get("expectancy", {}) if isinstance(payload.get("expectancy", {}), Mapping) else {}
    recovery = payload.get("learner_recovery", {}) if isinstance(payload.get("learner_recovery", {}), Mapping) else {}
    activity = _latest_activity(state)
    bridge = latest.get("bridge_heartbeat", {}) if isinstance(latest, Mapping) else {}
    bridge_payload = bridge.get("payload", {}) if isinstance(bridge, Mapping) else {}
    writer_status = state.get("writer_status", {}) if isinstance(state.get("writer_status", {}), Mapping) else {}
    events = list(state.get("events", []) or [])
    growth_count = 0
    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_payload = event.get("payload", {})
        if isinstance(event_payload, Mapping):
            growth_count += len(list(event_payload.get("growth_events", []) or []))
    return {
        "nodes": int(_finite(topology.get("node_count", 0))),
        "edges": int(_finite(topology.get("edge_count", 0))),
        "active_nodes": int(_finite(activity.get("active_node_count", len(activity.get("nodes", []) or [])))),
        "active_edges": int(_finite(activity.get("active_edge_count", len(activity.get("edges", []) or [])))),
        "action": str(action.get("name", "—") or "—"),
        "exp_lcb": _finite(expectancy.get("predicted_lcb", 0.0)),
        "exp_samples": int(_finite(expectancy.get("samples", 0))),
        "confidence": _finite(action.get("confidence", 0.0)),
        "growth_events": int(growth_count),
        "hold_reason": str(recovery.get("hold_reason", "") or ""),
        "recovery_replacement_applied": bool(recovery.get("replacement_applied", False)),
        "recovery_global_hold_penalty": _finite(recovery.get("global_hold_penalty", 0.0)),
        "bridge_sequence": int(_finite(bridge_payload.get("sequence", -1), -1.0)),
        "bridge_stage": str(bridge_payload.get("stage", "") or ""),
        "bridge_event_age_seconds": max(0.0, _finite(state.get("server_time", 0.0)) - _finite(bridge.get("created_at", 0.0))),
        "decision_event_age_seconds": max(0.0, _finite(state.get("server_time", 0.0)) - _finite(decision_event.get("created_at", 0.0))),
        "exists": bool(state.get("exists", False)),
        "last_error": str(state.get("last_error", "") or ""),
        "writer_event_sequence": int(_finite(writer_status.get("event_sequence", -1), -1.0)),
        "writer_last_event_at": _finite(writer_status.get("last_event_at", 0.0)),
        "writer_last_error": str(writer_status.get("last_error", "") or ""),
        "writer_run_id": str(writer_status.get("run_id", "") or ""),
    }


def live_view_button_label(enabled: bool) -> str:
    """Label for the explicit UI-only live-view toggle.

    This controls rendering in the desktop window, not the trainer's decisions.
    Telemetry capture is configured separately because a running MT5 trial cannot
    safely change its telemetry contract midway through a decision sequence.
    """
    return "Tắt xem Live" if bool(enabled) else "Bật xem Live"


def live_view_status_text(*, enabled: bool, training: bool, telemetry_available: bool) -> str:
    """Return an unambiguous UI status for the live-view control."""
    if enabled and telemetry_available:
        return "Live đang hiển thị trace thật; train không bị điều khiển bởi UI."
    if enabled and training:
        return "Live đã mở nhưng lượt train này không ghi telemetry; thay đổi có hiệu lực từ lượt tiếp theo."
    if enabled:
        return "Live đã mở; chờ telemetry hoặc nạp JSONL có sẵn."
    if training:
        return "Đã tắt hiển thị Live; train và telemetry hiện tại vẫn tiếp tục chạy."
    return "Live đang tắt; bấm Bật xem Live để mở phần quan sát."


def with_ui_observability(
    config: AppConfig,
    *,
    enabled: bool,
    event_path: str,
    decision_sample_every: int,
) -> AppConfig:
    """Return a config whose read-only telemetry is controlled by the desktop UI."""
    return replace(
        config,
        observability=replace(
            config.observability,
            enabled=bool(enabled),
            event_path=str(event_path or DEFAULT_EVENT_PATH),
            decision_sample_every=max(1, int(decision_sample_every)),
            # The UI performs an explicit pre-run reset when requested.  Keeping
            # this false prevents nested train-to-goal calls from erasing history.
            reset_on_process_start=False,
        ),
    )


class EmbeddedNeuralObservatory(ttk.Frame):
    """A read-only live neuron/edge viewer embedded in the training UI."""

    POLL_MS = 450

    def __init__(self, parent: tk.Misc, event_path_var: tk.StringVar) -> None:
        super().__init__(parent, padding=8)
        self.event_path_var = event_path_var
        self.reader = LiveEventReader(event_path_var.get() or DEFAULT_EVENT_PATH)
        self._reader_path = str(self.reader.path)
        self._paused = tk.BooleanVar(value=False)
        # The parent Train MT5 screen can hide/pause the viewer without
        # touching the event writer inside the trainer.
        self._view_enabled = tk.BooleanVar(value=False)
        self._active_only = tk.BooleanVar(value=False)
        self._max_nodes = tk.IntVar(value=700)
        self._max_edges = tk.IntVar(value=1400)
        self._state: dict[str, Any] = {}
        self._positions: dict[str, tuple[float, float]] = {}
        self._display_nodes: dict[str, dict[str, Any]] = {}
        self._display_edges: dict[str, dict[str, Any]] = {}
        self._activity_nodes: dict[str, dict[str, Any]] = {}
        self._activity_edges: dict[str, dict[str, Any]] = {}
        self._selected: tuple[str, str] | None = None
        self._metrics = {name: tk.StringVar(value="—") for name in (
            "nodes", "edges", "active_nodes", "active_edges", "action", "exp_lcb", "exp_samples", "confidence", "growth_events"
        )}
        self.status_var = tk.StringVar(value="Chờ event stream…")
        self._build()
        self.after(self.POLL_MS, self._poll)

    def _build(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(header, text="Live Neural Observatory", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.status_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(header, text="Tạm dừng khung hình", variable=self._paused).pack(side=tk.RIGHT)
        ttk.Checkbutton(header, text="Chỉ active", variable=self._active_only, command=self._redraw).pack(side=tk.RIGHT, padx=(0, 10))

        stream = ttk.Frame(self)
        stream.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(stream, text="Telemetry JSONL", width=16).pack(side=tk.LEFT)
        ttk.Entry(stream, textvariable=self.event_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(stream, text="…", width=4, command=self._browse_event_path).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(stream, text="Nạp lại", command=self.reload_stream).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(stream, text="Nodes").pack(side=tk.LEFT, padx=(12, 3))
        ttk.Spinbox(stream, from_=100, to=3000, increment=100, textvariable=self._max_nodes, width=6, command=self._redraw).pack(side=tk.LEFT)
        ttk.Label(stream, text="Edges").pack(side=tk.LEFT, padx=(8, 3))
        ttk.Spinbox(stream, from_=200, to=6000, increment=200, textvariable=self._max_edges, width=6, command=self._redraw).pack(side=tk.LEFT)

        metric_frame = ttk.Frame(self)
        metric_frame.pack(fill=tk.X, pady=(0, 7))
        labels = [
            ("Neuron", "nodes"), ("Liên kết", "edges"), ("Node sáng", "active_nodes"),
            ("Edge sáng", "active_edges"), ("Action", "action"), ("EXP LCB", "exp_lcb"),
            ("EXP samples", "exp_samples"), ("Confidence", "confidence"), ("Growth", "growth_events"),
        ]
        for index, (label, key) in enumerate(labels):
            card = ttk.Frame(metric_frame, padding=(8, 4), relief=tk.GROOVE)
            card.grid(row=0, column=index, padx=(0 if index == 0 else 4, 0), sticky="nsew")
            ttk.Label(card, textvariable=self._metrics[key], font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
            ttk.Label(card, text=label).pack(anchor=tk.W)
            metric_frame.columnconfigure(index, weight=1)

        split = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        split.pack(fill=tk.BOTH, expand=True)
        graph_box = ttk.Frame(split, padding=(0, 0, 6, 0))
        detail_box = ttk.Frame(split, padding=(6, 0, 0, 0))
        split.add(graph_box, weight=3)
        split.add(detail_box, weight=2)

        canvas_box = ttk.LabelFrame(graph_box, text="Mạng neuron trực tiếp — click node hoặc edge để xem chi tiết", padding=1)
        canvas_box.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_box, background="#0b0e14", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())
        self.canvas.bind("<Button-1>", self._select_canvas_item)
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self._zoom = 1.0

        self.detail_tabs = ttk.Notebook(detail_box)
        self.detail_tabs.pack(fill=tk.BOTH, expand=True)
        self.decision_text = self._create_text_tab("Decision", "Action, hypothesis, critic, context và EXP dự báo.")
        self.growth_text = self._create_text_tab("Growth / Trial", "Birth, plasticity, fade, prune và trial outcome.")
        self.selected_text = self._create_text_tab("Node / Edge", "Click vào graph để inspect dữ liệu thật của phần tử.")
        self.events_tree = self._create_timeline_tab()

    def _create_text_tab(self, title: str, hint: str) -> tk.Text:
        tab = ttk.Frame(self.detail_tabs, padding=8)
        self.detail_tabs.add(tab, text=title)
        ttk.Label(tab, text=hint, wraplength=400).pack(anchor=tk.W, pady=(0, 6))
        text = tk.Text(tab, wrap=tk.WORD, height=18, background="#101722", foreground="#e9eef8", insertbackground="#e9eef8", relief=tk.FLAT)
        scroll = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scroll.set, state=tk.DISABLED)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        return text

    def _create_timeline_tab(self) -> ttk.Treeview:
        tab = ttk.Frame(self.detail_tabs, padding=8)
        self.detail_tabs.add(tab, text="Timeline")
        cols = ("time", "event", "detail")
        tree = ttk.Treeview(tab, columns=cols, show="headings", height=18)
        tree.heading("time", text="Time")
        tree.heading("event", text="Event")
        tree.heading("detail", text="Detail")
        tree.column("time", width=80, stretch=False)
        tree.column("event", width=105, stretch=False)
        tree.column("detail", width=300, stretch=True)
        scroll = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.bind("<<TreeviewSelect>>", self._select_timeline_event)
        return tree

    def _browse_event_path(self) -> None:
        value = filedialog.askopenfilename(filetypes=[("JSON lines", "*.jsonl"), ("All", "*.*")])
        if value:
            self.event_path_var.set(value)
            self.reload_stream()

    def reload_stream(self) -> None:
        path = self.event_path_var.get().strip() or DEFAULT_EVENT_PATH
        self.reader = LiveEventReader(path)
        self._reader_path = str(self.reader.path)
        self._state = {}
        self._selected = None
        self._set_text(self.selected_text, "Chưa chọn node hoặc edge.")
        self._redraw()

    def reset_stream_for_new_run(self) -> None:
        """Clear only before the UI starts a new background run."""
        path = Path(self.event_path_var.get().strip() or DEFAULT_EVENT_PATH)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            self.status_var.set(f"Không thể xóa telemetry cũ: {exc}")
        self.reload_stream()

    def set_view_enabled(self, enabled: bool) -> None:
        """Enable/disable only desktop rendering; never alter training telemetry."""
        self._view_enabled.set(bool(enabled))
        if not enabled:
            self.status_var.set("Live tạm ẩn từ màn hình Train; telemetry/train vẫn có thể tiếp tục.")
            return
        try:
            self._state = self.reader.state(limit=240)
            self._update_panels()
            self._redraw()
            self.status_var.set("Đang xem live telemetry.")
        except Exception as exc:
            self.status_var.set(f"Không thể nạp live telemetry: {exc}")

    def _poll(self) -> None:
        try:
            current_path = self.event_path_var.get().strip() or DEFAULT_EVENT_PATH
            if current_path != self._reader_path:
                self.reload_stream()
            if self._view_enabled.get() and not self._paused.get():
                self._state = self.reader.state(limit=240)
                self._update_panels()
                self._redraw()
        except Exception as exc:  # UI must remain useful if a half-written event arrives.
            self.status_var.set(f"Lỗi đọc telemetry: {exc}")
        finally:
            self.after(self.POLL_MS, self._poll)

    @staticmethod
    def _set_text(widget: tk.Text, value: Any) -> None:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state=tk.DISABLED)

    def _update_panels(self) -> None:
        summary = summarize_live_state(self._state)
        self._metrics["nodes"].set(str(summary["nodes"]))
        self._metrics["edges"].set(str(summary["edges"]))
        self._metrics["active_nodes"].set(str(summary["active_nodes"]))
        self._metrics["active_edges"].set(str(summary["active_edges"]))
        self._metrics["action"].set(str(summary["action"]))
        self._metrics["exp_lcb"].set(f"{summary['exp_lcb']:.4f}")
        self._metrics["exp_samples"].set(str(summary["exp_samples"]))
        self._metrics["confidence"].set(f"{summary['confidence']:.3f}")
        self._metrics["growth_events"].set(str(summary["growth_events"]))
        if not summary["exists"]:
            self.status_var.set("Chưa có telemetry. Start Train MT5 trong UI hoặc trỏ tới JSONL của lệnh CLI đang chạy.")
        elif summary["last_error"]:
            self.status_var.set(f"Lỗi telemetry: {summary['last_error']}")
        elif summary["writer_last_error"]:
            self.status_var.set(
                f"LIVE TELEMETRY WRITE ERROR · run={summary['writer_run_id'] or 'unknown'} · "
                f"{summary['writer_last_error']}"
            )
        elif summary["bridge_sequence"] >= 0 and summary["bridge_event_age_seconds"] > 10.0:
            self.status_var.set(
                f"LIVE DATA STALE · bridge seq={summary['bridge_sequence']} · "
                f"heartbeat cũ {summary['bridge_event_age_seconds']:.1f}s"
            )
        elif summary["bridge_sequence"] >= 0:
            recovery_status = (
                f" · HOLD={summary['hold_reason']}"
                if summary["hold_reason"] else ""
            )
            self.status_var.set(
                f"Live bridge seq={summary['bridge_sequence']} ({summary['bridge_stage'] or 'unknown'}) · "
                f"decision age={summary['decision_event_age_seconds']:.1f}s{recovery_status}"
            )
        else:
            self.status_var.set(f"Đang theo dõi {Path(str(self._state.get('path', ''))).name} · telemetry read-only")

        latest = self._state.get("latest", {}) if isinstance(self._state.get("latest", {}), Mapping) else {}
        decision = latest.get("decision", {}) if isinstance(latest, Mapping) else {}
        decision_payload = decision.get("payload", {}) if isinstance(decision, Mapping) else {}
        self._set_text(self.decision_text, {
            "frame_id": decision_payload.get("frame_id", ""),
            "trace_id": decision_payload.get("trace_id", ""),
            "action": decision_payload.get("action", {}),
            "context": decision_payload.get("context", {}),
            "expectancy": decision_payload.get("expectancy", {}),
            "learner_recovery": decision_payload.get("learner_recovery", {}),
            "network": decision_payload.get("network", {}),
        })
        recent = [event for event in list(self._state.get("events", []) or [])[-100:] if isinstance(event, Mapping)]
        development = []
        for event in recent:
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), Mapping) else {}
            if event.get("event_type") in {"feedback", "development", "lifecycle", "trial_started", "trial_finished", "topology"}:
                development.append({
                    "time": event.get("created_at"),
                    "event": event.get("event_type"),
                    "trace_id": payload.get("trace_id", ""),
                    "growth_events": payload.get("growth_events", []),
                    "metrics": payload.get("metrics", {}),
                    "fitness": payload.get("fitness"),
                    "reason": payload.get("reason"),
                })
        self._set_text(self.growth_text, list(reversed(development[-40:])))
        self._update_timeline(recent)

    def _update_timeline(self, events: Iterable[Mapping[str, Any]]) -> None:
        selection = self.events_tree.selection()
        for item in self.events_tree.get_children():
            self.events_tree.delete(item)
        for index, event in enumerate(reversed(list(events)[-100:])):
            created = _finite(event.get("created_at", 0.0))
            when = time.strftime("%H:%M:%S", time.localtime(created)) if created else "—"
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), Mapping) else {}
            growth = list(payload.get("growth_events", []) or [])
            detail = str(payload.get("trace_id") or payload.get("policy_id") or payload.get("reason") or "")
            if growth:
                names = [str(item.get("type") or item.get("action") or "growth") for item in growth if isinstance(item, Mapping)]
                detail = "growth: " + ", ".join(names[:3])
            self.events_tree.insert("", tk.END, iid=f"event-{index}", values=(when, str(event.get("event_type", "")), detail))
        if selection:
            try:
                self.events_tree.selection_set(selection)
            except tk.TclError:
                pass

    def _canvas_size(self) -> tuple[float, float]:
        width = max(400, int(self.canvas.winfo_width() or 800))
        height = max(300, int(self.canvas.winfo_height() or 600))
        return float(width), float(height)

    def _redraw(self) -> None:
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("graph")
        topology = self._state.get("topology", {}) if isinstance(self._state.get("topology", {}), Mapping) else {}
        activity = _latest_activity(self._state)
        nodes, edges, active_nodes, active_edges = project_live_graph(
            topology,
            activity,
            max_nodes=max(100, int(self._max_nodes.get() or 700)),
            max_edges=max(200, int(self._max_edges.get() or 1400)),
            active_only=bool(self._active_only.get()),
        )
        self._display_nodes = {str(item.get("id", "")): item for item in nodes}
        self._display_edges = {str(item.get("id", "")): item for item in edges}
        self._activity_nodes = active_nodes
        self._activity_edges = active_edges
        width, height = self._canvas_size()
        if not nodes:
            self.canvas.create_text(width / 2.0, height / 2.0, text="Chưa có topology/decision telemetry", fill="#94a3b8", tags="graph")
            return

        groups: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            groups.setdefault(str(node.get("region", "unknown")), []).append(node)
        ordered_regions = sorted(groups, key=lambda region: (REGION_ORDER.get(region, 99), region))
        margin_x, top_y, bottom_y = 50.0, 52.0, 34.0
        usable_w = max(1.0, width - margin_x * 2)
        usable_h = max(1.0, height - top_y - bottom_y)
        self._positions = {}
        for region_index, region in enumerate(ordered_regions):
            items = sorted(
                groups[region],
                key=lambda item: (
                    0 if str(item.get("id", "")) in active_nodes else 1,
                    -abs(_finite(active_nodes.get(str(item.get("id", "")), {}).get("activation", 0.0))),
                    str(item.get("id", "")),
                ),
            )
            x = margin_x + usable_w * (region_index / max(1, len(ordered_regions) - 1))
            self.canvas.create_text(x, 18.0, text=region, fill=REGION_COLORS.get(region, "#7dd3fc"), tags="graph")
            for index, node in enumerate(items):
                y = top_y + usable_h * ((index + 1) / (len(items) + 1))
                self._positions[str(node.get("id", ""))] = (x, y)

        for edge in edges:
            edge_id = str(edge.get("id", ""))
            src = self._positions.get(str(edge.get("src", "")))
            dst = self._positions.get(str(edge.get("dst", "")))
            if src is None or dst is None:
                continue
            live = active_edges.get(edge_id, {})
            active = edge_id in active_edges
            enabled = bool(edge.get("enabled", True))
            weight = _finite(edge.get("weight", 0.0))
            if active:
                color = "#34d399"
                line_width = 2.2 + min(2.2, abs(_finite(live.get("contribution", 0.0))) * 2.0)
            elif not enabled:
                color = "#334155"
                line_width = 0.6
            elif str(edge.get("lifecycle", "")).lower().startswith("candidate"):
                color = "#eab308"
                line_width = 1.1
            elif weight < 0.0:
                color = "#ef4444"
                line_width = 0.8 + min(1.2, abs(weight))
            else:
                color = "#3b82f6"
                line_width = 0.8 + min(1.2, abs(weight))
            dash = (3, 3) if not enabled else ()
            self.canvas.create_line(*src, *dst, fill=color, width=line_width * self._zoom, dash=dash, tags=("graph", f"edge:{edge_id}"))

        for node in nodes:
            node_id = str(node.get("id", ""))
            pos = self._positions.get(node_id)
            if pos is None:
                continue
            live = active_nodes.get(node_id, {})
            active = node_id in active_nodes
            activation = abs(_finite(live.get("activation", 0.0)))
            radius = (4.0 + min(7.0, activation * 7.0)) if active else 3.0
            radius *= self._zoom
            region = str(node.get("region", "unknown"))
            color = "#34d399" if active else REGION_COLORS.get(region, "#7dd3fc")
            x, y = pos
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="" if self._selected != ("node", node_id) else "#ffffff", width=2 if self._selected == ("node", node_id) else 1, tags=("graph", f"node:{node_id}"))

        status = f"Hiển thị {len(nodes)}/{int(_finite(topology.get('node_count', 0)))} neuron · {len(edges)}/{int(_finite(topology.get('edge_count', 0)))} edges"
        self.canvas.create_text(10, height - 11, anchor=tk.W, text=status, fill="#94a3b8", tags="graph")

    def _mouse_wheel(self, event: tk.Event) -> None:
        direction = 1.08 if getattr(event, "delta", 0) > 0 else 0.92
        self._zoom = min(2.2, max(0.55, self._zoom * direction))
        self._redraw()

    def _select_canvas_item(self, event: tk.Event) -> None:
        x, y = float(event.x), float(event.y)
        nearest_node: tuple[str, float] | None = None
        for node_id, (px, py) in self._positions.items():
            distance = math.hypot(x - px, y - py)
            if nearest_node is None or distance < nearest_node[1]:
                nearest_node = (node_id, distance)
        if nearest_node is not None and nearest_node[1] <= 18.0:
            self._selected = ("node", nearest_node[0])
            node = dict(self._display_nodes.get(nearest_node[0], {}))
            node.update(self._activity_nodes.get(nearest_node[0], {}))
            self._set_text(self.selected_text, {"kind": "node", **node})
            self._redraw()
            return
        nearest_edge: tuple[str, float] | None = None
        for edge_id, edge in self._display_edges.items():
            src = self._positions.get(str(edge.get("src", "")))
            dst = self._positions.get(str(edge.get("dst", "")))
            if src is None or dst is None:
                continue
            distance = self._distance_to_segment(x, y, src[0], src[1], dst[0], dst[1])
            if nearest_edge is None or distance < nearest_edge[1]:
                nearest_edge = (edge_id, distance)
        if nearest_edge is not None and nearest_edge[1] <= 7.0:
            self._selected = ("edge", nearest_edge[0])
            edge = dict(self._display_edges.get(nearest_edge[0], {}))
            edge.update(self._activity_edges.get(nearest_edge[0], {}))
            self._set_text(self.selected_text, {"kind": "edge", **edge})
            self._redraw()

    @staticmethod
    def _distance_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        dx, dy = x2 - x1, y2 - y1
        denominator = dx * dx + dy * dy
        if denominator <= 1e-12:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denominator))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    def _select_timeline_event(self, _event: tk.Event) -> None:
        selected = self.events_tree.selection()
        if not selected:
            return
        item_id = selected[0]
        try:
            index = int(item_id.rsplit("-", 1)[-1])
        except (IndexError, ValueError):
            return
        events = list(reversed([item for item in list(self._state.get("events", []) or [])[-100:] if isinstance(item, Mapping)]))
        if 0 <= index < len(events):
            self._set_text(self.growth_text, events[index])


class SelfGrowingTraderApp(tk.Tk):
    """One-window trainer: controls, embedded neural graph, and structured log."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Self-Growing Trader — Training UI")
        self.geometry("1540x920")
        self.minsize(1180, 700)
        self.config_path = tk.StringVar(value="configs/default.yaml")
        self.data_path = tk.StringVar(value="")
        self.policy_path = tk.StringVar(value="outputs/winner.pkl")
        self.output_path = tk.StringVar(value="outputs/mt5_trained.pkl")
        self.rounds = tk.IntVar(value=1)
        self.trials = tk.IntVar(value=1)
        self.timeout_sec = tk.IntVar(value=600)
        self.runner_dir = tk.StringVar(value="mt5_v2_runner")
        self.from_date = tk.StringVar(value="")
        self.to_date = tk.StringVar(value="")
        self.visual_mt5 = tk.BooleanVar(value=False)
        self.curriculum = tk.BooleanVar(value=False)
        # Telemetry is configured before a run; the explicit button below only
        # toggles whether the user is currently watching it.
        self.live_enabled = tk.BooleanVar(value=True)
        self.live_view_enabled = tk.BooleanVar(value=False)
        self.live_reset = tk.BooleanVar(value=True)
        self.live_events_path = tk.StringVar(value=DEFAULT_EVENT_PATH)
        self.live_sample_every = tk.IntVar(value=4)
        self.status = tk.StringVar(value="Sẵn sàng")
        self.events: queue.Queue[object] = queue.Queue()
        self._training_running = False
        self._active_run_telemetry = False
        self.live_view_status = tk.StringVar(value=live_view_status_text(
            enabled=False, training=False, telemetry_available=bool(self.live_enabled.get())
        ))
        self._build()
        self.after(200, self._drain_events)

    def _build(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self.train_tab = ttk.Frame(self.tabs, padding=12)
        self.live_tab = ttk.Frame(self.tabs, padding=0)
        self.output_tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(self.train_tab, text="Train MT5")
        self.tabs.add(self.live_tab, text="Mạng sống")
        self.tabs.add(self.output_tab, text="Kết quả / Log")
        self._build_train_tab()
        self.observatory = EmbeddedNeuralObservatory(self.live_tab, self.live_events_path)
        self.observatory.pack(fill=tk.BOTH, expand=True)
        self.tabs.tab(self.live_tab, state="hidden")
        self._build_output_tab()

    def _build_train_tab(self) -> None:
        form = ttk.LabelFrame(self.train_tab, text="Cấu hình train", padding=10)
        form.pack(fill=tk.X)
        self._path_row(form, "Config", self.config_path, [("YAML", "*.yaml"), ("All", "*.*")])
        self._path_row(form, "Data CSV", self.data_path, [("CSV", "*.csv"), ("All", "*.*")])
        self._path_row(form, "Seed policy", self.policy_path, [("Pickle", "*.pkl"), ("All", "*.*")])
        self._path_row(form, "Output policy", self.output_path, [("Pickle", "*.pkl"), ("All", "*.*")], save=True)
        self._path_row(form, "MT5 runner", self.runner_dir, [("Terminal", "terminal64.exe"), ("All", "*.*")], directory=True)

        runtime = ttk.LabelFrame(self.train_tab, text="MT5 runtime", padding=10)
        runtime.pack(fill=tk.X, pady=(10, 0))
        self._entry_row(runtime, "Rounds", self.rounds, width=8, column=0, row=0)
        self._entry_row(runtime, "Trials", self.trials, width=8, column=2, row=0)
        self._entry_row(runtime, "Timeout (sec)", self.timeout_sec, width=8, column=4, row=0)
        self._entry_row(runtime, "From date", self.from_date, width=15, column=0, row=1)
        self._entry_row(runtime, "To date", self.to_date, width=15, column=2, row=1)
        ttk.Checkbutton(runtime, text="MT5 Visual", variable=self.visual_mt5).grid(row=1, column=4, sticky=tk.W, padx=(12, 4), pady=4)
        ttk.Checkbutton(runtime, text="Curriculum learning", variable=self.curriculum).grid(row=1, column=5, sticky=tk.W, padx=(12, 4), pady=4)

        live = ttk.LabelFrame(self.train_tab, text="Theo dõi mạng neuron", padding=10)
        live.pack(fill=tk.X, pady=(10, 0))
        self.live_view_button = ttk.Button(live, text=live_view_button_label(False), command=self.toggle_live_view)
        self.live_view_button.grid(row=0, column=0, sticky=tk.W)
        ttk.Label(live, textvariable=self.live_view_status, foreground="#556070").grid(row=0, column=1, columnspan=3, sticky=tk.W, padx=(12, 0))
        ttk.Checkbutton(live, text="Ghi telemetry khi train", variable=self.live_enabled).grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Checkbutton(live, text="Xóa event cũ trước run", variable=self.live_reset).grid(row=1, column=1, sticky=tk.W, padx=(14, 0), pady=(8, 0))
        ttk.Label(live, text="Sample mỗi N decision").grid(row=1, column=2, sticky=tk.E, padx=(20, 4), pady=(8, 0))
        ttk.Spinbox(live, from_=1, to=1000, textvariable=self.live_sample_every, width=8).grid(row=1, column=3, sticky=tk.W, pady=(8, 0))
        ttk.Label(live, text="Telemetry file").grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(live, textvariable=self.live_events_path).grid(row=2, column=1, columnspan=3, sticky=tk.EW, pady=(8, 0), padx=(8, 0))
        live.columnconfigure(1, weight=1)
        ttk.Label(live, text="Nút xem Live chỉ bật/tắt hiển thị trong UI. Train, policy và MT5 không bị dừng hoặc bị điều khiển.", foreground="#556070").grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(8, 0))

        actions = ttk.Frame(self.train_tab)
        actions.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(actions, text="Evaluate CSV", command=self.run_evaluate).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="Goal Audit", command=self.run_audit).pack(side=tk.LEFT, padx=(0, 8))
        self.train_button = ttk.Button(actions, text="Train MT5", command=self.run_train)
        self.train_button.pack(side=tk.LEFT)
        ttk.Label(self.train_tab, textvariable=self.status, font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(12, 0))

    def _refresh_live_view_status(self) -> None:
        telemetry_available = self._active_run_telemetry if self._training_running else bool(self.live_enabled.get())
        self.live_view_status.set(live_view_status_text(
            enabled=bool(self.live_view_enabled.get()),
            training=bool(self._training_running),
            telemetry_available=bool(telemetry_available),
        ))

    def toggle_live_view(self) -> None:
        """Toggle the embedded live view without modifying train-time behavior."""
        enabled = not bool(self.live_view_enabled.get())
        self.live_view_enabled.set(enabled)
        self.live_view_button.configure(text=live_view_button_label(enabled))
        self.observatory.set_view_enabled(enabled)
        if enabled:
            self.tabs.tab(self.live_tab, state="normal")
            self.tabs.select(self.live_tab)
        else:
            if self.tabs.select() == str(self.live_tab):
                self.tabs.select(self.train_tab)
            self.tabs.tab(self.live_tab, state="hidden")
        self._refresh_live_view_status()

    def _build_output_tab(self) -> None:
        ttk.Label(self.output_tab, text="Kết quả train, lỗi MT5 và dữ liệu trả về từ runner.").pack(anchor=tk.W)
        container = ttk.Frame(self.output_tab)
        container.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.output = tk.Text(container, wrap=tk.WORD, background="#101722", foreground="#e9eef8", insertbackground="#e9eef8", relief=tk.FLAT)
        scroll = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.output.yview)
        self.output.configure(yscrollcommand=scroll.set)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _path_row(self, parent: ttk.Frame, label: str, var: tk.StringVar, filetypes, save: bool = False, directory: bool = False) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        def browse() -> None:
            if directory:
                value = filedialog.askdirectory()
            elif save:
                value = filedialog.asksaveasfilename(filetypes=filetypes)
            else:
                value = filedialog.askopenfilename(filetypes=filetypes)
            if value:
                var.set(value)

        ttk.Button(row, text="…", width=4, command=browse).pack(side=tk.LEFT, padx=(6, 0))

    @staticmethod
    def _entry_row(parent: ttk.Frame, label: str, var: tk.Variable, *, width: int, column: int, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky=tk.W, padx=(0 if column == 0 else 12, 4), pady=4)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=column + 1, sticky=tk.W, pady=4)

    def _config(self) -> AppConfig:
        config = load_config(self.config_path.get())
        data_path = self.data_path.get().strip()
        if data_path:
            config = replace(config, csv_path=data_path)
        return with_ui_observability(
            config,
            enabled=bool(self.live_enabled.get()),
            event_path=self.live_events_path.get().strip() or DEFAULT_EVENT_PATH,
            decision_sample_every=max(1, int(self.live_sample_every.get() or 1)),
        )

    def _run_background(self, title: str, fn, *, marks_training: bool = False) -> None:
        if marks_training:
            self._training_running = True
            self.train_button.configure(state=tk.DISABLED)
            self._refresh_live_view_status()
        self.status.set(f"{title} đang chạy…")

        def worker() -> None:
            try:
                result = fn()
                self.events.put({"kind": "result", "title": title, "payload": result})
            except Exception as exc:
                self.events.put({"kind": "error", "title": title, "message": str(exc)})
            finally:
                if marks_training:
                    self.events.put({"kind": "training_finished", "title": title})

        threading.Thread(target=worker, daemon=True, name=f"sgt-{title.replace(' ', '-').lower()}").start()

    def run_evaluate(self) -> None:
        def work():
            config = self._config()
            data = load_market_data(config)
            policy_path = Path(self.policy_path.get())
            policy = load_policy_callable(policy_path) if policy_path.exists() else constant_hold_policy
            return evaluate_policy(data, config, policy)

        self._run_background("Evaluate", work)

    def run_audit(self) -> None:
        def work():
            config = self._config()
            data = load_market_data(config)
            policy_path = Path(self.policy_path.get())
            if not policy_path.exists():
                raise FileNotFoundError(f"Không tìm thấy policy: {policy_path}")
            return audit_policy(data, config, load_policy_callable(policy_path))

        self._run_background("Goal Audit", work)

    def run_train(self) -> None:
        if self._training_running:
            return
        config = self._config()
        self._active_run_telemetry = bool(config.observability.enabled)
        if bool(config.observability.enabled) and bool(self.live_reset.get()):
            self.observatory.reset_stream_for_new_run()
        if bool(self.live_view_enabled.get()):
            self.tabs.tab(self.live_tab, state="normal")
            self.tabs.select(self.live_tab)
            self.observatory.set_view_enabled(True)
        self._refresh_live_view_status()

        def work():
            policy_path = Path(self.policy_path.get())
            return train_self_growing_mt5(
                config,
                rounds=max(1, int(self.rounds.get())),
                output_path=self.output_path.get(),
                trials=max(1, int(self.trials.get())),
                seed_policy_path=self.policy_path.get() if policy_path.exists() else None,
                runner_dir=self.runner_dir.get(),
                timeout_sec=max(1, int(self.timeout_sec.get())),
                visual=bool(self.visual_mt5.get()),
                shutdown_terminal=True,
                curriculum_learning=bool(self.curriculum.get()),
                from_date=self.from_date.get().strip() or None,
                to_date=self.to_date.get().strip() or None,
                model=2,
                policy_core="current",
                m1_backend="graph",
                mt5_close_scope="trial",
            )

        self._run_background("Train MT5", work, marks_training=True)

    def _drain_events(self) -> None:
        while True:
            try:
                item = self.events.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, Mapping):
                kind = str(item.get("kind", ""))
                title = str(item.get("title", ""))
                if kind == "result":
                    rendered = json.dumps(item.get("payload", {}), ensure_ascii=False, indent=2, default=str)
                    self.output.insert(tk.END, f"[{title}]\n{rendered}\n\n")
                    self.output.see(tk.END)
                    self.status.set(f"{title} xong")
                    self.tabs.select(self.output_tab)
                elif kind == "error":
                    message = str(item.get("message", "Unknown error"))
                    self.output.insert(tk.END, f"[{title}] LỖI\n{message}\n\n")
                    self.output.see(tk.END)
                    self.status.set(f"{title} lỗi")
                    self.tabs.select(self.output_tab)
                    messagebox.showerror("Self-Growing Trader", f"{title}: {message}")
                elif kind == "training_finished":
                    self._training_running = False
                    self._active_run_telemetry = False
                    self.train_button.configure(state=tk.NORMAL)
                    self._refresh_live_view_status()
            else:
                self.output.insert(tk.END, str(item) + "\n\n")
                self.output.see(tk.END)
        self.after(200, self._drain_events)


def main() -> None:
    app = SelfGrowingTraderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
