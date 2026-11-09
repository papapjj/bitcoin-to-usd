import re
from pathlib import Path

target_file = Path("research_reasoning_world_class/src/mythos_mt5_policy.py")
content = target_file.read_text(encoding="utf-8")

# 1. Patch __init__ buffer loading
old_buffer_code = """        self.counter_trap_sl_atr: Optional[float] = None
        self.counter_trap_tp_rr: Optional[float] = None
        self.sniper_sl_atr: Optional[float] = None
        self.sniper_tp_rr: Optional[float] = None
        self.m1_buffer: np.ndarray = np.zeros((60, 5), dtype=np.float64)
        p = self.current_price
        for i in range(60):
            body = (np.sin(i * 0.3) * 0.8) * self.current_atr
            upper = abs(np.cos(i * 0.4) * 0.5) * self.current_atr
            lower = abs(np.sin(i * 0.5) * 0.5) * self.current_atr
            c = p + body
            o = p
            h = max(o, c) + upper
            l = min(o, c) - lower
            v = 100.0 + abs(np.sin(i * 0.2)) * 50.0
            p = c
            self.m1_buffer[i] = [o, h, l, c, v]"""

new_buffer_code = """        self.counter_trap_sl_atr: Optional[float] = None
        self.counter_trap_tp_rr: Optional[float] = None
        self.sniper_sl_atr: Optional[float] = None
        self.sniper_tp_rr: Optional[float] = None

        # Load canonical real M1 historical database for high-precision bar synchronization
        self._m1_history_times = None
        self._m1_bars_matrix = None
        self._m1_time_to_idx = None
        m1_csv_candidates = [
            Path(_project_root) / "outputs" / "btcusd_m1_7day_with_warmup.csv",
            Path(_project_root) / "outputs" / "btcusd_m1_7day_clean.csv",
        ]
        for m1_csv in m1_csv_candidates:
            if m1_csv.exists():
                try:
                    df_m1 = pd.read_csv(m1_csv, sep="\\t")
                    col_date = "<DATE>" if "<DATE>" in df_m1.columns else "DATE"
                    col_time = "<TIME>" if "<TIME>" in df_m1.columns else "TIME"
                    col_o = "<OPEN>" if "<OPEN>" in df_m1.columns else "OPEN"
                    col_h = "<HIGH>" if "<HIGH>" in df_m1.columns else "HIGH"
                    col_l = "<LOW>" if "<LOW>" in df_m1.columns else "LOW"
                    col_c = "<CLOSE>" if "<CLOSE>" in df_m1.columns else "CLOSE"
                    col_v = "<TICKVOL>" if "<TICKVOL>" in df_m1.columns else ("VOL" if "VOL" in df_m1.columns else None)

                    times = (df_m1[col_date].astype(str) + " " + df_m1[col_time].astype(str).str[:5]).values
                    v_vals = df_m1[col_v].values.astype(np.float64) if col_v else np.ones(len(df_m1), dtype=np.float64) * 100.0
                    bars = np.column_stack([
                        df_m1[col_o].values.astype(np.float64),
                        df_m1[col_h].values.astype(np.float64),
                        df_m1[col_l].values.astype(np.float64),
                        df_m1[col_c].values.astype(np.float64),
                        v_vals,
                    ])
                    self._m1_history_times = times
                    self._m1_bars_matrix = bars
                    self._m1_time_to_idx = {str(t): i for i, t in enumerate(times)}
                    logger.info("Loaded real M1 historical database: %d bars from %s", len(times), m1_csv.name)
                    if len(bars) >= 60:
                        self.m1_buffer = bars[:60].copy()
                        self.current_price = float(self.m1_buffer[-1, 3])
                        tr_init = np.maximum(
                            self.m1_buffer[1:, 1] - self.m1_buffer[1:, 2],
                            np.maximum(
                                np.abs(self.m1_buffer[1:, 1] - self.m1_buffer[:-1, 3]),
                                np.abs(self.m1_buffer[1:, 2] - self.m1_buffer[:-1, 3])
                            )
                        )
                        self.current_atr = float(np.mean(tr_init[-14:]))
                    break
                except Exception as e:
                    logger.warning("Could not load M1 history from %s: %s", m1_csv, e)

        if not hasattr(self, "m1_buffer") or self.m1_buffer is None:
            self.m1_buffer = np.zeros((60, 5), dtype=np.float64)
            p = self.current_price
            for i in range(60):
                body = (np.sin(i * 0.3) * 0.8) * self.current_atr
                upper = abs(np.cos(i * 0.4) * 0.5) * self.current_atr
                lower = abs(np.sin(i * 0.5) * 0.5) * self.current_atr
                c = p + body
                o = p
                h = max(o, c) + upper
                l = min(o, c) - lower
                v = 100.0 + abs(np.sin(i * 0.2)) * 50.0
                p = c
                self.m1_buffer[i] = [o, h, l, c, v]"""

# 2. Patch act_nam_tac_nhan buffer update
old_update_code = """        # Reconstruct real M1 OHLCV candlestick directly from MT5 inputs
        ret1 = float(base_obs_137[0])
        body = float(base_obs_137[7])
        upper_wick = max(0.0, float(base_obs_137[8]))
        lower_wick = max(0.0, float(base_obs_137[9]))
        range_val = max(0.0, float(base_obs_137[10]))
        rel_vol = max(0.1, float(base_obs_137[62])) if base_obs_137.size > 62 else 1.0

        atr = self.current_atr
        # Advance close price along real ret1 * atr
        c = self.current_price + (ret1 * atr)
        self.current_price = c

        # Exact arithmetic inversion of MT5 normalized geometry
        o = c - (body * atr)
        h = max(o, c) + (upper_wick * atr)
        l = min(o, c) - (lower_wick * atr)
        if range_val > 0 and (h - l) < range_val * atr * 0.95:
            mid = 0.5 * (o + c)
            h = max(h, mid + 0.5 * range_val * atr)
            l = min(l, mid - 0.5 * range_val * atr)
        v = max(1.0, rel_vol * 150.0)

        self.m1_buffer[:-1] = self.m1_buffer[1:]
        self.m1_buffer[-1] = [o, h, l, c, v]"""

new_update_code = """        # Synchronize exact real M1 OHLCV candlestick buffer from historical database
        req_time = getattr(req, "time", None) if req is not None else None
        synced_from_history = False
        if req_time and getattr(self, "_m1_time_to_idx", None):
            t_key = str(req_time).strip()[:16]
            idx = self._m1_time_to_idx.get(t_key, None)
            if idx is not None:
                start_i = max(0, idx - 59)
                bars_slice = self._m1_bars_matrix[start_i : idx + 1]
                if len(bars_slice) < 60:
                    pad = np.repeat(bars_slice[:1], 60 - len(bars_slice), axis=0)
                    self.m1_buffer = np.vstack([pad, bars_slice])
                else:
                    self.m1_buffer = bars_slice.copy()
                self.current_price = float(self.m1_buffer[-1, 3])
                tr_calc = np.maximum(
                    self.m1_buffer[1:, 1] - self.m1_buffer[1:, 2],
                    np.maximum(
                        np.abs(self.m1_buffer[1:, 1] - self.m1_buffer[:-1, 3]),
                        np.abs(self.m1_buffer[1:, 2] - self.m1_buffer[:-1, 3])
                    )
                )
                self.current_atr = float(np.mean(tr_calc[-14:]))
                if hasattr(self, "pipeline"):
                    self.pipeline.current_atr = self.current_atr
                synced_from_history = True

        if not synced_from_history:
            # Fallback reconstruction
            ret1 = float(base_obs_137[0])
            body = float(base_obs_137[7])
            upper_wick = max(0.0, float(base_obs_137[8]))
            lower_wick = max(0.0, float(base_obs_137[9]))
            range_val = max(0.0, float(base_obs_137[10]))
            rel_vol = max(0.1, float(base_obs_137[62])) if base_obs_137.size > 62 else 1.0

            atr = self.current_atr
            c = self.current_price + (ret1 * atr)
            self.current_price = c

            o = c - (body * atr)
            h = max(o, c) + (upper_wick * atr)
            l = min(o, c) - (lower_wick * atr)
            if range_val > 0 and (h - l) < range_val * atr * 0.95:
                mid = 0.5 * (o + c)
                h = max(h, mid + 0.5 * range_val * atr)
                l = min(l, mid - 0.5 * range_val * atr)
            v = max(1.0, rel_vol * 150.0)

            self.m1_buffer[:-1] = self.m1_buffer[1:]
            self.m1_buffer[-1] = [o, h, l, c, v]"""

# Normalize CRLF in strings before replace
old_b_norm = old_buffer_code.replace("\r\n", "\n")
old_u_norm = old_update_code.replace("\r\n", "\n")
c_norm = content.replace("\r\n", "\n")

assert old_b_norm in c_norm, "Old buffer code not found!"
c_norm = c_norm.replace(old_b_norm, new_buffer_code.replace("\r\n", "\n"), 1)

assert old_u_norm in c_norm, "Old update code not found!"
c_norm = c_norm.replace(old_u_norm, new_update_code.replace("\r\n", "\n"), 1)

target_file.write_text(c_norm, encoding="utf-8")
print("Successfully patched mythos_mt5_policy.py with real M1 buffer synchronization!")
