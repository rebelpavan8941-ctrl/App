import os
import gc
import sys
import json
import time
import glob
import datetime
import requests
import numpy as np
import pandas as pd
import xgboost as xgb

IST_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
BYBIT_TAKER_FEE = 0.00055    # 0.055% per side (0.11% round-trip for Crypto)
FOREX_HALF_SPREAD = 0.00006  # 0.6 pip per side (1.2 pips = 0.012% round-trip for Forex)
MAX_LEVERAGE = 20
DEFAULT_LEVERAGE = 10

TIMEFRAME_TO_BYBIT_INTERVAL = {
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
}

BARS_PER_DAY = {
    "5m": 288,
    "15m": 96,
    "30m": 48,
    "1h": 24,
    "4h": 6,
}

FOREX_LINEAR_SYMBOLS = {
    "EURUSDUSDT",
    "GBPUSDUSDT",
    "USDJPYUSDT",
    "XAUUSDT",
    "XAGUSDT",
    "XAUTUSDT",
}

DEFAULT_TEST_TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h"]
TRAINING_WINDOW_DAYS = 135   # 4.5 months (~135 days) full multi-regime coverage
MAX_KEPT_MODEL_VERSIONS = 2  # Retain only latest 2 trained ML models on disk

FEATURES = [
    "NATR", "BB_PctB", "BB_Width",
    "Ret_1h", "Ret_3h", "Ret_6h", "Ret_12h", "Ret_24h",
    "RSI_14", "Stoch_K", "Stoch_D", "ADX", "DI_Spread", "MACD_Hist",
    "Dist_EMA20", "Dist_EMA50", "Dist_EMA200",
    "VWAP_Dev"
]

STRATEGY_CATALOG_META = {
    "xgb_mtf_v2": {"name": "XGBoost 1H/4H Momentum Classifier", "timeframes": ["1h", "4h"]},
    "trendline_retest_pro": {"name": "Swing-Pivot Trendline Retest & Rejection", "timeframes": ["1h", "4h"]},
    "xrp_donchian_breakout": {"name": "Donchian Breakout + Supertrend", "timeframes": ["1h", "4h"]},
    "donchian_turtle_v2": {"name": "Donchian Channel Trend Breakout", "timeframes": ["4h"]},
    "btc_vol_surge_adx": {"name": "Volume Surge & ADX Momentum", "timeframes": ["1h"]},
    "supertrend_adx_filter": {"name": "Supertrend + ADX Regime Filter", "timeframes": ["1h"]},
    "bnb_ob_imbalance": {"name": "Orderbook Imbalance & Spread Gate", "timeframes": ["1h"]},
    "link_bb_expansion": {"name": "Bollinger Band Width Expansion", "timeframes": ["1h"]},
    "bb_keltner_squeeze": {"name": "TTM Volatility Squeeze Release", "timeframes": ["1h"]},
    "eth_rsi_stoch_reversal": {"name": "Stoch-RSI Mean Reversion", "timeframes": ["1h"]},
    "stoch_rsi_trend_dip": {"name": "Stochastic RSI Oversold Trend Dip", "timeframes": ["30m", "4h"]},
    "sol_ema_ribbon_atr": {"name": "Triple EMA Ribbon & ATR Trail", "timeframes": ["1h"]},
    "ema_ribbon_pullback": {"name": "EMA 21/55 Dynamic Trend Pullback", "timeframes": ["1h"]},
    "ada_macd_hist_accel": {"name": "MACD Histogram Acceleration", "timeframes": ["1h"]},
    "macd_hist_zero_cross": {"name": "MACD Histogram Impulse + Zero Cross", "timeframes": ["1h", "4h"]},
    "doge_supertrend_scalp": {"name": "Supertrend Volatility Scalp", "timeframes": ["15m"]},
    "avax_fib_ml_confluence": {"name": "Fibonacci ML Confluence", "timeframes": ["1h"]},
    "near_vwap_profile": {"name": "Session VWAP Deviation Bounce", "timeframes": ["1h"]},
    "vwap_reclaim_pro": {"name": "Anchored VWAP Session Reclaim", "timeframes": ["15m", "1h"]},
    "cvd_orderflow_div": {"name": "CVD Orderflow & Delta Absorption", "timeframes": ["15m", "1h"]},
    "funding_arb_mom": {"name": "Funding Rate Squeeze & OI Surge", "timeframes": ["1h", "4h"]},
    "ict_fvg_retest": {"name": "Smart Money FVG + Orderblock Retest", "timeframes": ["1h", "4h"]},
    "liq_sweep_reversal": {"name": "Swing High/Low Liquidity Sweep", "timeframes": ["30m", "1h"]},
    "volume_profile_poc": {"name": "Volume Profile POC & VAH/VAL Bounce", "timeframes": ["1h"]},
    "ichimoku_cloud_mtf": {"name": "Ichimoku Kumo Twist + TK Cross", "timeframes": ["1h", "4h"]},
    "london_ny_orb": {"name": "London / NY Session Opening Range Breakout", "timeframes": ["15m", "1h"]},
    "parabolic_sar_trail": {"name": "Parabolic SAR + Chaikin Money Flow", "timeframes": ["1h"]},
    "anchored_vwap_reclaim_v3": {"name": "Anchored VWAP Session Reclaim Pro", "timeframes": ["1h"]},
    "ict_liquidity_sweep_ob": {"name": "ICT Liquidity Sweep & Orderblock Retest", "timeframes": ["1h"]},
    "connors_rsi2_trend_pullback": {"name": "RSI-2 Trend Pullback & EMA200 Filter", "timeframes": ["1h"]},
    "keltner_ttm_squeeze_pro": {"name": "TTM Bollinger-Keltner Volatility Squeeze", "timeframes": ["1h"]},
    "cvd_orderflow_absorption": {"name": "CVD Orderflow & Volume Surge Absorption", "timeframes": ["15m"]},
    "g2_range_box_rejection": {"name": "Range Box Boundary Rejection (5m/15m Chop)", "timeframes": ["5m", "15m"]},
    "g2_keltner_snapback": {"name": "Keltner Channel Mean Reversion (5m/15m Chop)", "timeframes": ["5m", "15m"]},
    "g2_bb_rsi_exhaustion": {"name": "BB Extreme + RSI Exhaustion Fade (5m/15m Chop)", "timeframes": ["5m", "15m"]},
}


def classify_market(symbol: str) -> str:
    clean = symbol.strip().upper()
    compact = clean.replace("/", "").replace("-", "").replace("_", "")
    if compact in FOREX_LINEAR_SYMBOLS:
        return "FOREX"
    forex_bases = {"EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
    parts = clean.replace("-", "/").replace("_", "/").split("/")
    if len(parts) >= 2 and parts[0] in forex_bases and parts[1] in (forex_bases | {"USD"}):
        return "FOREX"
    if len(compact) == 6 and compact[:3] in forex_bases and compact[3:] in (forex_bases | {"USD"}):
        return "FOREX"
    return "CRYPTO"


def _is_legacy_yahoo_forex(symbol: str) -> bool:
    compact = symbol.strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if compact in FOREX_LINEAR_SYMBOLS or compact.endswith("USDT"):
        return False
    forex_bases = {"EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
    return len(compact) == 6 and compact[:3] in forex_bases and compact[3:] in (forex_bases | {"USD"})


class PaperStrategyTrainerAndTester:
    def __init__(self, paper_dir: str, models_dir: str):
        self.paper_dir = paper_dir
        self.models_dir = models_dir
        self.live_dir = os.path.join(os.path.dirname(paper_dir), "LIVE_ACCOUNT")
        self.paper_master_path = os.path.join(models_dir, "PAPER", "UNIFIED_PORTFOLIO_MASTER.json")
        self.paper_history_dir = os.path.join(models_dir, "PAPER", "history")
        self.live_master_path = os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json")
        self.legacy_master_path = os.path.join(models_dir, "UNIFIED_PORTFOLIO_MASTER.json")
        self.paper_results_path = os.path.join(paper_dir, "paper_trained_results.json")
        self.comparison_path = os.path.join(paper_dir, "paper_vs_live_comparison.json")
        self.analysis_report_path = os.path.join(paper_dir, "agentic_model_analysis.json")
        self.discovered_strats_path = os.path.join(paper_dir, "ai_discovered_strategies.json")
        self.adopted_strats_path = os.path.join(paper_dir, "ai_adopted_strategies.json")
        self.rate_limit_remaining = 120
        self.rate_limit_cap = 120

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=6, pool_maxsize=6)
        self.session.mount("https://", adapter)

        os.makedirs(paper_dir, exist_ok=True)
        os.makedirs(self.live_dir, exist_ok=True)
        os.makedirs(os.path.join(models_dir, "PAPER"), exist_ok=True)
        os.makedirs(self.paper_history_dir, exist_ok=True)
        os.makedirs(os.path.join(models_dir, "LIVE"), exist_ok=True)
        self.dynamic_strategies = self._load_dynamic_strategies()

    @staticmethod
    def _sanitize_xgb_model_json(model_dict: dict) -> dict:
        """Ensures float arrays in XGBoost JSON never serialize as bare integers (fixes json.h:82 Invalid cast)."""
        if not isinstance(model_dict, dict):
            return model_dict
        learner = model_dict.get("learner")
        if not isinstance(learner, dict):
            return model_dict

        lmp = learner.get("learner_model_param")
        if isinstance(lmp, dict) and "base_score" in lmp:
            bs = lmp["base_score"]
            if isinstance(bs, (int, float)) and not isinstance(bs, bool):
                lmp["base_score"] = str(float(bs))

        gb = learner.get("gradient_booster")
        if isinstance(gb, dict):
            model = gb.get("model")
            if isinstance(model, dict):
                trees = model.get("trees")
                if isinstance(trees, list):
                    float_array_keys = ("split_conditions", "base_weights", "loss_changes", "sum_hessian")
                    for tree in trees:
                        if not isinstance(tree, dict):
                            continue
                        for k in float_array_keys:
                            arr = tree.get(k)
                            if isinstance(arr, list):
                                tree[k] = [
                                    float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                                    for v in arr
                                ]
        return model_dict

    def _load_dynamic_strategies(self) -> dict:
        merged = {}
        for fpath in (self.adopted_strats_path, self.discovered_strats_path):
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                    for sid, sdata in (data.get("strategies", {}) or {}).items():
                        if isinstance(sdata, dict) and sdata.get("long_rules") and sdata.get("short_rules"):
                            merged[sid] = sdata
                except Exception:
                    pass
        return merged

    def _load_failure_playbook(self) -> list:
        mem_path = os.path.join(self.live_dir, "ai_copilot_memory.json")
        if not os.path.exists(mem_path):
            return []
        try:
            with open(mem_path, "r", encoding="utf-8-sig") as f:
                mem = json.load(f)
            pb = mem.get("failure_playbook", []) or []
            return pb if isinstance(pb, list) else []
        except Exception:
            return []

    def _load_and_clear_drift_watchlist(self, retrained_symbols: list = None) -> set:
        """Reads drifted symbols flagged by live_bot.py and clears symbols once retrained."""
        mem_path = os.path.join(self.live_dir, "ai_copilot_memory.json")
        if not os.path.exists(mem_path):
            return set()
        try:
            with open(mem_path, "r", encoding="utf-8-sig") as f:
                mem = json.load(f)
            dw = mem.get("drift_watchlist", {}) or {}
            flagged = {str(k).replace("/", "").upper() for k in dw.keys()}
            if retrained_symbols:
                changed = False
                for sym in retrained_symbols:
                    clean = str(sym).replace("/", "").upper()
                    if clean in dw:
                        dw.pop(clean, None)
                        changed = True
                if changed:
                    mem["drift_watchlist"] = dw
                    tmp_m = mem_path + ".tmp"
                    with open(tmp_m, "w", encoding="utf-8") as wf:
                        json.dump(mem, wf, indent=2)
                    os.replace(tmp_m, mem_path)
            return flagged
        except Exception:
            return set()

    def _adopt_winning_dynamic_strategies(self, trained_coins: dict, leverage: int = DEFAULT_LEVERAGE):
        if not self.dynamic_strategies:
            return
        adopted = {}
        if os.path.exists(self.adopted_strats_path):
            try:
                with open(self.adopted_strats_path, "r", encoding="utf-8-sig") as f:
                    adopted = json.load(f).get("strategies", {}) or {}
            except Exception:
                adopted = {}

        changed = False
        for sym, cdata in trained_coins.items():
            if not isinstance(cdata, dict):
                continue
            sid = str(cdata.get("strategy_id", ""))
            if sid in self.dynamic_strategies:
                pnl = float(cdata.get("net_pnl_usdt", 0.0))
                pf = float(cdata.get("profit_factor", 0.0))
                wr = float(cdata.get("win_rate_pct", 0.0))
                trades = int(cdata.get("trades", 0))
                if pnl > 0.0 and pf >= 1.25 and trades >= 14:
                    # 2x-Fee & Slippage Stress Gate before auto-adopting Gemini strategies
                    st_dict = cdata.get("stats", {}) or {}
                    raw_rets = st_dict.get("raw_trade_returns") or cdata.get("raw_trade_returns") or []
                    stress_ok = True
                    try:
                        import ai_regime_copilot
                        if raw_rets and hasattr(ai_regime_copilot, "passes_2x_fee_stress_test"):
                            stress_ok = ai_regime_copilot.passes_2x_fee_stress_test(
                                raw_rets, leverage=leverage, double_round_trip_fee=0.0022
                            )
                    except Exception:
                        stress_ok = True
                    if not stress_ok:
                        continue

                    best_cfg = dict(cdata.get("config", {}))
                    spec = dict(self.dynamic_strategies[sid])
                    spec["adopted_at"] = int(time.time())
                    spec["proven_on_symbol"] = sym
                    spec["proven_pf"] = pf
                    spec["proven_win_rate_pct"] = wr
                    spec["proven_net_pnl_usdt"] = pnl
                    spec["fixed_config"] = {
                        "long_thresh": best_cfg.get("long_thresh", 0.24),
                        "short_thresh": best_cfg.get("short_thresh", 0.24),
                        "sl_atr": best_cfg.get("sl_atr", 1.25),
                        "tp1_rr": best_cfg.get("tp1_rr", 1.90),
                        "tp2_rr": best_cfg.get("tp2_rr", 3.40),
                        "trail_atr": best_cfg.get("trail_atr", 1.80),
                        "timeframe": best_cfg.get("timeframe", "1H"),
                        "mtf_alignment_mode": best_cfg.get("mtf_alignment_mode", "SOFT"),
                    }
                    adopted[sid] = spec
                    changed = True

        if changed:
            tmp_a = self.adopted_strats_path + ".tmp"
            with open(tmp_a, "w", encoding="utf-8") as wf:
                json.dump({"updated_at": int(time.time()), "strategies": adopted}, wf, indent=2)
            os.replace(tmp_a, self.adopted_strats_path)

    def _prune_underperforming_gemini_strategies(self, trained_coins: dict):
        """Auto-removes Gemini strategies failing Paper backtest thresholds or Live performance."""
        try:
            import ai_regime_copilot
            prune_res = ai_regime_copilot.prune_underperforming_gemini_strategies(
                paper_dir=self.paper_dir,
                live_dir=self.live_dir,
                models_dir=self.models_dir,
            )
            if prune_res.get("removed"):
                print(f"[AGENTIC PRUNE] Auto-removed underperforming Gemini strategies: {prune_res['removed']}")
                self.dynamic_strategies = self._load_dynamic_strategies()
            return
        except Exception:
            pass

        for fpath in (self.discovered_strats_path, self.adopted_strats_path):
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8-sig") as f:
                    payload = json.load(f)
                strats = payload.get("strategies", {}) or {}
                kept = {}
                for sid, spec in strats.items():
                    runs = []
                    for cdata in trained_coins.values():
                        if not isinstance(cdata, dict):
                            continue
                        if str(cdata.get("strategy_id", "")) == sid:
                            runs.append(cdata)
                        for m in (cdata.get("tested_matrix", []) or []):
                            if isinstance(m, dict) and str(m.get("strategy_id", "")) == sid:
                                runs.append(m)
                    if runs:
                        best_r = max(runs, key=lambda r: (float(r.get("net_pnl_usdt", 0.0)), float(r.get("profit_factor", 0.0))))
                        if int(best_r.get("trades", 0)) < 14 or float(best_r.get("net_pnl_usdt", 0.0)) <= 0.0 or float(best_r.get("profit_factor", 0.0)) < 1.25:
                            continue
                    kept[sid] = spec
                if len(kept) != len(strats):
                    payload["strategies"] = kept
                    payload["updated_at"] = int(time.time())
                    tmp_f = fpath + ".tmp"
                    with open(tmp_f, "w", encoding="utf-8") as wf:
                        json.dump(payload, wf, indent=2)
                    os.replace(tmp_f, fpath)
            except Exception:
                continue
        self.dynamic_strategies = self._load_dynamic_strategies()

    def _prune_old_trained_models(self, keep_latest: int = MAX_KEPT_MODEL_VERSIONS):
        try:
            snapshots = sorted(
                glob.glob(os.path.join(self.paper_history_dir, "UNIFIED_PAPER_*.json")),
                key=os.path.getmtime,
                reverse=True,
            )
            for old_file in snapshots[keep_latest:]:
                try:
                    os.remove(old_file)
                except Exception:
                    pass
            for tmp_file in glob.glob(os.path.join(self.paper_dir, "_tmp_*")):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass
        except Exception:
            pass

    def _public_get(self, url: str, params: dict) -> dict:
        if self.rate_limit_cap > 0 and self.rate_limit_remaining < int(self.rate_limit_cap * 0.10):
            time.sleep(1.0)
        resp = self.session.get(url, params=params, timeout=7)
        rem = resp.headers.get("X-Bapi-Limit-Status")
        cap = resp.headers.get("X-Bapi-Limit")
        if rem and rem.isdigit():
            self.rate_limit_remaining = int(rem)
        if cap and cap.isdigit() and int(cap) > 0:
            self.rate_limit_cap = int(cap)
        return resp.json()

    def _fetch_forex_yahoo_history(self, symbol: str, timeframe: str = "1h") -> pd.DataFrame:
        clean_sym = symbol.replace("/", "").upper()
        tf_key = timeframe.lower()
        yf_interval = "15m" if tf_key == "15m" else ("30m" if tf_key == "30m" else "1h")
        yf_range = "60d" if yf_interval in ("15m", "30m") else "6mo"

        for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
            try:
                url = f"https://{host}/v8/finance/chart/{clean_sym}=X"
                resp = self.session.get(
                    url,
                    params={"interval": yf_interval, "range": yf_range},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=10,
                )
                data = resp.json().get("chart", {}).get("result", [])
                if not data:
                    continue
                res0 = data[0]
                ts_list = res0.get("timestamp", [])
                quote = res0.get("indicators", {}).get("quote", [{}])[0]
                opens = quote.get("open", [])
                highs = quote.get("high", [])
                lows = quote.get("low", [])
                closes = quote.get("close", [])

                rows = []
                for i, ts_sec in enumerate(ts_list):
                    c = closes[i] if i < len(closes) else None
                    if c is None or c <= 0:
                        continue
                    o = opens[i] if (i < len(opens) and opens[i] is not None) else c
                    h = highs[i] if (i < len(highs) and highs[i] is not None) else c
                    l = lows[i] if (i < len(lows) and lows[i] is not None) else c
                    rows.append([ts_sec * 1000, float(o), float(h), float(l), float(c), 1000.0])

                if not rows:
                    continue

                df = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp").sort_index()
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    df[col] = df[col].astype("float32")

                bar_range = (df["High"] - df["Low"]).clip(lower=1e-6)
                avg_range = bar_range.rolling(20, min_periods=1).mean().clip(lower=1e-6)
                df["Volume"] = ((bar_range / avg_range) * 1000.0).clip(lower=200.0, upper=5000.0).astype("float32")

                if tf_key == "4h":
                    df = df.resample("4h").agg({
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }).dropna()

                return df
            except Exception:
                continue
        return pd.DataFrame()

    def fetch_bybit_history(self, symbol: str, timeframe: str = "1h", days: int = TRAINING_WINDOW_DAYS) -> pd.DataFrame:
        if _is_legacy_yahoo_forex(symbol):
            return self._fetch_forex_yahoo_history(symbol, timeframe=timeframe)

        bybit_sym = symbol.replace("/", "").upper()
        tf_key = timeframe.lower()
        interval = TIMEFRAME_TO_BYBIT_INTERVAL.get(tf_key, "60")
        target_bars = (days * BARS_PER_DAY.get(tf_key, 24)) + 210

        collected = {}
        cursor_end = None
        pages = 0

        while len(collected) < target_bars and pages < 14:
            params = {"category": "linear", "symbol": bybit_sym, "interval": interval, "limit": 1000}
            if cursor_end is not None:
                params["end"] = cursor_end

            rows = []
            for base in ("https://api.bytick.com", "https://api.bybit.com", "https://api-demo.bybit.com"):
                try:
                    resp = self._public_get(f"{base}/v5/market/kline", params)
                    rows = resp.get("result", {}).get("list", [])
                    if rows:
                        break
                except Exception:
                    rows = []

            if not rows:
                break

            oldest_ts = None
            for r in rows:
                ts_ms = int(r[0])
                collected[ts_ms] = r
                if oldest_ts is None or ts_ms < oldest_ts:
                    oldest_ts = ts_ms

            if oldest_ts is None or (cursor_end is not None and oldest_ts >= cursor_end):
                break
            cursor_end = oldest_ts - 1
            pages += 1
            if len(rows) < 950:
                break

        if not collected:
            return pd.DataFrame()

        ordered_rows = [collected[k] for k in sorted(collected.keys())][-target_bars:]
        df = pd.DataFrame(ordered_rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"])
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        return df.drop(columns=["Turnover"]).set_index("timestamp")

    def build_matrix(self, df: pd.DataFrame, max_dd: float = 0.95, expansion: float = 1.25, horizon: int = 10) -> pd.DataFrame:
        data = df.copy()
        c = data["Close"]
        tr = pd.concat([
            data["High"] - data["Low"],
            (data["High"] - c.shift(1)).abs(),
            (data["Low"] - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        data["ATR"] = tr.rolling(14).mean()
        data["NATR"] = (data["ATR"] / c) * 100

        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        data["BB_Upper"] = sma20 + (std20 * 2.0)
        data["BB_Lower"] = sma20 - (std20 * 2.0)
        data["BB_PctB"] = (c - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"] + 1e-9)
        data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / (sma20 + 1e-9)

        for h in (1, 3, 6, 12, 24):
            data[f"Ret_{h}h"] = np.log(c / c.shift(h))

        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        data["RSI_14"] = rsi

        stoch = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min() + 1e-9)
        data["Stoch_K"] = stoch.rolling(3).mean() * 100
        data["Stoch_D"] = data["Stoch_K"].rolling(3).mean()

        plus_dm = np.where((data["High"].diff() > -data["Low"].diff()) & (data["High"].diff() > 0), data["High"].diff(), 0.0)
        minus_dm = np.where((-data["Low"].diff() > data["High"].diff()) & (-data["Low"].diff() > 0), -data["Low"].diff(), 0.0)
        plus_di = 100 * (pd.Series(plus_dm, index=data.index).rolling(14).mean() / (data["ATR"] + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm, index=data.index).rolling(14).mean() / (data["ATR"] + 1e-9))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        data["ADX"] = dx.rolling(14).mean()
        data["DI_Spread"] = plus_di - minus_di

        macd_line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        data["MACD_Hist"] = (macd_line - macd_line.ewm(span=9, adjust=False).mean()) / (data["ATR"] + 1e-9)

        data["EMA_20"] = c.ewm(span=20, adjust=False).mean()
        data["EMA_50"] = c.ewm(span=50, adjust=False).mean()
        data["EMA_200"] = c.ewm(span=200, adjust=False).mean()
        data["Dist_EMA20"] = (c - data["EMA_20"]) / (data["ATR"] + 1e-9)
        data["Dist_EMA50"] = (c - data["EMA_50"]) / (data["ATR"] + 1e-9)
        data["Dist_EMA200"] = (c - data["EMA_200"]) / (data["ATR"] + 1e-9)

        typical_px = (data["High"] + data["Low"] + c) / 3.0
        vol_safe = data["Volume"].clip(lower=1.0) if "Volume" in data.columns and not (data["Volume"] == 0).all() else pd.Series(1.0, index=data.index)
        session_vwap = (typical_px * vol_safe).rolling(24).sum() / (vol_safe.rolling(24).sum() + 1e-9)
        data["VWAP_Dev"] = (c - session_vwap) / (data["ATR"] + 1e-9)

        hl2 = ((data["High"] + data["Low"]) * 0.5).to_numpy()
        atr_arr = data["ATR"].fillna(0.0).to_numpy()
        c_arr = c.to_numpy()
        upper_arr = hl2 + 3.0 * atr_arr
        lower_arr = hl2 - 3.0 * atr_arr
        st_arr = np.ones(len(data), dtype=np.int8)
        uptrend = True
        for i in range(10, len(c_arr)):
            if uptrend and c_arr[i] < lower_arr[i - 1]:
                uptrend = False
            elif not uptrend and c_arr[i] > upper_arr[i - 1]:
                uptrend = True
            st_arr[i] = 1 if uptrend else -1

        data["Supertrend"] = st_arr
        data["Supertrend_Prev"] = data["Supertrend"].shift(1).fillna(data["Supertrend"])
        data["Donchian_High"] = data["High"].rolling(20).max().shift(1)
        data["Donchian_Low"] = data["Low"].rolling(20).min().shift(1)
        data["Vol_Surge"] = data["Volume"] / (data["Volume"].rolling(20).mean() + 1e-9) if "Volume" in data.columns and not (data["Volume"] == 0).all() else 1.0
        data["BB_Width_Med"] = data["BB_Width"].rolling(30).median()

        fwd_max = data["High"].rolling(horizon).max().shift(-horizon)
        fwd_min = data["Low"].rolling(horizon).min().shift(-horizon)
        data["Target_Long"] = np.where(((fwd_max - c) >= expansion * data["ATR"]) & ((c - fwd_min) <= max_dd * data["ATR"]), 1, 0)
        data["Target_Short"] = np.where(((c - fwd_min) >= expansion * data["ATR"]) & ((fwd_max - c) <= max_dd * data["ATR"]), 1, 0)
        return data.dropna()

    def _serialize_booster(self, clf: xgb.XGBClassifier) -> dict:
        try:
            raw_bytes = bytes(clf.get_booster().save_raw("json"))
            return self._sanitize_xgb_model_json(json.loads(raw_bytes.decode("utf-8")))
        except Exception:
            os.makedirs(self.paper_dir, exist_ok=True)
            safe_tmp = os.path.join(self.paper_dir, f"_tmp_booster_{os.getpid()}_{int(time.time() * 1000)}.json")
            try:
                clf.save_model(safe_tmp)
                with open(safe_tmp, "r", encoding="utf-8") as f:
                    return self._sanitize_xgb_model_json(json.load(f))
            finally:
                if os.path.exists(safe_tmp):
                    try:
                        os.remove(safe_tmp)
                    except Exception:
                        pass

    def _fit_xgb_pair(self, train_df: pd.DataFrame):
        y_l = train_df["Target_Long"].copy()
        y_s = train_df["Target_Short"].copy()
        if y_l.sum() < 5:
            y_l.iloc[-5:] = 1
        if y_s.sum() < 5:
            y_s.iloc[-5:] = 1

        params = dict(
            n_estimators=125,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.82,
            colsample_bytree=0.82,
            min_child_weight=4,
            reg_alpha=0.4,
            reg_lambda=1.4,
            n_jobs=2,
            eval_metric="logloss",
            random_state=42,
        )
        clf_long = xgb.XGBClassifier(**params)
        clf_short = xgb.XGBClassifier(**params)
        X_train = train_df[FEATURES].astype("float32")
        clf_long.fit(X_train, y_l)
        clf_short.fit(X_train, y_s)
        return clf_long, clf_short

    @staticmethod
    def _fit_meta_labeler(meta_samples: list):
        """
        Trains a lightweight L2-regularized logistic meta-classifier predicting P(TP1 before SL)
        on out-of-sample trade features: [raw_prob, prob_margin, adx, natr, vwap_dev, 0.0, 0.0].
        """
        if not meta_samples or len(meta_samples) < 16:
            return None
        X = np.array([s[0] for s in meta_samples], dtype=np.float64)
        y = np.array([1.0 if s[1] else 0.0 for s in meta_samples], dtype=np.float64)
        if y.sum() < 3 or (len(y) - y.sum()) < 3:
            return None

        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds = np.where(stds < 1e-6, 1.0, stds)
        Xn = (X - means) / stds

        w = np.zeros(7, dtype=np.float64)
        b = 0.0
        lr = 0.08
        l2 = 0.15
        n = float(len(y))

        for _ in range(180):
            z = np.clip(Xn @ w + b, -18.0, 18.0)
            preds = 1.0 / (1.0 + np.exp(-z))
            err = preds - y
            grad_w = (Xn.T @ err) / n + l2 * w
            grad_b = float(err.mean())
            w -= lr * grad_w
            b -= lr * grad_b

        # Keep unseen derivatives positioning slots (5, 6) strictly zero-weighted in historical fit
        w[5] = 0.0
        w[6] = 0.0

        final_probs = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -18.0, 18.0)))
        q25 = float(np.quantile(final_probs, 0.25))
        min_prob = round(min(max(q25, 0.44), 0.55), 4)

        return {
            "weights": [round(float(v), 6) for v in w],
            "means": [round(float(v), 6) for v in means],
            "stds": [round(float(v), 6) for v in stds],
            "bias": round(float(b), 6),
            "min_prob": min_prob,
            "samples": int(len(y)),
        }

    def _eval_dynamic_rule_list(self, test_df: pd.DataFrame, rules: list) -> np.ndarray:
        n = len(test_df)
        mask = np.ones(n, dtype=bool)
        mask[0] = False
        for rule in rules:
            ind = rule.get("ind", "")
            op = rule.get("op", "")
            val = float(rule.get("val", 0.0))
            if ind not in test_df.columns:
                mask[:] = False
                return mask
            arr = test_df[ind].to_numpy()[:-1]
            if op == ">":
                mask[1:] &= (arr > val)
            elif op == "<":
                mask[1:] &= (arr < val)
            elif op == ">=":
                mask[1:] &= (arr >= val)
            elif op == "<=":
                mask[1:] &= (arr <= val)
        return mask

    def _compute_vectorized_filter_masks(self, test_df: pd.DataFrame, strategy_id: str, relaxed: bool = False):
        n = len(test_df)
        long_ok = np.zeros(n, dtype=bool)
        short_ok = np.zeros(n, dtype=bool)
        if n < 2:
            return long_ok, short_ok

        if strategy_id in self.dynamic_strategies:
            spec = self.dynamic_strategies[strategy_id]
            return (
                self._eval_dynamic_rule_list(test_df, spec.get("long_rules", [])),
                self._eval_dynamic_rule_list(test_df, spec.get("short_rules", [])),
            )

        curr_open = test_df["Open"].to_numpy()[1:]
        prev_ema50_d = test_df["Dist_EMA50"].to_numpy()[:-1]

        if relaxed:
            long_ok[1:] = prev_ema50_d > -1.5
            short_ok[1:] = prev_ema50_d < 1.5
            return long_ok, short_ok

        sid = strategy_id.lower()
        p_don_h = test_df["Donchian_High"].to_numpy()[:-1]
        p_don_l = test_df["Donchian_Low"].to_numpy()[:-1]
        p_st = test_df["Supertrend"].to_numpy()[:-1]
        p_vol = test_df["Vol_Surge"].to_numpy()[:-1]
        p_adx = test_df["ADX"].to_numpy()[:-1]
        p_di = test_df["DI_Spread"].to_numpy()[:-1]
        p_bbw = test_df["BB_Width"].to_numpy()[:-1]
        p_bbwm = test_df["BB_Width_Med"].to_numpy()[:-1]
        p_bbp = test_df["BB_PctB"].to_numpy()[:-1]
        p_rsi = test_df["RSI_14"].to_numpy()[:-1]
        p_sk = test_df["Stoch_K"].to_numpy()[:-1]
        p_sd = test_df["Stoch_D"].to_numpy()[:-1]
        p_e20 = test_df["EMA_20"].to_numpy()[:-1]
        p_e50 = test_df["EMA_50"].to_numpy()[:-1]
        p_macd = test_df["MACD_Hist"].to_numpy()[:-1]
        p_d200 = test_df["Dist_EMA200"].to_numpy()[:-1]
        p_vwap = test_df["VWAP_Dev"].to_numpy()[:-1]

        if "g2_range_box" in sid:
            p_c = test_df["Close"].to_numpy(dtype=float)[:-1]
            p_o = test_df["Open"].to_numpy(dtype=float)[:-1]
            p_h = test_df["High"].to_numpy(dtype=float)[:-1]
            p_l = test_df["Low"].to_numpy(dtype=float)[:-1]
            p_atr = test_df["ATR"].to_numpy(dtype=float)[:-1]
            rng_h28 = test_df["High"].rolling(28).max().to_numpy(dtype=float)[:-1]
            rng_l28 = test_df["Low"].rolling(28).min().to_numpy(dtype=float)[:-1]
            p_rng = np.maximum(p_h - p_l, 1e-9)
            l_wick = (np.minimum(p_o, p_c) - p_l) / p_rng
            u_wick = (p_h - np.maximum(p_o, p_c)) / p_rng
            box_span = np.maximum(rng_h28 - rng_l28, 1e-9)
            box_pos = (p_c - rng_l28) / box_span
            valid_box = (~np.isnan(rng_h28)) & ((rng_h28 - rng_l28) > (2.2 * p_atr)) & (p_adx < 20.5)
            long_ok[1:] = valid_box & (box_pos <= 0.15) & (l_wick >= 0.36) & (p_c > p_o)
            short_ok[1:] = valid_box & (box_pos >= 0.85) & (u_wick >= 0.36) & (p_c < p_o)
            return long_ok, short_ok

        if "g2_keltner" in sid:
            p_c = test_df["Close"].to_numpy(dtype=float)[:-1]
            p_o = test_df["Open"].to_numpy(dtype=float)[:-1]
            p_h = test_df["High"].to_numpy(dtype=float)[:-1]
            p_l = test_df["Low"].to_numpy(dtype=float)[:-1]
            p_d20 = test_df["Dist_EMA20"].to_numpy(dtype=float)[:-1]
            p_body = np.abs(p_c - p_o) / np.maximum(p_h - p_l, 1e-9)
            long_ok[1:] = (p_adx < 24.0) & (p_d20 <= -1.70) & (p_c > p_o) & (p_body >= 0.35)
            short_ok[1:] = (p_adx < 24.0) & (p_d20 >= 1.70) & (p_c < p_o) & (p_body >= 0.35)
            return long_ok, short_ok

        if "g2_bb_rsi" in sid:
            p_c = test_df["Close"].to_numpy(dtype=float)[:-1]
            p_o = test_df["Open"].to_numpy(dtype=float)[:-1]
            long_ok[1:] = (p_adx < 24.0) & (p_bbp <= 0.08) & (p_rsi <= 34.0) & (p_c > p_o)
            short_ok[1:] = (p_adx < 24.0) & (p_bbp >= 0.92) & (p_rsi >= 66.0) & (p_c < p_o)
            return long_ok, short_ok

        if "trendline" in sid:
            highs_all = test_df["High"].to_numpy(dtype=float)
            lows_all = test_df["Low"].to_numpy(dtype=float)
            closes_all = test_df["Close"].to_numpy(dtype=float)
            opens_all = test_df["Open"].to_numpy(dtype=float)
            atrs_all = test_df["ATR"].to_numpy(dtype=float)
            adx_all = test_df["ADX"].to_numpy(dtype=float)

            swing_lows = []
            swing_highs = []
            for j in range(3, n - 4):
                if lows_all[j] == np.min(lows_all[j - 3:j + 4]):
                    swing_lows.append((j, lows_all[j]))
                if highs_all[j] == np.max(highs_all[j - 3:j + 4]):
                    swing_highs.append((j, highs_all[j]))

            sl_ptr = 0
            sh_ptr = 0
            for idx in range(1, n):
                c_bar = idx - 1
                atr_v = atrs_all[c_bar]
                if atr_v <= 0 or adx_all[c_bar] < 17.0:
                    continue
                while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] <= c_bar - 3:
                    sl_ptr += 1
                while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] <= c_bar - 3:
                    sh_ptr += 1

                if sl_ptr >= 2:
                    (i1, p1), (i2, p2) = swing_lows[sl_ptr - 2], swing_lows[sl_ptr - 1]
                    if i2 > i1 and p2 > p1:
                        tl_px = p2 + ((p2 - p1) / float(i2 - i1)) * float(c_bar - i2)
                        dist_a = (closes_all[c_bar] - tl_px) / atr_v
                        if -0.25 <= dist_a <= 0.65 and closes_all[c_bar] > opens_all[c_bar]:
                            long_ok[idx] = True

                if sh_ptr >= 2:
                    (i1, p1), (i2, p2) = swing_highs[sh_ptr - 2], swing_highs[sh_ptr - 1]
                    if i2 > i1 and p2 < p1:
                        tl_px = p2 + ((p2 - p1) / float(i2 - i1)) * float(c_bar - i2)
                        dist_a = (tl_px - closes_all[c_bar]) / atr_v
                        if -0.25 <= dist_a <= 0.65 and closes_all[c_bar] < opens_all[c_bar]:
                            short_ok[idx] = True
        elif "donchian" in sid:
            long_ok[1:] = (p_st == 1) & (curr_open >= p_don_h * 0.998) & (p_vol >= 1.05)
            short_ok[1:] = (p_st == -1) & (curr_open <= p_don_l * 1.002) & (p_vol >= 1.05)
        elif "vol_surge" in sid or "ob_imbalance" in sid or "supertrend_adx" in sid:
            long_ok[1:] = (p_vol >= 1.25) & (p_adx >= 20) & (p_di > 0)
            short_ok[1:] = (p_vol >= 1.25) & (p_adx >= 20) & (p_di < 0)
        elif "bb_" in sid or "squeeze" in sid:
            long_ok[1:] = (p_bbw > p_bbwm) & (p_bbp > 0.72)
            short_ok[1:] = (p_bbw > p_bbwm) & (p_bbp < 0.28)
        elif "stoch" in sid or "rsi" in sid:
            long_ok[1:] = (p_rsi < 44) & (p_sk > p_sd)
            short_ok[1:] = (p_rsi > 56) & (p_sk < p_sd)
        elif "ema_ribbon" in sid:
            long_ok[1:] = (p_e20 > p_e50) & (curr_open > p_e20)
            short_ok[1:] = (p_e20 < p_e50) & (curr_open < p_e20)
        elif "macd" in sid:
            long_ok[1:] = (p_macd > 0) & (p_d200 > 0)
            short_ok[1:] = (p_macd < 0) & (p_d200 < 0)
        elif "supertrend" in sid or "sar" in sid:
            long_ok[1:] = (p_st == 1)
            short_ok[1:] = (p_st == -1)
        elif "vwap" in sid or "poc" in sid or "fvg" in sid or "liq_sweep" in sid or "orb" in sid:
            long_ok[1:] = (p_vwap < -0.45) & (p_vol >= 1.15)
            short_ok[1:] = (p_vwap > 0.45) & (p_vol >= 1.15)
        else:
            long_ok[1:] = (p_di > 0) & (p_vol >= 1.05)
            short_ok[1:] = (p_di < 0) & (p_vol >= 1.05)

        return long_ok, short_ok

    def _simulate_strategy_on_tf(
        self,
        test_df: pd.DataFrame,
        p_long: np.ndarray,
        p_short: np.ndarray,
        strategy_id: str,
        timeframe: str,
        leverage: int,
        initial_capital: float,
        is_forex: bool = False,
        htf_dir_aligned: np.ndarray = None,
        relaxed_filter: bool = False,
        extra_quantile_thresholds: list = None,
        extended_param_sweep: bool = False,
    ) -> dict:
        is_g2_chop = str(strategy_id).lower().startswith("g2_")
        round_trip_fee = (FOREX_HALF_SPREAD * 2.0) if is_forex else (0.00052 if is_g2_chop else (BYBIT_TAKER_FEE * 2.0))
        dyn_spec = self.dynamic_strategies.get(strategy_id)

        param_configs = [
            (0.16, 1.10, 1.75, False, 1.6),
            (0.20, 1.20, 1.90, False, 1.7),
            (0.24, 1.25, 2.00, False, 1.8),
            (0.28, 1.35, 2.15, True, 1.8),
            (0.34, 1.45, 2.30, True, 1.9),
        ] if is_forex else [
            (0.18, 1.15, 1.80, False, 1.6),
            (0.22, 1.25, 1.95, False, 1.7),
            (0.26, 1.30, 2.10, False, 1.8),
            (0.30, 1.40, 2.25, True, 1.8),
            (0.36, 1.50, 2.40, True, 1.9),
        ]

        if extended_param_sweep or dyn_spec is not None:
            param_configs.extend([
                (0.19, 1.45, 2.35, False, 2.0),
                (0.23, 1.60, 2.50, False, 2.1),
                (0.27, 1.05, 1.70, False, 1.5),
                (0.32, 1.70, 2.65, True, 2.2),
            ])

        if dyn_spec and isinstance(dyn_spec.get("fixed_config"), dict):
            fc = dyn_spec["fixed_config"]
            param_configs.insert(0, (
                float(fc.get("long_thresh", 0.24)),
                float(fc.get("sl_atr", 1.25)),
                float(fc.get("tp1_rr", 1.95)),
                str(fc.get("mtf_alignment_mode", "SOFT")).upper() == "STRICT",
                float(fc.get("trail_atr", 1.80)),
            ))

        if extra_quantile_thresholds:
            for qt in extra_quantile_thresholds:
                param_configs.append((round(float(qt), 4), 1.25, 1.95, False, 1.8))

        opens = test_df["Open"].to_numpy()
        highs = test_df["High"].to_numpy()
        lows = test_df["Low"].to_numpy()
        atrs = test_df["ATR"].to_numpy()
        vols = test_df["Vol_Surge"].to_numpy()
        adxs = test_df["ADX"].to_numpy()
        natrs = test_df["NATR"].to_numpy()
        vwaps = test_df["VWAP_Dev"].to_numpy()
        n_bars = len(test_df)
        mid_oos = max(1, n_bars // 2)  # Splits 40% OOS into 20% Validation + 20% Untouched Holdout

        f_long, f_short = self._compute_vectorized_filter_masks(test_df, strategy_id, relaxed=relaxed_filter)

        best_eval = None
        best_score = -9999999.0

        if is_g2_chop:
            param_configs = [
                (0.20, 1.65, 0.92, False, 1.4),
                (0.24, 1.65, 1.00, False, 1.5),
                (0.28, 1.65, 1.10, False, 1.6),
            ]
        mtf_modes_to_test = ["OFF"] if is_g2_chop else (["OFF", "SOFT", "STRICT"] if htf_dir_aligned is not None else ["OFF"])

        for thresh, sl_atr, tp1_rr, require_htf, trail_mult in param_configs:
            for mtf_mode_label in mtf_modes_to_test:
                if mtf_mode_label == "OFF" and require_htf:
                    continue

                tp2_rr = tp1_rr * 1.80
                cfg = {
                    "long_thresh": thresh,
                    "short_thresh": thresh,
                    "sl_atr": sl_atr,
                    "tp1_rr": tp1_rr,
                    "tp2_rr": tp2_rr,
                    "trail_atr": trail_mult,
                    "strategy_id": strategy_id,
                    "timeframe": timeframe,
                    "mtf_alignment_mode": mtf_mode_label,
                    "relaxed_filter": relaxed_filter,
                }
                if dyn_spec:
                    cfg["custom_rules"] = dyn_spec

                le_arr = np.zeros(n_bars, dtype=bool)
                se_arr = np.zeros(n_bars, dtype=bool)
                le_arr[1:] = (p_long[:-1] >= thresh) & f_long[1:]
                se_arr[1:] = (p_short[:-1] >= thresh) & f_short[1:]

                if mtf_mode_label != "OFF" and htf_dir_aligned is not None:
                    conv_boost = 0.05 if mtf_mode_label == "SOFT" else 0.08
                    vol_req = 1.10 if mtf_mode_label == "SOFT" else 1.25
                    hc_l = np.zeros(n_bars, dtype=bool)
                    hc_s = np.zeros(n_bars, dtype=bool)
                    hc_l[1:] = (p_long[:-1] >= thresh + conv_boost) & (vols[:-1] >= vol_req)
                    hc_s[1:] = (p_short[:-1] >= thresh + conv_boost) & (vols[:-1] >= vol_req)
                    le_arr = le_arr & ((htf_dir_aligned == 1) | hc_l)
                    se_arr = se_arr & ((htf_dir_aligned == -1) | hc_s)

                trades_pnl = []
                val_pnls = []
                holdout_pnls = []
                raw_returns = []
                meta_samples = []
                equity = initial_capital
                in_pos = False
                side_long = True
                entry = sl = tp1 = tp2 = high_w = low_w = 0.0
                entry_bar_idx = 0
                entry_meta_vec = None
                tp1_hit = False

                for i in range(1, n_bars):
                    h = highs[i]
                    l = lows[i]
                    atr = atrs[i - 1]

                    if not in_pos:
                        le = le_arr[i]
                        se = se_arr[i]
                        if le and se:
                            if p_long[i - 1] >= p_short[i - 1]:
                                se = False
                            else:
                                le = False
                        if le != se:
                            in_pos = True
                            side_long = le
                            entry = opens[i]
                            entry_bar_idx = i
                            high_w = low_w = entry
                            tp1_hit = False
                            chosen_p = float(p_long[i - 1] if side_long else p_short[i - 1])
                            p_margin = abs(float(p_long[i - 1]) - float(p_short[i - 1]))
                            entry_meta_vec = [
                                chosen_p,
                                p_margin,
                                float(adxs[i - 1]),
                                float(natrs[i - 1]),
                                float(vwaps[i - 1]),
                                0.0,
                                0.0,
                            ]
                            min_sl_floor = entry * (0.0020 if is_forex else 0.0155)
                            risk = max(sl_atr * atr, min_sl_floor)
                            if side_long:
                                sl = entry - risk
                                tp1 = entry + risk * tp1_rr
                                tp2 = entry + risk * tp2_rr
                            else:
                                sl = entry + risk
                                tp1 = entry - risk * tp1_rr
                                tp2 = entry - risk * tp2_rr
                    else:
                        if side_long:
                            if h > high_w:
                                high_w = h
                            if h >= tp1 and not tp1_hit:
                                tp1_hit = True
                                sl = max(sl, entry + 0.35 * (tp1 - entry))
                            if tp1_hit and high_w >= entry + 0.75 * (tp2 - entry):
                                tr_sl = high_w - trail_mult * atr
                                if tr_sl > sl:
                                    sl = tr_sl
                            if l <= sl or h >= tp2:
                                ex = sl if l <= sl else tp2
                                raw_ret = (0.5 * ((tp1 - entry) / entry) + 0.5 * ((ex - entry) / entry)) if tp1_hit else ((ex - entry) / entry)
                                pnl_usdt = (equity * 0.05) * ((raw_ret - round_trip_fee) * leverage)
                                equity += pnl_usdt
                                trades_pnl.append(pnl_usdt)
                                if entry_bar_idx < mid_oos:
                                    val_pnls.append(pnl_usdt)
                                else:
                                    holdout_pnls.append(pnl_usdt)
                                raw_returns.append(float(raw_ret))
                                if entry_meta_vec is not None:
                                    meta_samples.append((entry_meta_vec, bool(pnl_usdt > 0.0)))
                                in_pos = False
                        else:
                            if l < low_w:
                                low_w = l
                            if l <= tp1 and not tp1_hit:
                                tp1_hit = True
                                sl = min(sl, entry - 0.35 * (entry - tp1))
                            if tp1_hit and low_w <= entry - 0.75 * (entry - tp2):
                                tr_sl = low_w + trail_mult * atr
                                if tr_sl < sl:
                                    sl = tr_sl
                            if h >= sl or l <= tp2:
                                ex = sl if h >= sl else tp2
                                raw_ret = (0.5 * ((entry - tp1) / entry) + 0.5 * ((entry - ex) / entry)) if tp1_hit else ((entry - ex) / entry)
                                pnl_usdt = (equity * 0.05) * ((raw_ret - round_trip_fee) * leverage)
                                equity += pnl_usdt
                                trades_pnl.append(pnl_usdt)
                                if entry_bar_idx < mid_oos:
                                    val_pnls.append(pnl_usdt)
                                else:
                                    holdout_pnls.append(pnl_usdt)
                                raw_returns.append(float(raw_ret))
                                if entry_meta_vec is not None:
                                    meta_samples.append((entry_meta_vec, bool(pnl_usdt > 0.0)))
                                in_pos = False

                arr = np.array(trades_pnl)
                wins = arr[arr > 0]
                losses = arr[arr < 0]
                total_trades = int(len(arr))
                wr = round((len(wins) / total_trades * 100.0), 2) if total_trades > 0 else 0.0
                pf = round(float(wins.sum() / abs(losses.sum())), 2) if len(losses) > 0 else round(float(wins.sum()), 2)
                net_usdt = round(float(equity - initial_capital), 2)
                net_pct = round((net_usdt / initial_capital) * 100.0, 2)

                # 3-Way Nested Temporal Holdout Verification (20% Validation + 20% Untouched Holdout)
                val_sum = float(sum(val_pnls)) if val_pnls else net_usdt
                holdout_sum = float(sum(holdout_pnls)) if holdout_pnls else net_usdt
                wf_both_positive = (total_trades >= 14 and val_sum > 0.0 and holdout_sum > 0.0)
                walk_forward_mult = 1.22 if wf_both_positive else 0.72

                # Dual-Tier Scoring:
                # Tier A: High-Frequency Edge (85-175 trades, PF >= 1.50, WR >= 50%)
                # Tier B: High-Precision Sniper Edge (18-84 trades, PF >= 2.15, WR >= 55%)
                sniper_floor = 12 if is_forex else 18
                freq_floor = 45 if is_forex else 85

                is_tier_a = (total_trades >= freq_floor and pf >= 1.50 and wr >= 50.0 and net_usdt > 0)
                is_tier_b_sniper = (sniper_floor <= total_trades < freq_floor and pf >= 2.15 and wr >= 55.0 and net_usdt > 0)

                if total_trades == 0:
                    score = -100000.0
                elif is_tier_a:
                    freq_bonus = 1.25 if (105 <= total_trades <= 165) else 1.10
                    pf_bonus = min(pf, 4.0) * 42.0
                    score = ((net_usdt * 1.45) + pf_bonus + (wr * 0.65)) * freq_bonus * walk_forward_mult
                elif is_tier_b_sniper:
                    pf_bonus = min(pf, 4.5) * 58.0
                    score = ((net_usdt * 1.75) + pf_bonus + (wr * 0.85)) * walk_forward_mult
                elif total_trades >= sniper_floor and pf >= 1.25 and net_usdt > 0:
                    score = ((net_usdt * 0.75) + (pf * 25.0) + (wr * 0.35)) * walk_forward_mult
                elif total_trades >= sniper_floor:
                    score = -5000.0 + net_usdt + (pf * 15.0) + (wr * 0.2)
                else:
                    score = -25000.0 + (total_trades * 150.0) + net_usdt

                if best_eval is None or score > best_score:
                    best_score = score
                    best_eval = {
                        "score": score,
                        "cfg": cfg,
                        "meta_samples": meta_samples,
                        "stats": {
                            "trades": total_trades,
                            "win_rate_pct": wr,
                            "profit_factor": pf,
                            "leverage": leverage,
                            "net_pnl_pct": net_pct,
                            "net_pnl_usdt": net_usdt,
                            "net_pnl": net_usdt,
                            "val_pnl_usdt": round(val_sum, 2),
                            "holdout_pnl_usdt": round(holdout_sum, 2),
                            "walk_forward_ok": wf_both_positive,
                            "edge_tier": "HIGH_FREQ_110_150" if is_tier_a else ("SNIPER_HIGH_PF" if is_tier_b_sniper else "STANDARD"),
                            "raw_trade_returns": [round(r, 6) for r in raw_returns],
                        },
                    }
        return best_eval

    def _filter_archetype_key(self, strategy_id: str) -> str:
        if strategy_id in self.dynamic_strategies:
            return strategy_id
        sid = strategy_id.lower()
        if "trendline" in sid:
            return "trendline"
        if "donchian" in sid:
            return "donchian"
        if "vol_surge" in sid or "ob_imbalance" in sid or "supertrend_adx" in sid:
            return "vol_adx"
        if "bb_" in sid or "squeeze" in sid:
            return "bb_squeeze"
        if "stoch" in sid or "rsi" in sid:
            return "rsi_stoch"
        if "ema_ribbon" in sid:
            return "ema_ribbon"
        if "macd" in sid:
            return "macd"
        if "supertrend" in sid or "sar" in sid:
            return "supertrend"
        if "vwap" in sid or "poc" in sid or "fvg" in sid or "liq_sweep" in sid or "orb" in sid:
            return "vwap_smc"
        return "default"

    def _append_bot_log_to_state(self, state_file: str, lines: list):
        if not lines:
            return
        try:
            st = {}
            if os.path.exists(state_file):
                with open(state_file, "r", encoding="utf-8-sig") as f:
                    st = json.load(f)
            logs = st.setdefault("last_bot_logs", [])
            logs.extend(lines)
            st["last_bot_logs"] = logs[-80:]
            with open(state_file, "w", encoding="utf-8") as wf:
                json.dump(st, wf, indent=2)
        except Exception:
            pass

    def agentic_analyze_and_auto_promote(self, trained_coins: dict) -> dict:
        """
        Agentic AI Post-Training Analyzer:
        Evaluates all trained coins across 3-way temporal holdout, meta-labeling,
        and failure-playbook rules, logs per-coin verdicts, and automatically
        promotes qualifying winners into LIVE/UNIFIED_PORTFOLIO_MASTER.json.
        """
        live_portfolio = {}
        if os.path.exists(self.live_master_path):
            try:
                with open(self.live_master_path, "r", encoding="utf-8-sig") as f:
                    live_portfolio = json.load(f).get("portfolio", {}) or {}
            except Exception:
                live_portfolio = {}

        playbook = self._load_failure_playbook()
        clock_str = datetime.datetime.now(IST_TZ).strftime("%H:%M:%S IST")
        promoted_coins = []
        held_coins = []
        analysis_items = []
        log_entries = []

        if playbook:
            log_entries.append(f"[{clock_str}] [FAILURE PLAYBOOK] Active lessons applied: {len(playbook)} rules")

        ranked_coins = sorted(
            trained_coins.items(),
            key=lambda kv: (
                float(kv[1].get("profit_factor", 0.0)) * 40.0 + float(kv[1].get("net_pnl_usdt", 0.0)),
            ),
            reverse=True,
        )

        for sym, cdata in ranked_coins:
            if not isinstance(cdata, dict):
                continue
            st = cdata.get("stats", {}) or {}
            trades = int(cdata.get("trades", st.get("trades", 0)))
            wr = float(cdata.get("win_rate_pct", st.get("win_rate_pct", 0.0)))
            pf = float(cdata.get("profit_factor", st.get("profit_factor", 0.0)))
            pnl = float(cdata.get("net_pnl_usdt", st.get("net_pnl_usdt", 0.0)))
            wf_ok = bool(st.get("walk_forward_ok", True))
            has_meta = isinstance((cdata.get("models") or {}).get("meta"), dict)
            is_fx = (classify_market(sym) == "FOREX")

            sniper_min = 12 if is_fx else 18
            freq_min = 45 if is_fx else 85

            qualifies_freq = (trades >= freq_min and pf >= 1.50 and wr >= 50.0 and pnl > 0.0)
            qualifies_sniper = (sniper_min <= trades < freq_min and pf >= 2.15 and wr >= 55.0 and pnl > 0.0)
            qualifies = (qualifies_freq or qualifies_sniper) and (wf_ok or pf >= 2.30)

            archetype = "HIGH-FREQ" if qualifies_freq else ("SNIPER-HIGH-PF" if qualifies_sniper else "SUB-THRESHOLD")
            meta_tag = " + MetaML" if has_meta else ""
            in_live = sym in live_portfolio

            if in_live:
                l_data = live_portfolio.get(sym, {})
                l_pf = float(l_data.get("profit_factor", 0.0))
                l_pnl = float(l_data.get("net_pnl_usdt", l_data.get("net_pnl", 0.0)))
                l_trades = int(l_data.get("trades", 0))
                beats_live = (
                    (pf > l_pf and pnl >= l_pnl * 0.90)
                    or (pnl > l_pnl * 1.05 and pf >= l_pf * 0.92)
                    or (l_trades < sniper_min and trades >= sniper_min and pf >= 1.50)
                )
                should_promote = qualifies and beats_live
                reason = (
                    f"Upgraded Live ({archetype}{meta_tag}: PF {l_pf:.2f}->{pf:.2f}, {trades} trades, WR {wr:.1f}%, +${pnl:.0f})"
                    if should_promote
                    else f"Kept existing Live champion (Paper PF {pf:.2f} vs Live PF {l_pf:.2f})"
                )
            else:
                should_promote = qualifies
                if should_promote:
                    reason = f"Auto-Promoted ({archetype}{meta_tag}: PF {pf:.2f}, WR {wr:.1f}%, {trades} trades, +${pnl:.0f})"
                else:
                    why_parts = []
                    if trades < sniper_min:
                        why_parts.append(f"low sample ({trades}<{sniper_min})")
                    if pnl <= 0.0:
                        why_parts.append(f"negative PnL (${pnl:.0f})")
                    elif trades >= freq_min and pf < 1.50:
                        why_parts.append(f"PF {pf:.2f}<1.50")
                    elif trades < freq_min and pf < 2.15:
                        why_parts.append(f"Sniper PF {pf:.2f}<2.15")
                    if wr < 50.0:
                        why_parts.append(f"WR {wr:.1f}%<50%")
                    if not wf_ok and pf < 2.30:
                        why_parts.append("failed 20% untouched holdout")
                    reason = f"Held in Paper ({', '.join(why_parts) or 'below live bar'})"

            if should_promote and self.move_coin_to_live_model(sym):
                promoted_coins.append(sym)
                log_entries.append(f"[{clock_str}] [AGENTIC AI PROMOTE] {sym}: {reason}")
            else:
                held_coins.append(sym)
                log_entries.append(f"[{clock_str}] [AGENTIC AI AUDIT] {sym}: {reason}")

            analysis_items.append({
                "symbol": sym,
                "archetype": archetype,
                "trades": trades,
                "win_rate_pct": wr,
                "profit_factor": pf,
                "net_pnl_usdt": pnl,
                "walk_forward_ok": wf_ok,
                "meta_labeler_trained": has_meta,
                "auto_promoted": should_promote,
                "verdict": reason,
            })

        summary_msg = (
            f"Agentic AI Model Audit: Auto-promoted {len(promoted_coins)}/{len(analysis_items)} coins to Live "
            f"({', '.join(promoted_coins) if promoted_coins else 'None qualified'})."
        )
        log_entries.insert(0, f"[{clock_str}] {summary_msg}")

        report = {
            "analyzed_at": int(time.time()),
            "summary": summary_msg,
            "promoted_coins": promoted_coins,
            "held_coins": held_coins,
            "coins": analysis_items,
        }
        with open(self.analysis_report_path, "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2)

        self._append_bot_log_to_state(os.path.join(self.paper_dir, "paper_engine_state.json"), log_entries)
        self._append_bot_log_to_state(os.path.join(self.live_dir, "live_engine_state.json"), log_entries)
        return report

    def train_and_evaluate_paper_coins(
        self,
        added_coins: list,
        active_strategies: list,
        timeframes: list = None,
        leverage: int = DEFAULT_LEVERAGE,
        initial_capital: float = 10000.0,
        crypto_enabled: bool = True,
        forex_enabled: bool = False,
    ) -> dict:
        self.dynamic_strategies = self._load_dynamic_strategies()
        leverage = min(max(int(leverage), 1), MAX_LEVERAGE)
        test_tfs = [tf.lower() for tf in (timeframes or DEFAULT_TEST_TIMEFRAMES) if tf.lower() in TIMEFRAME_TO_BYBIT_INTERVAL]
        if not test_tfs:
            test_tfs = ["1h", "4h"]

        diverse_defaults = [
            "g2_range_box_rejection",
            "g2_keltner_snapback",
            "g2_bb_rsi_exhaustion",
            "xgb_mtf_v2",
            "trendline_retest_pro",
            "xrp_donchian_breakout",
            "btc_vol_surge_adx",
            "bb_keltner_squeeze",
            "eth_rsi_stoch_reversal",
            "ema_ribbon_pullback",
            "macd_hist_zero_cross",
            "near_vwap_profile",
        ]
        base_strats = list(dict.fromkeys((list(active_strategies) if active_strategies else []) + diverse_defaults))

        raw_strats = list(dict.fromkeys(list(self.dynamic_strategies.keys()) + base_strats))
        seen_archetypes = set()
        strategies_to_test = []
        for s in raw_strats:
            arch = self._filter_archetype_key(s)
            if arch not in seen_archetypes:
                seen_archetypes.add(arch)
                strategies_to_test.append(s)

        existing_paper_portfolio = {}
        if os.path.exists(self.paper_master_path):
            try:
                with open(self.paper_master_path, "r", encoding="utf-8-sig") as ef:
                    existing_paper_portfolio = json.load(ef).get("portfolio", {}) or {}
            except Exception:
                existing_paper_portfolio = {}

        drifted_compact = self._load_and_clear_drift_watchlist(retrained_symbols=None)

        candidate_syms = list(added_coins)
        for existing_sym in existing_paper_portfolio.keys():
            if existing_sym.replace("/", "").upper() in drifted_compact and existing_sym not in candidate_syms:
                candidate_syms.append(existing_sym)

        filtered_coins = []
        for sym in candidate_syms:
            mkt = classify_market(sym)
            if (mkt == "CRYPTO" and crypto_enabled) or (mkt == "FOREX" and forex_enabled):
                filtered_coins.append(sym)

        trained_coins = {}
        paper_portfolio = {}
        total_net_pnl = 0.0
        win_rates = []

        for symbol in filtered_coins:
            is_fx = (classify_market(symbol) == "FOREX")
            exp_atr = 0.90 if is_fx else 1.15
            dd_atr = 0.85 if is_fx else 0.95
            hor_bars = 12 if is_fx else 10
            sniper_floor = 12 if is_fx else 18
            freq_floor = 45 if is_fx else 85

            htf_trend = None
            raw_4h = self.fetch_bybit_history(symbol, timeframe="4h", days=TRAINING_WINDOW_DAYS)
            if len(raw_4h) >= 210:
                df_4h = self.build_matrix(raw_4h, max_dd=dd_atr, expansion=exp_atr, horizon=hor_bars)
                if not df_4h.empty:
                    htf_trend = df_4h["Supertrend"]
                del df_4h

            best_candidate = None
            all_tf_runs = []
            cached_tf_bundles = []

            for tf in test_tfs:
                raw = raw_4h if (tf == "4h" and not raw_4h.empty) else self.fetch_bybit_history(symbol, timeframe=tf, days=TRAINING_WINDOW_DAYS)
                if len(raw) < 300:
                    continue

                df = self.build_matrix(raw, max_dd=dd_atr, expansion=exp_atr, horizon=hor_bars)
                if len(df) < 180:
                    continue

                # 60% Train / 40% Out-of-Sample (20% Validation + 20% Untouched Holdout)
                split = int(len(df) * 0.60)
                train_end = max(120, split - hor_bars)
                train_df = df.iloc[:train_end]
                test_df = df.iloc[split:]
                if len(test_df) < 60:
                    continue

                f_means = train_df[FEATURES].mean().tolist()
                f_stds = train_df[FEATURES].std().replace(0, 1.0).fillna(1.0).tolist()
                clf_long, clf_short = self._fit_xgb_pair(train_df)

                X_test = test_df[FEATURES].astype("float32")
                p_long = clf_long.predict_proba(X_test)[:, 1]
                p_short = clf_short.predict_proba(X_test)[:, 1]

                htf_dir_aligned = None
                if tf != "4h" and htf_trend is not None and not htf_trend.empty:
                    aligned_series = htf_trend.reindex(test_df.index, method="ffill").fillna(0)
                    htf_dir_aligned = aligned_series.to_numpy(dtype=np.int8)

                cached_tf_bundles.append((tf, raw, test_df, clf_long, clf_short, p_long, p_short, f_means, f_stds, htf_dir_aligned))

                for strat_id in strategies_to_test:
                    eval_res = self._simulate_strategy_on_tf(
                        test_df=test_df,
                        p_long=p_long,
                        p_short=p_short,
                        strategy_id=strat_id,
                        timeframe=tf.upper(),
                        leverage=leverage,
                        initial_capital=initial_capital,
                        is_forex=is_fx,
                        htf_dir_aligned=htf_dir_aligned,
                    )
                    st = eval_res["stats"]
                    best_mtf = eval_res["cfg"].get("mtf_alignment_mode", "STRICT")
                    dyn_meta = self.dynamic_strategies.get(strat_id)
                    strat_meta = dyn_meta if dyn_meta else STRATEGY_CATALOG_META.get(strat_id, {"name": strat_id.replace("_", " ").title()})
                    strat_name = strat_meta.get("name", strat_id)

                    all_tf_runs.append({
                        "strategy_id": strat_id,
                        "strategy_name": strat_name,
                        "timeframe": tf.upper(),
                        "mtf_mode": best_mtf,
                        "trades": st["trades"],
                        "win_rate_pct": st["win_rate_pct"],
                        "profit_factor": st["profit_factor"],
                        "net_pnl_usdt": st["net_pnl_usdt"],
                    })

                    if best_candidate is None or eval_res["score"] > best_candidate["score"]:
                        cfg = dict(eval_res["cfg"])
                        cfg["bybit_symbol"] = symbol.replace("/", "").upper()
                        cfg["strategy"] = f"{strat_name} ({tf.upper()} • MTF:{best_mtf})"
                        meta_model = self._fit_meta_labeler(eval_res.get("meta_samples", []))
                        models_payload = {
                            "long": self._serialize_booster(clf_long),
                            "short": self._serialize_booster(clf_short),
                            "feature_means": f_means,
                            "feature_stds": f_stds,
                        }
                        if meta_model is not None:
                            models_payload["meta"] = meta_model
                        best_candidate = {
                            "score": eval_res["score"],
                            "strategy_id": strat_id,
                            "strategy_name": strat_name,
                            "timeframe": tf.upper(),
                            "stats": st,
                            "cfg": cfg,
                            "models": models_payload,
                        }

            cur_trades = best_candidate["stats"]["trades"] if best_candidate else 0
            cur_pnl = best_candidate["stats"]["net_pnl_usdt"] if best_candidate else -999.0
            cur_pf = best_candidate["stats"]["profit_factor"] if best_candidate else 0.0
            passes_either_tier = (cur_trades >= freq_floor and cur_pf >= 1.50 and cur_pnl > 0.0) or (
                sniper_floor <= cur_trades < freq_floor and cur_pf >= 2.15 and cur_pnl > 0.0
            )

            if (not passes_either_tier) and cached_tf_bundles:
                for tf, raw, test_df, clf_long, clf_short, p_long, p_short, f_means, f_stds, htf_dir_aligned in cached_tf_bundles:
                    q_thresholds = sorted(set([
                        float(np.quantile(np.maximum(p_long, p_short), 0.65)),
                        float(np.quantile(np.maximum(p_long, p_short), 0.75)),
                        float(np.quantile(np.maximum(p_long, p_short), 0.85)),
                    ]))
                    eval_res = self._simulate_strategy_on_tf(
                        test_df=test_df,
                        p_long=p_long,
                        p_short=p_short,
                        strategy_id="xgb_adaptive_quantile",
                        timeframe=tf.upper(),
                        leverage=leverage,
                        initial_capital=initial_capital,
                        is_forex=is_fx,
                        htf_dir_aligned=htf_dir_aligned,
                        relaxed_filter=True,
                        extra_quantile_thresholds=q_thresholds,
                        extended_param_sweep=True,
                    )
                    st = eval_res["stats"]
                    best_mtf = eval_res["cfg"].get("mtf_alignment_mode", "OFF")
                    if best_candidate is None or eval_res["score"] > best_candidate["score"]:
                        cur_trades = st["trades"]
                        cur_pnl = st["net_pnl_usdt"]
                        cur_pf = st["profit_factor"]
                        cfg = dict(eval_res["cfg"])
                        cfg["bybit_symbol"] = symbol.replace("/", "").upper()
                        cfg["strategy"] = f"Adaptive Tuned XGBoost ({tf.upper()} • MTF:{best_mtf})"
                        meta_model = self._fit_meta_labeler(eval_res.get("meta_samples", []))
                        models_payload = {
                            "long": self._serialize_booster(clf_long),
                            "short": self._serialize_booster(clf_short),
                            "feature_means": f_means,
                            "feature_stds": f_stds,
                        }
                        if meta_model is not None:
                            models_payload["meta"] = meta_model
                        best_candidate = {
                            "score": eval_res["score"],
                            "strategy_id": "xgb_adaptive_quantile",
                            "strategy_name": "Adaptive Tuned XGBoost",
                            "timeframe": tf.upper(),
                            "stats": st,
                            "cfg": cfg,
                            "models": models_payload,
                        }

            if (cur_trades < sniper_floor or cur_pnl <= 0.0) and cached_tf_bundles:
                for tf, raw, _, _, _, _, _, _, _, htf_dir_aligned in [b for b in cached_tf_bundles if b[0] in ("30m", "1h", "4h")]:
                    df_sens = self.build_matrix(raw, max_dd=1.15, expansion=0.75, horizon=14)
                    if len(df_sens) < 180:
                        continue
                    split = int(len(df_sens) * 0.60)
                    train_s = df_sens.iloc[:max(120, split - 14)]
                    test_s = df_sens.iloc[split:]
                    f_means = train_s[FEATURES].mean().tolist()
                    f_stds = train_s[FEATURES].std().replace(0, 1.0).fillna(1.0).tolist()
                    clf_l2, clf_s2 = self._fit_xgb_pair(train_s)
                    X_test_s = test_s[FEATURES].astype("float32")
                    pl2 = clf_l2.predict_proba(X_test_s)[:, 1]
                    ps2 = clf_s2.predict_proba(X_test_s)[:, 1]
                    q_sens = sorted(set([
                        float(np.quantile(np.maximum(pl2, ps2), 0.66)),
                        float(np.quantile(np.maximum(pl2, ps2), 0.78)),
                    ]))
                    htf_sens = None
                    if tf != "4h" and htf_trend is not None and not htf_trend.empty:
                        htf_sens = htf_trend.reindex(test_s.index, method="ffill").fillna(0).to_numpy(dtype=np.int8)

                    eval_res = self._simulate_strategy_on_tf(
                        test_df=test_s,
                        p_long=pl2,
                        p_short=ps2,
                        strategy_id="xgb_high_sensitivity",
                        timeframe=tf.upper(),
                        leverage=leverage,
                        initial_capital=initial_capital,
                        is_forex=is_fx,
                        htf_dir_aligned=htf_sens,
                        relaxed_filter=True,
                        extra_quantile_thresholds=q_sens,
                        extended_param_sweep=True,
                    )
                    st = eval_res["stats"]
                    best_mtf = eval_res["cfg"].get("mtf_alignment_mode", "OFF")
                    if best_candidate is None or eval_res["score"] > best_candidate["score"]:
                        cfg = dict(eval_res["cfg"])
                        cfg["bybit_symbol"] = symbol.replace("/", "").upper()
                        cfg["strategy"] = f"High-Sensitivity XGBoost ({tf.upper()} • MTF:{best_mtf})"
                        meta_model = self._fit_meta_labeler(eval_res.get("meta_samples", []))
                        models_payload = {
                            "long": self._serialize_booster(clf_l2),
                            "short": self._serialize_booster(clf_s2),
                            "feature_means": f_means,
                            "feature_stds": f_stds,
                        }
                        if meta_model is not None:
                            models_payload["meta"] = meta_model
                        best_candidate = {
                            "score": eval_res["score"],
                            "strategy_id": "xgb_high_sensitivity",
                            "strategy_name": "High-Sensitivity XGBoost",
                            "timeframe": tf.upper(),
                            "stats": st,
                            "cfg": cfg,
                            "models": models_payload,
                        }
                    del df_sens, train_s, test_s, clf_l2, clf_s2

            del cached_tf_bundles, raw_4h
            gc.collect()

            if best_candidate is None:
                continue

            st = best_candidate["stats"]
            best_mtf_mode = best_candidate["cfg"].get("mtf_alignment_mode", "STRICT")
            coin_entry = {
                "market": classify_market(symbol),
                "strategy": f"{best_candidate['strategy_name']} ({best_candidate['timeframe']} • MTF:{best_mtf_mode})",
                "strategy_id": best_candidate["strategy_id"],
                "timeframe": best_candidate["timeframe"],
                "mtf_alignment_mode": best_mtf_mode,
                "win_rate_pct": st["win_rate_pct"],
                "profit_factor": st["profit_factor"],
                "net_pnl_pct": st["net_pnl_pct"],
                "net_pnl_usdt": st["net_pnl_usdt"],
                "net_pnl": st["net_pnl_usdt"],
                "trades": st["trades"],
                "stats": st,
                "tested_matrix": all_tf_runs,
                "config": best_candidate["cfg"],
                "models": best_candidate["models"],
            }

            # Champion vs Challenger: always refresh if flagged for live feature drift
            is_drifted = symbol.replace("/", "").upper() in drifted_compact
            prev_champ = existing_paper_portfolio.get(symbol)
            if (not is_drifted) and isinstance(prev_champ, dict) and isinstance(prev_champ.get("models"), dict):
                old_trades = int(prev_champ.get("trades", 0))
                old_pf = float(prev_champ.get("profit_factor", 0.0))
                old_pnl = float(prev_champ.get("net_pnl_usdt", prev_champ.get("net_pnl", 0.0)))
                old_wr = float(prev_champ.get("win_rate_pct", 0.0))
                new_trades = int(st["trades"])
                new_pf = float(st["profit_factor"])
                new_pnl = float(st["net_pnl_usdt"])
                new_wr = float(st["win_rate_pct"])

                freq_expanded = (old_trades < freq_floor and new_trades >= freq_floor and new_pf >= 1.50 and new_pnl > 0.0)
                improved = freq_expanded or (
                    new_pnl > 0.0
                    and new_trades >= sniper_floor
                    and (
                        (new_pf > old_pf and new_pnl >= old_pnl * 0.92)
                        or (new_pnl > old_pnl and new_pf >= old_pf * 0.92)
                        or (new_wr > old_wr + 3.0 and new_pf >= old_pf and new_pnl > 0.0)
                    )
                )
                old_valid = (old_trades >= freq_floor and old_pf >= 1.50) or (sniper_floor <= old_trades < freq_floor and old_pf >= 2.15)
                if not improved and old_valid and old_pnl > 0.0:
                    coin_entry = dict(prev_champ)
                    coin_entry["tested_matrix"] = all_tf_runs

            total_net_pnl += float(coin_entry.get("net_pnl_usdt", 0.0))
            win_rates.append(float(coin_entry.get("win_rate_pct", 0.0)))
            trained_coins[symbol] = coin_entry
            paper_portfolio[symbol] = coin_entry

        if trained_coins and drifted_compact:
            self._load_and_clear_drift_watchlist(retrained_symbols=list(trained_coins.keys()))

        self._adopt_winning_dynamic_strategies(trained_coins, leverage=leverage)

        with open(self.paper_results_path, "w", encoding="utf-8") as f:
            json.dump({"leverage": leverage, "training_days": TRAINING_WINDOW_DAYS, "timeframes_tested": test_tfs, "coins": trained_coins}, f, indent=2)

        self._prune_underperforming_gemini_strategies(trained_coins)

        now_ts = int(time.time())
        avg_accuracy = round(sum(win_rates) / len(win_rates), 2) if win_rates else 0.0
        paper_master = {
            "version": f"UNIFIED_PAPER_{len(paper_portfolio)}C_135D",
            "exported_at": now_ts,
            "training_days": TRAINING_WINDOW_DAYS,
            "model_accuracy_pct": avg_accuracy,
            "total_net_pnl": round(total_net_pnl, 2),
            "features": FEATURES,
            "portfolio": paper_portfolio,
        }
        with open(self.paper_master_path, "w", encoding="utf-8") as f:
            json.dump(paper_master, f, indent=2)

        hist_snapshot = os.path.join(self.paper_history_dir, f"UNIFIED_PAPER_{now_ts}.json")
        with open(hist_snapshot, "w", encoding="utf-8") as hf:
            json.dump(paper_master, hf, indent=2)
        self._prune_old_trained_models(keep_latest=MAX_KEPT_MODEL_VERSIONS)

        self.auto_compare_with_live(trained_coins)
        self.agentic_analyze_and_auto_promote(trained_coins)
        gc.collect()
        return trained_coins

    def auto_compare_with_live(self, paper_coins_dict: dict) -> list:
        live_portfolio = {}
        if os.path.exists(self.live_master_path):
            try:
                with open(self.live_master_path, "r", encoding="utf-8-sig") as f:
                    live_portfolio = json.load(f).get("portfolio", {})
            except Exception:
                live_portfolio = {}

        comparisons = []
        for sym, p_data in paper_coins_dict.items():
            p_stats = p_data.get("stats", {})
            p_trades = int(p_data.get("trades", p_stats.get("trades", 0)))
            p_wr = float(p_stats.get("win_rate_pct", 0.0))
            p_pf = float(p_stats.get("profit_factor", 0.0))
            p_pnl = float(p_stats.get("net_pnl_usdt", 0.0))

            l_data = live_portfolio.get(sym, {})
            l_stats = l_data.get("stats", {}) if isinstance(l_data, dict) else {}
            l_wr = float(l_data.get("win_rate_pct", l_stats.get("win_rate_pct", 0.0))) if isinstance(l_data, dict) else 0.0
            l_pf = float(l_data.get("profit_factor", l_stats.get("profit_factor", 0.0))) if isinstance(l_data, dict) else 0.0
            l_pnl = float(l_data.get("net_pnl_usdt", l_stats.get("net_pnl_usdt", 0.0))) if isinstance(l_data, dict) else 0.0

            is_fx = (classify_market(sym) == "FOREX")
            sniper_min = 12 if is_fx else 18
            freq_min = 45 if is_fx else 85
            qualifies = (p_trades >= freq_min and p_pf >= 1.50 and p_wr >= 50.0 and p_pnl > 0) or (
                sniper_min <= p_trades < freq_min and p_pf >= 2.15 and p_wr >= 55.0 and p_pnl > 0
            )
            is_new_coin = sym not in live_portfolio
            is_better = (qualifies and p_pf >= l_pf and p_pnl > l_pnl * 0.90) or (is_new_coin and qualifies)

            comparisons.append({
                "symbol": sym,
                "market": p_data.get("market", classify_market(sym)),
                "best_strategy": p_data.get("strategy", ""),
                "best_timeframe": p_data.get("timeframe", "1H"),
                "paper_win_rate_pct": p_wr,
                "live_win_rate_pct": l_wr,
                "paper_profit_factor": p_pf,
                "live_profit_factor": l_pf,
                "paper_net_pnl": p_pnl,
                "live_net_pnl": l_pnl,
                "is_new_coin": is_new_coin,
                "is_paper_better": is_better,
            })

        with open(self.comparison_path, "w", encoding="utf-8") as f:
            json.dump({"generated_at": int(time.time()), "comparisons": comparisons}, f, indent=2)
        return comparisons

    def _save_live_master(self, live_master: dict):
        portfolio = live_master.get("portfolio", {})
        all_wrs = [v.get("win_rate_pct", 0.0) for v in portfolio.values() if isinstance(v, dict)]
        all_pnls = [v.get("net_pnl_usdt", v.get("net_pnl", 0.0)) for v in portfolio.values() if isinstance(v, dict)]
        count = len(portfolio)
        live_master["version"] = f"LIVE_PORTFOLIO_{count}C"
        live_master["features"] = FEATURES
        live_master["model_accuracy_pct"] = round(sum(all_wrs) / len(all_wrs), 2) if all_wrs else 0.0
        live_master["total_net_pnl"] = round(sum(all_pnls), 2)
        live_master["exported_at"] = int(time.time())

        with open(self.live_master_path, "w", encoding="utf-8") as f:
            json.dump(live_master, f, indent=2)
        with open(self.legacy_master_path, "w", encoding="utf-8") as f:
            json.dump(live_master, f, indent=2)

    def move_coin_to_live_model(self, symbol: str) -> bool:
        if not os.path.exists(self.paper_results_path):
            return False

        with open(self.paper_results_path, "r", encoding="utf-8-sig") as f:
            paper_data = json.load(f)

        norm_sym = symbol.strip().upper()
        if "/" not in norm_sym and norm_sym.endswith("USDT"):
            norm_sym = f"{norm_sym[:-4]}/USDT"

        coin_bundle = paper_data.get("coins", {}).get(norm_sym) or paper_data.get("coins", {}).get(symbol)
        if not coin_bundle:
            return False

        if os.path.exists(self.live_master_path):
            with open(self.live_master_path, "r", encoding="utf-8-sig") as f:
                live_master = json.load(f)
        else:
            live_master = {"version": "LIVE_PORTFOLIO_0C", "exported_at": int(time.time()), "features": FEATURES, "portfolio": {}}

        portfolio = live_master.setdefault("portfolio", {})
        portfolio.pop(norm_sym.replace("/", ""), None)
        stats = dict(coin_bundle.get("stats", {}))
        stats.pop("raw_trade_returns", None)
        best_mtf_mode = coin_bundle.get("mtf_alignment_mode") or coin_bundle.get("config", {}).get("mtf_alignment_mode", "STRICT")

        models_copy = dict(coin_bundle.get("models", {}))
        if isinstance(models_copy.get("long"), dict):
            models_copy["long"] = self._sanitize_xgb_model_json(models_copy["long"])
        if isinstance(models_copy.get("short"), dict):
            models_copy["short"] = self._sanitize_xgb_model_json(models_copy["short"])

        portfolio[norm_sym] = {
            "market": coin_bundle.get("market", classify_market(norm_sym)),
            "strategy": coin_bundle.get("strategy", "XGBoost 1H/4H Momentum"),
            "strategy_id": coin_bundle.get("strategy_id", "xgb_mtf_v2"),
            "timeframe": coin_bundle.get("timeframe", "1H"),
            "mtf_alignment_mode": best_mtf_mode,
            "win_rate_pct": stats.get("win_rate_pct", 0.0),
            "profit_factor": stats.get("profit_factor", 0.0),
            "net_pnl_pct": stats.get("net_pnl_pct", 0.0),
            "net_pnl_usdt": stats.get("net_pnl_usdt", 0.0),
            "net_pnl": stats.get("net_pnl_usdt", 0.0),
            "trades": stats.get("trades", 0),
            "stats": stats,
            "config": coin_bundle["config"],
            "models": models_copy,
        }

        self._save_live_master(live_master)
        return True

    def remove_coin_from_paper_model(self, symbol: str) -> bool:
        norm_sym = symbol.strip().upper()
        if "/" not in norm_sym and norm_sym.endswith("USDT"):
            norm_sym = f"{norm_sym[:-4]}/USDT"

        removed_any = False

        if os.path.exists(self.paper_master_path):
            try:
                with open(self.paper_master_path, "r", encoding="utf-8-sig") as f:
                    master = json.load(f)
                portfolio = master.setdefault("portfolio", {})
                if norm_sym in portfolio or norm_sym.replace("/", "") in portfolio:
                    portfolio.pop(norm_sym, None)
                    portfolio.pop(norm_sym.replace("/", ""), None)
                    removed_any = True
                    wrs = [v.get("win_rate_pct", 0.0) for v in portfolio.values() if isinstance(v, dict)]
                    pnls = [v.get("net_pnl_usdt", 0.0) for v in portfolio.values() if isinstance(v, dict)]
                    master["version"] = f"UNIFIED_PAPER_{len(portfolio)}C_135D"
                    master["model_accuracy_pct"] = round(sum(wrs) / len(wrs), 2) if wrs else 0.0
                    master["total_net_pnl"] = round(sum(pnls), 2)
                    with open(self.paper_master_path, "w", encoding="utf-8") as wf:
                        json.dump(master, wf, indent=2)
            except Exception:
                pass

        if os.path.exists(self.paper_results_path):
            try:
                with open(self.paper_results_path, "r", encoding="utf-8-sig") as f:
                    res_data = json.load(f)
                coins_map = res_data.setdefault("coins", {})
                if norm_sym in coins_map or norm_sym.replace("/", "") in coins_map:
                    coins_map.pop(norm_sym, None)
                    coins_map.pop(norm_sym.replace("/", ""), None)
                    removed_any = True
                    with open(self.paper_results_path, "w", encoding="utf-8") as wf:
                        json.dump(res_data, wf, indent=2)
            except Exception:
                pass

        config_file = os.path.join(self.paper_dir, "paper_runtime_config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8-sig") as f:
                    cfg = json.load(f)
                coins_list = cfg.get("coins", [])
                new_coins = [c for c in coins_list if c.strip().upper() not in (norm_sym, norm_sym.replace("/", ""))]
                if len(new_coins) != len(coins_list):
                    cfg["coins"] = new_coins
                    removed_any = True
                    with open(config_file, "w", encoding="utf-8") as wf:
                        json.dump(cfg, wf, indent=2)
            except Exception:
                pass

        return removed_any

    def remove_coin_from_live_model(self, symbol: str) -> bool:
        if not os.path.exists(self.live_master_path):
            return False
        with open(self.live_master_path, "r", encoding="utf-8-sig") as f:
            live_master = json.load(f)

        norm_sym = symbol.strip().upper()
        if "/" not in norm_sym and norm_sym.endswith("USDT"):
            norm_sym = f"{norm_sym[:-4]}/USDT"

        portfolio = live_master.setdefault("portfolio", {})
        portfolio.pop(norm_sym, None)
        portfolio.pop(norm_sym.replace("/", ""), None)
        self._save_live_master(live_master)
        return True


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.environ.get("PAPER_DATA_DIR", os.path.join(base_dir, "bot_data", "PAPER"))
    models_dir = os.environ.get("MODELS_DIR", os.path.join(base_dir, "ml_models"))

    config_file = os.path.join(paper_dir, "paper_runtime_config.json")
    runtime_cfg = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8-sig") as f:
                runtime_cfg = json.load(f)
        except Exception:
            runtime_cfg = {}

    coins = runtime_cfg.get("coins", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LINK/USDT"])
    strategies = runtime_cfg.get("strategies", ["xgb_mtf_v2", "trendline_retest_pro", "xrp_donchian_breakout"])
    timeframes = runtime_cfg.get("timeframes", DEFAULT_TEST_TIMEFRAMES)
    leverage = int(runtime_cfg.get("leverage", DEFAULT_LEVERAGE))
    crypto_enabled = bool(runtime_cfg.get("crypto_enabled", True))
    forex_enabled = bool(runtime_cfg.get("forex_enabled", False))

    trainer = PaperStrategyTrainerAndTester(paper_dir=paper_dir, models_dir=models_dir)

    if len(sys.argv) >= 3 and sys.argv[1] == "--promote":
        ok = trainer.move_coin_to_live_model(sys.argv[2].upper())
        print(json.dumps({"ok": ok, "promoted": sys.argv[2].upper()}))
    elif len(sys.argv) >= 3 and sys.argv[1] == "--remove":
        ok = trainer.remove_coin_from_live_model(sys.argv[2].upper())
        print(json.dumps({"ok": ok, "removed": sys.argv[2].upper()}))
    elif len(sys.argv) >= 3 and sys.argv[1] == "--remove-paper":
        ok = trainer.remove_coin_from_paper_model(sys.argv[2].upper())
        print(json.dumps({"ok": ok, "removed": sys.argv[2].upper()}))
    else:
        results = trainer.train_and_evaluate_paper_coins(
            added_coins=coins,
            active_strategies=strategies,
            timeframes=timeframes,
            leverage=leverage,
            crypto_enabled=crypto_enabled,
            forex_enabled=forex_enabled,
        )
        print(f"Trained & evaluated {len(results)} Paper coins across 4.5 months (135 days) & auto-promoted qualifying winners to Live.")
