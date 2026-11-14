import os
import sys
import json
import time
import tempfile
import requests
import numpy as np
import pandas as pd
import xgboost as xgb

BYBIT_TAKER_FEE = 0.00055  # 0.055% per side (0.11% round-trip)
MAX_LEVERAGE = 20
DEFAULT_LEVERAGE = 10

TIMEFRAME_TO_BYBIT_INTERVAL = {
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
}

DEFAULT_TEST_TIMEFRAMES = ["15m", "30m", "1h", "4h"]

FEATURES = [
    "NATR", "BB_PctB", "BB_Width",
    "Ret_1h", "Ret_3h", "Ret_6h", "Ret_12h", "Ret_24h",
    "RSI_14", "Stoch_K", "Stoch_D", "ADX", "DI_Spread", "MACD_Hist",
    "Dist_EMA20", "Dist_EMA50", "Dist_EMA200"
]

# Base catalog meta with dynamic fallback for any online/custom strategy
STRATEGY_CATALOG_META = {
    "xgb_mtf_v2": {"name": "XGBoost 1H/4H Momentum Classifier", "timeframes": ["1h", "4h"]},
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
}


def classify_market(symbol: str) -> str:
    clean = symbol.strip().upper()
    forex_bases = {"EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
    parts = clean.replace("-", "/").replace("_", "/").split("/")
    if len(parts) >= 2 and parts[0] in forex_bases and parts[1] in (forex_bases | {"USD"}):
        return "FOREX"
    if len(clean) == 6 and clean[:3] in forex_bases and clean[3:] in (forex_bases | {"USD"}):
        return "FOREX"
    return "CRYPTO"


class PaperStrategyTrainerAndTester:
    def __init__(self, paper_dir: str, models_dir: str):
        self.paper_dir = paper_dir
        self.models_dir = models_dir
        self.paper_master_path = os.path.join(models_dir, "PAPER", "UNIFIED_PORTFOLIO_MASTER.json")
        self.live_master_path = os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json")
        self.legacy_master_path = os.path.join(models_dir, "UNIFIED_PORTFOLIO_MASTER.json")
        self.paper_results_path = os.path.join(paper_dir, "paper_trained_results.json")
        self.comparison_path = os.path.join(paper_dir, "paper_vs_live_comparison.json")

        os.makedirs(paper_dir, exist_ok=True)
        os.makedirs(os.path.join(models_dir, "PAPER"), exist_ok=True)
        os.makedirs(os.path.join(models_dir, "LIVE"), exist_ok=True)

    def fetch_bybit_history(self, symbol: str, timeframe: str = "1h", bars: int = 1000) -> pd.DataFrame:
        bybit_sym = symbol.replace("/", "").upper()
        interval = TIMEFRAME_TO_BYBIT_INTERVAL.get(timeframe.lower(), "60")
        try:
            resp = requests.get(
                "https://api.bybit.com/v5/market/kline",
                params={"category": "linear", "symbol": bybit_sym, "interval": interval, "limit": min(bars, 1000)},
                timeout=12,
            ).json()
            rows = resp.get("result", {}).get("list", [])
        except Exception:
            rows = []

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"])
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("timestamp").set_index("timestamp")

    def build_matrix(self, df: pd.DataFrame, max_dd: float = 0.95, expansion: float = 1.6, horizon: int = 8) -> pd.DataFrame:
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
        data["MACD_Hist"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

        data["EMA_20"] = c.ewm(span=20, adjust=False).mean()
        data["EMA_50"] = c.ewm(span=50, adjust=False).mean()
        data["EMA_200"] = c.ewm(span=200, adjust=False).mean()
        data["Dist_EMA20"] = (c - data["EMA_20"]) / (data["ATR"] + 1e-9)
        data["Dist_EMA50"] = (c - data["EMA_50"]) / (data["ATR"] + 1e-9)
        data["Dist_EMA200"] = (c - data["EMA_200"]) / (data["ATR"] + 1e-9)

        hl2 = (data["High"] + data["Low"]) / 2
        upper = hl2 + 3.0 * data["ATR"]
        lower = hl2 - 3.0 * data["ATR"]
        st_dir = pd.Series(1, index=data.index)
        uptrend = True
        for i in range(10, len(data)):
            if uptrend and c.iloc[i] < lower.iloc[i - 1]:
                uptrend = False
            elif not uptrend and c.iloc[i] > upper.iloc[i - 1]:
                uptrend = True
            st_dir.iloc[i] = 1 if uptrend else -1
        data["Supertrend"] = st_dir
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            path = tmp.name
        try:
            clf.save_model(path)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def _strategy_entry_filter(self, strategy_id: str, row: pd.Series, prev: pd.Series, side: str) -> bool:
        sid = strategy_id.lower()
        c = row["Close"]

        if "donchian" in sid:
            return (c > prev["Donchian_High"] and prev["Supertrend"] == 1) if side == "LONG" else (
                c < prev["Donchian_Low"] and prev["Supertrend"] == -1
            )
        if "vol_surge" in sid or "ob_imbalance" in sid or "supertrend_adx" in sid:
            return bool(prev["Vol_Surge"] >= 1.25 and prev["ADX"] >= 20)
        if "bb_" in sid or "squeeze" in sid:
            return bool(prev["BB_Width"] > prev["BB_Width_Med"] and (
                (prev["BB_PctB"] > 0.75) if side == "LONG" else (prev["BB_PctB"] < 0.25)
            ))
        if "stoch" in sid or "rsi" in sid:
            return bool((prev["RSI_14"] < 42 and prev["Stoch_K"] > prev["Stoch_D"]) if side == "LONG" else (
                prev["RSI_14"] > 58 and prev["Stoch_K"] < prev["Stoch_D"]
            ))
        if "ema_ribbon" in sid:
            return bool((prev["EMA_20"] > prev["EMA_50"] and c > prev["EMA_20"]) if side == "LONG" else (
                prev["EMA_20"] < prev["EMA_50"] and c < prev["EMA_20"]
            ))
        if "macd" in sid:
            return bool((prev["MACD_Hist"] > 0 and prev["Dist_EMA200"] > 0) if side == "LONG" else (
                prev["MACD_Hist"] < 0 and prev["Dist_EMA200"] < 0
            ))
        if "supertrend" in sid or "sar" in sid:
            return bool((prev["Supertrend"] == 1) if side == "LONG" else (prev["Supertrend"] == -1))
        # Fallback for online strategies or custom indicators added dynamically
        return bool((prev["DI_Spread"] > 0) if side == "LONG" else (prev["DI_Spread"] < 0))

    def _simulate_strategy_on_tf(
        self,
        test_df: pd.DataFrame,
        p_long: np.ndarray,
        p_short: np.ndarray,
        strategy_id: str,
        timeframe: str,
        leverage: int,
        initial_capital: float,
        htf_trend_series: pd.Series = None,
    ) -> dict:
        round_trip_fee = BYBIT_TAKER_FEE * 2.0
        cfg = {
            "long_thresh": 0.270,
            "short_thresh": 0.270,
            "sl_atr": 0.35,
            "tp1_rr": 2.0,
            "tp2_rr": 3.5,
            "trail_atr": 1.8,
            "strategy_id": strategy_id,
            "timeframe": timeframe,
        }

        trades_pnl = []
        equity = initial_capital
        in_pos = False
        side = None
        entry = sl = tp1 = tp2 = high_w = low_w = 0.0
        tp1_hit = False

        for i in range(1, len(test_df)):
            row = test_df.iloc[i]
            prev = test_df.iloc[i - 1]
            c, h, l, atr = row["Close"], row["High"], row["Low"], row["ATR"]

            if not in_pos:
                le = (p_long[i - 1] >= cfg["long_thresh"]) and self._strategy_entry_filter(strategy_id, row, prev, "LONG")
                se = (p_short[i - 1] >= cfg["short_thresh"]) and self._strategy_entry_filter(strategy_id, row, prev, "SHORT")

                if htf_trend_series is not None:
                    ts = test_df.index[i]
                    prior_htf = htf_trend_series[htf_trend_series.index <= ts]
                    if len(prior_htf) > 0:
                        htf_dir = prior_htf.iloc[-1]
                        le = le and (htf_dir == 1)
                        se = se and (htf_dir == -1)

                if le != se:
                    in_pos = True
                    side = "LONG" if le else "SHORT"
                    entry = row["Open"]
                    high_w = low_w = entry
                    tp1_hit = False
                    risk = cfg["sl_atr"] * atr
                    sl = entry - risk if side == "LONG" else entry + risk
                    tp1 = entry + risk * cfg["tp1_rr"] if side == "LONG" else entry - risk * cfg["tp1_rr"]
                    tp2 = entry + risk * cfg["tp2_rr"] if side == "LONG" else entry - risk * cfg["tp2_rr"]
            else:
                if side == "LONG":
                    high_w = max(high_w, h)
                    if h >= tp1 and not tp1_hit:
                        tp1_hit = True
                        sl = entry + 0.20 * (tp1 - entry)
                    if tp1_hit:
                        sl = max(sl, high_w - cfg["trail_atr"] * atr)
                    if l <= sl or h >= tp2:
                        ex = sl if l <= sl else tp2
                        raw_ret = (0.5 * ((tp1 - entry) / entry) + 0.5 * ((ex - entry) / entry)) if tp1_hit else ((ex - entry) / entry)
                        net_ret = (raw_ret - round_trip_fee) * leverage
                        pnl_usdt = (equity * 0.05) * net_ret
                        equity += pnl_usdt
                        trades_pnl.append(pnl_usdt)
                        in_pos = False
                else:
                    low_w = min(low_w, l)
                    if l <= tp1 and not tp1_hit:
                        tp1_hit = True
                        sl = entry - 0.20 * (entry - tp1)
                    if tp1_hit:
                        sl = min(sl, low_w + cfg["trail_atr"] * atr)
                    if h >= sl or l <= tp2:
                        ex = sl if h >= sl else tp2
                        raw_ret = (0.5 * ((entry - tp1) / entry) + 0.5 * ((entry - ex) / entry)) if tp1_hit else ((entry - ex) / entry)
                        net_ret = (raw_ret - round_trip_fee) * leverage
                        pnl_usdt = (equity * 0.05) * net_ret
                        equity += pnl_usdt
                        trades_pnl.append(pnl_usdt)
                        in_pos = False

        arr = np.array(trades_pnl)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        wr = round((len(wins) / len(arr) * 100.0), 2) if len(arr) > 0 else 0.0
        pf = round(float(wins.sum() / abs(losses.sum())), 2) if len(losses) > 0 else round(float(wins.sum()), 2)
        net_usdt = round(float(equity - initial_capital), 2)
        net_pct = round((net_usdt / initial_capital) * 100.0, 2)

        return {
            "cfg": cfg,
            "stats": {
                "trades": int(len(arr)),
                "win_rate_pct": wr,
                "profit_factor": pf,
                "leverage": leverage,
                "net_pnl_pct": net_pct,
                "net_pnl_usdt": net_usdt,
                "net_pnl": net_usdt,
            }
        }

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
        leverage = min(max(int(leverage), 1), MAX_LEVERAGE)
        test_tfs = [tf.lower() for tf in (timeframes or DEFAULT_TEST_TIMEFRAMES) if tf.lower() in TIMEFRAME_TO_BYBIT_INTERVAL]
        if not test_tfs:
            test_tfs = ["1h"]

        strategies_to_test = active_strategies if active_strategies else ["xgb_mtf_v2"]

        filtered_coins = []
        for sym in added_coins:
            mkt = classify_market(sym)
            if (mkt == "CRYPTO" and crypto_enabled) or (mkt == "FOREX" and forex_enabled):
                filtered_coins.append(sym)

        trained_coins = {}
        paper_portfolio = {}
        total_net_pnl = 0.0
        win_rates = []

        for symbol in filtered_coins:
            htf_trend = None
            raw_4h = self.fetch_bybit_history(symbol, timeframe="4h", bars=600)
            if len(raw_4h) >= 60:
                df_4h = self.build_matrix(raw_4h)
                if not df_4h.empty:
                    htf_trend = df_4h["Supertrend"]

            best_candidate = None
            all_tf_runs = []

            for tf in test_tfs:
                raw = raw_4h if (tf == "4h" and not raw_4h.empty) else self.fetch_bybit_history(symbol, timeframe=tf, bars=1000)
                if len(raw) < 350:
                    continue

                df = self.build_matrix(raw)
                if len(df) < 250:
                    continue

                split = int(len(df) * 0.70)
                train_df = df.iloc[:split]
                test_df = df.iloc[split:]

                clf_long = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
                clf_short = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
                clf_long.fit(train_df[FEATURES], train_df["Target_Long"])
                clf_short.fit(train_df[FEATURES], train_df["Target_Short"])

                p_long = clf_long.predict_proba(test_df[FEATURES])[:, 1]
                p_short = clf_short.predict_proba(test_df[FEATURES])[:, 1]

                for strat_id in strategies_to_test:
                    eval_res = self._simulate_strategy_on_tf(
                        test_df=test_df,
                        p_long=p_long,
                        p_short=p_short,
                        strategy_id=strat_id,
                        timeframe=tf.upper(),
                        leverage=leverage,
                        initial_capital=initial_capital,
                        htf_trend_series=htf_trend if tf != "4h" else None,
                    )
                    st = eval_res["stats"]
                    strat_meta = STRATEGY_CATALOG_META.get(strat_id, {"name": strat_id.replace("_", " ").title()})
                    strat_name = strat_meta.get("name", strat_id)

                    all_tf_runs.append({
                        "strategy_id": strat_id,
                        "strategy_name": strat_name,
                        "timeframe": tf.upper(),
                        "trades": st["trades"],
                        "win_rate_pct": st["win_rate_pct"],
                        "profit_factor": st["profit_factor"],
                        "net_pnl_usdt": st["net_pnl_usdt"],
                    })

                    score = (st["net_pnl_usdt"] if st["trades"] >= 2 else -9999.0) + (st["win_rate_pct"] * 0.5)
                    if best_candidate is None or score > best_candidate["score"]:
                        cfg = dict(eval_res["cfg"])
                        cfg["bybit_symbol"] = symbol.replace("/", "").upper()
                        cfg["strategy"] = f"{strat_name} ({tf.upper()})"
                        best_candidate = {
                            "score": score,
                            "strategy_id": strat_id,
                            "strategy_name": strat_name,
                            "timeframe": tf.upper(),
                            "stats": st,
                            "cfg": cfg,
                            "models": {
                                "long": self._serialize_booster(clf_long),
                                "short": self._serialize_booster(clf_short),
                            },
                        }

            if best_candidate is None:
                continue

            st = best_candidate["stats"]
            total_net_pnl += st["net_pnl_usdt"]
            win_rates.append(st["win_rate_pct"])

            coin_entry = {
                "market": classify_market(symbol),
                "strategy": f"{best_candidate['strategy_name']} ({best_candidate['timeframe']})",
                "strategy_id": best_candidate["strategy_id"],
                "timeframe": best_candidate["timeframe"],
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
            trained_coins[symbol] = coin_entry
            paper_portfolio[symbol] = coin_entry

        with open(self.paper_results_path, "w", encoding="utf-8") as f:
            json.dump({"leverage": leverage, "timeframes_tested": test_tfs, "coins": trained_coins}, f, indent=2)

        avg_accuracy = round(sum(win_rates) / len(win_rates), 2) if win_rates else 0.0
        paper_master = {
            "version": f"UNIFIED_PAPER_{len(paper_portfolio)}C",
            "exported_at": int(time.time()),
            "model_accuracy_pct": avg_accuracy,
            "total_net_pnl": round(total_net_pnl, 2),
            "features": FEATURES,
            "portfolio": paper_portfolio,
        }
        with open(self.paper_master_path, "w", encoding="utf-8") as f:
            json.dump(paper_master, f, indent=2)

        self.auto_compare_with_live(trained_coins)
        return trained_coins

    def auto_compare_with_live(self, paper_coins_dict: dict) -> list:
        live_portfolio = {}
        target_live = self.live_master_path if os.path.exists(self.live_master_path) else self.legacy_master_path
        if os.path.exists(target_live):
            try:
                with open(target_live, "r", encoding="utf-8") as f:
                    live_portfolio = json.load(f).get("portfolio", {})
            except Exception:
                live_portfolio = {}

        comparisons = []
        for sym, p_data in paper_coins_dict.items():
            p_stats = p_data.get("stats", {})
            p_wr = float(p_stats.get("win_rate_pct", 0.0))
            p_pf = float(p_stats.get("profit_factor", 0.0))
            p_pnl = float(p_stats.get("net_pnl_usdt", 0.0))

            l_data = live_portfolio.get(sym, {})
            l_stats = l_data.get("stats", {}) if isinstance(l_data, dict) else {}
            l_wr = float(l_data.get("win_rate_pct", l_stats.get("win_rate_pct", 0.0))) if isinstance(l_data, dict) else 0.0
            l_pf = float(l_data.get("profit_factor", l_stats.get("profit_factor", 0.0))) if isinstance(l_data, dict) else 0.0
            l_pnl = float(l_data.get("net_pnl_usdt", l_stats.get("net_pnl_usdt", 0.0))) if isinstance(l_data, dict) else 0.0

            is_new_coin = sym not in live_portfolio
            is_better = (p_wr > l_wr and p_pf >= l_pf and p_pnl > 0) or (is_new_coin and p_wr >= 54.0 and p_pnl > 0)

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

    def move_coin_to_live_model(self, symbol: str) -> bool:
        """
        Explicit user confirmation required: copies ONLY the promoted coin's config,
        accuracy, Net PnL, and XGBoost weights from Paper into Live UNIFIED_PORTFOLIO_MASTER.json
        while preserving all previous Live coins, configs, and strategies.
        """
        if not os.path.exists(self.paper_results_path):
            return False

        with open(self.paper_results_path, "r", encoding="utf-8") as f:
            paper_data = json.load(f)

        coin_bundle = paper_data.get("coins", {}).get(symbol)
        if not coin_bundle:
            return False

        target_live_path = self.live_master_path if os.path.exists(self.live_master_path) else self.legacy_master_path
        if os.path.exists(target_live_path):
            with open(target_live_path, "r", encoding="utf-8") as f:
                live_master = json.load(f)
        else:
            live_master = {"version": "2.0_LIVE_ONLY", "exported_at": int(time.time()), "features": FEATURES, "portfolio": {}}

        portfolio = live_master.setdefault("portfolio", {})
        stats = coin_bundle.get("stats", {})
        portfolio[symbol] = {
            "market": coin_bundle.get("market", classify_market(symbol)),
            "strategy": coin_bundle.get("strategy", "XGBoost 1H/4H Momentum"),
            "timeframe": coin_bundle.get("timeframe", "1H"),
            "win_rate_pct": stats.get("win_rate_pct", 0.0),
            "profit_factor": stats.get("profit_factor", 0.0),
            "net_pnl_pct": stats.get("net_pnl_pct", 0.0),
            "net_pnl_usdt": stats.get("net_pnl_usdt", 0.0),
            "net_pnl": stats.get("net_pnl_usdt", 0.0),
            "trades": stats.get("trades", 0),
            "stats": stats,
            "config": coin_bundle["config"],
            "models": coin_bundle["models"],
        }

        all_wrs = [v.get("win_rate_pct", 0.0) for v in portfolio.values() if isinstance(v, dict)]
        all_pnls = [v.get("net_pnl_usdt", v.get("net_pnl", 0.0)) for v in portfolio.values() if isinstance(v, dict)]
        live_master["model_accuracy_pct"] = round(sum(all_wrs) / len(all_wrs), 2) if all_wrs else 0.0
        live_master["total_net_pnl"] = round(sum(all_pnls), 2)
        live_master["exported_at"] = int(time.time())

        with open(self.live_master_path, "w", encoding="utf-8") as f:
            json.dump(live_master, f, indent=2)
        with open(self.legacy_master_path, "w", encoding="utf-8") as f:
            json.dump(live_master, f, indent=2)
        return True


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.environ.get("PAPER_DATA_DIR", os.path.join(base_dir, "bot_data", "PAPER"))
    models_dir = os.environ.get("MODELS_DIR", os.path.join(base_dir, "ml_models"))

    # Dynamically read user coins and online strategies from app runtime config if available
    config_file = os.path.join(paper_dir, "paper_runtime_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                runtime_cfg = json.load(f)
        except Exception:
            runtime_cfg = {}
    else:
        runtime_cfg = {}

    coins = runtime_cfg.get("coins", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LINK/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT"])
    strategies = runtime_cfg.get("strategies", ["xgb_mtf_v2", "xrp_donchian_breakout", "cvd_orderflow_div", "funding_arb_mom"])
    timeframes = runtime_cfg.get("timeframes", DEFAULT_TEST_TIMEFRAMES)
    leverage = int(runtime_cfg.get("leverage", DEFAULT_LEVERAGE))
    crypto_enabled = bool(runtime_cfg.get("crypto_enabled", True))
    forex_enabled = bool(runtime_cfg.get("forex_enabled", False))

    trainer = PaperStrategyTrainerAndTester(paper_dir=paper_dir, models_dir=models_dir)
    
    # Promotion happens ONLY when explicitly invoked with --promote <SYMBOL>
    if len(sys.argv) >= 3 and sys.argv[1] == "--promote":
        ok = trainer.move_coin_to_live_model(sys.argv[2].upper())
        print(json.dumps({"ok": ok, "promoted": sys.argv[2].upper()}))
    else:
        results = trainer.train_and_evaluate_paper_coins(
            added_coins=coins,
            active_strategies=strategies,
            timeframes=timeframes,
            leverage=leverage,
            crypto_enabled=crypto_enabled,
            forex_enabled=forex_enabled,
        )
        print(f"Trained & evaluated {len(results)} Paper coins successfully across dynamic strategies and timeframes.")