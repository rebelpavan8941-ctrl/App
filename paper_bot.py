import os
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

FEATURES = [
    "NATR", "BB_PctB", "BB_Width",
    "Ret_1h", "Ret_3h", "Ret_6h", "Ret_12h", "Ret_24h",
    "RSI_14", "Stoch_K", "Stoch_D", "ADX", "DI_Spread", "MACD_Hist",
    "Dist_EMA20", "Dist_EMA50", "Dist_EMA200"
]


class PaperStrategyTrainerAndTester:
    def __init__(self, paper_dir: str, models_dir: str):
        self.paper_dir = paper_dir
        self.models_dir = models_dir
        self.live_master_path = os.path.join(models_dir, "UNIFIED_PORTFOLIO_MASTER.json")
        self.paper_results_path = os.path.join(paper_dir, "paper_trained_results.json")
        os.makedirs(paper_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

    def fetch_bybit_history(self, symbol: str, bars: int = 1000) -> pd.DataFrame:
        bybit_sym = symbol.replace("/", "").upper()
        resp = requests.get(
            "https://api.bybit.com/v5/market/kline",
            params={"category": "linear", "symbol": bybit_sym, "interval": "60", "limit": bars},
            timeout=12,
        ).json()
        rows = resp.get("result", {}).get("list", [])
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

        data["Dist_EMA20"] = (c - c.ewm(span=20, adjust=False).mean()) / (data["ATR"] + 1e-9)
        data["Dist_EMA50"] = (c - c.ewm(span=50, adjust=False).mean()) / (data["ATR"] + 1e-9)
        data["Dist_EMA200"] = (c - c.ewm(span=200, adjust=False).mean()) / (data["ATR"] + 1e-9)

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
        data["Vol_Surge"] = data["Volume"] / (data["Volume"].rolling(20).mean() + 1e-9)

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

    def train_and_evaluate_paper_coins(
        self,
        added_coins: list,
        active_strategies: list,
        leverage: int = DEFAULT_LEVERAGE,
        initial_capital: float = 10000.0
    ) -> dict:
        """
        Trains & evaluates user-added Paper coins using the selected strategies,
        Bybit VIP 0 taker fees (0.11% round-trip), and leverage capped at 20x (default 10x).
        """
        leverage = min(max(int(leverage), 1), MAX_LEVERAGE)
        round_trip_fee = BYBIT_TAKER_FEE * 2.0

        use_vol_filter = any("ob_imbalance" in s or "supertrend_adx" in s for s in active_strategies)
        use_donchian = any("donchian" in s for s in active_strategies)

        trained_coins = {}

        for symbol in added_coins:
            raw = self.fetch_bybit_history(symbol, bars=1000)
            if len(raw) < 350:
                continue

            df = self.build_matrix(raw)
            split = int(len(df) * 0.70)
            train_df = df.iloc[:split]
            test_df = df.iloc[split:]

            clf_long = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
            clf_short = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
            clf_long.fit(train_df[FEATURES], train_df["Target_Long"])
            clf_short.fit(train_df[FEATURES], train_df["Target_Short"])

            p_long = clf_long.predict_proba(test_df[FEATURES])[:, 1]
            p_short = clf_short.predict_proba(test_df[FEATURES])[:, 1]

            # Simulate out-of-sample test window with Bybit fees & leverage
            cfg = {
                "bybit_symbol": symbol.replace("/", "").upper(),
                "long_thresh": 0.275,
                "short_thresh": 0.285,
                "sl_atr": 0.40,
                "tp1_rr": 1.5,
                "tp2_rr": 2.6,
                "trail_atr": 1.6,
                "vol_filter": use_vol_filter,
                "is_xrp_momentum": use_donchian,
            }

            trades_pnl = []
            in_pos = False
            side = None
            entry = sl = tp1 = tp2 = high_w = low_w = 0.0
            tp1_hit = False

            for i in range(1, len(test_df)):
                row = test_df.iloc[i]
                prev = test_df.iloc[i - 1]
                c, h, l, atr = row["Close"], row["High"], row["Low"], row["ATR"]

                if not in_pos:
                    le = p_long[i - 1] >= cfg["long_thresh"]
                    se = p_short[i - 1] >= cfg["short_thresh"]
                    if use_vol_filter:
                        v_ok = prev["Vol_Surge"] >= 1.25 and prev["ADX"] >= 20
                        le = le and v_ok
                        se = se and v_ok
                    if use_donchian:
                        le = le and (c > prev["Donchian_High"]) and (prev["Supertrend"] == 1)
                        se = se and (c < prev["Donchian_Low"]) and (prev["Supertrend"] == -1)

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
                            sl = entry + 0.15 * (tp1 - entry)
                        if tp1_hit:
                            sl = max(sl, high_w - cfg["trail_atr"] * atr)
                        if l <= sl or h >= tp2:
                            ex = sl if l <= sl else tp2
                            raw_ret = (0.5 * ((tp1 - entry) / entry) + 0.5 * ((ex - entry) / entry)) if tp1_hit else ((ex - entry) / entry)
                            net_ret = (raw_ret - round_trip_fee) * leverage
                            trades_pnl.append(net_ret)
                            in_pos = False
                    else:
                        low_w = min(low_w, l)
                        if l <= tp1 and not tp1_hit:
                            tp1_hit = True
                            sl = entry - 0.15 * (entry - tp1)
                        if tp1_hit:
                            sl = min(sl, low_w + cfg["trail_atr"] * atr)
                        if h >= sl or l <= tp2:
                            ex = sl if h >= sl else tp2
                            raw_ret = (0.5 * ((entry - tp1) / entry) + 0.5 * ((entry - ex) / entry)) if tp1_hit else ((entry - ex) / entry)
                            net_ret = (raw_ret - round_trip_fee) * leverage
                            trades_pnl.append(net_ret)
                            in_pos = False

            arr = np.array(trades_pnl)
            wins = arr[arr > 0]
            losses = arr[arr < 0]
            wr = round((len(wins) / len(arr) * 100.0), 2) if len(arr) > 0 else 0.0
            pf = round(float(wins.sum() / abs(losses.sum())), 2) if len(losses) > 0 else round(float(wins.sum()), 2)
            net_pct = round(float(arr.sum() * 100.0), 2)
            net_usdt = round(initial_capital * 0.10 * float(arr.sum()), 2)

            trained_coins[symbol] = {
                "stats": {
                    "trades": int(len(arr)),
                    "win_rate_pct": wr,
                    "profit_factor": pf,
                    "leverage": leverage,
                    "net_pnl_pct": net_pct,
                    "net_pnl_usdt": net_usdt
                },
                "config": cfg,
                "models": {
                    "long": self._serialize_booster(clf_long),
                    "short": self._serialize_booster(clf_short)
                }
            }

        with open(self.paper_results_path, "w", encoding="utf-8") as f:
            json.dump({"leverage": leverage, "coins": trained_coins}, f, indent=2)

        return trained_coins

    def move_coin_to_live_model(self, symbol: str) -> bool:
        """
        One-click promotion: copies the trained coin's config & XGBoost weights from Paper
        into the Live UNIFIED_PORTFOLIO_MASTER.json so live_bot.py starts trading it immediately.
        """
        if not os.path.exists(self.paper_results_path):
            return False

        with open(self.paper_results_path, "r", encoding="utf-8") as f:
            paper_data = json.load(f)

        coin_bundle = paper_data.get("coins", {}).get(symbol)
        if not coin_bundle:
            return False

        if os.path.exists(self.live_master_path):
            with open(self.live_master_path, "r", encoding="utf-8") as f:
                live_master = json.load(f)
        else:
            live_master = {"version": "2.0_LIVE_ONLY", "exported_at": int(time.time()), "features": FEATURES, "portfolio": {}}

        live_master["portfolio"][symbol] = {
            "config": coin_bundle["config"],
            "models": coin_bundle["models"]
        }
        live_master["exported_at"] = int(time.time())

        with open(self.live_master_path, "w", encoding="utf-8") as f:
            json.dump(live_master, f, indent=2)
        return True