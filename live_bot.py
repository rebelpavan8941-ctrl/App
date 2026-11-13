import os
import json
import time
import hmac
import hashlib
import tempfile
import urllib.parse
import requests
import numpy as np
import pandas as pd
import xgboost as xgb


class LiveAccountTradingBot:
    def __init__(self, api_key: str, api_secret: str, models_dir: str, endpoint: str = "Mainnet"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.models_dir = models_dir
        self.master_path = os.path.join(models_dir, "UNIFIED_PORTFOLIO_MASTER.json")
        self.base_url = "https://api-testnet.bybit.com" if endpoint.lower() == "testnet" else "https://api.bybit.com"
        self.category = "linear"
        self.active_positions = {}

    def _sign(self, ts: str, recv_window: str, payload: str) -> str:
        raw = f"{ts}{self.api_key}{recv_window}{payload}"
        return hmac.new(self.api_secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()

    def _bybit_request(self, method: str, path: str, params: dict = None, body: dict = None) -> dict:
        ts = str(int(time.time() * 1000))
        recv_window = "5000"
        if method == "GET":
            qs = urllib.parse.urlencode(params or {})
            sig = self._sign(ts, recv_window, qs)
            url = f"{self.base_url}{path}?{qs}" if qs else f"{self.base_url}{path}"
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-SIGN": sig,
                "X-BAPI-TIMESTAMP": ts,
                "X-BAPI-RECV-WINDOW": recv_window,
            }
            return requests.get(url, headers=headers, timeout=10).json()
        else:
            raw_body = json.dumps(body or {})
            sig = self._sign(ts, recv_window, raw_body)
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-SIGN": sig,
                "X-BAPI-TIMESTAMP": ts,
                "X-BAPI-RECV-WINDOW": recv_window,
                "Content-Type": "application/json",
            }
            return requests.post(f"{self.base_url}{path}", headers=headers, data=raw_body, timeout=10).json()

    def fetch_bybit_1h_candles(self, bybit_symbol: str, limit: int = 250) -> pd.DataFrame:
        resp = requests.get(
            f"{self.base_url}/v5/market/kline",
            params={"category": self.category, "symbol": bybit_symbol, "interval": "60", "limit": limit},
            timeout=10,
        ).json()
        rows = resp.get("result", {}).get("list", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"])
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("timestamp").set_index("timestamp")

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
        return data.dropna()

    def _load_booster(self, model_dict: dict) -> xgb.Booster:
        booster = xgb.Booster()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(model_dict, tmp)
            tmp_path = tmp.name
        try:
            booster.load_model(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return booster

    def run_live_cycle(self, leverage: int = 10, risk_per_trade_pct: float = 1.0, max_open_positions: int = 2):
        """
        Dynamically reads whatever coins are present in UNIFIED_PORTFOLIO_MASTER.json.
        No coins are hardcoded; moving any coin from Paper to Live in the app adds it here automatically.
        """
        if not os.path.exists(self.master_path):
            raise FileNotFoundError(f"Live Master ML Model not found at {self.master_path}")

        leverage = min(max(int(leverage), 1), 20)  # Enforce max 20x leverage (default 10x)

        with open(self.master_path, "r", encoding="utf-8") as f:
            master = json.load(f)

        features = master["features"]
        portfolio = master.get("portfolio", {})
        if not portfolio:
            return

        bal_res = self._bybit_request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": "USDT"})
        avail_usdt = float(bal_res["result"]["list"][0]["totalAvailableBalance"])

        for symbol, entry in portfolio.items():
            cfg = entry["config"]
            bybit_sym = cfg.get("bybit_symbol") or symbol.replace("/", "")
            df_raw = self.fetch_bybit_1h_candles(bybit_sym)
            if len(df_raw) < 210:
                continue

            df = self.build_features(df_raw)
            prev = df.iloc[-2]
            curr = df.iloc[-1]

            dmat = xgb.DMatrix(df.iloc[[-2]][features])
            p_long = float(self._load_booster(entry["models"]["long"]).predict(dmat)[0])
            p_short = float(self._load_booster(entry["models"]["short"]).predict(dmat)[0])

            if cfg.get("is_xrp_momentum", False):
                long_edge = (curr["Close"] > prev["Donchian_High"]) and (prev["Supertrend"] == 1) and (p_long >= cfg["long_thresh"])
                short_edge = (curr["Close"] < prev["Donchian_Low"]) and (prev["Supertrend"] == -1) and (p_short >= cfg["short_thresh"])
            else:
                long_edge = p_long >= cfg["long_thresh"]
                short_edge = p_short >= cfg["short_thresh"]

            if cfg.get("vol_filter", False):
                vol_ok = prev["Vol_Surge"] >= 1.25 and prev["ADX"] >= 20
                long_edge = long_edge and vol_ok
                short_edge = short_edge and vol_ok

            if bybit_sym not in self.active_positions and len(self.active_positions) < max_open_positions:
                price = float(curr["Close"])
                atr = float(prev["ATR"])
                risk_dist = float(cfg["sl_atr"]) * atr

                if long_edge and not short_edge:
                    sl = round(price - risk_dist, 4)
                    tp2 = round(price + (risk_dist * float(cfg["tp2_rr"])), 4)
                    notional = avail_usdt * (risk_per_trade_pct / 100.0) * leverage
                    qty = str(round(notional / price, 2))
                    self._bybit_request("POST", "/v5/position/set-leverage", body={
                        "category": self.category, "symbol": bybit_sym,
                        "buyLeverage": str(leverage), "sellLeverage": str(leverage)
                    })
                    self._bybit_request("POST", "/v5/order/create", body={
                        "category": self.category, "symbol": bybit_sym, "side": "Buy",
                        "orderType": "Market", "qty": qty, "stopLoss": str(sl),
                        "takeProfit": str(tp2), "positionIdx": 0
                    })
                    self.active_positions[bybit_sym] = {"side": "LONG", "entry": price, "sl": sl, "tp2": tp2}

                elif short_edge and not long_edge:
                    sl = round(price + risk_dist, 4)
                    tp2 = round(price - (risk_dist * float(cfg["tp2_rr"])), 4)
                    notional = avail_usdt * (risk_per_trade_pct / 100.0) * leverage
                    qty = str(round(notional / price, 2))
                    self._bybit_request("POST", "/v5/position/set-leverage", body={
                        "category": self.category, "symbol": bybit_sym,
                        "buyLeverage": str(leverage), "sellLeverage": str(leverage)
                    })
                    self._bybit_request("POST", "/v5/order/create", body={
                        "category": self.category, "symbol": bybit_sym, "side": "Sell",
                        "orderType": "Market", "qty": qty, "stopLoss": str(sl),
                        "takeProfit": str(tp2), "positionIdx": 0
                    })
                    self.active_positions[bybit_sym] = {"side": "SHORT", "entry": price, "sl": sl, "tp2": tp2}