import os
import gc
import sys
import json
import time
import math
import hmac
import hashlib
import urllib.parse
import datetime
import requests
import numpy as np
import pandas as pd
import xgboost as xgb

TIMEFRAME_TO_BYBIT_INTERVAL = {
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
}

FEATURES = [
    "NATR", "BB_PctB", "BB_Width",
    "Ret_1h", "Ret_3h", "Ret_6h", "Ret_12h", "Ret_24h",
    "RSI_14", "Stoch_K", "Stoch_D", "ADX", "DI_Spread", "MACD_Hist",
    "Dist_EMA20", "Dist_EMA50", "Dist_EMA200",
    "VWAP_Dev"
]

BYBIT_ERROR_HINTS = {
    10001: "Invalid order parameter / quantity step or price precision",
    10003: "API key invalid or Mainnet/Demo endpoint mismatch",
    10004: "API signature mismatch — check API Secret",
    10006: "Bybit API rate limit exceeded",
    110001: "Order does not exist or already filled",
    110003: "Order price exceeds allowable range",
    110004: "Wallet balance is insufficient for margin",
    110007: "Available USDT balance is insufficient to open position",
    110012: "Insufficient available balance",
    110013: "Cannot set leverage due to risk limit tier — lower leverage",
    110016: "Order quantity exceeds max risk limit tier",
    110043: "Leverage unchanged or position margin mode conflict",
    110044: "Available margin is insufficient",
    110132: "Contract not available in your region",
}


def is_forex_symbol(symbol: str) -> bool:
    clean = symbol.replace("/", "").upper()
    forex_bases = {"EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
    return len(clean) == 6 and clean[:3] in forex_bases and clean[3:] in (forex_bases | {"USD"})


def is_valid_xgb_model(model_dict) -> bool:
    return isinstance(model_dict, dict) and "learner" in model_dict and isinstance(model_dict.get("learner"), dict)


class LiveAccountTradingBot:
    def __init__(self, api_key: str, api_secret: str, models_dir: str, live_dir: str = None, endpoint: str = "Mainnet"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.models_dir = models_dir
        self.live_dir = live_dir or os.path.join(os.path.dirname(models_dir), "bot_data", "LIVE_ACCOUNT")
        os.makedirs(self.live_dir, exist_ok=True)
        self.state_path = os.path.join(self.live_dir, "live_engine_state.json")

        p1 = os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json")
        p2 = os.path.join(models_dir, "UNIFIED_PORTFOLIO_MASTER.json")
        self.master_path = p1 if os.path.exists(p1) else p2

        self.is_demo = str(endpoint).lower() in ("testnet", "demo")
        self.host_candidates = (
            ["https://api-demo.bybit.com", "https://api.bytick.com", "https://api.bybit.com"]
            if self.is_demo
            else ["https://api.bytick.com", "https://api.bybit.com"]
        )
        self.base_url = self.host_candidates[0]
        self.category = "linear"
        self.rate_limit_remaining = 120
        self.rate_limit_cap = 120
        self._booster_cache = {}
        self._instrument_cache = {}
        self._trend_4h_cache = {}

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=6, pool_maxsize=6)
        self.session.mount("https://", adapter)

    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8-sig") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "traded_candle_ts": {},
            "last_trade_epoch_ms": {},
            "daily_coin_trades": {},
            "daily_total_trades": 0,
            "tp1_partial_done": {},
            "last_diagnostics": [],
            "last_order_rejections": [],
            "last_trade_blocks": [],
            "last_skip_notice_key": "",
            "last_skip_notice_ms": 0,
            "coins_scanned": 0,
        }

    def _save_state(self, state: dict):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def _sign(self, ts: str, recv_window: str, payload: str) -> str:
        raw = f"{ts}{self.api_key}{recv_window}{payload}"
        return hmac.new(self.api_secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()

    def _update_rate_limits(self, resp: requests.Response):
        rem = resp.headers.get("X-Bapi-Limit-Status")
        cap = resp.headers.get("X-Bapi-Limit")
        if rem and rem.isdigit():
            self.rate_limit_remaining = int(rem)
        if cap and cap.isdigit() and int(cap) > 0:
            self.rate_limit_cap = int(cap)

    def _public_get(self, path: str, params: dict, timeout: float = 5.0) -> dict:
        for base in self.host_candidates:
            try:
                resp = self.session.get(f"{base}{path}", params=params, timeout=timeout)
                self._update_rate_limits(resp)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                continue
        return {}

    def _bybit_request(self, method: str, path: str, params: dict = None, body: dict = None) -> dict:
        if self.rate_limit_cap > 0 and self.rate_limit_remaining < int(self.rate_limit_cap * 0.10):
            time.sleep(1.0)

        last_err = None
        for base in self.host_candidates:
            ts = str(int(time.time() * 1000))
            recv_window = "5000"
            try:
                if method == "GET":
                    qs = urllib.parse.urlencode(params or {})
                    sig = self._sign(ts, recv_window, qs)
                    url = f"{base}{path}?{qs}" if qs else f"{base}{path}"
                    headers = {
                        "X-BAPI-API-KEY": self.api_key,
                        "X-BAPI-SIGN": sig,
                        "X-BAPI-TIMESTAMP": ts,
                        "X-BAPI-RECV-WINDOW": recv_window,
                    }
                    resp = self.session.get(url, headers=headers, timeout=6)
                    self._update_rate_limits(resp)
                    if resp.status_code == 200:
                        return resp.json()
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
                    resp = self.session.post(f"{base}{path}", headers=headers, data=raw_body, timeout=6)
                    self._update_rate_limits(resp)
                    if resp.status_code == 200:
                        return resp.json()
            except Exception as e:
                last_err = e
                continue
        if last_err:
            raise last_err
        return {}

    def _get_instrument_filters(self, bybit_symbol: str):
        if bybit_symbol in self._instrument_cache:
            return self._instrument_cache[bybit_symbol]
        data = self._public_get("/v5/market/instruments-info", {"category": self.category, "symbol": bybit_symbol})
        items = data.get("result", {}).get("list", [])
        if not items:
            return None
        lot = items[0].get("lotSizeFilter", {})
        pf = items[0].get("priceFilter", {})
        res = (
            lot.get("qtyStep", "0.001"),
            float(lot.get("minOrderQty", "0.001")),
            float(lot.get("minNotionalValue", "5.0")),
            pf.get("tickSize", "0.0001"),
        )
        self._instrument_cache[bybit_symbol] = res
        return res

    @staticmethod
    def _format_step(val: float, step_str: str, round_up: bool = False) -> str:
        step = float(step_str)
        scaled = math.ceil(val / step) * step if round_up else round(val / step) * step
        clean_step = step_str.rstrip("0")
        decimals = len(clean_step.split(".")[1]) if "." in clean_step else 0
        return f"{scaled:.{decimals}f}"

    @staticmethod
    def _regime_exit_multipliers(adx_val: float, natr_val: float, btc_bearish: bool) -> tuple:
        if natr_val >= 2.8 or (adx_val < 14.0 and btc_bearish):
            return 1.15, 0.85, 0.35
        if not btc_bearish and adx_val >= 18.0 and natr_val < 2.6:
            return 1.05, 1.20, 0.55
        return 1.00, 1.00, 0.45

    @staticmethod
    def _conviction_risk_scale(raw_prob: float, req_thresh: float) -> float:
        edge = max(0.0, raw_prob - req_thresh)
        scale = 0.85 + min(edge * 2.2, 0.35)
        return round(min(max(scale, 0.85), 1.20), 4)

    def fetch_bybit_candles(self, bybit_symbol: str, interval: str = "60", limit: int = 230) -> pd.DataFrame:
        data = self._public_get(
            "/v5/market/kline",
            {"category": self.category, "symbol": bybit_symbol, "interval": interval, "limit": limit},
            timeout=5.0,
        )
        rows = data.get("result", {}).get("list", [])
        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"])
            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            return df.sort_values("timestamp").set_index("timestamp")

        if is_forex_symbol(bybit_symbol):
            yf_interval = "15m" if interval == "15" else ("30m" if interval == "30" else "1h")
            yf_range = "15d" if yf_interval in ("15m", "30m") else "60d"
            try:
                y_resp = self.session.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{bybit_symbol}=X",
                    params={"interval": yf_interval, "range": yf_range},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=6,
                )
                res0 = y_resp.json().get("chart", {}).get("result", [])[0]
                ts_list = res0.get("timestamp", [])
                quote = res0.get("indicators", {}).get("quote", [{}])[0]
                f_rows = []
                for i, ts_sec in enumerate(ts_list):
                    c = quote["close"][i]
                    if c is None or c <= 0:
                        continue
                    o = quote["open"][i] or c
                    h = quote["high"][i] or c
                    l = quote["low"][i] or c
                    f_rows.append([ts_sec * 1000, float(o), float(h), float(l), float(c), 1000.0])
                if f_rows:
                    df = pd.DataFrame(f_rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df = df.set_index("timestamp").sort_index()
                    bar_range = (df["High"] - df["Low"]).clip(lower=1e-6)
                    avg_range = bar_range.rolling(20, min_periods=1).mean().clip(lower=1e-6)
                    df["Volume"] = ((bar_range / avg_range) * 1000.0).clip(lower=200.0, upper=5000.0)
                    if interval == "240":
                        df = df.resample("4h").agg({
                            "Open": "first",
                            "High": "max",
                            "Low": "min",
                            "Close": "last",
                            "Volume": "sum",
                        }).dropna()
                    return df.tail(limit)
            except Exception:
                pass
        return pd.DataFrame()

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
        data["MACD_Hist"] = (macd_line - macd_line.ewm(span=9, adjust=False).mean()) / (data["ATR"] + 1e-9)

        data["EMA_20"] = c.ewm(span=20, adjust=False).mean()
        data["EMA_50"] = c.ewm(span=50, adjust=False).mean()
        data["EMA_200"] = c.ewm(span=200, adjust=False).mean()
        data["Dist_EMA20"] = (c - data["EMA_20"]) / (data["ATR"] + 1e-9)
        data["Dist_EMA50"] = (c - data["EMA_50"]) / (data["ATR"] + 1e-9)
        data["Dist_EMA200"] = (c - data["EMA_200"]) / (data["ATR"] + 1e-9)

        typical_px = (data["High"] + data["Low"] + c) / 3.0
        vol_safe = data["Volume"].clip(lower=1.0)
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
        data["Donchian_High"] = data["High"].rolling(20).max().shift(1)
        data["Donchian_Low"] = data["Low"].rolling(20).min().shift(1)
        data["Vol_Surge"] = data["Volume"] / (data["Volume"].rolling(20).mean() + 1e-9)
        data["BB_Width_Med"] = data["BB_Width"].rolling(30).median()
        return data.dropna()

    def _get_cached_4h_metrics(self, bybit_sym: str) -> tuple:
        now_s = time.time()
        cached = self._trend_4h_cache.get(bybit_sym)
        if cached and (now_s - cached[0]) < 600.0:
            return cached[1], cached[2], cached[3]
        raw_4h = self.fetch_bybit_candles(bybit_sym, interval="240", limit=220)
        if len(raw_4h) >= 210:
            df_4h = self.build_features(raw_4h)
            if not df_4h.empty:
                st = int(df_4h.iloc[-1]["Supertrend"])
                adx = float(df_4h.iloc[-1].get("ADX", 20.0))
                natr = float(df_4h.iloc[-1].get("NATR", 1.8))
                self._trend_4h_cache[bybit_sym] = (now_s, st, adx, natr)
                return st, adx, natr
        return 0, 20.0, 1.8

    def _eval_custom_rules(self, prev: pd.Series, rules: list) -> bool:
        if not rules:
            return False
        for rule in rules:
            ind = str(rule.get("ind", ""))
            op = str(rule.get("op", ""))
            val = float(rule.get("val", 0.0))
            if ind not in prev.index:
                return False
            actual = float(prev[ind])
            if op == ">" and not (actual > val):
                return False
            elif op == "<" and not (actual < val):
                return False
            elif op == ">=" and not (actual >= val):
                return False
            elif op == "<=" and not (actual <= val):
                return False
        return True

    def _strategy_entry_filter(
        self,
        strategy_id: str,
        curr: pd.Series,
        prev: pd.Series,
        side: str,
        relaxed: bool = False,
        custom_rules: dict = None,
    ) -> bool:
        if isinstance(custom_rules, dict):
            rule_key = "long_rules" if side == "LONG" else "short_rules"
            if custom_rules.get(rule_key):
                return self._eval_custom_rules(prev, custom_rules[rule_key])

        sid = strategy_id.lower()
        if relaxed or "adaptive_quantile" in sid or "high_sensitivity" in sid:
            return bool((prev["Dist_EMA50"] > -1.5) if side == "LONG" else (prev["Dist_EMA50"] < 1.5))

        c = curr["Close"]
        if "donchian" in sid:
            return (c > prev["Donchian_High"] and prev["Supertrend"] == 1) if side == "LONG" else (
                c < prev["Donchian_Low"] and prev["Supertrend"] == -1
            )
        if "vol_surge" in sid or "ob_imbalance" in sid or "supertrend_adx" in sid:
            return bool(prev["Vol_Surge"] >= 1.25 and prev["ADX"] >= 20 and (
                (prev["DI_Spread"] > 0) if side == "LONG" else (prev["DI_Spread"] < 0)
            ))
        if "bb_" in sid or "squeeze" in sid:
            return bool(prev["BB_Width"] > prev["BB_Width_Med"] and (
                (prev["BB_PctB"] > 0.72) if side == "LONG" else (prev["BB_PctB"] < 0.28)
            ))
        if "stoch" in sid or "rsi" in sid:
            return bool((prev["RSI_14"] < 44 and prev["Stoch_K"] > prev["Stoch_D"]) if side == "LONG" else (
                prev["RSI_14"] > 56 and prev["Stoch_K"] < prev["Stoch_D"]
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
        if "vwap" in sid or "poc" in sid or "fvg" in sid or "liq_sweep" in sid or "orb" in sid:
            return bool((prev["VWAP_Dev"] < -0.45 and prev["Vol_Surge"] >= 1.15) if side == "LONG" else (
                prev["VWAP_Dev"] > 0.45 and prev["Vol_Surge"] >= 1.15
            ))
        return bool((prev["DI_Spread"] > 0 and prev["Vol_Surge"] >= 1.05) if side == "LONG" else (
            prev["DI_Spread"] < 0 and prev["Vol_Surge"] >= 1.05
        ))

    def _load_booster(self, cache_key: str, model_dict: dict) -> xgb.Booster:
        if not is_valid_xgb_model(model_dict):
            raise ValueError(f"Invalid XGBoost booster format for {cache_key} (missing 'learner'). Retrain Paper model.")
        if cache_key in self._booster_cache:
            return self._booster_cache[cache_key]
        if len(self._booster_cache) >= 24:
            oldest_key = next(iter(self._booster_cache))
            self._booster_cache.pop(oldest_key, None)

        booster = xgb.Booster()
        os.makedirs(self.live_dir, exist_ok=True)
        tmp_path = os.path.join(self.live_dir, f"_tmp_live_{cache_key}_{os.getpid()}.json")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(model_dict, f)
            booster.load_model(tmp_path)
            self._booster_cache[cache_key] = booster
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        return booster

    def sync_positions_and_trail_stops(self, state: dict, trail_ratio: float = 0.45) -> list:
        res = self._bybit_request("GET", "/v5/position/list", {"category": self.category, "settleCoin": "USDT"})
        tp1_done_map = state.setdefault("tp1_partial_done", {})
        active_list = []

        for item in res.get("result", {}).get("list", []):
            size = float(item.get("size", 0.0))
            if size <= 0.0:
                continue

            bybit_sym = item.get("symbol", "").upper()
            side = "LONG" if item.get("side", "").lower() == "buy" else "SHORT"
            entry = float(item.get("avgPrice", 0.0))
            mark = float(item.get("markPrice", entry))
            cur_sl = float(item.get("stopLoss", 0.0) or 0.0)
            cur_tp = float(item.get("takeProfit", 0.0) or 0.0)

            if entry > 0.0 and mark > 0.0 and cur_tp > 0.0:
                tp_dist = abs(cur_tp - entry)
                profit_dist = (mark - entry) if side == "LONG" else (entry - mark)
                if tp_dist > 0.0 and profit_dist >= tp_dist * 0.50:
                    filters = self._get_instrument_filters(bybit_sym)
                    if filters:
                        qty_step, min_qty, min_notional, tick_size = filters
                        pos_key = f"{bybit_sym}_{entry:.6f}"
                        if not tp1_done_map.get(pos_key, False):
                            half_qty = size * 0.5
                            if half_qty >= min_qty and (half_qty * mark) >= min_notional:
                                close_side = "Sell" if side == "LONG" else "Buy"
                                p_res = self._bybit_request("POST", "/v5/order/create", body={
                                    "category": self.category,
                                    "symbol": bybit_sym,
                                    "side": close_side,
                                    "orderType": "Market",
                                    "qty": self._format_step(half_qty, qty_step),
                                    "timeInForce": "IOC",
                                    "reduceOnly": True,
                                    "positionIdx": 0,
                                })
                                if p_res.get("retCode") == 0:
                                    tp1_done_map[pos_key] = True

                        trail_offset = tp_dist * trail_ratio
                        cand_sl = max(entry + tp_dist * 0.15, mark - trail_offset) if side == "LONG" else min(
                            entry - tp_dist * 0.15, mark + trail_offset
                        )
                        should_tighten = (cur_sl == 0.0 or cand_sl > cur_sl * 1.0005) if side == "LONG" else (
                            cur_sl == 0.0 or cand_sl < cur_sl * 0.9995
                        )
                        if should_tighten:
                            new_sl_str = self._format_step(cand_sl, tick_size)
                            self._bybit_request("POST", "/v5/position/trading-stop", body={
                                "category": self.category,
                                "symbol": bybit_sym,
                                "stopLoss": new_sl_str,
                                "positionIdx": 0,
                            })

            active_list.append({
                "symbol": bybit_sym,
                "side": side,
                "qty": item.get("size", "0"),
                "entry_price": entry,
                "mark_price": mark,
                "unrealized_pnl": float(item.get("unrealisedPnl", 0.0)),
            })
        return active_list

    def emergency_close_all(self) -> list:
        state = self._load_state()
        open_positions = self.sync_positions_and_trail_stops(state)
        closed = []
        for pos in open_positions:
            close_side = "Sell" if pos["side"] == "LONG" else "Buy"
            res = self._bybit_request("POST", "/v5/order/create", body={
                "category": self.category,
                "symbol": pos["symbol"],
                "side": close_side,
                "orderType": "Market",
                "qty": str(pos["qty"]),
                "timeInForce": "IOC",
                "reduceOnly": True,
                "positionIdx": 0,
            })
            closed.append({"symbol": pos["symbol"], "ok": res.get("retCode") == 0})
        self._save_state(state)
        return closed

    def run_live_cycle(
        self,
        leverage: int = 10,
        risk_per_trade_pct: float = 1.5,
        max_open_positions: int = 2,
        crypto_enabled: bool = True,
        forex_enabled: bool = False,
        live_orders_armed: bool = False,
        mtf_mode: str = "STRICT",
        btc_macro_guard: bool = True,
        daily_drawdown_pct: float = 0.0,
        max_daily_loss_pct: float = 3.0,
    ) -> dict:
        if not os.path.exists(self.master_path):
            raise FileNotFoundError(f"Master ML Model not found at {self.master_path}")

        leverage = min(max(int(leverage), 1), 20)
        with open(self.master_path, "r", encoding="utf-8-sig") as f:
            master = json.load(f)

        features = master.get("features", FEATURES)
        portfolio = master.get("portfolio", {})
        if not portfolio:
            raise ValueError("No coins found in Live ML Model. Promote trained coins from Paper to Live first.")

        state = self._load_state()
        traded_candle_ts = state.setdefault("traded_candle_ts", {})
        last_trade_epoch_ms = state.setdefault("last_trade_epoch_ms", {})
        daily_coin_trades = state.setdefault("daily_coin_trades", {})
        diagnostics = []
        order_rejections = []
        trade_blocks = []
        radar_signals = []
        today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        if daily_coin_trades.get("_date") != today_str:
            daily_coin_trades.clear()
            daily_coin_trades["_date"] = today_str
            state["daily_total_trades"] = 0

        daily_total_trades = int(state.get("daily_total_trades", 0))
        need_daily_first_trade = (daily_total_trades == 0)

        now_ms = int(time.time() * 1000)

        bal_res = self._bybit_request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": "USDT"})
        bal_ret_code = int(bal_res.get("retCode", -1)) if bal_res else -1
        if bal_ret_code != 0:
            bal_msg = bal_res.get("retMsg", "Unable to fetch Unified USDT balance")
            order_rejections.append(f"Bybit Account Balance Check [{bal_ret_code}]: {BYBIT_ERROR_HINTS.get(bal_ret_code, bal_msg)}")
        avail_usdt = float(bal_res.get("result", {}).get("list", [{}])[0].get("totalAvailableBalance", 0.0))

        btc_macro_bearish = False
        btc_adx = 20.0
        btc_natr = 1.8
        if btc_macro_guard:
            btc_st, btc_adx, btc_natr = self._get_cached_4h_metrics("BTCUSDT")
            btc_macro_bearish = bool(btc_st == -1)

        sl_mult, tp_mult, trail_ratio = self._regime_exit_multipliers(btc_adx, btc_natr, btc_macro_bearish)
        open_positions = self.sync_positions_and_trail_stops(state, trail_ratio=trail_ratio)
        open_syms = {p["symbol"] for p in open_positions}
        if open_positions and need_daily_first_trade:
            need_daily_first_trade = False

        candidates = []
        daily_hunter_backups = []
        top_blocked_info = None
        coins_scanned_count = 0

        for symbol, entry in portfolio.items():
            if not isinstance(entry, dict):
                continue

            market_type = entry.get("market", "FOREX" if is_forex_symbol(symbol) else "CRYPTO").upper()
            if (market_type == "CRYPTO" and not crypto_enabled) or (market_type == "FOREX" and not forex_enabled):
                continue

            coins_scanned_count += 1
            cfg = entry.get("config", {})
            bybit_sym = (cfg.get("bybit_symbol") or symbol.replace("/", "")).upper()

            models_map = entry.get("models", {})
            long_data = models_map.get("long") or models_map.get("long_model")
            short_data = models_map.get("short") or models_map.get("short_model")
            if not is_valid_xgb_model(long_data) or not is_valid_xgb_model(short_data):
                diagnostics.append(f"{bybit_sym}: Missing valid XGBoost booster weights")
                continue

            tf_str = (cfg.get("timeframe") or entry.get("timeframe") or "1h").lower()
            interval = TIMEFRAME_TO_BYBIT_INTERVAL.get(tf_str, "60")
            strat_id = cfg.get("strategy_id") or entry.get("strategy_id") or "xgb_mtf_v2"
            is_relaxed = bool(cfg.get("relaxed_filter", False))
            custom_rules = cfg.get("custom_rules")

            df_raw = self.fetch_bybit_candles(bybit_sym, interval=interval, limit=230)
            if len(df_raw) < 210:
                diagnostics.append(f"{bybit_sym}: Insufficient candles ({len(df_raw)}/210)")
                continue

            df = self.build_features(df_raw)
            if len(df) < 2:
                continue
            prev = df.iloc[-2]
            curr = df.iloc[-1]
            latest_bar_ms = int(df.index[-1].timestamp() * 1000)

            try:
                dmat = xgb.DMatrix(df.iloc[[-1]][features].to_numpy(dtype="float32"))
                p_long = float(self._load_booster(f"{bybit_sym}_long", long_data).predict(dmat)[0])
                p_short = float(self._load_booster(f"{bybit_sym}_short", short_data).predict(dmat)[0])
            except Exception as e:
                diagnostics.append(f"{bybit_sym}: Booster predict error ({e})")
                continue

            direction = "LONG" if p_long >= p_short else "SHORT"
            raw_prob = p_long if direction == "LONG" else p_short
            default_th = 0.24 if market_type == "FOREX" else 0.30
            req_thresh = float(cfg.get("long_thresh" if direction == "LONG" else "short_thresh", default_th))

            if interval == "240":
                trend_4h = int(curr["Supertrend"])
            else:
                trend_4h, _, _ = self._get_cached_4h_metrics(bybit_sym)

            raw_4h_ok = (direction == "LONG" and trend_4h == 1) or (direction == "SHORT" and trend_4h == -1)
            high_conv_rev = (raw_prob >= req_thresh + 0.07) and (float(curr["Vol_Surge"]) >= 1.20)
            guard_blocked = btc_macro_bearish and direction == "LONG" and market_type == "CRYPTO" and not bybit_sym.startswith("BTC")

            effective_mtf = str(cfg.get("mtf_alignment_mode") or entry.get("mtf_alignment_mode") or mtf_mode).upper()
            if effective_mtf == "OFF":
                mtf_aligned = not guard_blocked
            elif effective_mtf == "SOFT":
                mtf_aligned = (raw_4h_ok or (raw_prob >= (req_thresh + 0.05) and float(curr["Vol_Surge"]) >= 1.10)) and not guard_blocked
            else:
                mtf_aligned = (raw_4h_ok or high_conv_rev) and not guard_blocked

            # Build radar signal directly here so dynamic_bot_runner skips duplicate candle downloads
            adx_val = float(curr.get("ADX", 0.0))
            disp_sym = symbol if "/" in symbol else (f"{bybit_sym[:-4]}/USDT" if bybit_sym.endswith("USDT") else bybit_sym)
            conv_pct = int(min(99, max(45, round((raw_prob / max(req_thresh, 0.16)) * 76 + (10 if raw_4h_ok else -8) + min(adx_val * 0.25, 8)))))
            radar_signals.append({
                "symbol": disp_sym,
                "side": direction,
                "conviction_pct": conv_pct,
                "aligned_4h": bool(raw_4h_ok and not guard_blocked),
                "raw_prob": round(raw_prob, 4),
            })

            if bybit_sym in open_syms:
                trade_blocks.append(f"{bybit_sym}: Already has an open position")
                continue

            strat_ok = self._strategy_entry_filter(
                strat_id,
                curr,
                prev,
                direction,
                relaxed=is_relaxed,
                custom_rules=custom_rules,
            )
            effective_min_prob = max(req_thresh, 0.18)
            quality_edge = (raw_prob - effective_min_prob) + (0.03 if raw_4h_ok else 0.0) + (0.02 if adx_val >= 20.0 else 0.0)

            if mtf_aligned and strat_ok and raw_prob >= effective_min_prob:
                candidates.append({
                    "edge": quality_edge,
                    "raw_prob": raw_prob,
                    "req_thresh": effective_min_prob,
                    "bybit_sym": bybit_sym,
                    "market_type": market_type,
                    "direction": direction,
                    "cfg": cfg,
                    "curr": curr.copy(),
                    "latest_bar_ms": latest_bar_ms,
                })
            else:
                # Calibrated Daily 1-Trade Hunter: qualifies top 4H-aligned setup (raw_prob >= 0.12) when 0 trades fired today
                if (
                    need_daily_first_trade
                    and not guard_blocked
                    and (raw_4h_ok or effective_mtf == "OFF")
                    and raw_prob >= 0.12
                ):
                    daily_hunter_backups.append({
                        "edge": quality_edge,
                        "raw_prob": raw_prob,
                        "req_thresh": 0.12,
                        "bybit_sym": bybit_sym,
                        "market_type": market_type,
                        "direction": direction,
                        "cfg": cfg,
                        "curr": curr.copy(),
                        "latest_bar_ms": latest_bar_ms,
                    })

                reasons = []
                if guard_blocked:
                    reasons.append("BTC 4H Macro Guard")
                elif not mtf_aligned:
                    reasons.append(f"4H MTF misaligned ({effective_mtf})")
                if not strat_ok:
                    reasons.append(f"Filter ({strat_id})")
                if raw_prob < effective_min_prob:
                    reasons.append(f"ML prob {raw_prob:.2f} < {effective_min_prob:.2f}")
                block_msg = f"{bybit_sym} {direction} waiting: {', '.join(reasons)} [ADX:{adx_val:.1f}, Avail:${avail_usdt:.2f}]"
                diagnostics.append(block_msg)
                trade_blocks.append(block_msg)
                if top_blocked_info is None or raw_prob > top_blocked_info[0]:
                    top_blocked_info = (raw_prob, f"{bybit_sym}_{direction}_{','.join(reasons)}", block_msg)

        radar_signals.sort(key=lambda x: (x["aligned_4h"], x["conviction_pct"]), reverse=True)
        candidates.sort(key=lambda x: x["edge"], reverse=True)
        if not candidates and need_daily_first_trade and daily_hunter_backups:
            daily_hunter_backups.sort(key=lambda x: x["edge"], reverse=True)
            candidates = [daily_hunter_backups[0]]
            diagnostics.append(f"Daily 1-Trade Hunter selected top setup: {candidates[0]['bybit_sym']} {candidates[0]['direction']}")

        live_trades_this_cycle = 0

        for cand in candidates:
            bybit_sym = cand["bybit_sym"]
            if not live_orders_armed:
                msg = f"{bybit_sym} {cand['direction']} blocked: Live Orders Armed switch is OFF"
                diagnostics.append(msg)
                trade_blocks.append(msg)
                order_rejections.append(msg)
                break
            if daily_drawdown_pct >= max_daily_loss_pct and max_daily_loss_pct > 0.0:
                msg = f"{bybit_sym} {cand['direction']} blocked: Daily loss limit reached ({daily_drawdown_pct:.2f}% >= {max_daily_loss_pct:.2f}%)"
                diagnostics.append(msg)
                trade_blocks.append(msg)
                order_rejections.append(msg)
                break
            if len(open_syms) >= max_open_positions:
                msg = f"{bybit_sym} {cand['direction']} blocked: Max open positions reached ({len(open_syms)}/{max_open_positions})"
                diagnostics.append(msg)
                trade_blocks.append(msg)
                break
            if avail_usdt < 2.0:
                msg = f"{bybit_sym} {cand['direction']} blocked: Available USDT balance too low (${avail_usdt:.2f} < $2.00)"
                diagnostics.append(msg)
                trade_blocks.append(msg)
                order_rejections.append(msg)
                break
            if traded_candle_ts.get(bybit_sym) == cand["latest_bar_ms"]:
                trade_blocks.append(f"{bybit_sym}: Already traded on this candle")
                continue
            if (now_ms - int(last_trade_epoch_ms.get(bybit_sym, 0))) < 15 * 60 * 1000:
                trade_blocks.append(f"{bybit_sym}: 15-minute cooldown active")
                continue

            filters = self._get_instrument_filters(bybit_sym)
            if filters is None:
                msg = f"{bybit_sym} blocked: Instrument specs unavailable on Bybit"
                diagnostics.append(msg)
                trade_blocks.append(msg)
                order_rejections.append(msg)
                continue
            qty_step, min_qty, min_notional, tick_size = filters

            curr = cand["curr"]
            cfg = cand["cfg"]
            price = float(curr["Close"])
            atr = float(curr["ATR"])
            sl_atr = max(float(cfg.get("sl_atr", 1.35)) * sl_mult, 0.80)
            tp_rr = max(float(cfg.get("tp2_rr", 3.0)) * tp_mult, 1.8)
            min_sl_floor = price * (0.0010 if cand["market_type"] == "FOREX" else 0.0075)
            risk_dist = max(sl_atr * atr, min_sl_floor)

            max_buying_power = avail_usdt * leverage * 0.85
            min_required_notional = max(min_notional * 1.05, min_qty * price)
            if max_buying_power < min_required_notional:
                msg = f"{bybit_sym} {cand['direction']} blocked: Min order ${min_required_notional:.2f} > buying power ${max_buying_power:.2f} (Avail ${avail_usdt:.2f})"
                diagnostics.append(msg)
                trade_blocks.append(msg)
                order_rejections.append(msg)
                continue

            conv_scale = self._conviction_risk_scale(float(cand.get("raw_prob", 0.3)), float(cand.get("req_thresh", 0.3)))
            max_risk_cap = max(risk_per_trade_pct * 1.20, 2.50)
            scaled_risk_pct = min(max(risk_per_trade_pct * conv_scale, 0.50), max_risk_cap)
            risk_usdt = max(avail_usdt * (scaled_risk_pct / 100.0), 0.15)
            notional = min(max(risk_usdt / (risk_dist / price), min_required_notional, 5.5), max_buying_power)
            qty_str = self._format_step(max(notional / price, min_qty), qty_step, round_up=True)

            is_long = cand["direction"] == "LONG"
            sl_raw = (price - risk_dist) if is_long else (price + risk_dist)
            tp_raw = (price + risk_dist * tp_rr) if is_long else (price - risk_dist * tp_rr)
            sl_str = self._format_step(sl_raw, tick_size)
            tp_str = self._format_step(tp_raw, tick_size)

            self._bybit_request("POST", "/v5/position/set-leverage", body={
                "category": self.category,
                "symbol": bybit_sym,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage),
            })
            res = self._bybit_request("POST", "/v5/order/create", body={
                "category": self.category,
                "symbol": bybit_sym,
                "side": "Buy" if is_long else "Sell",
                "orderType": "Market",
                "qty": qty_str,
                "timeInForce": "IOC",
                "stopLoss": sl_str,
                "takeProfit": tp_str,
                "positionIdx": 0,
            })
            if res.get("retCode") == 0:
                open_syms.add(bybit_sym)
                traded_candle_ts[bybit_sym] = cand["latest_bar_ms"]
                last_trade_epoch_ms[bybit_sym] = now_ms
                daily_coin_trades[bybit_sym] = int(daily_coin_trades.get(bybit_sym, 0)) + 1
                daily_total_trades += 1
                state["daily_total_trades"] = daily_total_trades
                live_trades_this_cycle += 1
                avail_usdt = max(0.0, avail_usdt - (notional / leverage))
            else:
                code = int(res.get("retCode", -1))
                raw_msg = str(res.get("retMsg", "Unknown Bybit rejection"))
                hint = BYBIT_ERROR_HINTS.get(code, raw_msg)
                rej_msg = f"{bybit_sym} {cand['direction']} (qty {qty_str}) rejected by Bybit [{code}]: {hint} ({raw_msg})"
                diagnostics.append(rej_msg)
                order_rejections.append(rej_msg)

        if live_orders_armed and live_trades_this_cycle == 0 and len(open_syms) == 0 and not order_rejections and top_blocked_info is not None:
            _, skip_key, skip_msg = top_blocked_info
            order_rejections.append(f"Scan Status: {skip_msg}")
            state["last_skip_notice_key"] = skip_key
            state["last_skip_notice_ms"] = now_ms

        state["coins_scanned"] = coins_scanned_count
        state["last_diagnostics"] = diagnostics[-25:]
        state["last_order_rejections"] = order_rejections[-10:]
        state["last_trade_blocks"] = trade_blocks[-15:]
        self._save_state(state)
        gc.collect()

        return {
            "ok": True,
            "coins_scanned": coins_scanned_count,
            "daily_total_trades": daily_total_trades,
            "open_positions_count": len(open_syms),
            "radar_signals": radar_signals,
            "ml_model_version": master.get("version", f"LIVE_PORTFOLIO_{len(portfolio)}C"),
            "order_rejections": order_rejections,
            "trade_blocks": trade_blocks,
            "diagnostics": diagnostics[-10:],
        }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    live_dir = os.environ.get("LIVE_DATA_DIR", os.path.join(base_dir, "bot_data", "LIVE_ACCOUNT"))
    models_dir = os.environ.get("MODELS_DIR", os.path.join(base_dir, "ml_models"))

    cfg_file = os.path.join(live_dir, "live_runtime_config.json")
    runtime_cfg = {}
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8-sig") as f:
                runtime_cfg = json.load(f)
        except Exception:
            runtime_cfg = {}

    api_key = os.environ.get("BYBIT_API_KEY", runtime_cfg.get("api_key", ""))
    api_secret = os.environ.get("BYBIT_API_SECRET", runtime_cfg.get("api_secret", ""))
    endpoint = runtime_cfg.get("endpoint", "Mainnet")

    bot = LiveAccountTradingBot(
        api_key=api_key,
        api_secret=api_secret,
        models_dir=models_dir,
        live_dir=live_dir,
        endpoint=endpoint,
    )

    if len(sys.argv) >= 2 and sys.argv[1] == "--kill":
        print(json.dumps({"ok": True, "closed": bot.emergency_close_all()}))
    elif api_key and api_secret:
        out = bot.run_live_cycle(
            leverage=int(runtime_cfg.get("leverage", 10)),
            risk_per_trade_pct=float(runtime_cfg.get("risk_per_trade_pct", 1.5)),
            max_open_positions=int(runtime_cfg.get("max_open_positions", 2)),
            crypto_enabled=bool(runtime_cfg.get("crypto_enabled", True)),
            forex_enabled=bool(runtime_cfg.get("forex_enabled", False)),
            live_orders_armed=bool(runtime_cfg.get("live_orders_armed", False)),
            mtf_mode=str(runtime_cfg.get("mtf_mode", "STRICT")),
            btc_macro_guard=bool(runtime_cfg.get("btc_macro_guard", True)),
            daily_drawdown_pct=float(runtime_cfg.get("daily_drawdown_pct", 0.0)),
            max_daily_loss_pct=float(runtime_cfg.get("max_daily_loss_pct", 3.0)),
        )
        print(json.dumps(out))
        