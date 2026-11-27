import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
import pandas as pd
import xgboost as xgb

TIMEFRAME_TO_BYBIT_INTERVAL = {
    "5m": "5",
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
    "VWAP_Dev",
]

import socket as _sock, urllib3.util.connection as _u3c
_u3c.HAS_IPV6 = True
_u3c.allowed_gai_family = lambda: _sock.AF_UNSPEC
_SESSION = requests.Session()
_ADAPTER = requests.adapters.HTTPAdapter(pool_connections=12, pool_maxsize=12)
_SESSION.mount("https://", _ADAPTER)

def reset_network_pools():
    global _SESSION
    old_s = _SESSION
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=12, pool_maxsize=12))
    _SESSION = s
    try:
        old_s.close()
    except Exception:
        pass

_MASTER_CACHE = {}   # path -> (mtime, dict)
_BOOSTER_CACHE = {}  # (path, mtime, key) -> xgb.Booster
_TREND_4H_CACHE = {} # symbol -> (ts, trend_int)
_RADAR_CACHE = {}    # cache_key -> (ts, signals, ml_connected, ml_version)


def _is_forex(symbol: str) -> bool:
    clean = symbol.replace("/", "").upper()
    if clean in {"EURUSDUSDT", "GBPUSDUSDT", "USDJPYUSDT", "XAUUSDT", "XAGUSDT", "XAUTUSDT"}:
        return True
    fx = {"EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
    return len(clean) == 6 and clean[:3] in fx and clean[3:] in (fx | {"USD"})


def _normalize_signal(raw: dict) -> dict:
    sym = str(raw.get("symbol", "")).strip()
    direction = str(raw.get("direction") or raw.get("side") or "LONG").upper()
    if direction not in ("LONG", "SHORT"):
        direction = "LONG"
    conf = raw.get("confidence_pct")
    if conf is None or int(conf) == 0:
        conf = raw.get("conviction_pct", 50)
    conf_int = int(max(1, min(99, int(conf or 50))))
    aligned = bool(raw.get("mtf_aligned") if "mtf_aligned" in raw else raw.get("aligned_4h", False))
    return {
        "symbol": sym,
        "direction": direction,
        "confidence_pct": conf_int,
        "mtf_aligned": aligned,
        "side": direction,
        "conviction_pct": conf_int,
        "aligned_4h": aligned,
    }


def _sanitize_xgb(model_dict: dict) -> dict:
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
        trees = (gb.get("model") or {}).get("trees")
        if isinstance(trees, list):
            for tree in trees:
                if not isinstance(tree, dict):
                    continue
                for k in ("split_conditions", "base_weights", "loss_changes", "sum_hessian"):
                    arr = tree.get(k)
                    if isinstance(arr, list):
                        tree[k] = [float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v for v in arr]
    return model_dict


def _load_master_cached(master_path: str) -> tuple:
    mtime = os.path.getmtime(master_path)
    cached = _MASTER_CACHE.get(master_path)
    if cached and cached[0] == mtime:
        return mtime, cached[1]
    with open(master_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    _MASTER_CACHE[master_path] = (mtime, data)
    return mtime, data


def _load_booster_cached(master_path: str, mtime: float, key: str, model_dict: dict, tmp_dir: str) -> xgb.Booster:
    ck = (master_path, mtime, key)
    if ck in _BOOSTER_CACHE:
        return _BOOSTER_CACHE[ck]
    if len(_BOOSTER_CACHE) >= 48:
        _BOOSTER_CACHE.pop(next(iter(_BOOSTER_CACHE)), None)
    booster = xgb.Booster()
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"_tmp_radar_{key}_{os.getpid()}.json")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_sanitize_xgb(model_dict), f)
        booster.load_model(tmp_path)
        _BOOSTER_CACHE[ck] = booster
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    return booster


def _fetch_candles(bybit_sym: str, interval: str = "60", limit: int = 220, is_demo: bool = False) -> pd.DataFrame:
    hosts = ["https://api-demo.bybit.com", "https://api.bybit.com"] if is_demo else ["https://api.bybit.com", "https://api.bytick.com"]
    for base in hosts:
        try:
            resp = _SESSION.get(
                f"{base}/v5/market/kline",
                params={"category": "linear", "symbol": bybit_sym, "interval": interval, "limit": limit},
                timeout=3.5,
            )
            if resp.status_code == 200:
                rows = resp.json().get("result", {}).get("list", [])
                if rows:
                    df = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"])
                    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
                    for col in ("Open", "High", "Low", "Close", "Volume"):
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
                    return df.sort_values("timestamp").set_index("timestamp")
        except Exception:
            continue

    if _is_forex(bybit_sym):
        yf_int = "15m" if interval == "15" else ("30m" if interval == "30" else "1h")
        yf_rng = "15d" if yf_int in ("15m", "30m") else "60d"
        try:
            y_resp = _SESSION.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{bybit_sym}=X",
                params={"interval": yf_int, "range": yf_rng},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=4.5,
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
                if interval == "240":
                    df = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
                return df.tail(limit)
        except Exception:
            pass
    return pd.DataFrame()


def _build_features_fast(df: pd.DataFrame) -> pd.DataFrame:
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
    bb_u = sma20 + (std20 * 2.0)
    bb_l = sma20 - (std20 * 2.0)
    data["BB_PctB"] = (c - bb_l) / (bb_u - bb_l + 1e-9)
    data["BB_Width"] = (bb_u - bb_l) / (sma20 + 1e-9)

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

    h_diff = data["High"].diff()
    l_diff = -data["Low"].diff()
    plus_dm = np.where((h_diff > l_diff) & (h_diff > 0), h_diff, 0.0)
    minus_dm = np.where((l_diff > h_diff) & (l_diff > 0), l_diff, 0.0)
    plus_di = 100 * (pd.Series(plus_dm, index=data.index).rolling(14).mean() / (data["ATR"] + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=data.index).rolling(14).mean() / (data["ATR"] + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    data["ADX"] = dx.rolling(14).mean()
    data["DI_Spread"] = plus_di - minus_di

    macd_line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    data["MACD_Hist"] = (macd_line - macd_line.ewm(span=9, adjust=False).mean()) / (data["ATR"] + 1e-9)

    data["Dist_EMA20"] = (c - c.ewm(span=20, adjust=False).mean()) / (data["ATR"] + 1e-9)
    data["Dist_EMA50"] = (c - c.ewm(span=50, adjust=False).mean()) / (data["ATR"] + 1e-9)
    data["Dist_EMA200"] = (c - c.ewm(span=200, adjust=False).mean()) / (data["ATR"] + 1e-9)

    typical_px = (data["High"] + data["Low"] + c) / 3.0
    vol_safe = data["Volume"].clip(lower=1.0)
    vwap = (typical_px * vol_safe).rolling(24).sum() / (vol_safe.rolling(24).sum() + 1e-9)
    data["VWAP_Dev"] = (c - vwap) / (data["ATR"] + 1e-9)

    hl2 = ((data["High"] + data["Low"]) * 0.5).to_numpy()
    atr_arr = data["ATR"].fillna(0.0).to_numpy()
    c_arr = c.to_numpy()
    upper = hl2 + 3.0 * atr_arr
    lower = hl2 - 3.0 * atr_arr
    st = np.ones(len(data), dtype=np.int8)
    uptrend = True
    for i in range(10, len(c_arr)):
        if uptrend and c_arr[i] < lower[i - 1]:
            uptrend = False
        elif not uptrend and c_arr[i] > upper[i - 1]:
            uptrend = True
        st[i] = 1 if uptrend else -1
    data["Supertrend"] = st
    return data.dropna()


def get_fast_radar_signals(
    models_dir: str,
    live_dir: str,
    mode_name: str = "LIVE",
    endpoint: str = "Mainnet",
    crypto_en: bool = True,
    forex_en: bool = True,
    ttl_sec: float = 30.0,
) -> tuple:
    """
    Ultra-fast, non-blocking radar engine:
    1) Uses fresh live_engine_state.json (<45s old) normalized to exact UI keys.
    2) Uses 20s in-memory TTL cache (<2 ms response).
    3) Parallel 8-thread fetch + cached XGBoost boosters (~0.8s refresh).
    """
    now_s = time.time()
    cache_key = (models_dir, mode_name, str(endpoint).lower(), bool(crypto_en), bool(forex_en))

    # 1. Fast path: fresh live_engine_state.json from live_bot.py
    if mode_name == "LIVE" and live_dir:
        st_path = os.path.join(live_dir, "live_engine_state.json")
        if os.path.exists(st_path) and (now_s - os.path.getmtime(st_path)) < 45.0:
            try:
                with open(st_path, "r", encoding="utf-8-sig") as sf:
                    st_obj = json.load(sf)
                raw_sigs = st_obj.get("radar_signals") or []
                if raw_sigs:
                    norm = [_normalize_signal(s) for s in raw_sigs if isinstance(s, dict) and s.get("symbol")]
                    norm.sort(key=lambda x: (x["mtf_aligned"], x["confidence_pct"]), reverse=True)
                    if norm:
                        ver = f"LIVE_PORTFOLIO_{len(norm)}C"
                        _RADAR_CACHE[cache_key] = (now_s, norm, True, ver)
                        return norm, True, ver
            except Exception:
                pass

    # 2. Fast path: 20-second in-memory TTL cache
    cached = _RADAR_CACHE.get(cache_key)
    if cached and (now_s - cached[0]) < ttl_sec:
        return cached[1], cached[2], cached[3]

    # 3. Parallel compute with cached XGBoost boosters
    p1 = os.path.join(models_dir, mode_name, "UNIFIED_PORTFOLIO_MASTER.json")
    p2 = os.path.join(models_dir, "UNIFIED_PORTFOLIO_MASTER.json")
    master_path = p1 if os.path.exists(p1) else (p2 if mode_name == "LIVE" and os.path.exists(p2) else "")
    if not master_path or not os.path.exists(master_path):
        return [], False, "NONE"

    try:
        mtime, master = _load_master_cached(master_path)
    except Exception:
        return [], False, "NONE"

    portfolio = master.get("portfolio", {})
    features = master.get("features", FEATURES)
    if not portfolio:
        return [], False, "NONE"

    is_demo = str(endpoint).lower() in ("testnet", "demo")
    items = []
    tasks = {}
    for sym, entry in portfolio.items():
        if not isinstance(entry, dict):
            continue
        mtype = entry.get("market", "FOREX" if _is_forex(sym) else "CRYPTO").upper()
        if (mtype == "CRYPTO" and not crypto_en) or (mtype == "FOREX" and not forex_en):
            continue
        cfg = entry.get("config", {})
        bybit_sym = (cfg.get("bybit_symbol") or sym.replace("/", "")).upper()
        tf = (cfg.get("timeframe") or entry.get("timeframe") or "1h").lower()
        interval = TIMEFRAME_TO_BYBIT_INTERVAL.get(tf, "60")
        items.append((sym, entry, mtype, cfg, bybit_sym, interval))
        tasks[(bybit_sym, interval)] = (bybit_sym, interval, 220)
        if interval != "240":
            c4 = _TREND_4H_CACHE.get(bybit_sym)
            if not c4 or (now_s - c4[0]) >= 600.0:
                tasks[(bybit_sym, "240")] = (bybit_sym, "240", 215)

    fetched = {}
    if tasks:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {k: pool.submit(_fetch_candles, v[0], v[1], v[2], is_demo) for k, v in tasks.items()}
            for k, f in futs.items():
                try:
                    fetched[k] = f.result()
                except Exception:
                    fetched[k] = pd.DataFrame()

    for (sym_k, int_k), df4_raw in fetched.items():
        if int_k == "240" and len(df4_raw) >= 205:
            try:
                df4 = _build_features_fast(df4_raw)
                if not df4.empty:
                    bar4 = df4.iloc[-2] if len(df4) >= 2 else df4.iloc[-1]
                    _TREND_4H_CACHE[sym_k] = (now_s, int(bar4["Supertrend"]))
            except Exception:
                pass

    signals = []
    tmp_dir = live_dir or models_dir
    for sym, entry, mtype, cfg, bybit_sym, interval in items:
        try:
            models_map = entry.get("models", {})
            l_data = models_map.get("long") or models_map.get("long_model")
            s_data = models_map.get("short") or models_map.get("short_model")
            if not isinstance(l_data, dict) or not isinstance(s_data, dict):
                continue
            df_raw = fetched.get((bybit_sym, interval), pd.DataFrame())
            if len(df_raw) < 205:
                continue
            df = _build_features_fast(df_raw)
            if len(df) < 2:
                continue
            bar = df.iloc[-2]
            x_row = df.iloc[[-2]][features].to_numpy(dtype="float32")
            dmat = xgb.DMatrix(x_row, feature_names=features)
            p_long = float(_load_booster_cached(master_path, mtime, f"{bybit_sym}_L", l_data, tmp_dir).predict(dmat)[0])
            p_short = float(_load_booster_cached(master_path, mtime, f"{bybit_sym}_S", s_data, tmp_dir).predict(dmat)[0])
            direction = "LONG" if p_long >= p_short else "SHORT"
            raw_prob = p_long if direction == "LONG" else p_short
            req_th = float(cfg.get("long_thresh" if direction == "LONG" else "short_thresh", 0.24 if mtype == "FOREX" else 0.28))

            trend_4h = int(bar["Supertrend"]) if interval == "240" else (_TREND_4H_CACHE.get(bybit_sym, (0, 0))[1])
            mtf_ok = (direction == "LONG" and trend_4h == 1) or (direction == "SHORT" and trend_4h == -1)
            adx_val = float(bar.get("ADX", 0.0))
            conf_pct = int(min(99, max(45, round((raw_prob / max(req_th, 0.16)) * 76 + (10 if mtf_ok else -8) + min(adx_val * 0.25, 8)))))
            disp_sym = sym if "/" in sym else (f"{bybit_sym[:-4]}/USDT" if bybit_sym.endswith("USDT") else bybit_sym)
            signals.append(_normalize_signal({
                "symbol": disp_sym,
                "direction": direction,
                "confidence_pct": conf_pct,
                "mtf_aligned": bool(mtf_ok),
            }))
        except Exception:
            continue

    signals.sort(key=lambda x: (x["mtf_aligned"], x["confidence_pct"]), reverse=True)
    ver = master.get("version", f"{mode_name}_PORTFOLIO_{len(portfolio)}C")
    _RADAR_CACHE[cache_key] = (now_s, signals, True, ver)
    return signals, True, ver
