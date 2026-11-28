import os as _at_os, sys as _at_sys
_SYNCED_MODS_DIR = _at_os.path.join(_at_os.path.dirname(_at_os.path.abspath(__file__)), "synced_modules")
if _at_os.path.isdir(_SYNCED_MODS_DIR) and _SYNCED_MODS_DIR not in _at_sys.path:
    _at_sys.path.insert(0, _SYNCED_MODS_DIR)
def _safe_float(v, d=0.0):
    try:
        s = "" if v is None else str(v).strip()
        return float(s) if s else d
    except Exception:
        return d

import gc
import os

import threading as _at_threading
import uuid as _at_uuid
_ATOMIC_IO_LOCK = _at_threading.RLock()

def _locked_replace(src_tmp: str, dst_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    with _ATOMIC_IO_LOCK:
        os.replace(src_tmp, dst_path)

import sys
import json
import time
import hmac
import hashlib
import zipfile
import shutil
import tempfile
import importlib.util
import urllib.parse
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
import urllib3.util.connection as _urllib3_cn
import urllib3.connection as _urllib3_conn
import pandas as pd
import xgboost as xgb

# Dual-stack IPv4/IPv6 support (compatible with both home Wi-Fi and 4G+/5G IPv6/464XLAT)
_urllib3_cn.HAS_IPV6 = True
_urllib3_cn.allowed_gai_family = lambda: socket.AF_UNSPEC

# Enable TCP_NODELAY + OS-level TCP Keep-Alive probes so packets flush immediately
try:
    _urllib3_conn.HTTPConnection.default_socket_options = [
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ]
except Exception:
    pass

# Short-TTL Dual-Stack DNS cache (flushed automatically on Wi-Fi <-> Mobile Data handover)
_ORIG_GETADDRINFO = socket.getaddrinfo
_DNS_CACHE = {}
_BYBIT_HOSTS_SET = {"api.bytick.com", "api.bybit.com", "api-demo.bybit.com"}

def _fast_dualstack_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host in _BYBIT_HOSTS_SET:
        key = (host, port)
        cached = _DNS_CACHE.get(key)
        if cached and (time.time() - cached[0]) < 25.0:
            return cached[1]
        res = _ORIG_GETADDRINFO(host, port, family or socket.AF_UNSPEC, type or socket.SOCK_STREAM, proto, flags)
        if res:
            # Prefer IPv4 first when dual-stack is available, with automatic IPv6 fallback on cellular
            res_sorted = sorted(res, key=lambda r: 0 if r[0] == socket.AF_INET else 1)
            _DNS_CACHE[key] = (time.time(), res_sorted)
            return res_sorted
    return _ORIG_GETADDRINFO(host, port, family, type, proto, flags)

socket.getaddrinfo = _fast_dualstack_getaddrinfo

def _build_fresh_http_session() -> requests.Session:
    s = requests.Session()
    ad = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
    s.mount("https://", ad)
    s.headers.update({"Connection": "keep-alive"})
    return s

_HTTP_SESSION = _build_fresh_http_session()
_NET_RESET_LOCK = threading.Lock()
_LAST_NET_RESET_TS = 0.0

def reset_network_pools(force: bool = False):
    """Flushes stale DNS entries and pooled sockets after Wi-Fi <-> Mobile Data handover."""
    global _HTTP_SESSION, _HOST_BENCH_TS, _LAST_NET_RESET_TS
    now = time.time()
    with _NET_RESET_LOCK:
        if not force and (now - _LAST_NET_RESET_TS) < 1.0:
            return
        _LAST_NET_RESET_TS = now
        _DNS_CACHE.clear()
        _HOST_BENCH_TS = 0.0
        old_sess = _HTTP_SESSION
        _HTTP_SESSION = _build_fresh_http_session()
        try:
            old_sess.close()
        except Exception:
            pass

_FAST_MAINNET_HOSTS = ["https://api.bybit.com", "https://api.bytick.com"]
_FAST_DEMO_HOSTS = ["https://api-demo.bybit.com"]
_HOST_BENCH_TS = 0.0
_WARM_LAST_TS = 0.0
_ACTIVE_WARM_ENDPOINT = "Mainnet"
_WARM_THREAD_STARTED = False
_WARM_LOCK = threading.Lock()


def _benchmark_and_warm_bybit(endpoint: str = "Mainnet", force: bool = False):
    global _FAST_MAINNET_HOSTS, _HOST_BENCH_TS, _WARM_LAST_TS, _ACTIVE_WARM_ENDPOINT
    _ACTIVE_WARM_ENDPOINT = endpoint or "Mainnet"
    is_demo = str(endpoint).lower() in ("testnet", "demo")
    now = time.time()

    if not is_demo and (force or (now - _HOST_BENCH_TS) > 300.0):
        candidates = ["https://api.bybit.com", "https://api.bytick.com"]

        def _probe(host: str):
            try:
                # Warm TCP/TLS first, then measure warm RTT
                _HTTP_SESSION.get(f"{host}/v5/market/time", timeout=2.5)
                t0 = time.perf_counter()
                r = _HTTP_SESSION.get(f"{host}/v5/market/time", timeout=2.5)
                if r.status_code == 200:
                    return (host, (time.perf_counter() - t0) * 1000.0)
            except Exception:
                pass
            return (host, 9999.0)

        with ThreadPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(_probe, candidates))
        results.sort(key=lambda x: x[1])
        ordered = [h for h, _ in results]
        if ordered:
            _FAST_MAINNET_HOSTS = ordered
        _HOST_BENCH_TS = time.time()
        _WARM_LAST_TS = _HOST_BENCH_TS
        return

    if (now - _WARM_LAST_TS) >= 8.0:
        host = _FAST_DEMO_HOSTS[0] if is_demo else _FAST_MAINNET_HOSTS[0]
        try:
            _HTTP_SESSION.get(f"{host}/v5/market/time", timeout=2.5)
            _WARM_LAST_TS = time.time()
        except Exception:
            pass


def _start_keepalive_thread():
    global _WARM_THREAD_STARTED
    with _WARM_LOCK:
        if _WARM_THREAD_STARTED:
            return
        _WARM_THREAD_STARTED = True

        def _keepalive_loop():
            while True:
                try:
                    _benchmark_and_warm_bybit(_ACTIVE_WARM_ENDPOINT, force=False)
                except Exception:
                    pass
                time.sleep(55.0)

        t = threading.Thread(target=_keepalive_loop, name="BybitKeepAliveWarmer", daemon=True)
        t.start()


class _NonClosingSessionProxy:
    def __init__(self, sess: requests.Session):
        self._sess = sess

    def __getattr__(self, name):
        return getattr(self._sess, name)

    def _rewrite_url(self, url: str) -> str:
        if isinstance(url, str) and _FAST_MAINNET_HOSTS:
            fastest = _FAST_MAINNET_HOSTS[0]
            if url.startswith("https://api.bybit.com"):
                return fastest + url[len("https://api.bybit.com"):]
            if url.startswith("https://api.bytick.com"):
                return fastest + url[len("https://api.bytick.com"):]
        return url

    def request(self, method, url, **kwargs):
        rw_url = self._rewrite_url(url)
        try:
            return _HTTP_SESSION.request(method, rw_url, **kwargs)
        except (requests.exceptions.RequestException, OSError):
            # Flush stale Wi-Fi/cellular sockets, pause briefly for route handover, and retry
            reset_network_pools(force=False)
            time.sleep(0.55)
            retry_urls = [self._rewrite_url(url)]
            if isinstance(url, str) and ".com" in url:
                suffix = url.split(".com", 1)[1]
                if "api-demo.bybit.com" in url:
                    retry_urls.append("https://api-demo.bybit.com" + suffix)
                else:
                    for h in ("https://api.bybit.com", "https://api.bytick.com"):
                        cand_u = h + suffix
                        if cand_u not in retry_urls:
                            retry_urls.append(cand_u)
            last_exc = None
            for ru in retry_urls:
                try:
                    return _HTTP_SESSION.request(method, ru, **kwargs)
                except Exception as ex:
                    last_exc = ex
            if last_exc:
                raise last_exc
            raise

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def close(self):
        pass


class _PooledRequestsProxy:
    def __init__(self, real_mod):
        self._real = real_mod
        self._sess_proxy = _NonClosingSessionProxy(_HTTP_SESSION)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def Session(self):
        return self._sess_proxy

    def request(self, method, url, **kwargs):
        return self._sess_proxy.request(method, url, **kwargs)

    def get(self, url, **kwargs):
        return self._sess_proxy.get(url, **kwargs)

    def post(self, url, **kwargs):
        return self._sess_proxy.post(url, **kwargs)

_RADAR_DEFAULT_FEATURES = [
    "NATR", "BB_PctB", "BB_Width",
    "Ret_1h", "Ret_3h", "Ret_6h", "Ret_12h", "Ret_24h",
    "RSI_14", "Stoch_K", "Stoch_D", "ADX", "DI_Spread", "MACD_Hist",
    "Dist_EMA20", "Dist_EMA50", "Dist_EMA200",
    "VWAP_Dev"
]

def _radar_is_forex(symbol: str) -> bool:
    clean = symbol.replace("/", "").upper()
    if clean in {"EURUSDUSDT", "GBPUSDUSDT", "USDJPYUSDT", "XAUUSDT", "XAGUSDT", "XAUTUSDT"}:
        return True
    forex_bases = {"EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
    return len(clean) == 6 and clean[:3] in forex_bases and clean[3:] in (forex_bases | {"USD"})

def _radar_fetch_candles(bybit_sym: str, interval: str = "60", limit: int = 220) -> pd.DataFrame:
    for base in ("https://api.bybit.com", "https://api.bytick.com", "https://api-demo.bybit.com"):
        try:
            r = _HTTP_SESSION.get(
                f"{base}/v5/market/kline",
                params={"category": "linear", "symbol": bybit_sym, "interval": interval, "limit": limit},
                timeout=5.5,
            )
            rows = r.json().get("result", {}).get("list", [])
            if rows:
                df = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"])
                df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
                for col in ("Open", "High", "Low", "Close", "Volume"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
                return df.sort_values("timestamp").set_index("timestamp")
        except Exception:
            continue

    if _radar_is_forex(bybit_sym):
        yf_int = "15m" if interval == "15" else ("30m" if interval == "30" else "1h")
        yf_rng = "15d" if yf_int in ("15m", "30m") else "60d"
        try:
            yr = _HTTP_SESSION.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{bybit_sym}=X",
                params={"interval": yf_int, "range": yf_rng},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=6.0,
            )
            res0 = yr.json().get("chart", {}).get("result", [])[0]
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

def _radar_build_features(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    data = df.copy()
    c = data["Close"]
    tr = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - c.shift(1)).abs(),
        (data["Low"] - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    data["ATR"] = tr.rolling(14).mean()
    data["NATR"] = (data["ATR"] / c) * 100.0

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_up = sma20 + (std20 * 2.0)
    bb_lo = sma20 - (std20 * 2.0)
    data["BB_PctB"] = (c - bb_lo) / (bb_up - bb_lo + 1e-9)
    data["BB_Width"] = (bb_up - bb_lo) / (sma20 + 1e-9)

    for h in (1, 3, 6, 12, 24):
        data[f"Ret_{h}h"] = np.log(c / c.shift(h))

    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rsi = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-9))))
    data["RSI_14"] = rsi

    stoch = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min() + 1e-9)
    data["Stoch_K"] = stoch.rolling(3).mean() * 100.0
    data["Stoch_D"] = data["Stoch_K"].rolling(3).mean()

    plus_dm = np.where((data["High"].diff() > -data["Low"].diff()) & (data["High"].diff() > 0), data["High"].diff(), 0.0)
    minus_dm = np.where((-data["Low"].diff() > data["High"].diff()) & (-data["Low"].diff() > 0), -data["Low"].diff(), 0.0)
    plus_di = 100.0 * (pd.Series(plus_dm, index=data.index).rolling(14).mean() / (data["ATR"] + 1e-9))
    minus_di = 100.0 * (pd.Series(minus_dm, index=data.index).rolling(14).mean() / (data["ATR"] + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100.0
    data["ADX"] = dx.rolling(14).mean()
    data["DI_Spread"] = plus_di - minus_di

    macd_line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    data["MACD_Hist"] = (macd_line - macd_line.ewm(span=9, adjust=False).mean()) / (data["ATR"] + 1e-9)

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    data["Dist_EMA20"] = (c - ema20) / (data["ATR"] + 1e-9)
    data["Dist_EMA50"] = (c - ema50) / (data["ATR"] + 1e-9)
    data["Dist_EMA200"] = (c - ema200) / (data["ATR"] + 1e-9)

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
    return data.dropna()
def _validate_and_clean_master_json(models_dir: str):
    candidates = [
        os.path.join(models_dir, "PAPER", "UNIFIED_PORTFOLIO_MASTER.json"),
        os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json"),
        os.path.join(models_dir, "UNIFIED_PORTFOLIO_MASTER.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    json.load(f)
            except Exception:
                try:
                    os.remove(path)
                except Exception:
                    pass
def _safe_load_json(path: str, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    for _ in range(3):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            time.sleep(0.15)
        except Exception:
            break
    return default

BYBIT_HOSTS_MAINNET = ["https://api.bybit.com", "https://api.bytick.com"]
BYBIT_HOSTS_DEMO = ["https://api-demo.bybit.com", "https://api.bytick.com"]
_BOOSTER_CACHE = {}


def _hosts_for(endpoint: str, is_paper: bool = False):
    if is_paper or str(endpoint).lower() in ("testnet", "demo"):
        return list(_FAST_DEMO_HOSTS)
    return list(_FAST_MAINNET_HOSTS)


def _public_bybit_get(path: str, params: dict, endpoint: str = "Mainnet", is_paper: bool = False, timeout: float = 6.0) -> dict:
    last_err = None
    for base in _hosts_for(endpoint, is_paper=is_paper):
        try:
            r = _HTTP_SESSION.get(f"{base}{path}", params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return {}


def _sanitize_bom_if_needed(file_path: str):
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as bf:
                raw_bytes = bf.read()
            if raw_bytes.startswith(b"\xef\xbb\xbf"):
                with open(file_path, "wb") as wf:
                    wf.write(raw_bytes[3:])
    except Exception:
        pass

def _load_synced_module(module_name: str, file_path: str):
    _sanitize_bom_if_needed(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Synced bot file not found at {file_path}. Please sync from GitHub in Settings.")
    parent_dir = os.path.dirname(file_path)
    shared_dir = os.path.join(os.path.dirname(parent_dir), "synced_modules")
    for p_candidate in (shared_dir, parent_dir):
        if os.path.isdir(p_candidate) and p_candidate not in sys.path:
            sys.path.insert(0, p_candidate)
    if module_name in sys.modules:
        del sys.modules[module_name]
    _start_keepalive_thread()
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    try:
        mod.requests = _PooledRequestsProxy(requests)
        if getattr(mod, "Session", None) is requests.Session:
            mod.Session = lambda *a, **kw: _NonClosingSessionProxy(_HTTP_SESSION)
    except Exception:
        pass
    return mod


def _is_valid_xgb_model(model_dict) -> bool:
    return isinstance(model_dict, dict) and "learner" in model_dict and isinstance(model_dict.get("learner"), dict)


def _get_cached_booster(cache_key: str, model_dict: dict, work_dir: str = None):
    if not _is_valid_xgb_model(model_dict):
        return None
    if cache_key in _BOOSTER_CACHE:
        return _BOOSTER_CACHE[cache_key]
    if len(_BOOSTER_CACHE) >= 20:
        _BOOSTER_CACHE.pop(next(iter(_BOOSTER_CACHE)), None)
    booster = xgb.Booster()
    base_tmp = work_dir if (work_dir and os.path.isdir(work_dir)) else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_tmp, exist_ok=True)
    tmp_path = os.path.join(base_tmp, f"_radar_tmp_{cache_key}_{os.getpid()}.json")
    try:
        with open(tmp_path, "w", encoding="utf-8") as tmp:
            json.dump(model_dict, tmp)
        booster.load_model(tmp_path)
        _BOOSTER_CACHE[cache_key] = booster
        return booster
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _compute_radar_signals(live_mod, models_dir: str, mode_name: str, endpoint: str, crypto_en: bool, forex_en: bool) -> tuple:
    import numpy as np
    if mode_name == "LIVE":
        state_p = os.path.join(os.path.dirname(models_dir), "bot_data", "LIVE_ACCOUNT", "live_engine_state.json")
        if os.path.exists(state_p):
            try:
                with open(state_p, "r", encoding="utf-8-sig") as sf:
                    st_data = json.load(sf)
                st_radar = st_data.get("radar_signals")
                if isinstance(st_radar, list):
                    p_m = os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json")
                    ver_str = "LIVE_ML"
                    if os.path.exists(p_m):
                        with open(p_m, "r", encoding="utf-8-sig") as mf:
                            ver_str = json.load(mf).get("version", "LIVE_ML")
                    return st_radar, True, ver_str
            except Exception:
                pass
    p1 = os.path.join(models_dir, mode_name, "UNIFIED_PORTFOLIO_MASTER.json")
    p2 = os.path.join(models_dir, "UNIFIED_PORTFOLIO_MASTER.json")
    master_path = p1 if os.path.exists(p1) else (p2 if mode_name == "LIVE" and os.path.exists(p2) else "")
    if not master_path or not os.path.exists(master_path):
        return [], False, "NONE"

    try:
        with open(master_path, "r", encoding="utf-8") as f:
            master = json.load(f)
    except Exception:
        return [], False, "NONE"

    portfolio = master.get("portfolio", {})
    version = master.get("version", f"{mode_name}_PORTFOLIO_{len(portfolio)}C")
    features = master.get("features", _RADAR_DEFAULT_FEATURES)
    if not portfolio:
        return [], False, version

    tf_map = {"15m": "15", "30m": "30", "1h": "60", "4h": "240"}
    signals = []
    for sym, entry in portfolio.items():
        if not isinstance(entry, dict):
            continue
        is_fx = _radar_is_forex(sym)
        mkt = entry.get("market", "FOREX" if is_fx else "CRYPTO").upper()
        if (mkt == "CRYPTO" and not crypto_en) or (mkt == "FOREX" and not forex_en):
            continue

        cfg = entry.get("config", {})
        bybit_sym = (cfg.get("bybit_symbol") or sym.replace("/", "")).upper()
        tf_str = (cfg.get("timeframe") or entry.get("timeframe") or "1h").lower()
        interval = tf_map.get(tf_str, "60")
        disp_sym = sym if "/" in sym else (f"{sym[:-4]}/USDT" if sym.endswith("USDT") else sym)

        try:
            df_raw = _radar_fetch_candles(bybit_sym, interval=interval, limit=220)
            if len(df_raw) < 35:
                continue

            # AI Copilot Technical & Momentum Telemetry (works even on newer listings < 200 bars)
            c = df_raw["Close"]
            delta = c.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rsi_val = float((100.0 - (100.0 / (1.0 + (gain / (loss + 1e-9))))).iloc[-1])
            ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
            last_c = float(c.iloc[-1])
            ret_3 = float(((last_c - float(c.iloc[-4])) / (float(c.iloc[-4]) + 1e-9)) * 100.0) if len(c) >= 5 else 0.0
            vol_s = float((df_raw["Volume"].iloc[-1] / (df_raw["Volume"].rolling(20).mean().iloc[-1] + 1e-9))) if len(df_raw) >= 20 else 1.0

            ai_long_score = 0.50 + ((ema9 - ema21) / (ema21 + 1e-9)) * 14.0 + (ret_3 * 0.03) + ((rsi_val - 50.0) * 0.004) + (min(vol_s - 1.0, 1.5) * 0.04 if ret_3 > 0 else 0.0)
            ai_short_score = 0.50 + ((ema21 - ema9) / (ema21 + 1e-9)) * 14.0 - (ret_3 * 0.03) + ((50.0 - rsi_val) * 0.004) + (min(vol_s - 1.0, 1.5) * 0.04 if ret_3 < 0 else 0.0)
            ai_long_prob = float(np.clip(ai_long_score, 0.12, 0.94))
            ai_short_prob = float(np.clip(ai_short_score, 0.12, 0.94))

            # Blend with XGBoost Booster if 205+ bars and valid booster weights exist
            xgb_long_prob = None
            xgb_short_prob = None
            trend_4h = 1 if ema9 >= ema21 else -1

            if len(df_raw) >= 205:
                df = _radar_build_features(df_raw)
                if not df.empty:
                    if interval == "240":
                        trend_4h = int(df.iloc[-1]["Supertrend"])
                    models_map = entry.get("models", {})
                    long_data = models_map.get("long") or models_map.get("long_model")
                    short_data = models_map.get("short") or models_map.get("short_model")
                    if _is_valid_xgb_model(long_data) and _is_valid_xgb_model(short_data):
                        b_long = _get_cached_booster(f"{mode_name}_{bybit_sym}_long", long_data, models_dir)
                        b_short = _get_cached_booster(f"{mode_name}_{bybit_sym}_short", short_data, models_dir)
                        if b_long is not None and b_short is not None:
                            dmat = xgb.DMatrix(df.iloc[[-1]][features].to_numpy(dtype="float32"))
                            xgb_long_prob = float(b_long.predict(dmat)[0])
                            xgb_short_prob = float(b_short.predict(dmat)[0])
                            del dmat
                    del df

            if interval != "240":
                raw_4h = _radar_fetch_candles(bybit_sym, interval="240", limit=60)
                if len(raw_4h) >= 25:
                    c4 = raw_4h["Close"]
                    e9_4 = float(c4.ewm(span=9, adjust=False).mean().iloc[-1])
                    e21_4 = float(c4.ewm(span=21, adjust=False).mean().iloc[-1])
                    trend_4h = 1 if e9_4 >= e21_4 else -1
                del raw_4h

            if xgb_long_prob is not None and xgb_short_prob is not None:
                # Calibrate raw XGBoost probability (typical 0.18-0.55 range) onto intuitive 35%-95% conviction scale + AI Copilot telemetry
                xgb_l_scaled = float(np.clip(0.32 + xgb_long_prob * 1.15, 0.15, 0.96))
                xgb_s_scaled = float(np.clip(0.32 + xgb_short_prob * 1.15, 0.15, 0.96))
                final_l = (0.65 * xgb_l_scaled) + (0.35 * ai_long_prob)
                final_s = (0.65 * xgb_s_scaled) + (0.35 * ai_short_prob)
            else:
                final_l = ai_long_prob
                final_s = ai_short_prob

            direction = "LONG" if final_l >= final_s else "SHORT"
            raw_prob = final_l if direction == "LONG" else final_s
            conf_pct = int(min(74, max(18, round(raw_prob * 100))))
            mtf_ok = (direction == "LONG" and trend_4h == 1) or (direction == "SHORT" and trend_4h == -1)

            signals.append({
                "symbol": disp_sym,
                "direction": direction,
                "confidence_pct": conf_pct,
                "mtf_aligned": bool(mtf_ok),
            })
            del df_raw
        except Exception:
            continue

    gc.collect()
    signals.sort(key=lambda x: x["confidence_pct"], reverse=True)
    return signals, True, version

def execute_single_paper_coin_training(paper_dir: str, models_dir: str, config_json: str) -> str:
    bot_path = os.path.join(paper_dir, "paper_bot.py")
    paper_mod = _load_synced_module("synced_paper_bot", bot_path)
    cfg = json.loads(config_json)
    sym = cfg.get("symbol", "")
    all_coins = cfg.get("all_coins", [sym])
    is_first = bool(cfg.get("is_first", False))

    paper_master_file = os.path.join(models_dir, "PAPER", "UNIFIED_PORTFOLIO_MASTER.json")
    paper_res_file = os.path.join(paper_dir, "paper_trained_results.json")
    prev_portfolio = {}
    prev_results = {}
    if not is_first:
        if os.path.exists(paper_master_file):
            try:
                with open(paper_master_file, "r", encoding="utf-8") as f:
                    prev_portfolio = json.load(f).get("portfolio", {})
            except Exception:
                prev_portfolio = {}
        if os.path.exists(paper_res_file):
            try:
                with open(paper_res_file, "r", encoding="utf-8") as f:
                    prev_results = json.load(f).get("coins", {})
            except Exception:
                prev_results = {}

    trainer = paper_mod.PaperStrategyTrainerAndTester(paper_dir=paper_dir, models_dir=models_dir)
    results = trainer.train_and_evaluate_paper_coins(
        added_coins=[sym],
        active_strategies=cfg.get("strategies", []),
        timeframes=cfg.get("timeframes", ["15m", "30m", "1h", "4h"]),
        leverage=int(cfg.get("leverage", 10)),
        crypto_enabled=bool(cfg.get("crypto_enabled", True)),
        forex_enabled=bool(cfg.get("forex_enabled", True)),
    )

    # Merge previously trained coins in this batch so UNIFIED_PORTFOLIO_MASTER.json holds ALL coins
    if prev_portfolio:
        merged_port = dict(prev_portfolio)
        if os.path.exists(paper_master_file):
            with open(paper_master_file, "r", encoding="utf-8") as f:
                cur_m = json.load(f)
            merged_port.update(cur_m.get("portfolio", {}))
            wrs = [v.get("win_rate_pct", 0.0) for v in merged_port.values() if isinstance(v, dict)]
            pnls = [v.get("net_pnl_usdt", 0.0) for v in merged_port.values() if isinstance(v, dict)]
            cur_m["version"] = f"UNIFIED_PAPER_{len(merged_port)}C_135D"
            cur_m["model_accuracy_pct"] = round(sum(wrs) / len(wrs), 2) if wrs else 0.0
            cur_m["total_net_pnl"] = round(sum(pnls), 2)
            cur_m["portfolio"] = merged_port
            with open(paper_master_file, "w", encoding="utf-8") as f:
                json.dump(cur_m, f, indent=2)

    if prev_results and os.path.exists(paper_res_file):
        with open(paper_res_file, "r", encoding="utf-8") as f:
            cur_r = json.load(f)
        merged_r = dict(prev_results)
        merged_r.update(cur_r.get("coins", {}))
        cur_r["coins"] = merged_r
        with open(paper_res_file, "w", encoding="utf-8") as f:
            json.dump(cur_r, f, indent=2)

    comparisons = trainer.auto_compare_with_live(results)
    return json.dumps({
        "ok": True,
        "symbol": sym,
        "trained_count": len(results),
        "comparisons": comparisons,
    })


def execute_paper_training(paper_dir: str, models_dir: str, config_json: str) -> str:
    bot_path = os.path.join(paper_dir, "paper_bot.py")
    paper_mod = _load_synced_module("synced_paper_bot", bot_path)
    cfg = json.loads(config_json)

    trainer = paper_mod.PaperStrategyTrainerAndTester(paper_dir=paper_dir, models_dir=models_dir)
    results = trainer.train_and_evaluate_paper_coins(
        added_coins=cfg.get("coins", []),
        active_strategies=cfg.get("strategies", []),
        timeframes=cfg.get("timeframes", ["1h", "4h"]),
        leverage=int(cfg.get("leverage", 10)),
        crypto_enabled=bool(cfg.get("crypto_enabled", True)),
        forex_enabled=bool(cfg.get("forex_enabled", True)),
    )
    comparisons = trainer.auto_compare_with_live(results)
    return json.dumps({
        "ok": True,
        "trained_count": len(results),
        "comparisons": comparisons,
    })


def promote_coin_with_paper_bot(paper_dir: str, models_dir: str, symbol: str) -> str:
    bot_path = os.path.join(paper_dir, "paper_bot.py")
    paper_mod = _load_synced_module("synced_paper_bot", bot_path)
    trainer = paper_mod.PaperStrategyTrainerAndTester(paper_dir=paper_dir, models_dir=models_dir)
    ok = trainer.move_coin_to_live_model(symbol.upper())
    return json.dumps({"ok": ok, "promoted": symbol.upper()})


def remove_coin_with_paper_bot(paper_dir: str, models_dir: str, symbol: str) -> str:
    bot_path = os.path.join(paper_dir, "paper_bot.py")
    paper_mod = _load_synced_module("synced_paper_bot", bot_path)
    trainer = paper_mod.PaperStrategyTrainerAndTester(paper_dir=paper_dir, models_dir=models_dir)
    ok = trainer.remove_coin_from_live_model(symbol.upper())
    return json.dumps({"ok": ok, "removed": symbol.upper()})


def execute_paper_simulation_tick(paper_dir: str, models_dir: str, config_json: str) -> str:
    """
    Runs Paper Account forward execution directly against Bybit Demo Trading (api-demo.bybit.com)
    using the Paper Account's Demo API keys and PAPER/UNIFIED_PORTFOLIO_MASTER.json.
    """
    os.makedirs(paper_dir, exist_ok=True)
    cfg = json.loads(config_json)
    api_key = cfg.get("api_key", "")
    api_secret = cfg.get("api_secret", "")
    crypto_en = bool(cfg.get("crypto_enabled", True))
    forex_en = bool(cfg.get("forex_enabled", True))

    live_bot_path = os.path.join(os.path.dirname(paper_dir), "LIVE_ACCOUNT", "live_bot.py")
    if not os.path.exists(live_bot_path):
        raise FileNotFoundError("live_bot.py not synced yet. Sync both PAPER and LIVE bots in Settings.")
    live_mod = _load_synced_module("synced_live_bot_for_paper", live_bot_path)

    if not api_key or not api_secret:
        return json.dumps({"ok": False, "network_ok": False, "error": "Missing Paper Bybit Demo API Key / Secret in Settings."})

    # Point master model lookup to PAPER/UNIFIED_PORTFOLIO_MASTER.json on Bybit Demo Trading
    paper_models_root = os.path.join(models_dir, "PAPER")
    os.makedirs(paper_models_root, exist_ok=True)
    bot = live_mod.LiveAccountTradingBot(
        api_key=api_key,
        api_secret=api_secret,
        models_dir=paper_models_root,
        live_dir=paper_dir,
        endpoint="Demo",
    )
    bot.master_path = os.path.join(paper_models_root, "UNIFIED_PORTFOLIO_MASTER.json")

    if os.path.exists(bot.master_path):
        bot.run_live_cycle(
            leverage=int(cfg.get("leverage", 10)),
            risk_per_trade_pct=_safe_float(cfg.get("risk_per_trade_pct", 1.5)),
            max_open_positions=int(cfg.get("max_open_positions", 3)),
            crypto_enabled=crypto_en,
            forex_enabled=forex_en,
            live_orders_armed=True,
            mtf_mode=str(cfg.get("mtf_mode", "STRICT")),
            btc_macro_guard=bool(cfg.get("btc_macro_guard", True)),
            daily_drawdown_pct=_safe_float(cfg.get("daily_drawdown_pct", 0.0)),
            max_daily_loss_pct=_safe_float(cfg.get("max_daily_loss_pct", 3.0)),
        bot_logs_enabled=bool(cfg.get("bot_logs_enabled", True)),
        )

    snap = json.loads(fetch_live_snapshot(paper_dir, models_dir, api_key, api_secret, "Demo"))
    signals, ml_connected, ml_version = _compute_radar_signals(live_mod, models_dir, "PAPER", "Demo", crypto_en, forex_en)
    snap["account_mode"] = "PAPER"
    snap["network_ok"] = snap.get("ok", False)
    snap["ml_connected"] = ml_connected
    snap["ml_model_version"] = ml_version
    snap["signals"] = signals
    return json.dumps(snap)



_LAST_DELIST_CHECK_TS = 0.0

def _auto_prune_delisted_bybit_contracts(models_dir: str, paper_dir: str = None):
    """Automatically removes delisted Bybit Linear contracts (status != Trading) from Live & Paper master models."""
    global _LAST_DELIST_CHECK_TS
    now_s = time.time()
    if (now_s - _LAST_DELIST_CHECK_TS) < 43200.0:  # Check every 12 hours
        return []
    try:
        resp = _public_bybit_get("/v5/market/instruments-info", {"category": "linear", "limit": 1000}, endpoint="Mainnet", is_paper=False, timeout=6.0)
        items = resp.get("result", {}).get("list", [])
        if not items:
            return []
        active_syms = {
            str(it.get("symbol", "")).upper()
            for it in items
            if str(it.get("status", "")).upper() == "TRADING"
        }
        if len(active_syms) < 50:
            return []
        _LAST_DELIST_CHECK_TS = now_s
    except Exception:
        return []

    removed_all = []
    master_paths = [
        os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json"),
        os.path.join(models_dir, "PAPER", "UNIFIED_PORTFOLIO_MASTER.json"),
    ]
    for mp in master_paths:
        if not os.path.exists(mp):
            continue
        try:
            with open(mp, "r", encoding="utf-8-sig") as f:
                mdata = json.load(f)
            port = mdata.get("portfolio", {})
            if not isinstance(port, dict):
                continue
            to_del = []
            for sym, entry in port.items():
                if _radar_is_forex(sym):
                    continue
                bsym = str((entry.get("config", {}) or {}).get("bybit_symbol") or sym.replace("/", "")).upper()
                if bsym and bsym not in active_syms:
                    to_del.append(sym)
            if to_del:
                for sym in to_del:
                    port.pop(sym, None)
                    if sym not in removed_all:
                        removed_all.append(sym)
                wrs = [float(v.get("win_rate_pct", 0.0)) for v in port.values() if isinstance(v, dict)]
                pnls = [float(v.get("net_pnl_usdt", v.get("net_pnl", 0.0))) for v in port.values() if isinstance(v, dict)]
                prefix = "LIVE_PORTFOLIO" if "LIVE" in mp else "UNIFIED_PAPER"
                suffix = "" if "LIVE" in mp else "_135D"
                mdata["version"] = f"{prefix}_{len(port)}C{suffix}"
                mdata["model_accuracy_pct"] = round(sum(wrs) / len(wrs), 2) if wrs else 0.0
                mdata["total_net_pnl"] = round(sum(pnls), 2)
                with open(mp, "w", encoding="utf-8") as wf:
                    json.dump(mdata, wf, indent=2)
        except Exception:
            continue

    if removed_all and paper_dir:
        cfg_p = os.path.join(paper_dir, "paper_runtime_config.json")
        if os.path.exists(cfg_p):
            try:
                with open(cfg_p, "r", encoding="utf-8-sig") as f:
                    pcfg = json.load(f)
                old_c = pcfg.get("coins", [])
                clean_removed = {s.replace("/", "").upper() for s in removed_all}
                new_c = [c for c in old_c if c.replace("/", "").upper() not in clean_removed]
                if len(new_c) != len(old_c):
                    pcfg["coins"] = new_c
                    with open(cfg_p, "w", encoding="utf-8") as wf:
                        json.dump(pcfg, wf, indent=2)
            except Exception:
                pass
    return removed_all


def execute_live_cycle(live_dir: str, models_dir: str, config_json: str) -> str:
    _auto_prune_delisted_bybit_contracts(models_dir, os.path.join(os.path.dirname(live_dir), 'PAPER'))
    bot_path = os.path.join(live_dir, "live_bot.py")
    live_mod = _load_synced_module("synced_live_bot", bot_path)
    cfg = json.loads(config_json)

    api_key = cfg.get("api_key", "")
    api_secret = cfg.get("api_secret", "")
    endpoint = cfg.get("endpoint", "Mainnet")
    crypto_en = bool(cfg.get("crypto_enabled", True))
    forex_en = bool(cfg.get("forex_enabled", False))
    if not api_key or not api_secret:
        return json.dumps({"ok": False, "network_ok": False, "error": "Missing Live Bybit Mainnet API credentials"})

    live_master_p1 = os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json")
    live_master_p2 = os.path.join(models_dir, "UNIFIED_PORTFOLIO_MASTER.json")
    active_live_master = live_master_p1 if os.path.exists(live_master_p1) else live_master_p2
    live_port = {}
    if os.path.exists(active_live_master):
        try:
            with open(active_live_master, "r", encoding="utf-8") as mf:
                live_port = json.load(mf).get("portfolio", {})
        except Exception:
            live_port = {}
    if not live_port:
        try:
            snap = fetch_live_snapshot(api_key, api_secret, endpoint)
        except Exception:
            snap = {}
        return json.dumps({
            "ok": True,
            "account_mode": "LIVE",
            "latency_ms": snap.get("latency_ms", 0),
            "total_equity": _safe_float(snap.get("total_equity", 0.0)),
            "wallet_balance": _safe_float(snap.get("wallet_balance", 0.0)),
            "unrealized_pnl": _safe_float(snap.get("unrealized_pnl", 0.0)),
            "open_positions": snap.get("open_positions", []),
            "closed_summary": snap.get("closed_summary", []),
            "signals": [],
            "ml_connected": False,
            "ml_model_version": "NONE (0 Coins)",
            "actions_taken": [],
            "diagnostics": ["No coins in Live ML Model yet — promote trained coins from Paper."],
        })

    bot = live_mod.LiveAccountTradingBot(
        api_key=api_key,
        api_secret=api_secret,
        models_dir=models_dir,
        live_dir=live_dir,
        endpoint=endpoint,
    )
    cycle_out = bot.run_live_cycle(
        leverage=int(cfg.get("leverage", 10)),
        risk_per_trade_pct=_safe_float(cfg.get("risk_per_trade_pct", 1.5)),
        max_open_positions=int(cfg.get("max_open_positions", 2)),
        crypto_enabled=crypto_en,
        forex_enabled=forex_en,
        live_orders_armed=bool(cfg.get("live_orders_armed", False)),
        mtf_mode=str(cfg.get("mtf_mode", "STRICT")),
        btc_macro_guard=bool(cfg.get("btc_macro_guard", True)),
        daily_drawdown_pct=_safe_float(cfg.get("daily_drawdown_pct", 0.0)),
        max_daily_loss_pct=_safe_float(cfg.get("max_daily_loss_pct", 3.0)),
        bot_logs_enabled=bool(cfg.get("bot_logs_enabled", True)),
    ) or {}
    rej_list = cycle_out.get("order_rejections") or []
    if rej_list:
        return json.dumps({
            "ok": False,
            "network_ok": True,
            "error": " | ".join(str(x) for x in rej_list[:3]),
            "order_rejections": rej_list,
            "trade_blocks": cycle_out.get("trade_blocks") or [],
        })
    snap = json.loads(fetch_live_snapshot(live_dir, models_dir, api_key, api_secret, endpoint))
    if cycle_out.get("radar_signals"):
        signals = cycle_out["radar_signals"]
        ml_connected = True
        ml_version = cycle_out.get("ml_model_version", "LIVE_ML")
    else:
        signals, ml_connected, ml_version = _compute_radar_signals(live_mod, models_dir, "LIVE", endpoint, crypto_en, forex_en)
    snap["network_ok"] = snap.get("ok", False)
    snap["ml_connected"] = ml_connected
    snap["ml_model_version"] = ml_version
    snap["signals"] = signals
    return json.dumps(snap)


def execute_live_kill_switch(live_dir: str, models_dir: str, config_json: str) -> str:
    bot_path = os.path.join(live_dir, "live_bot.py")
    live_mod = _load_synced_module("synced_live_bot", bot_path)
    cfg = json.loads(config_json)
    bot = live_mod.LiveAccountTradingBot(
        api_key=cfg.get("api_key", ""),
        api_secret=cfg.get("api_secret", ""),
        models_dir=models_dir,
        live_dir=live_dir,
        endpoint=cfg.get("endpoint", "Mainnet"),
    )
    closed = bot.emergency_close_all()
    return json.dumps({"ok": True, "closed": closed})


def close_single_live_position(live_dir: str, models_dir: str, config_json: str, symbol: str) -> str:
    bot_path = os.path.join(live_dir, "live_bot.py")
    live_mod = _load_synced_module("synced_live_bot", bot_path)
    cfg = json.loads(config_json)
    bot = live_mod.LiveAccountTradingBot(
        api_key=cfg.get("api_key", ""),
        api_secret=cfg.get("api_secret", ""),
        models_dir=models_dir,
        live_dir=live_dir,
        endpoint=cfg.get("endpoint", "Mainnet"),
    )
    clean_sym = symbol.replace("/", "").upper()
    state = bot._load_state()
    open_positions = bot.sync_positions_and_trail_stops(state)
    closed_ok = False
    for pos in open_positions:
        if pos["symbol"].upper() == clean_sym:
            close_side = "Sell" if pos["side"] == "LONG" else "Buy"
            res = bot._bybit_request("POST", "/v5/order/create", body={
                "category": bot.category,
                "symbol": clean_sym,
                "side": close_side,
                "orderType": "Market",
                "qty": str(pos["qty"]),
                "timeInForce": "IOC",
                "reduceOnly": True,
                "positionIdx": 0,
            })
            closed_ok = (res.get("retCode") == 0)
            break
    bot._save_state(state)
    return json.dumps({"ok": closed_ok, "symbol": symbol})


def _bybit_signed_get(api_key: str, api_secret: str, endpoint: str, path: str, query: str) -> dict:
    last_err = None
    for base in _hosts_for(endpoint):
        try:
            ts = str(int(time.time() * 1000))
            recv = "5000"
            raw = f"{ts}{api_key}{recv}{query}"
            sig = hmac.new(api_secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()
            headers = {
                "X-BAPI-API-KEY": api_key,
                "X-BAPI-SIGN": sig,
                "X-BAPI-TIMESTAMP": ts,
                "X-BAPI-RECV-WINDOW": recv,
            }
            resp = _HTTP_SESSION.get(f"{base}{path}?{query}", headers=headers, timeout=8)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return {}


def test_bybit_wallet(api_key: str, api_secret: str, endpoint: str) -> str:
    if not api_key or not api_secret:
        return json.dumps({"ok": False, "error": "API Key and Secret are required."})
    try:
        _start_keepalive_thread()
        _benchmark_and_warm_bybit(endpoint, force=False)
        t0 = time.perf_counter()
        data = _bybit_signed_get(api_key, api_secret, endpoint, "/v5/account/wallet-balance", "accountType=UNIFIED&coin=USDT")
        latency = max(12, int((time.perf_counter() - t0) * 1000))
        if data.get("retCode") != 0:
            return json.dumps({"ok": False, "error": f"Bybit ({data.get('retCode')}): {data.get('retMsg')}"})
        lst = data.get("result", {}).get("list", [])
        first = lst[0] if lst else {}
        eq = _safe_float(first.get("totalEquity", 0) or 0)
        wal = _safe_float(first.get("totalWalletBalance", 0) or eq)
        avail = _safe_float(first.get("totalAvailableBalance", 0) or wal)
        return json.dumps({
            "ok": True,
            "latency_ms": latency,
            "total_equity": eq,
            "wallet_balance": wal,
            "available_margin": avail,
        })
    except Exception as e:
        err_s = str(e)
        if "NameResolutionError" in err_s or "Errno 7" in err_s:
            return json.dumps({"ok": False, "error": "Network/DNS unreachable for api.bytick.com ? check mobile data/VPN."})
        return json.dumps({"ok": False, "error": err_s})


def fetch_live_snapshot(live_dir: str, models_dir: str, api_key: str, api_secret: str, endpoint: str) -> str:
    _start_keepalive_thread()
    _benchmark_and_warm_bybit(endpoint, force=False)

    # Check Exchange Live Orders toggle (live_orders_armed) before launching parallel queries
    live_orders_armed = True
    try:
        rcfg_file = os.path.join(live_dir, "live_runtime_config.json")
        if os.path.exists(rcfg_file):
            with open(rcfg_file, "r", encoding="utf-8-sig") as rf:
                live_orders_armed = bool(json.load(rf).get("live_orders_armed", True))
    except Exception:
        live_orders_armed = True

    def _safe_pos_call():
        try:
            return _bybit_signed_get(api_key, api_secret, endpoint, "/v5/position/list", "category=linear&settleCoin=USDT")
        except Exception:
            return {}

    def _safe_pnl_call():
        if not live_orders_armed:
            return {}
        now_t = time.time()
        cache_entry = getattr(fetch_live_snapshot, "_pnl_cache", None)
        if cache_entry and (now_t - cache_entry[0]) < 15.0:
            return cache_entry[1]
        try:
            res_p = _bybit_signed_get(api_key, api_secret, endpoint, "/v5/position/closed-pnl", "category=linear&limit=50")
            fetch_live_snapshot._pnl_cache = (now_t, res_p)
            return res_p
        except Exception:
            return cache_entry[1] if cache_entry else {}

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_w = ex.submit(test_bybit_wallet, api_key, api_secret, endpoint)
        fut_p = ex.submit(_safe_pos_call)
        fut_c = ex.submit(_safe_pnl_call)
        w_res = json.loads(fut_w.result())
        if not w_res.get("ok"):
            return json.dumps(w_res)
        pos_data = fut_p.result()
        pnl_data = fut_c.result()

    positions = []
    unreal_sum = 0.0
    try:
        if pos_data.get("retCode") == 0:
            for item in pos_data.get("result", {}).get("list", []):
                size = _safe_float(item.get("size", 0) or 0)
                if size <= 0:
                    continue
                raw_sym = item.get("symbol", "")
                disp_sym = f"{raw_sym[:-4]}/USDT" if raw_sym.endswith("USDT") else raw_sym
                side = "LONG" if item.get("side", "").lower() == "buy" else "SHORT"
                u_pnl = _safe_float(item.get("unrealisedPnl", 0) or 0)
                unreal_sum += u_pnl
                positions.append({
                    "symbol": disp_sym,
                    "side": side,
                    "entry_price": _safe_float(item.get("avgPrice", 0) or 0),
                    "mark_price": _safe_float(item.get("markPrice", 0) or 0),
                    "leverage": int(_safe_float(item.get("leverage", 10) or 10)),
                    "qty": size,
                    "margin": _safe_float(item.get("positionIM", 0) or 0),
                    "unrealized_pnl": round(u_pnl, 4),
                    "stop_loss": _safe_float(item.get("stopLoss", 0) or 0),
                    "take_profit": _safe_float(item.get("takeProfit", 0) or 0),
                })
    except Exception:
        pass

    closed_summary = []
    if live_orders_armed:
        # ACTUAL LIVE EXCHANGE REPORTS ONLY (never mixed with Dry-Run)
        try:
            if pnl_data.get("retCode") == 0:
                by_sym = {}
                reset_ts_ms = 0
                reset_file = os.path.join(live_dir, "reports_reset_ts.txt")
                if os.path.exists(reset_file):
                    try:
                        with open(reset_file, "r", encoding="utf-8") as rf:
                            reset_ts_ms = int(rf.read().strip() or "0")
                    except Exception:
                        reset_ts_ms = 0

                merged_positions = []
                for row in pnl_data.get("result", {}).get("list", []):
                    updated_ms = int(row.get("updatedTime", 0) or row.get("createdTime", 0) or 0)
                    if reset_ts_ms > 0 and updated_ms > 0 and updated_ms <= reset_ts_ms:
                        continue
                    raw_sym = row.get("symbol", "")
                    disp_sym = f"{raw_sym[:-4]}/USDT" if raw_sym.endswith("USDT") else raw_sym
                    close_side = str(row.get("side", "")).upper()
                    avg_entry = _safe_float(row.get("avgEntryPrice", 0) or 0)
                    cpnl = _safe_float(row.get("closedPnl", 0) or 0)

                    matched_bucket = None
                    for b in merged_positions:
                        if b["symbol"] == disp_sym and b["side"] == close_side:
                            same_entry = (avg_entry > 0.0 and b["entry"] > 0.0 and abs(avg_entry - b["entry"]) / b["entry"] <= 0.0003)
                            within_24h = (updated_ms > 0 and b["ts"] > 0 and abs(updated_ms - b["ts"]) <= 86_400_000)
                            if same_entry and within_24h:
                                matched_bucket = b
                                break
                    if matched_bucket is not None:
                        matched_bucket["pnl"] = round(matched_bucket["pnl"] + cpnl, 4)
                    else:
                        merged_positions.append({
                            "symbol": disp_sym,
                            "side": close_side,
                            "entry": avg_entry,
                            "ts": updated_ms,
                            "pnl": round(cpnl, 4),
                        })

                for mp in merged_positions:
                    disp_sym = mp["symbol"]
                    cpnl = mp["pnl"]
                    rec = by_sym.setdefault(disp_sym, {"symbol": disp_sym, "strategy": "XGBoost MTF", "pnl": 0.0, "realized_pnl": 0.0, "wins": 0, "losses": 0, "trades": 0, "win_rate_pct": 0.0})
                    rec["pnl"] = round(rec["pnl"] + cpnl, 4)
                    rec["realized_pnl"] = rec["pnl"]
                    if cpnl >= 0:
                        rec["wins"] += 1
                    else:
                        rec["losses"] += 1
                    rec["trades"] = rec["wins"] + rec["losses"]
                    rec["win_rate_pct"] = round((rec["wins"] / max(rec["trades"], 1)) * 100.0, 1)
                closed_summary = [v for v in by_sym.values() if v.get("trades", 0) > 0]
        except Exception:
            pass
    else:
        # VIRTUAL DRY-RUN REPORTS ONLY (auto-switched when Exchange Live Orders is OFF)
        try:
            reset_ts_ms = 0
            reset_file = os.path.join(live_dir, "reports_reset_ts.txt")
            if os.path.exists(reset_file):
                try:
                    with open(reset_file, "r", encoding="utf-8") as rf:
                        reset_ts_ms = int(rf.read().strip() or "0")
                except Exception:
                    reset_ts_ms = 0

            hist_file = os.path.join(live_dir, "live_reports_history.json")
            if os.path.exists(hist_file):
                with open(hist_file, "r", encoding="utf-8-sig") as hf:
                    h_obj = json.load(hf)
                v_trades = [t for t in (h_obj.get("trades") or []) if t.get("virtual")]
                by_v = {}
                for vt in v_trades:
                    closed_ms = int(vt.get("closed_at_ms", 0) or 0)
                    if reset_ts_ms > 0 and closed_ms > 0 and closed_ms <= reset_ts_ms:
                        continue
                    sym = f"{vt.get('symbol', '')} [DRY]"
                    pnl = _safe_float(vt.get("net_pnl", 0.0))
                    rec = by_v.setdefault(sym, {"symbol": sym, "strategy": "Dry-Run Sim", "pnl": 0.0, "realized_pnl": 0.0, "wins": 0, "losses": 0, "trades": 0, "win_rate_pct": 0.0})
                    rec["pnl"] = round(rec["pnl"] + pnl, 4)
                    rec["realized_pnl"] = rec["pnl"]
                    if pnl >= 0:
                        rec["wins"] += 1
                    else:
                        rec["losses"] += 1
                    rec["trades"] = rec["wins"] + rec["losses"]
                    rec["win_rate_pct"] = round((rec["wins"] / max(rec["trades"], 1)) * 100.0, 1)
                closed_summary = [v for v in by_v.values() if v.get("trades", 0) > 0]
        except Exception:
            pass

    try:
        import radar_engine
        signals, ml_connected, ml_version = radar_engine.get_fast_radar_signals(
            models_dir=models_dir,
            live_dir=live_dir,
            mode_name="LIVE",
            endpoint=endpoint,
            crypto_en=True,
            forex_en=True,
            ttl_sec=20.0,
        )
    except Exception:
        signals, ml_connected, ml_version = [], False, "LIVE_ML"

    # Show Virtual Dry-Run open positions ONLY when Exchange Live Orders is OFF
    if not live_orders_armed and not positions:
        try:
            st_file = os.path.join(live_dir, "live_engine_state.json")
            if os.path.exists(st_file):
                with open(st_file, "r", encoding="utf-8-sig") as sf:
                    st_obj = json.load(sf)
                for vp in (st_obj.get("virtual_open_positions") or []):
                    upnl = _safe_float(vp.get("unrealized_pnl", 0.0))
                    unreal_sum += upnl
                    positions.append({
                        "symbol": f"{vp.get('symbol', '')} [DRY-RUN]",
                        "side": vp.get("side", "LONG"),
                        "qty": str(vp.get("qty", "0")),
                        "entry_price": _safe_float(vp.get("entry_price", 0.0)),
                        "mark_price": _safe_float(vp.get("mark_price", 0.0)),
                        "unrealized_pnl": round(upnl, 4),
                        "stop_loss": _safe_float(vp.get("sl", 0.0)),
                        "take_profit": _safe_float(vp.get("tp", 0.0)),
                    })
        except Exception:
            pass

    w_res["unrealized_pnl"] = round(unreal_sum, 4)
    w_res["positions"] = positions
    w_res["open_positions"] = positions
    l1_learning_event = None
    try:
        mem_file = os.path.join(live_dir, "ai_copilot_memory.json")
        if os.path.exists(mem_file):
            with open(mem_file, "r", encoding="utf-8-sig") as mf:
                l1_learning_event = json.load(mf).get("last_layer1_learning_event")
    except Exception:
        l1_learning_event = None
    w_res["layer1_learning_event"] = l1_learning_event
    w_res["closed_summary"] = closed_summary
    w_res["signals"] = signals
    w_res["ml_connected"] = ml_connected
    w_res["ml_model_version"] = ml_version
    return json.dumps(w_res)


def sync_github_file(target_dir: str, repo_url: str, branch: str, secret_key: str, target_filename: str) -> str:
    import ast as _ast
    os.makedirs(target_dir, exist_ok=True)
    clean_repo = str(repo_url or "").strip().rstrip("/").replace(".git", "")
    parts = clean_repo.split("github.com/")
    if len(parts) != 2:
        return json.dumps({"ok": False, "error": "Invalid GitHub URL"})

    owner_repo = parts[1].strip("/")
    branch_clean = str(branch or "main").strip() or "main"
    tok = str(secret_key or "").strip()
    has_tok = bool(tok) and ("??" not in tok) and ("*" not in tok)

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "AutonomousAITrading-Android",
        "Cache-Control": "no-cache",
    }
    if has_tok:
        headers["Authorization"] = f"Bearer {tok}" if tok.startswith("github_pat_") else f"token {tok}"

    last_err = "Unknown GitHub sync error"

    def _validate_and_save(text_body: str) -> bool:
        clean_txt = text_body.lstrip("\ufeff")
        if not clean_txt.strip() or clean_txt.lstrip().startswith("<!DOCTYPE") or clean_txt.lstrip().startswith("<html"):
            return False
        if target_filename.endswith(".py"):
            _ast.parse(clean_txt)
        dest_path = os.path.join(target_dir, target_filename)
        tmp_path = dest_path + f".tmp.{os.getpid()}_{_at_threading.get_ident()}_{_at_uuid.uuid4().hex[:8]}"
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as wf:
            wf.write(clean_txt)
        _locked_replace(tmp_path, dest_path)
        return True

    # Method 1: Direct raw.githubusercontent.com with cache-buster
    try:
        raw_cdn = f"https://raw.githubusercontent.com/{owner_repo}/{branch_clean}/{target_filename}?t={int(time.time())}"
        r1 = _HTTP_SESSION.get(raw_cdn, headers=headers, timeout=12)
        if r1.status_code == 200 and _validate_and_save(r1.text):
            return json.dumps({"ok": True, "saved_to": os.path.join(target_dir, target_filename), "bytes": len(r1.text), "source": "raw_cdn"})
        last_err = f"Raw CDN HTTP {r1.status_code}"
    except Exception as e1:
        last_err = f"Raw CDN error: {e1}"

    # Method 2: GitHub Contents API
    try:
        api_url = f"https://api.github.com/repos/{owner_repo}/contents/{target_filename}?ref={branch_clean}"
        api_headers = dict(headers)
        api_headers["Accept"] = "application/vnd.github.raw+json"
        r2 = _HTTP_SESSION.get(api_url, headers=api_headers, timeout=12)
        if r2.status_code == 200 and _validate_and_save(r2.text):
            return json.dumps({"ok": True, "saved_to": os.path.join(target_dir, target_filename), "bytes": len(r2.text), "source": "contents_api"})
        last_err = f"Contents API HTTP {r2.status_code}: {r2.text[:120]}"
    except Exception as e2:
        last_err = f"Contents API error: {e2}"

    # Method 3: Zipball fallback
    try:
        zip_url = f"https://api.github.com/repos/{owner_repo}/zipball/{branch_clean}"
        resp = _HTTP_SESSION.get(zip_url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return json.dumps({"ok": False, "error": f"{last_err} | Zipball HTTP {resp.status_code}"})
        zip_path = os.path.join(target_dir, "synced_repo.zip")
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        extracted = False
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith("/" + target_filename) or member == target_filename:
                    raw_bytes = zf.read(member)
                    if _validate_and_save(raw_bytes.decode("utf-8-sig")):
                        extracted = True
                    break
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if not extracted:
            return json.dumps({"ok": False, "error": f"{target_filename} not found or invalid in branch '{branch_clean}'"})
        return json.dumps({"ok": True, "saved_to": os.path.join(target_dir, target_filename), "source": "zipball"})
    except Exception as e3:
        return json.dumps({"ok": False, "error": f"{last_err} | {e3}"})


def analyze_market_symbols(symbols_json: str) -> str:
    symbols = json.loads(symbols_json)
    out = {}

    def _one_sym(sym: str):
        clean = sym.replace("/", "").upper()
        try:
            data = _public_bybit_get("/v5/market/kline", {"category": "linear", "symbol": clean, "interval": "60", "limit": 50}, endpoint="Demo", is_paper=True, timeout=5.0)
            rows = data.get("result", {}).get("list", [])
            if len(rows) >= 25:
                closes = [float(x[4]) for x in reversed(rows)]
                highs = [float(x[2]) for x in reversed(rows)]
                lows = [float(x[3]) for x in reversed(rows)]
                trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])) for i in range(1, len(closes))]
                atr = sum(trs[-14:]) / 14.0
                natr = (atr / closes[-1]) * 100.0 if closes[-1] > 0 else 0.0
                ret24 = ((closes[-1] - closes[-24]) / closes[-24]) * 100.0 if closes[-24] > 0 else 0.0
                adx_proxy = min(45.0, max(12.0, abs(ret24) * 4.5 + natr * 6.0))
                return (sym, {"ok": True, "natr": round(natr, 3), "adx": round(adx_proxy, 2), "ret24h": round(ret24, 2)})
        except Exception:
            pass
        return (sym, None)

    if symbols:
        with ThreadPoolExecutor(max_workers=min(6, len(symbols))) as ex:
            for sym, res in ex.map(_one_sym, symbols):
                if res is not None:
                    out[sym] = res
    return json.dumps(out)

def compute_live_radar_json(live_dir: str, models_dir: str, endpoint: str = "Mainnet", crypto_en: bool = None, forex_en: bool = None) -> str:
    try:
        cfg_p = os.path.join(live_dir, "live_runtime_config.json")
        if not os.path.exists(cfg_p):
            cfg_p = os.path.join(live_dir, "paper_runtime_config.json")
        rcfg = _safe_load_json(cfg_p, {})
        if crypto_en is None:
            crypto_en = bool(rcfg.get("crypto_enabled", True))
        if forex_en is None:
            forex_en = bool(rcfg.get("forex_enabled", False))
        signals, ml_ok, ver = _compute_radar_signals(None, models_dir, "LIVE", endpoint, bool(crypto_en), bool(forex_en))
        return json.dumps({"ok": True, "ml_connected": ml_ok, "ml_model_version": ver, "signals": signals})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e), "signals": []})
