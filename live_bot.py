import os

import threading as _at_threading
import uuid as _at_uuid
_ATOMIC_IO_LOCK = _at_threading.RLock()

def _locked_replace(src_tmp: str, dst_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    with _ATOMIC_IO_LOCK:
        os.replace(src_tmp, dst_path)

import threading
import uuid
import gc
import sys
import json
import time
import math
import hmac
import hashlib
import urllib.parse
import datetime
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
import pandas as pd
import xgboost as xgb

IST_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

TIMEFRAME_TO_BYBIT_INTERVAL = {
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
}

FOREX_LINEAR_SYMBOLS = {
    "EURUSDUSDT",
    "GBPUSDUSDT",
    "USDJPYUSDT",
    "XAUUSDT",
    "XAGUSDT",
    "XAUTUSDT",
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
    10002: "Timestamp outside Bybit recv_window — resyncing server clock",
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
    110126: "Contract agreement not signed on Bybit (sign in Bybit app or remove stock-perp coin)",
    110132: "Contract not available in your region",
}


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val) if math.isfinite(float(val)) else default
    s = str(val).strip()
    if not s:
        return default
    try:
        num = float(s)
        return num if math.isfinite(num) else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default: int = 0) -> int:
    if val is None:
        return default
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    s = str(val).strip()
    if not s:
        return default
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return default


def get_ist_session_name(dt_ist: datetime.datetime) -> str:
    hm = dt_ist.hour * 60 + dt_ist.minute
    if 330 <= hm < 810:      # 05:30 - 13:30 IST
        return "ASIA"
    if 810 <= hm < 1110:     # 13:30 - 18:30 IST
        return "LONDON"
    return "NEW_YORK"        # 18:30 - 05:30 IST


def is_forex_symbol(symbol: str) -> bool:
    clean = symbol.replace("/", "").upper()
    if clean in FOREX_LINEAR_SYMBOLS:
        return True
    forex_bases = {"EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
    return len(clean) == 6 and clean[:3] in forex_bases and clean[3:] in (forex_bases | {"USD"})


def _coin_sector(symbol: str) -> str:
    clean = symbol.replace("/", "").upper()
    if is_forex_symbol(clean):
        return "FOREX"
    if clean.startswith(("BTC", "ETH")):
        return "MAJORS"
    if clean.startswith(("SOL", "AVAX", "LINK", "ADA", "DOT", "ATOM", "NEAR", "UNI", "AAVE", "SUI", "APT")):
        return "ALT_L1_DEFI"
    return "HIGH_BETA"


def is_valid_xgb_model(model_dict) -> bool:
    return isinstance(model_dict, dict) and "learner" in model_dict and isinstance(model_dict.get("learner"), dict)


def _sanitize_xgb_model_json(model_dict: dict) -> dict:
    """Repairs XGBoost JSON float arrays where integer tokens cause C++ json.h:82 Invalid cast."""
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


class LiveAccountTradingBot:
    def __init__(self, api_key: str, api_secret: str, models_dir: str, live_dir: str = None, endpoint: str = "Mainnet"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.models_dir = models_dir
        self.live_dir = live_dir or os.path.join(os.path.dirname(models_dir), "bot_data", "LIVE_ACCOUNT")
        os.makedirs(self.live_dir, exist_ok=True)
        self.state_path = os.path.join(self.live_dir, "live_engine_state.json")
        self.memory_path = os.path.join(self.live_dir, "ai_copilot_memory.json")
        self.reports_summary_path = os.path.join(self.live_dir, "live_reports_summary.json")

        is_paper_dir = os.path.basename(os.path.normpath(self.live_dir)).upper() == "PAPER"
        p_paper = os.path.join(models_dir, "PAPER", "UNIFIED_PORTFOLIO_MASTER.json")
        p1 = os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json")
        p2 = os.path.join(models_dir, "UNIFIED_PORTFOLIO_MASTER.json")
        if is_paper_dir and os.path.exists(p_paper):
            self.master_path = p_paper
        else:
            self.master_path = p1 if os.path.exists(p1) else p2

        self.is_demo = str(endpoint).lower() in ("testnet", "demo")
        self.host_candidates = (
            ["https://api-demo.bybit.com", "https://api.bybit.com"]
            if self.is_demo
            else ["https://api.bybit.com", "https://api.bytick.com"]
        )
        self.base_url = self.host_candidates[0]
        self.category = "linear"
        self.rate_limit_remaining = 120
        self.rate_limit_cap = 120
        self._server_time_offset_ms = 0
        self._last_time_sync_s = 0.0
        self._booster_cache = {}
        self._instrument_cache = {}
        self._trend_4h_cache = {}
        self._positioning_cache = {}
        self._ticker_cache = (0.0, {}, {})

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=12, pool_maxsize=12)
        self.session.mount("https://", adapter)

    def _reset_http_session(self):
        old_s = getattr(self, "session", None)
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=12, pool_maxsize=12))
        self.session = s
        try:
            if old_s is not None:
                old_s.close()
        except Exception:
            pass

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
            "position_opened_ms": {},
            "position_entry_meta": {},
            "loss_cooldown_until_ms": {},
            "agentic_veto_until_ms": {},
            "restricted_symbols_until_ms": {},
            "last_notified_block_layer": {},
            "last_shadow_eval_ms": {},
            "reflected_closed_ids": [],
            "daily_coin_trades": {},
            "daily_total_trades": 0,
            "peak_equity_usdt": 0.0,
            "tp1_partial_done": {},
            "native_trail_set": {},
            "virtual_open_positions": [],
            "radar_signals": [],
            "last_bot_logs": [],
            "last_diagnostics": [],
            "last_order_rejections": [],
            "last_trade_blocks": [],
            "coins_scanned": 0,
            "top_status_summary": "",
        }

    def _save_state(self, state: dict):
        os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
        tmp_path = f"{self.state_path}.tmp.{os.getpid()}_{_at_threading.get_ident()}_{_at_uuid.uuid4().hex[:8]}_{threading.get_ident()}_{uuid.uuid4().hex[:8]}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        _locked_replace(tmp_path, self.state_path)

    def _write_lightweight_live_summary(self, master: dict):
        try:
            port = master.get("portfolio", {}) or {}
            lite_port = {}
            for sym, entry in port.items():
                if not isinstance(entry, dict):
                    continue
                lite_entry = {k: v for k, v in entry.items() if k not in ("models", "tested_matrix")}
                if isinstance(lite_entry.get("stats"), dict):
                    lite_entry["stats"] = {k: v for k, v in lite_entry["stats"].items() if k != "raw_trade_returns"}
                if isinstance(lite_entry.get("gear2_model"), dict):
                    lite_entry["gear2_model"] = {k: v for k, v in lite_entry["gear2_model"].items() if k != "models"}
                lite_port[sym] = lite_entry
            tmp_s = f"{self.reports_summary_path}.tmp.{os.getpid()}_{_at_threading.get_ident()}_{_at_uuid.uuid4().hex[:8]}_{threading.get_ident()}_{uuid.uuid4().hex[:8]}"
            with open(tmp_s, "w", encoding="utf-8") as wf:
                json.dump({
                    "version": master.get("version", ""),
                    "exported_at": master.get("exported_at", int(time.time())),
                    "model_accuracy_pct": master.get("model_accuracy_pct", 0.0),
                    "total_net_pnl": master.get("total_net_pnl", 0.0),
                    "portfolio": lite_port,
                }, wf, indent=2)
            _locked_replace(tmp_s, self.reports_summary_path)
        except Exception:
            pass

    def _load_agentic_memory(self) -> dict:
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {}

    def _resolve_gemini_shadow_settings(self) -> tuple:
        """Reads Gemini Shadow toggle & API key across shadow config, copilot config, and runtime config."""
        enabled = False
        key = ""
        for fname in ("ai_shadow_config.json", "ai_copilot_config.json", "live_runtime_config.json"):
            fpath = os.path.join(self.live_dir, fname)
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8-sig") as f:
                        d = json.load(f)
                    if fname == "ai_shadow_config.json":
                        enabled = bool(d.get("gemini_shadow_enabled", False))
                    if not key:
                        key = str(d.get("gemini_api_key") or d.get("api_key_gemini") or "").strip()
                except Exception:
                    pass
        return enabled, key

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

    def _public_get(self, path: str, params: dict, timeout: float = 4.0) -> dict:
        for base in self.host_candidates:
            try:
                resp = self.session.get(f"{base}{path}", params=params, timeout=timeout)
                self._update_rate_limits(resp)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                continue
        return {}

    def _fetch_bybit_positioning_metrics(self, bybit_sym: str) -> dict:
        """Fetches cached 1H Open Interest 4H delta (%) and Top-Trader Long/Short skew (-100 to +100)."""
        now_s = time.time()
        cached = self._positioning_cache.get(bybit_sym)
        if cached and (now_s - cached[0]) < 600.0:
            return cached[1]

        oi_delta_4h = 0.0
        ls_skew = 0.0
        try:
            oi_res = self._public_get(
                "/v5/market/open-interest",
                {"category": self.category, "symbol": bybit_sym, "intervalTime": "1h", "limit": 5},
                timeout=3.0,
            )
            oi_list = oi_res.get("result", {}).get("list", []) or []
            if len(oi_list) >= 2:
                oi_now = _safe_float(oi_list[0].get("openInterest"), 0.0)
                oi_prev = _safe_float(oi_list[-1].get("openInterest"), 0.0)
                if oi_prev > 0.0:
                    oi_delta_4h = ((oi_now - oi_prev) / oi_prev) * 100.0

            ar_res = self._public_get(
                "/v5/market/account-ratio",
                {"category": self.category, "symbol": bybit_sym, "period": "1h", "limit": 1},
                timeout=3.0,
            )
            ar_list = ar_res.get("result", {}).get("list", []) or []
            if ar_list:
                buy_r = _safe_float(ar_list[0].get("buyRatio"), 0.5)
                sell_r = _safe_float(ar_list[0].get("sellRatio"), 0.5)
                ls_skew = (buy_r - sell_r) * 100.0
        except Exception:
            pass

        out = {"oi_delta_4h": round(oi_delta_4h, 3), "ls_skew": round(ls_skew, 2)}
        self._positioning_cache[bybit_sym] = (now_s, out)
        return out

    def _sync_bybit_server_time(self, force: bool = False):
        now_s = time.time()
        if not force and (now_s - self._last_time_sync_s) < 600.0:
            return
        try:
            local_before_ms = int(time.time() * 1000)
            data = self._public_get("/v5/market/time", {}, timeout=3.0)
            local_after_ms = int(time.time() * 1000)
            res = data.get("result", {})
            srv_nano = _safe_int(res.get("timeNano"), 0)
            srv_sec = _safe_int(res.get("timeSecond"), 0)
            srv_ms = (srv_nano // 1_000_000) if srv_nano > 0 else (srv_sec * 1000 if srv_sec > 0 else 0)
            if srv_ms > 0:
                mid_local_ms = (local_before_ms + local_after_ms) // 2
                self._server_time_offset_ms = srv_ms - mid_local_ms
                self._last_time_sync_s = now_s
        except Exception:
            pass

    def _check_orderbook_spread_and_depth(
        self,
        bybit_sym: str,
        order_notional_usdt: float,
        max_spread_pct: float = 0.08,
    ) -> tuple:
        """Blocks entry if bid-ask spread > max_spread_pct or top-5 orderbook depth < 3x order notional."""
        data = self._public_get(
            "/v5/market/orderbook",
            {"category": self.category, "symbol": bybit_sym, "limit": 5},
            timeout=3.5,
        )
        res = data.get("result", {})
        bids = res.get("b", [])
        asks = res.get("a", [])
        if not bids or not asks:
            return True, "Orderbook check skipped"

        best_bid = _safe_float(bids[0][0], 0.0)
        best_ask = _safe_float(asks[0][0], 0.0)
        if best_bid <= 0.0 or best_ask <= 0.0:
            return True, "Orderbook check skipped"

        spread_pct = ((best_ask - best_bid) / best_bid) * 100.0
        bid_depth_usdt = sum(_safe_float(px, 0.0) * _safe_float(sz, 0.0) for px, sz in bids[:5])
        ask_depth_usdt = sum(_safe_float(px, 0.0) * _safe_float(sz, 0.0) for px, sz in asks[:5])
        min_side_depth = min(bid_depth_usdt, ask_depth_usdt)

        if spread_pct > max_spread_pct:
            return False, f"Wide spread {spread_pct:.3f}% > {max_spread_pct:.2f}%"
        if order_notional_usdt > 0.0 and min_side_depth < (order_notional_usdt * 3.0):
            return False, f"Thin orderbook depth ${min_side_depth:.0f} < 3x notional (${order_notional_usdt * 3.0:.0f})"
        return True, f"Spread {spread_pct:.3f}%"

    @staticmethod
    def _passes_sector_correlation_cap(symbol: str, side: str, open_positions: list) -> tuple:
        """Prevents holding 2+ simultaneous positions in the same sector and same direction."""
        target_sec = _coin_sector(symbol)
        target_side = side.upper()
        for pos in (open_positions or []):
            if not isinstance(pos, dict):
                continue
            p_sym = str(pos.get("symbol", "")).upper()
            p_side = str(pos.get("side", "")).upper()
            if p_sym and _coin_sector(p_sym) == target_sec and p_side == target_side:
                return False, f"Correlated {target_sec} {target_side} already open ({p_sym})"
        return True, "Sector OK"

    @staticmethod
    def _passes_rolling_correlation_guard(
        bybit_sym: str,
        side: str,
        open_positions: list,
        returns_map: dict,
        max_corr: float = 0.82,
    ) -> tuple:
        """Computes rolling 72-bar log-return correlation across active pairs to block macro-correlated stacking."""
        target_ret = returns_map.get(bybit_sym)
        if target_ret is None or len(target_ret) < 24:
            return True, "Corr OK"
        target_side = side.upper()
        for pos in (open_positions or []):
            if not isinstance(pos, dict):
                continue
            p_sym = str(pos.get("symbol", "")).upper()
            p_side = str(pos.get("side", "")).upper()
            if not p_sym or p_sym == bybit_sym or p_side != target_side:
                continue
            other_ret = returns_map.get(p_sym)
            if other_ret is None or len(other_ret) < 24:
                continue
            n_min = min(len(target_ret), len(other_ret), 72)
            a = target_ret[-n_min:]
            b = other_ret[-n_min:]
            if np.std(a) > 1e-8 and np.std(b) > 1e-8:
                corr = float(np.corrcoef(a, b)[0, 1])
                if math.isfinite(corr) and corr >= max_corr:
                    return False, f"72h Correlation {corr:.2f} >= {max_corr:.2f} with open {p_sym} {p_side}"
        return True, "Corr OK"

    @staticmethod
    def _evaluate_meta_labeler(
        meta_bundle: dict,
        raw_prob: float,
        prob_margin: float,
        adx_val: float,
        natr_val: float,
        vwap_dev: float,
        oi_delta_4h: float,
        ls_skew: float,
    ) -> tuple:
        """Evaluates trained secondary meta-classifier P(TP1 before SL) when available."""
        if not isinstance(meta_bundle, dict):
            return None, 0.0
        weights = meta_bundle.get("weights")
        means = meta_bundle.get("means")
        stds = meta_bundle.get("stds")
        bias = _safe_float(meta_bundle.get("bias"), 0.0)
        min_meta_prob = _safe_float(meta_bundle.get("min_prob"), 0.52)
        if not isinstance(weights, list) or not isinstance(means, list) or not isinstance(stds, list) or len(weights) != 7:
            return None, 0.0
        x_vec = [raw_prob, prob_margin, adx_val, natr_val, vwap_dev, oi_delta_4h, ls_skew]
        z = bias
        for i in range(7):
            sd = max(_safe_float(stds[i], 1.0), 1e-5)
            z += _safe_float(weights[i], 0.0) * ((x_vec[i] - _safe_float(means[i], 0.0)) / sd)
        z = max(-18.0, min(18.0, z))
        meta_prob = 1.0 / (1.0 + math.exp(-z))
        return bool(meta_prob >= min_meta_prob), round(meta_prob, 4)

    def _fetch_tickers_cached(self) -> tuple:
        now_s = time.time()
        c_ts, c_fr, c_px = self._ticker_cache
        if c_px and (now_s - c_ts) < 45.0:
            return c_fr, c_px
        rates = {}
        prices = {}
        try:
            data = self._public_get("/v5/market/tickers", {"category": self.category}, timeout=3.5)
            for row in data.get("result", {}).get("list", []):
                sym = str(row.get("symbol", "")).upper()
                if not sym:
                    continue
                fr_raw = row.get("fundingRate", "")
                if str(fr_raw).strip() != "":
                    rates[sym] = _safe_float(fr_raw, 0.0)
                px_val = _safe_float(row.get("markPrice") or row.get("lastPrice"), 0.0)
                if px_val > 0.0:
                    prices[sym] = px_val
            if prices:
                self._ticker_cache = (now_s, rates, prices)
        except Exception:
            pass
        return (rates or c_fr), (prices or c_px)

    def _bybit_request(self, method: str, path: str, params: dict = None, body: dict = None) -> dict:
        if self.rate_limit_cap > 0 and self.rate_limit_remaining < int(self.rate_limit_cap * 0.10):
            time.sleep(0.4)

        self._sync_bybit_server_time(force=False)
        last_err = None
        for attempt in range(2):
            if attempt == 1:
                self._reset_http_session()
                time.sleep(0.75)
            for base in self.host_candidates:
                ts = str(int(time.time() * 1000) + self._server_time_offset_ms)
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
                        resp = self.session.get(url, headers=headers, timeout=5.0)
                        self._update_rate_limits(resp)
                        if resp.status_code == 200:
                            j_out = resp.json()
                            if _safe_int(j_out.get("retCode"), 0) == 10002:
                                self._sync_bybit_server_time(force=True)
                                continue
                            return j_out
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
                        resp = self.session.post(f"{base}{path}", headers=headers, data=raw_body, timeout=5.0)
                        self._update_rate_limits(resp)
                        if resp.status_code == 200:
                            j_out = resp.json()
                            if _safe_int(j_out.get("retCode"), 0) == 10002:
                                self._sync_bybit_server_time(force=True)
                                continue
                            return j_out
                except Exception as e:
                    last_err = e
                    continue
        if last_err:
            raise last_err
        return {}

    def _extract_wallet_metrics(self, bal_res: dict) -> tuple:
        """Returns (available_usdt, total_equity_usdt)."""
        acct_list = bal_res.get("result", {}).get("list", [])
        if not acct_list or not isinstance(acct_list[0], dict):
            return 0.0, 0.0
        acct = acct_list[0]
        total_eq = _safe_float(acct.get("totalEquity"), 0.0)
        raw_avail = str(acct.get("totalAvailableBalance", "")).strip()
        if raw_avail != "":
            return _safe_float(raw_avail, 0.0), total_eq
        for c_item in acct.get("coin", []) or []:
            if str(c_item.get("coin", "")).upper() == "USDT":
                wb = _safe_float(c_item.get("walletBalance"), 0.0)
                im = _safe_float(c_item.get("totalPositionIM"), 0.0) + _safe_float(c_item.get("totalOrderIM"), 0.0)
                eq = _safe_float(c_item.get("equity"), wb)
                return max(0.0, wb - im), (total_eq if total_eq > 0.0 else eq)
        return total_eq, total_eq

    def _get_top_of_book(self, bybit_symbol: str) -> tuple:
        """Returns (best_bid, best_ask) from Bybit v5 linear orderbook."""
        try:
            data = self._public_get(
                "/v5/market/orderbook",
                {"category": self.category, "symbol": bybit_symbol, "limit": 1},
                timeout=2.5,
            )
            res = data.get("result", {}) or {}
            bids = res.get("b", []) or []
            asks = res.get("a", []) or []
            bid = _safe_float(bids[0][0], 0.0) if bids else 0.0
            ask = _safe_float(asks[0][0], 0.0) if asks else 0.0
            return bid, ask
        except Exception:
            return 0.0, 0.0

    def _cancel_stale_entry_orders(self, bybit_symbol: str, max_age_ms: int = 600000):
        """Cancels resting unfilled Limit entry orders older than max_age_ms (default 10m)."""
        try:
            res = self._bybit_request(
                "GET",
                "/v5/order/realtime",
                params={"category": self.category, "symbol": bybit_symbol, "openOnly": 0, "limit": 10},
            )
            now_ms = int(time.time() * 1000)
            for ord_item in (res.get("result", {}) or {}).get("list", []) or []:
                if ord_item.get("reduceOnly"):
                    continue
                created_ms = _safe_int(ord_item.get("createdTime"), now_ms)
                oid = ord_item.get("orderId")
                if oid and (now_ms - created_ms) >= max_age_ms:
                    self._bybit_request(
                        "POST",
                        "/v5/order/cancel",
                        body={"category": self.category, "symbol": bybit_symbol, "orderId": oid},
                    )
        except Exception:
            pass

    def _get_instrument_filters(self, bybit_symbol: str):
        if bybit_symbol in self._instrument_cache:
            return self._instrument_cache[bybit_symbol]
        data = self._public_get("/v5/market/instruments-info", {"category": self.category, "symbol": bybit_symbol})
        items = data.get("result", {}).get("list", [])
        if not items:
            return None
        lot = items[0].get("lotSizeFilter", {})
        pf = items[0].get("priceFilter", {})
        qty_step = str(lot.get("qtyStep", "")).strip() or "0.001"
        tick_size = str(pf.get("tickSize", "")).strip() or "0.0001"
        res = (
            qty_step,
            _safe_float(lot.get("minOrderQty"), 0.001),
            _safe_float(lot.get("minNotionalValue"), 5.0),
            tick_size,
        )
        self._instrument_cache[bybit_symbol] = res
        return res

    @staticmethod
    def _format_step(val: float, step_str: str, round_up: bool = False) -> str:
        step = _safe_float(step_str, 0.001)
        if step <= 0.0:
            step = 0.001
        scaled = math.ceil(val / step) * step if round_up else round(val / step) * step
        clean_step = step_str.rstrip("0")
        decimals = len(clean_step.split(".")[1]) if "." in clean_step else 0
        return f"{scaled:.{decimals}f}"

    @staticmethod
    def _regime_exit_multipliers(adx_val: float, natr_val: float, btc_bearish: bool) -> tuple:
        if natr_val >= 2.8 or (adx_val < 15.0 and btc_bearish):
            return 1.25, 0.92, 0.42
        if not btc_bearish and adx_val >= 18.0 and natr_val < 2.6:
            return 1.15, 1.05, 0.45
        return 1.12, 1.00, 0.45

    @staticmethod
    def _conviction_risk_scale(raw_prob: float, req_thresh: float) -> float:
        edge = max(0.0, raw_prob - req_thresh)
        scale = 0.85 + min(edge * 2.2, 0.35)
        return round(min(max(scale, 0.85), 1.20), 4)

    @staticmethod
    def _passes_anti_chase_guard(closed_bar: pd.Series, live_bar: pd.Series, side: str, is_chop_gear: bool = False) -> tuple:
        rsi = _safe_float(closed_bar.get("RSI_14"), 50.0)
        dist_ema20 = _safe_float(closed_bar.get("Dist_EMA20"), 0.0)
        ret_1h = _safe_float(closed_bar.get("Ret_1h"), 0.0)
        di_spread = _safe_float(closed_bar.get("DI_Spread"), 0.0)

        closed_px = _safe_float(closed_bar.get("Close"), 0.0)
        live_px = _safe_float(live_bar.get("Close"), closed_px)
        intrabar_move = ((live_px - closed_px) / closed_px) if closed_px > 0 else 0.0
        max_bar_move = 0.022 if is_chop_gear else 0.012

        if side == "SHORT":
            if dist_ema20 < -2.35:
                return False, f"Stretched below EMA20 ({dist_ema20:.2f} ATR)"
            if rsi < 27.0:
                return False, f"RSI deeply oversold ({rsi:.1f} < 27)"
            if ret_1h > max_bar_move or intrabar_move > 0.009:
                return False, f"Sharp bounce (+{max(ret_1h, intrabar_move) * 100:.2f}%)"
            if di_spread > 10.0:
                return False, f"Strong bullish DI (+{di_spread:.1f})"
        else:
            if dist_ema20 > 2.35:
                return False, f"Stretched above EMA20 (+{dist_ema20:.2f} ATR)"
            if rsi > 73.0:
                return False, f"RSI deeply overbought ({rsi:.1f} > 73)"
            if ret_1h < -max_bar_move or intrabar_move < -0.009:
                return False, f"Sharp dump ({min(ret_1h, intrabar_move) * 100:.2f}%)"
            if di_spread < -10.0:
                return False, f"Strong bearish DI ({di_spread:.1f})"
        return True, ""

    def _save_closed_trade_history(self, order_id: str, sym: str, side_label: str, pnl: float, closed_ms: int, is_virtual: bool = False):
        try:
            hist_file = os.path.join(self.live_dir, "live_reports_history.json")
            hist_data = {"trades": []}
            if os.path.exists(hist_file):
                with open(hist_file, "r", encoding="utf-8-sig") as hf:
                    loaded = json.load(hf)
                    if isinstance(loaded, dict):
                        hist_data = loaded
            trades_list = hist_data.setdefault("trades", [])
            if not any(str(t.get("id")) == order_id for t in trades_list):
                disp = f"{sym[:-4]}/USDT" if sym.endswith("USDT") else sym
                trades_list.insert(0, {
                    "id": order_id,
                    "symbol": disp,
                    "side": side_label,
                    "net_pnl": round(pnl, 4),
                    "closed_at_ms": closed_ms,
                    "virtual": bool(is_virtual),
                })
                hist_data["trades"] = trades_list[:2000]
                tmp_hf = f"{hist_file}.tmp.{os.getpid()}_{_at_threading.get_ident()}_{_at_uuid.uuid4().hex[:8]}_{threading.get_ident()}_{uuid.uuid4().hex[:8]}"
                with open(tmp_hf, "w", encoding="utf-8") as wf:
                    json.dump(hist_data, wf, indent=2)
                _locked_replace(tmp_hf, hist_file)
        except Exception:
            pass

    def _sync_virtual_positions(self, state: dict, mark_prices: dict, trail_ratio: float, now_ms: int, add_bot_log, curr_session: str = "NEW_YORK", curr_regime: str = "CHOP") -> list:
        v_list = state.get("virtual_open_positions") or []
        if not v_list:
            return []
        remaining = []
        loss_cd = state.setdefault("loss_cooldown_until_ms", {})

        for vp in v_list:
            sym = str(vp.get("symbol", "")).upper()
            side = str(vp.get("side", "LONG")).upper()
            entry = _safe_float(vp.get("entry_price"), 0.0)
            qty = _safe_float(vp.get("qty"), 0.0)
            sl = _safe_float(vp.get("sl"), 0.0)
            tp = _safe_float(vp.get("tp"), 0.0)
            opened_ms = _safe_int(vp.get("opened_ms"), now_ms)
            vp_is_chop = bool(vp.get("is_chop_gear", False))
            vp_conf_regime = str(vp.get("conformal_regime") or (f"G2_CHOP_{curr_regime}" if vp_is_chop else curr_regime))
            mark = _safe_float(mark_prices.get(sym), _safe_float(vp.get("mark_price"), entry))
            if mark <= 0.0:
                mark = entry

            is_long = side == "LONG"
            gross_pnl = ((mark - entry) if is_long else (entry - mark)) * qty
            fee_rate = 0.00026 if vp_is_chop else 0.00055
            est_fees = (entry * qty + mark * qty) * fee_rate
            net_pnl = gross_pnl - est_fees

            tp_dist = abs(tp - entry)
            profit_dist = (mark - entry) if is_long else (entry - mark)

            if tp_dist > 0.0 and profit_dist >= tp_dist * 0.65:
                be_sl = (entry + tp_dist * 0.15) if is_long else (entry - tp_dist * 0.15)
                if (is_long and be_sl > sl) or ((not is_long) and be_sl < sl):
                    sl = be_sl
            if tp_dist > 0.0 and profit_dist >= tp_dist * 0.70:
                trail_sl = (mark - tp_dist * trail_ratio) if is_long else (mark + tp_dist * trail_ratio)
                if (is_long and trail_sl > sl) or ((not is_long) and trail_sl < sl):
                    sl = trail_sl

            hit_sl = (mark <= sl) if is_long else (mark >= sl)
            hit_tp = (mark >= tp) if is_long else (mark <= tp)
            age_hours = (now_ms - opened_ms) / 3_600_000.0
            max_stagnant_h = 2.0 if vp_is_chop else 8.0
            hit_time = (age_hours >= max_stagnant_h and tp_dist > 0.0 and profit_dist >= 0.0 and profit_dist < tp_dist * 0.25)

            if hit_sl or hit_tp or hit_time:
                reason = "TP HIT" if hit_tp else ("TIME-STOP" if hit_time else "SL HIT")
                vid = f"DRY_{sym}_{now_ms}"
                self._save_closed_trade_history(vid, sym, side, net_pnl, now_ms, is_virtual=True)
                if net_pnl < 0.0:
                    loss_cd[sym] = now_ms + (4 * 60 * 60 * 1000)
                try:
                    import importlib
                    ai_regime_copilot = importlib.import_module("ai_regime_copilot")
                    ref_res = ai_regime_copilot.record_post_trade_reflection(
                        self.live_dir, sym, side, net_pnl, entry, mark
                    )
                    if hasattr(ai_regime_copilot, "update_bayesian_bucket"):
                        ai_regime_copilot.update_bayesian_bucket(self.live_dir, sym, curr_session, net_pnl)
                    if hasattr(ai_regime_copilot, "record_conformal_outcome"):
                        ai_regime_copilot.record_conformal_outcome(
                            self.live_dir, vp_conf_regime, _safe_float(vp.get("raw_prob"), 0.35), bool(net_pnl > 0.0)
                        )
                    elif hasattr(ai_regime_copilot, "update_conformal_calibration"):
                        ai_regime_copilot.update_conformal_calibration(
                            self.live_dir, vp_conf_regime, _safe_float(vp.get("raw_prob"), 0.35), bool(net_pnl > 0.0)
                        )
                    if hasattr(ai_regime_copilot, "synthesize_failure_playbook"):
                        ai_regime_copilot.synthesize_failure_playbook(self.live_dir, max_items=6)
                    if ref_res.get("lesson"):
                        add_bot_log(f"[DRY-RUN REFLECTION] {ref_res['lesson']}")
                except Exception:
                    pass
                add_bot_log(f"[DRY-RUN CLOSED • {reason}] {sym} {side} @ {mark:.4f} | Net PnL: {net_pnl:+.4f} USDT")
            else:
                vp["mark_price"] = mark
                vp["sl"] = sl
                vp["unrealized_pnl"] = round(net_pnl, 4)
                remaining.append(vp)

        state["virtual_open_positions"] = remaining
        return remaining

    def _record_recent_stop_losses_and_reflect(self, state: dict, now_ms: int, add_bot_log, curr_session: str = "NEW_YORK", curr_regime: str = "CHOP"):
        loss_cd = state.setdefault("loss_cooldown_until_ms", {})
        reflected = state.setdefault("reflected_closed_ids", [])
        entry_meta_map = state.setdefault("position_entry_meta", {})
        try:
            res = self._bybit_request("GET", "/v5/position/closed-pnl", {"category": self.category, "limit": 50})
        except Exception:
            return
        for row in res.get("result", {}).get("list", []):
            sym = str(row.get("symbol", "")).upper()
            pnl = _safe_float(row.get("closedPnl"), 0.0)
            updated_ms = _safe_int(row.get("updatedTime"), 0)
            order_id = str(row.get("orderId", f"{sym}_{updated_ms}"))
            if sym and pnl < 0.0 and updated_ms > 0 and (now_ms - updated_ms) < 4 * 60 * 60 * 1000:
                target_cd = updated_ms + (4 * 60 * 60 * 1000)
                if target_cd > _safe_int(loss_cd.get(sym), 0):
                    loss_cd[sym] = target_cd

            if sym and updated_ms > 0 and order_id not in reflected:
                reflected.append(order_id)
                side_label = "SHORT" if str(row.get("side", "")).lower() == "buy" else "LONG"
                meta_info = entry_meta_map.pop(sym, {}) or {}
                conf_reg = str(meta_info.get("conformal_regime") or meta_info.get("regime") or curr_regime)
                try:
                    ai_regime_copilot = __import__("ai_regime_copilot")
                    ref_res = ai_regime_copilot.record_post_trade_reflection(
                        self.live_dir,
                        sym,
                        side_label,
                        pnl,
                        _safe_float(row.get("avgEntryPrice"), 0.0),
                        _safe_float(row.get("avgExitPrice"), 0.0),
                    )
                    if hasattr(ai_regime_copilot, "update_bayesian_bucket"):
                        ai_regime_copilot.update_bayesian_bucket(
                            self.live_dir, sym, str(meta_info.get("session", curr_session)), pnl
                        )
                    if hasattr(ai_regime_copilot, "record_conformal_outcome"):
                        ai_regime_copilot.record_conformal_outcome(
                            self.live_dir,
                            conf_reg,
                            _safe_float(meta_info.get("raw_prob"), 0.35),
                            bool(pnl > 0.0),
                        )
                    elif hasattr(ai_regime_copilot, "update_conformal_calibration"):
                        ai_regime_copilot.update_conformal_calibration(
                            self.live_dir,
                            conf_reg,
                            _safe_float(meta_info.get("raw_prob"), 0.35),
                            bool(pnl > 0.0),
                        )
                    if hasattr(ai_regime_copilot, "synthesize_failure_playbook"):
                        ai_regime_copilot.synthesize_failure_playbook(self.live_dir, max_items=6)
                    if ref_res.get("lesson"):
                        add_bot_log(f"[AGENTIC REFLECTION] {ref_res['lesson']}")
                except Exception:
                    pass
                self._save_closed_trade_history(order_id, sym, side_label, pnl, updated_ms, is_virtual=False)
        state["reflected_closed_ids"] = reflected[-40:]

    def fetch_bybit_candles(self, bybit_symbol: str, interval: str = "60", limit: int = 230) -> pd.DataFrame:
        data = self._public_get(
            "/v5/market/kline",
            {"category": self.category, "symbol": bybit_symbol, "interval": interval, "limit": limit},
            timeout=4.0,
        )
        rows = data.get("result", {}).get("list", [])
        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"])
            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            return df.sort_values("timestamp").set_index("timestamp")

        if is_forex_symbol(bybit_symbol) and not bybit_symbol.endswith("USDT"):
            yf_interval = "5m" if interval == "5" else ("15m" if interval == "15" else ("30m" if interval == "30" else "1h"))
            yf_range = "5d" if yf_interval == "5m" else ("15d" if yf_interval in ("15m", "30m") else "60d")
            try:
                y_resp = self.session.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{bybit_symbol}=X",
                    params={"interval": yf_interval, "range": yf_range},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=5.0,
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
                ref_bar = df_4h.iloc[-2] if len(df_4h) >= 2 else df_4h.iloc[-1]
                st = int(ref_bar["Supertrend"])
                adx = _safe_float(ref_bar.get("ADX"), 20.0)
                natr = _safe_float(ref_bar.get("NATR"), 1.8)
                self._trend_4h_cache[bybit_sym] = (now_s, st, adx, natr)
                return st, adx, natr
        return 0, 20.0, 1.8

    def _eval_custom_rules(self, bar: pd.Series, rules: list) -> bool:
        if not rules:
            return False
        for rule in rules:
            ind = str(rule.get("ind", ""))
            op = str(rule.get("op", ""))
            val = _safe_float(rule.get("val"), 0.0)
            if ind not in bar.index:
                return False
            actual = _safe_float(bar[ind], 0.0)
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
        curr_open: float,
        closed_bar: pd.Series,
        side: str,
        relaxed: bool = False,
        custom_rules: dict = None,
    ) -> bool:
        if isinstance(custom_rules, dict):
            rule_key = "long_rules" if side == "LONG" else "short_rules"
            if custom_rules.get(rule_key):
                return self._eval_custom_rules(closed_bar, custom_rules[rule_key])

        sid = strategy_id.lower()
        if relaxed or "adaptive_quantile" in sid or "high_sensitivity" in sid:
            return bool((closed_bar["Dist_EMA50"] > -1.5) if side == "LONG" else (closed_bar["Dist_EMA50"] < 1.5))

        if "g2_range_box" in sid:
            adx_v = _safe_float(closed_bar.get("ADX"), 25.0)
            atr_v = max(_safe_float(closed_bar.get("ATR"), 0.0), 1e-9)
            rh28 = _safe_float(closed_bar.get("RangeHigh28"), 0.0)
            rl28 = _safe_float(closed_bar.get("RangeLow28"), 0.0)
            c_v = _safe_float(closed_bar.get("Close"), 0.0)
            o_v = _safe_float(closed_bar.get("Open"), 0.0)
            h_v = _safe_float(closed_bar.get("High"), 0.0)
            l_v = _safe_float(closed_bar.get("Low"), 0.0)
            rng_v = max(h_v - l_v, 1e-9)
            if rh28 <= rl28 or (rh28 - rl28) <= (1.8 * atr_v) or adx_v >= 23.5:
                return False
            box_pos = (c_v - rl28) / max(rh28 - rl28, 1e-9)
            if side == "LONG":
                l_wick = (min(o_v, c_v) - l_v) / rng_v
                return bool(box_pos <= 0.15 and l_wick >= 0.36 and c_v > o_v)
            else:
                u_wick = (h_v - max(o_v, c_v)) / rng_v
                return bool(box_pos >= 0.85 and u_wick >= 0.36 and c_v < o_v)

        if "g2_keltner" in sid:
            adx_v = _safe_float(closed_bar.get("ADX"), 25.0)
            d20_v = _safe_float(closed_bar.get("Dist_EMA20"), 0.0)
            c_v = _safe_float(closed_bar.get("Close"), 0.0)
            o_v = _safe_float(closed_bar.get("Open"), 0.0)
            h_v = _safe_float(closed_bar.get("High"), 0.0)
            l_v = _safe_float(closed_bar.get("Low"), 0.0)
            body_v = abs(c_v - o_v) / max(h_v - l_v, 1e-9)
            if adx_v >= 24.0 or body_v < 0.35:
                return False
            return bool((d20_v <= -1.70 and c_v > o_v) if side == "LONG" else (d20_v >= 1.70 and c_v < o_v))

        if "g2_bb_rsi" in sid:
            adx_v = _safe_float(closed_bar.get("ADX"), 25.0)
            bbp_v = _safe_float(closed_bar.get("BB_PctB"), 0.5)
            rsi_v = _safe_float(closed_bar.get("RSI_14"), 50.0)
            c_v = _safe_float(closed_bar.get("Close"), 0.0)
            o_v = _safe_float(closed_bar.get("Open"), 0.0)
            if adx_v >= 24.0:
                return False
            return bool((bbp_v <= 0.08 and rsi_v <= 34.0 and c_v > o_v) if side == "LONG" else (bbp_v >= 0.92 and rsi_v >= 66.0 and c_v < o_v))

        if "trendline" in sid:
            swing_lows = closed_bar.get("SwingLows")
            swing_highs = closed_bar.get("SwingHighs")
            adx_v = _safe_float(closed_bar.get("ADX"), 0.0)
            atr_v = max(_safe_float(closed_bar.get("ATR"), 0.0), 1e-9)
            c_v = _safe_float(closed_bar.get("Close"), 0.0)
            o_v = _safe_float(closed_bar.get("Open"), 0.0)
            c_bar_idx = _safe_int(closed_bar.get("ClosedBarIdx"), -1)
            if adx_v < 17.0 or c_bar_idx < 0:
                return False
            if side == "LONG" and isinstance(swing_lows, list) and len(swing_lows) >= 2:
                (i1, p1), (i2, p2) = swing_lows[-2], swing_lows[-1]
                if i2 > i1 and p2 > p1:
                    tl_px = p2 + ((p2 - p1) / float(i2 - i1)) * float(c_bar_idx - i2)
                    dist_a = (c_v - tl_px) / atr_v
                    return bool(-0.25 <= dist_a <= 0.65 and c_v > o_v)
            elif side == "SHORT" and isinstance(swing_highs, list) and len(swing_highs) >= 2:
                (i1, p1), (i2, p2) = swing_highs[-2], swing_highs[-1]
                if i2 > i1 and p2 < p1:
                    tl_px = p2 + ((p2 - p1) / float(i2 - i1)) * float(c_bar_idx - i2)
                    dist_a = (tl_px - c_v) / atr_v
                    return bool(-0.25 <= dist_a <= 0.65 and c_v < o_v)
            return False

        if "donchian" in sid:
            if side == "LONG":
                return bool(
                    closed_bar["Supertrend"] == 1
                    and (
                        curr_open >= closed_bar["Donchian_High"] * 0.996
                        or (closed_bar["Close"] > closed_bar["EMA_20"] and closed_bar["DI_Spread"] > 0)
                    )
                )
            else:
                return bool(
                    closed_bar["Supertrend"] == -1
                    and (
                        curr_open <= closed_bar["Donchian_Low"] * 1.004
                        or (closed_bar["Close"] < closed_bar["EMA_20"] and closed_bar["DI_Spread"] < 0)
                    )
                )
        if "vol_surge" in sid or "ob_imbalance" in sid or "supertrend_adx" in sid:
            return bool(closed_bar["Vol_Surge"] >= 1.25 and closed_bar["ADX"] >= 20 and (
                (closed_bar["DI_Spread"] > 0) if side == "LONG" else (closed_bar["DI_Spread"] < 0)
            ))
        if "bb_" in sid or "squeeze" in sid:
            return bool(closed_bar["BB_Width"] > closed_bar["BB_Width_Med"] and (
                (closed_bar["BB_PctB"] > 0.72) if side == "LONG" else (closed_bar["BB_PctB"] < 0.28)
            ))
        if "stoch" in sid or "rsi" in sid:
            return bool((closed_bar["RSI_14"] < 44 and closed_bar["Stoch_K"] > closed_bar["Stoch_D"]) if side == "LONG" else (
                closed_bar["RSI_14"] > 56 and closed_bar["Stoch_K"] < closed_bar["Stoch_D"]
            ))
        if "ema_ribbon" in sid:
            return bool((closed_bar["EMA_20"] > closed_bar["EMA_50"] and curr_open > closed_bar["EMA_20"]) if side == "LONG" else (
                closed_bar["EMA_20"] < closed_bar["EMA_50"] and curr_open < closed_bar["EMA_20"]
            ))
        if "macd" in sid:
            return bool((closed_bar["MACD_Hist"] > 0 and closed_bar["Dist_EMA200"] > 0) if side == "LONG" else (
                closed_bar["MACD_Hist"] < 0 and closed_bar["Dist_EMA200"] < 0
            ))
        if "supertrend" in sid or "sar" in sid:
            return bool((closed_bar["Supertrend"] == 1) if side == "LONG" else (closed_bar["Supertrend"] == -1))
        if "vwap" in sid or "poc" in sid or "fvg" in sid or "liq_sweep" in sid or "orb" in sid:
            return bool((closed_bar["VWAP_Dev"] < -0.45 and closed_bar["Vol_Surge"] >= 1.15) if side == "LONG" else (
                closed_bar["VWAP_Dev"] > 0.45 and closed_bar["Vol_Surge"] >= 1.15
            ))
        return bool((closed_bar["DI_Spread"] > 0 and closed_bar["Vol_Surge"] >= 1.05) if side == "LONG" else (
            closed_bar["DI_Spread"] < 0 and closed_bar["Vol_Surge"] >= 1.05
        ))

    def _load_booster(self, cache_key: str, model_dict: dict) -> xgb.Booster:
        if not is_valid_xgb_model(model_dict):
            raise ValueError(f"Invalid XGBoost booster format for {cache_key} (missing 'learner'). Retrain Paper model.")
        if cache_key in self._booster_cache:
            return self._booster_cache[cache_key]
        if len(self._booster_cache) >= 32:
            oldest_key = next(iter(self._booster_cache))
            self._booster_cache.pop(oldest_key, None)

        sanitized_dict = _sanitize_xgb_model_json(model_dict)
        booster = xgb.Booster()
        os.makedirs(self.live_dir, exist_ok=True)
        tmp_path = os.path.join(self.live_dir, f"_tmp_live_{cache_key}_{os.getpid()}.json")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(sanitized_dict, f)
            booster.load_model(tmp_path)
            self._booster_cache[cache_key] = booster
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        return booster

    def _arm_native_trailing_stop(self, bybit_sym: str, side: str, entry: float, tp_dist: float, tick_size: str, trail_ratio: float) -> bool:
        if entry <= 0.0 or tp_dist <= 0.0:
            return False
        act_px = (entry + tp_dist * 0.70) if side == "LONG" else (entry - tp_dist * 0.70)
        trail_dist = max(tp_dist * min(max(trail_ratio, 0.40), 0.50), _safe_float(tick_size, 0.0001) * 2.0)
        try:
            t_res = self._bybit_request("POST", "/v5/position/trading-stop", body={
                "category": self.category,
                "symbol": bybit_sym,
                "trailingStop": self._format_step(trail_dist, tick_size),
                "activePrice": self._format_step(act_px, tick_size),
                "positionIdx": 0,
            })
            return t_res.get("retCode") == 0
        except Exception:
            return False

    def _get_pending_entry_order_symbols(self) -> set:
        if not self.api_key or not self.api_secret:
            return set()
        pending = set()
        try:
            res = self._bybit_request(
                "GET",
                "/v5/order/realtime",
                params={"category": self.category, "settleCoin": "USDT", "openOnly": 0, "limit": 20},
            )
            for ord_item in (res.get("result", {}) or {}).get("list", []) or []:
                if not ord_item.get("reduceOnly"):
                    sym = str(ord_item.get("symbol", "")).upper()
                    if sym:
                        pending.add(sym)
        except Exception:
            pass
        return pending

    def sync_positions_and_trail_stops(self, state: dict, trail_ratio: float = 0.45, add_bot_log=None) -> list:
        if not self.api_key or not self.api_secret:
            return []
        res = self._bybit_request("GET", "/v5/position/list", {"category": self.category, "settleCoin": "USDT"})
        tp1_done_map = state.setdefault("tp1_partial_done", {})
        native_trail_map = state.setdefault("native_trail_set", {})
        pos_opened_map = state.setdefault("position_opened_ms", {})
        entry_meta_map = state.setdefault("position_entry_meta", {})
        now_ms = int(time.time() * 1000)
        active_list = []

        for item in res.get("result", {}).get("list", []):
            size = _safe_float(item.get("size"), 0.0)
            if size <= 0.0:
                continue

            bybit_sym = str(item.get("symbol", "")).upper()
            side = "LONG" if str(item.get("side", "")).lower() == "buy" else "SHORT"
            entry = _safe_float(item.get("avgPrice"), 0.0)
            mark = _safe_float(item.get("markPrice"), entry)
            cur_sl = _safe_float(item.get("stopLoss"), 0.0)
            cur_tp = _safe_float(item.get("takeProfit"), 0.0)
            cur_trail = _safe_float(item.get("trailingStop"), 0.0)
            created_ms = _safe_int(item.get("createdTime"), 0)
            opened_ms = _safe_int(pos_opened_map.get(bybit_sym), created_ms or now_ms)
            pos_opened_map[bybit_sym] = opened_ms

            if entry > 0.0 and mark > 0.0 and cur_tp > 0.0:
                tp_dist = abs(cur_tp - entry)
                profit_dist = (mark - entry) if side == "LONG" else (entry - mark)
                filters = self._get_instrument_filters(bybit_sym)
                pos_key = f"{bybit_sym}_{entry:.6f}"

                if filters and cur_trail <= 0.0 and not native_trail_map.get(pos_key, False):
                    _, _, _, tick_size = filters
                    if self._arm_native_trailing_stop(bybit_sym, side, entry, tp_dist, tick_size, trail_ratio):
                        native_trail_map[pos_key] = True

                age_hours = (now_ms - opened_ms) / 3_600_000.0
                pos_is_chop = bool((entry_meta_map.get(bybit_sym) or {}).get("is_chop_gear", False))
                max_stagnant_h = 2.0 if pos_is_chop else 8.0
                if filters and age_hours >= max_stagnant_h and tp_dist > 0.0 and profit_dist < (tp_dist * 0.25):
                    qty_step, _, _, tick_size = filters
                    if profit_dist >= 0.0:
                        close_side = "Sell" if side == "LONG" else "Buy"
                        c_res = self._bybit_request("POST", "/v5/order/create", body={
                            "category": self.category,
                            "symbol": bybit_sym,
                            "side": close_side,
                            "orderType": "Market",
                            "qty": self._format_step(size, qty_step),
                            "timeInForce": "IOC",
                            "reduceOnly": True,
                            "positionIdx": 0,
                        })
                        if c_res.get("retCode") == 0:
                            if add_bot_log:
                                add_bot_log(f"TIME-STOP EXIT ({age_hours:.1f}h stagnant): Closed {bybit_sym} {side} near breakeven/profit to free capital.")
                            continue
                    else:
                        tight_sl = (entry - tp_dist * 0.25) if side == "LONG" else (entry + tp_dist * 0.25)
                        valid_vs_mark = (tight_sl < mark * 0.9985) if side == "LONG" else (tight_sl > mark * 1.0015)
                        better_than_cur = (cur_sl == 0.0 or tight_sl > cur_sl * 1.0020) if side == "LONG" else (cur_sl == 0.0 or tight_sl < cur_sl * 0.9980)
                        if valid_vs_mark and better_than_cur:
                            self._bybit_request("POST", "/v5/position/trading-stop", body={
                                "category": self.category,
                                "symbol": bybit_sym,
                                "stopLoss": self._format_step(tight_sl, tick_size),
                                "slTriggerBy": "MarkPrice",
                                "positionIdx": 0,
                            })
                            cur_sl = tight_sl
                            if add_bot_log:
                                add_bot_log(f"TIME-STOP GUARD ({age_hours:.1f}h stagnant): Tightened {bybit_sym} {side} SL to {self._format_step(tight_sl, tick_size)}")

                if filters and tp_dist > 0.0 and profit_dist >= tp_dist * 0.65:
                    qty_step, min_qty, min_notional, tick_size = filters
                    be_sl = (entry + tp_dist * 0.15) if side == "LONG" else (entry - tp_dist * 0.15)
                    need_be = (cur_sl == 0.0 or be_sl > cur_sl * 1.0020) if side == "LONG" else (
                        cur_sl == 0.0 or be_sl < cur_sl * 0.9980
                    )
                    if need_be:
                        self._bybit_request("POST", "/v5/position/trading-stop", body={
                            "category": self.category,
                            "symbol": bybit_sym,
                            "stopLoss": self._format_step(be_sl, tick_size),
                            "slTriggerBy": "MarkPrice",
                            "positionIdx": 0,
                        })
                        cur_sl = be_sl

                if filters and tp_dist > 0.0 and profit_dist >= tp_dist * 0.85:
                    qty_step, min_qty, min_notional, tick_size = filters
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
                    cand_sl = max(entry + tp_dist * 0.55, mark - trail_offset) if side == "LONG" else min(
                        entry - tp_dist * 0.55, mark + trail_offset
                    )
                    should_tighten = (cur_sl == 0.0 or cand_sl > cur_sl * 1.0020) if side == "LONG" else (
                        cur_sl == 0.0 or cand_sl < cur_sl * 0.9980
                    )
                    if should_tighten:
                        self._bybit_request("POST", "/v5/position/trading-stop", body={
                            "category": self.category,
                            "symbol": bybit_sym,
                            "stopLoss": self._format_step(cand_sl, tick_size),
                            "slTriggerBy": "MarkPrice",
                            "positionIdx": 0,
                        })

            active_list.append({
                "symbol": bybit_sym,
                "side": side,
                "qty": str(item.get("size", "0")),
                "entry_price": entry,
                "mark_price": mark,
                "unrealized_pnl": _safe_float(item.get("unrealisedPnl"), 0.0),
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
        state["virtual_open_positions"] = []
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
        bot_logs_enabled: bool = True,
    ) -> dict:
        if not os.path.exists(self.master_path):
            raise FileNotFoundError(f"Master ML Model not found at {self.master_path}")

        leverage = min(max(int(leverage), 1), 20)
        effective_max_positions = max(1, int(max_open_positions))
        with open(self.master_path, "r", encoding="utf-8-sig") as f:
            master = json.load(f)

        self._write_lightweight_live_summary(master)

        features = master.get("features", FEATURES)
        portfolio = master.get("portfolio", {})
        if not portfolio:
            raise ValueError("No coins found in Live ML Model. Promote trained coins from Paper to Live first.")

        state = self._load_state()
        traded_candle_ts = state.setdefault("traded_candle_ts", {})
        last_trade_epoch_ms = state.setdefault("last_trade_epoch_ms", {})
        pos_opened_map = state.setdefault("position_opened_ms", {})
        entry_meta_map = state.setdefault("position_entry_meta", {})
        loss_cooldown_until_ms = state.setdefault("loss_cooldown_until_ms", {})
        agentic_veto_until_ms = state.setdefault("agentic_veto_until_ms", {})
        restricted_until_ms = state.setdefault("restricted_symbols_until_ms", {})
        notified_block_layers = state.setdefault("last_notified_block_layer", {})
        last_shadow_eval_ms = state.setdefault("last_shadow_eval_ms", {})
        daily_coin_trades = state.setdefault("daily_coin_trades", {})
        native_trail_map = state.setdefault("native_trail_set", {})
        bot_logs = state.setdefault("last_bot_logs", [])

        agentic_mem = self._load_agentic_memory()
        agentic_coin_scores = agentic_mem.get("coin_scores", {}) or {}
        session_stats = agentic_mem.get("session_stats", {}) or {}
        quarantine_mem = agentic_mem.get("auto_quarantine", {}) or {}

        gem_shadow_on, gem_shadow_key = self._resolve_gemini_shadow_settings()

        diagnostics = []
        order_rejections = []
        trade_blocks = []
        radar_signals = []
        radar_by_disp = {}
        per_coin_summaries = []
        returns_72h_map = {}

        now_ist = datetime.datetime.now(IST_TZ)
        today_ist_str = now_ist.strftime("%Y-%m-%d")
        clock_str = now_ist.strftime("%H:%M:%S IST")
        curr_session = get_ist_session_name(now_ist)

        sess_data = session_stats.get(curr_session, {}) if isinstance(session_stats, dict) else {}
        sess_wins = _safe_int(sess_data.get("wins"), 0)
        sess_losses = _safe_int(sess_data.get("losses"), 0)
        sess_total = sess_wins + sess_losses
        session_prob_penalty = 0.03 if (sess_total >= 4 and (sess_wins / max(sess_total, 1)) < 0.35) else 0.0

        def add_bot_log(msg: str):
            if bot_logs_enabled:
                bot_logs.append(f"[{clock_str}] {msg}")

        def emit_layer_block_alert(bybit_sym: str, side: str, layer_key: str, alert_msg: str):
            """Sends ONLY ONE notification per (coin, side, blocking_layer); re-alerts only if a new layer blocks."""
            coin_side_key = f"{bybit_sym}_{side}"
            prev_layer = str(notified_block_layers.get(coin_side_key, ""))
            if prev_layer != layer_key:
                notified_block_layers[coin_side_key] = layer_key
                trade_blocks.append(alert_msg)

        if daily_coin_trades.get("_date_ist") != today_ist_str:
            daily_coin_trades.clear()
            daily_coin_trades["_date_ist"] = today_ist_str
            state["daily_total_trades"] = 0
            add_bot_log(f"Daily Counter Auto-Renewed for {today_ist_str} (12:00 AM IST).")

        daily_total_trades = _safe_int(state.get("daily_total_trades"), 0)
        now_ms = int(time.time() * 1000)

        btc_macro_bearish = False
        btc_adx = 20.0
        btc_natr = 1.8
        if btc_macro_guard:
            btc_st, btc_adx, btc_natr = self._get_cached_4h_metrics("BTCUSDT")
            btc_macro_bearish = bool(btc_st == -1)

        curr_regime = (
            "VOLATILE" if btc_natr >= 2.8 else (
                "BEAR_TREND" if (btc_macro_bearish and btc_adx >= 18.0) else (
                    "BULL_TREND" if ((not btc_macro_bearish) and btc_adx >= 18.0) else "CHOP"
                )
            )
        )

        has_api_creds = bool(self.api_key and self.api_secret)
        if has_api_creds:
            self._record_recent_stop_losses_and_reflect(state, now_ms, add_bot_log, curr_session=curr_session, curr_regime=curr_regime)
        funding_rates, mark_prices = self._fetch_tickers_cached() if (crypto_enabled or forex_enabled) else ({}, {})

        gate_adjustments = {}
        try:
            import importlib
            ai_regime_copilot = importlib.import_module("ai_regime_copilot")
            if hasattr(ai_regime_copilot, "evaluate_shadow_counterfactuals"):
                gate_adjustments = ai_regime_copilot.evaluate_shadow_counterfactuals(self.live_dir, mark_prices) or {}
        except Exception:
            gate_adjustments = {}

        if has_api_creds:
            bal_res = self._bybit_request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": "USDT"})
            bal_ret_code = _safe_int(bal_res.get("retCode"), -1) if bal_res else -1
            if bal_ret_code != 0:
                bal_msg = bal_res.get("retMsg", "Unable to fetch Unified USDT balance")
                err_str = f"Bybit Wallet API Error [{bal_ret_code}]: {BYBIT_ERROR_HINTS.get(bal_ret_code, bal_msg)}"
                order_rejections.append(err_str)
                add_bot_log(err_str)
            avail_usdt, total_equity_usdt = self._extract_wallet_metrics(bal_res) if bal_ret_code == 0 else (0.0, 0.0)
        else:
            avail_usdt, total_equity_usdt = 10000.0, 10000.0

        peak_equity = _safe_float(state.get("peak_equity_usdt"), 0.0)
        if total_equity_usdt > peak_equity:
            peak_equity = total_equity_usdt
            state["peak_equity_usdt"] = round(peak_equity, 4)
        peak_dd_pct = ((peak_equity - total_equity_usdt) / peak_equity * 100.0) if (peak_equity > 0.0 and total_equity_usdt > 0.0) else 0.0
        effective_risk_pct = risk_per_trade_pct * 0.5 if peak_dd_pct >= 8.0 else risk_per_trade_pct
        peak_dd_paused = bool(peak_dd_pct >= 10.0)

        sl_mult, tp_mult, trail_ratio = self._regime_exit_multipliers(btc_adx, btc_natr, btc_macro_bearish)
        open_positions = self.sync_positions_and_trail_stops(state, trail_ratio=trail_ratio, add_bot_log=add_bot_log)
        virtual_positions = self._sync_virtual_positions(state, mark_prices, trail_ratio, now_ms, add_bot_log, curr_session=curr_session, curr_regime=curr_regime)
        pending_order_syms = self._get_pending_entry_order_symbols() if live_orders_armed else set()
        active_pos_records = list(open_positions) + list(virtual_positions)
        open_syms = {p["symbol"] for p in active_pos_records} | pending_order_syms

        active_items = []
        fetch_tasks = {}
        for symbol, entry in portfolio.items():
            if not isinstance(entry, dict):
                continue
            market_type = "FOREX" if is_forex_symbol(symbol) else entry.get("market", "CRYPTO").upper()
            if (market_type == "CRYPTO" and not crypto_enabled) or (market_type == "FOREX" and not forex_enabled):
                continue
            cfg = entry.get("config", {})
            bybit_sym = (cfg.get("bybit_symbol") or symbol.replace("/", "")).upper()
            tf_str = (cfg.get("timeframe") or entry.get("timeframe") or "1h").lower()
            interval = TIMEFRAME_TO_BYBIT_INTERVAL.get(tf_str, "60")
            active_items.append((symbol, entry, market_type, cfg, bybit_sym, tf_str, interval))
            fetch_tasks[(bybit_sym, interval)] = (bybit_sym, interval, 230)
            g2_sub = entry.get("gear2_model")
            if isinstance(g2_sub, dict) and isinstance(g2_sub.get("models"), dict):
                g2_cfg = g2_sub.get("config", {}) or {}
                g2_tf_str = str(g2_sub.get("timeframe") or g2_cfg.get("timeframe") or "5m").lower()
                g2_int = TIMEFRAME_TO_BYBIT_INTERVAL.get(g2_tf_str, "5")
                if (bybit_sym, g2_int) not in fetch_tasks:
                    fetch_tasks[(bybit_sym, g2_int)] = (bybit_sym, g2_int, 230)
            if interval != "240":
                cached = self._trend_4h_cache.get(bybit_sym)
                if not cached or (time.time() - cached[0]) >= 600.0:
                    fetch_tasks[(bybit_sym, "240")] = (bybit_sym, "240", 220)

        fetched_candles = {}
        if fetch_tasks:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {
                    k: pool.submit(self.fetch_bybit_candles, v[0], v[1], v[2])
                    for k, v in fetch_tasks.items()
                }
                for k, fut in futures.items():
                    try:
                        fetched_candles[k] = fut.result()
                    except Exception:
                        fetched_candles[k] = pd.DataFrame()

        now_s = time.time()
        for (sym_k, int_k), df_4h_raw in fetched_candles.items():
            if int_k == "240" and len(df_4h_raw) >= 210:
                df_4h = self.build_features(df_4h_raw)
                if not df_4h.empty:
                    ref_bar = df_4h.iloc[-2] if len(df_4h) >= 2 else df_4h.iloc[-1]
                    self._trend_4h_cache[sym_k] = (
                        now_s,
                        int(ref_bar["Supertrend"]),
                        _safe_float(ref_bar.get("ADX"), 20.0),
                        _safe_float(ref_bar.get("NATR"), 1.8),
                    )

        candidates = []
        top_wait_tuple = None
        coins_scanned_count = 0
        any_conformal_tight = False

        for symbol, entry, market_type, cfg, bybit_sym, tf_str, interval in active_items:
            coins_scanned_count += 1
            disp_sym = symbol if "/" in symbol else (f"{bybit_sym[:-4]}/USDT" if bybit_sym.endswith("USDT") else bybit_sym)

            # Dual-Gear Regime Switcher: Automatically shift to 5m/15m Gear 2 chop model when ADX < 20.5 & 4H ADX < 25.0
            active_entry = entry
            active_cfg = cfg
            active_tf_str = tf_str
            active_interval = interval
            booster_prefix = f"{bybit_sym}_g1"
            gear_badge = ""

            g2_sub = entry.get("gear2_model")
            primary_is_g2 = bool(entry.get("is_chop_gear", False)) or str(cfg.get("strategy_id") or entry.get("strategy_id") or "").lower().startswith("g2_")
            if (not primary_is_g2) and isinstance(g2_sub, dict) and isinstance(g2_sub.get("models"), dict):
                g2_models_chk = g2_sub.get("models", {})
                if is_valid_xgb_model(g2_models_chk.get("long")) and is_valid_xgb_model(g2_models_chk.get("short")):
                    df_g1_chk = fetched_candles.get((bybit_sym, interval), pd.DataFrame())
                    cached_4h_chk = self._trend_4h_cache.get(bybit_sym)
                    adx_4h_chk = cached_4h_chk[2] if cached_4h_chk else 20.0
                    g1_adx_chk = 25.0
                    if len(df_g1_chk) >= 210:
                        df_g1_feat = self.build_features(df_g1_chk)
                        if len(df_g1_feat) >= 2:
                            g1_adx_chk = _safe_float(df_g1_feat.iloc[-2].get("ADX"), 25.0)
                    if g1_adx_chk < 23.5 and adx_4h_chk < 27.0:
                        g2_cfg_cand = g2_sub.get("config", {}) or {}
                        g2_tf_cand = str(g2_sub.get("timeframe") or g2_cfg_cand.get("timeframe") or "5m").lower()
                        g2_int_cand = TIMEFRAME_TO_BYBIT_INTERVAL.get(g2_tf_cand, "5")
                        if len(fetched_candles.get((bybit_sym, g2_int_cand), pd.DataFrame())) >= 210:
                            active_entry = g2_sub
                            active_cfg = g2_cfg_cand
                            active_tf_str = g2_tf_cand
                            active_interval = g2_int_cand
                            booster_prefix = f"{bybit_sym}_g2"
                            gear_badge = f" [G2-{g2_tf_cand.upper()}]"

            entry = active_entry
            cfg = active_cfg
            tf_str = active_tf_str
            interval = active_interval
            models_map = entry.get("models", {})
            long_data = models_map.get("long") or models_map.get("long_model")
            short_data = models_map.get("short") or models_map.get("short_model")
            if not is_valid_xgb_model(long_data) or not is_valid_xgb_model(short_data):
                err_msg = f"{bybit_sym}: Missing valid XGBoost booster weights"
                diagnostics.append(err_msg)
                order_rejections.append(err_msg)
                per_coin_summaries.append(f"{bybit_sym}: No ML weights")
                sig_entry = {
                    "symbol": disp_sym,
                    "direction": "WAIT",
                    "confidence_pct": 18,
                    "setup_score": 18,
                    "mtf_aligned": False,
                    "side": "WAIT",
                    "conviction_pct": 18,
                    "aligned_4h": False,
                    "raw_prob": 0.0,
                }
                radar_signals.append(sig_entry)
                radar_by_disp[disp_sym] = sig_entry
                continue

            strat_id = cfg.get("strategy_id") or entry.get("strategy_id") or "xgb_mtf_v2"
            is_relaxed = bool(cfg.get("relaxed_filter", False))
            custom_rules = cfg.get("custom_rules")

            df_raw = fetched_candles.get((bybit_sym, interval), pd.DataFrame())
            if len(df_raw) < 210:
                skip_msg = f"{bybit_sym}: Insufficient candles ({len(df_raw)}/210 — newly listed or delisted on Bybit)"
                diagnostics.append(skip_msg)
                per_coin_summaries.append(f"{bybit_sym}: {len(df_raw)}/210 bars")
                sig_entry = {
                    "symbol": disp_sym,
                    "direction": "WAIT",
                    "confidence_pct": 22,
                    "setup_score": 22,
                    "mtf_aligned": False,
                    "side": "WAIT",
                    "conviction_pct": 22,
                    "aligned_4h": False,
                    "raw_prob": 0.0,
                }
                radar_signals.append(sig_entry)
                radar_by_disp[disp_sym] = sig_entry
                continue

            df = self.build_features(df_raw)
            if len(df) < 3:
                continue

            returns_72h_map[bybit_sym] = df["Ret_1h"].iloc[-73:-1].to_numpy(dtype=np.float64)

            closed_bar = df.iloc[-2]
            live_bar = df.iloc[-1]
            latest_bar_ms = int(df.index[-1].timestamp() * 1000)

            try:
                import importlib
                ai_regime_copilot = importlib.import_module("ai_regime_copilot")
                if hasattr(ai_regime_copilot, "check_coin_feature_drift"):
                    ai_regime_copilot.check_coin_feature_drift(
                        self.live_dir,
                        bybit_sym,
                        closed_bar.to_dict(),
                        features,
                        models_map.get("feature_means", []),
                        models_map.get("feature_stds", []),
                    )
            except Exception:
                pass

            try:
                x_row = df.iloc[[-2]][features].to_numpy(dtype="float32")
                dmat = xgb.DMatrix(x_row, feature_names=features)
                p_long = float(self._load_booster(f"{booster_prefix}_long", long_data).predict(dmat)[0])
                p_short = float(self._load_booster(f"{booster_prefix}_short", short_data).predict(dmat)[0])
            except Exception as e:
                err_msg = f"{bybit_sym}: Booster error - {e}"
                diagnostics.append(err_msg)
                order_rejections.append(err_msg)
                per_coin_summaries.append(f"{bybit_sym}: Booster err")
                continue

            direction = "LONG" if p_long >= p_short else "SHORT"
            raw_prob = p_long if direction == "LONG" else p_short
            prob_margin = abs(p_long - p_short)
            default_th = 0.24 if market_type == "FOREX" else 0.28
            req_thresh = _safe_float(cfg.get("long_thresh" if direction == "LONG" else "short_thresh"), default_th)

            adx_val = _safe_float(closed_bar.get("ADX"), 0.0)
            natr_val = _safe_float(closed_bar.get("NATR"), 1.5)
            vwap_dev = _safe_float(closed_bar.get("VWAP_Dev"), 0.0)
            is_chop_gear = str(strat_id).lower().startswith("g2_") or (interval in ("5", "15") and adx_val < 23.0)

            is_quarantined = (
                now_ms < _safe_int(restricted_until_ms.get(bybit_sym), 0)
                or now_ms < _safe_int(quarantine_mem.get(bybit_sym), 0)
            )
            raw_cd_until = _safe_int(loss_cooldown_until_ms.get(bybit_sym), 0)
            eff_cd_until = max(0, raw_cd_until - (195 * 60 * 1000)) if is_chop_gear else raw_cd_until
            in_cooldown = (
                is_quarantined
                or now_ms < eff_cd_until
                or now_ms < _safe_int(agentic_veto_until_ms.get(bybit_sym), 0)
                or bybit_sym in open_syms
            )

            oi_delta_4h = 0.0
            ls_skew = 0.0
            if market_type == "CRYPTO" and (not in_cooldown) and raw_prob >= (req_thresh * 0.85):
                pos_metrics = self._fetch_bybit_positioning_metrics(bybit_sym)
                oi_delta_4h = _safe_float(pos_metrics.get("oi_delta_4h"), 0.0)
                ls_skew = _safe_float(pos_metrics.get("ls_skew"), 0.0)

            if interval == "240":
                trend_4h = int(closed_bar["Supertrend"])
                adx_4h = adx_val
            else:
                cached_4h = self._trend_4h_cache.get(bybit_sym)
                trend_4h = cached_4h[1] if cached_4h else 0
                adx_4h = cached_4h[2] if cached_4h else 20.0

            raw_4h_ok = (direction == "LONG" and trend_4h == 1) or (direction == "SHORT" and trend_4h == -1)
            high_conv_rev = (raw_prob >= req_thresh + 0.06) and (_safe_float(closed_bar["Vol_Surge"], 0.0) >= 1.15)

            decoupled_leader = raw_4h_ok and (adx_val >= 22.0) and (raw_prob >= req_thresh + 0.04)
            guard_blocked = (
                btc_macro_bearish
                and direction == "LONG"
                and market_type == "CRYPTO"
                and not bybit_sym.startswith("BTC")
                and not decoupled_leader
            )

            fr_val = _safe_float(funding_rates.get(bybit_sym), 0.0) if market_type == "CRYPTO" else 0.0
            crowded_oi_top = (direction == "LONG" and ls_skew > 48.0 and oi_delta_4h > 4.0) or (
                direction == "SHORT" and ls_skew < -48.0 and oi_delta_4h > 4.0
            )
            funding_blocked = (direction == "LONG" and fr_val > 0.0004) or (direction == "SHORT" and fr_val < -0.0004) or crowded_oi_top

            if (is_chop_gear or "trendline" in str(strat_id).lower()) and len(df) >= 30:
                closed_bar = closed_bar.copy()
                if is_chop_gear and "RangeHigh28" not in closed_bar.index:
                    closed_bar["RangeHigh28"] = float(df["High"].iloc[-29:-1].max())
                    closed_bar["RangeLow28"] = float(df["Low"].iloc[-29:-1].min())
                if "trendline" in str(strat_id).lower():
                    h_arr = df["High"].to_numpy(dtype=float)
                    l_arr = df["Low"].to_numpy(dtype=float)
                    c_bar_idx = len(df) - 2
                    s_lows = [(j, float(l_arr[j])) for j in range(3, max(4, c_bar_idx - 2)) if l_arr[j] == np.min(l_arr[j - 3:j + 4])]
                    s_highs = [(j, float(h_arr[j])) for j in range(3, max(4, c_bar_idx - 2)) if h_arr[j] == np.max(h_arr[j - 3:j + 4])]
                    closed_bar["SwingLows"] = s_lows
                    closed_bar["SwingHighs"] = s_highs
                    closed_bar["ClosedBarIdx"] = c_bar_idx

            chop_4h_runaway = bool(is_chop_gear and adx_4h >= 25.0)
            effective_mtf = str(cfg.get("mtf_alignment_mode") or entry.get("mtf_alignment_mode") or mtf_mode).upper()
            if is_chop_gear:
                mtf_aligned = (not guard_blocked) and (not chop_4h_runaway)
            elif effective_mtf == "OFF":
                mtf_aligned = not guard_blocked
            elif effective_mtf == "SOFT":
                mtf_aligned = (raw_4h_ok or (raw_prob >= (req_thresh + 0.04) and _safe_float(closed_bar["Vol_Surge"], 0.0) >= 1.10)) and not guard_blocked
            else:
                mtf_aligned = (raw_4h_ok or high_conv_rev) and not guard_blocked

            anti_chase_ok, chase_reason = self._passes_anti_chase_guard(closed_bar, live_bar, direction, is_chop_gear=is_chop_gear)
            strat_ok = self._strategy_entry_filter(
                strat_id,
                _safe_float(live_bar["Open"], 0.0),
                closed_bar,
                direction,
                relaxed=is_relaxed,
                custom_rules=custom_rules,
            )

            trained_meta_ok, trained_meta_prob = self._evaluate_meta_labeler(
                models_map.get("meta"),
                raw_prob,
                prob_margin,
                adx_val,
                natr_val,
                vwap_dev,
                oi_delta_4h,
                ls_skew,
            )
            if trained_meta_ok is not None:
                meta_ok = trained_meta_ok
                meta_prob = trained_meta_prob
            else:
                meta_prob = 0.65
                try:
                    import ai_regime_copilot  # type: ignore[import-not-found]
                    if hasattr(ai_regime_copilot, "compute_meta_label_score"):
                        meta_prob = ai_regime_copilot.compute_meta_label_score(
                            p_long, p_short, adx_val, natr_val, vwap_dev, raw_4h_ok, curr_session
                        )
                except Exception:
                    meta_prob = 0.65
                meta_ok = bool(meta_prob >= 0.48)

            learned_score = _safe_float(agentic_coin_scores.get(disp_sym), 0.0)
            gate_prob_delta = _safe_float(gate_adjustments.get("PROB_THRESH"), 0.0)
            base_min_prob = max((req_thresh * 0.94) + gate_prob_delta, 0.19) + session_prob_penalty

            est_tp_rr = (
                min(max(_safe_float(cfg.get("tp1_rr"), 0.95), 0.88), 1.15)
                if is_chop_gear
                else min(max(_safe_float(cfg.get("tp1_rr"), 1.65) * tp_mult, 1.50), 1.85)
            )
            conformal_regime = f"G2_CHOP_{curr_regime}" if is_chop_gear else curr_regime
            conformal_abstain = False
            effective_min_prob = base_min_prob
            kelly_scale = 1.0
            try:
                import ai_regime_copilot  # type: ignore[import-not-found]
                if hasattr(ai_regime_copilot, "get_conformal_threshold_and_kelly"):
                    conformal_abstain, effective_min_prob, kelly_scale, _ = ai_regime_copilot.get_conformal_threshold_and_kelly(
                        self.live_dir, conformal_regime, raw_prob, base_min_prob, rr_ratio=est_tp_rr
                    )
                    if effective_min_prob > base_min_prob + 0.001:
                        any_conformal_tight = True
            except Exception:
                conformal_abstain = False
                effective_min_prob = base_min_prob
                kelly_scale = 1.0

            # 4-Pillar Setup & Confidence Score (12%-97% while scanning; 98%-100% ONLY when all gates pass)
            prob_ratio = raw_prob / max(effective_min_prob, 0.16)
            prob_pts = min(36.0, max(0.0, (prob_ratio - 0.55) * 45.0))
            mtf_pts = 24.0 if mtf_aligned else (8.0 if raw_4h_ok else 0.0)
            strat_pts = 20.0 if strat_ok else 4.0
            adx_pts = (
                min(9.0, max(0.0, ((25.0 - adx_val) / 14.0) * 9.0))
                if is_chop_gear
                else min(9.0, max(0.0, (adx_val / 28.0) * 9.0))
            )
            clean_pts = 8.0 if (anti_chase_ok and not funding_blocked and meta_ok and not conformal_abstain) else 0.0
            setup_score = int(round(min(97.0, max(12.0, prob_pts + mtf_pts + strat_pts + adx_pts + clean_pts))))

            mtf_or_momentum_ok = is_chop_gear or raw_4h_ok or (raw_prob >= effective_min_prob + 0.05 and adx_val >= 18.0)
            min_adx_req = 8.0 if is_chop_gear else 14.5
            grade_ok = (setup_score >= 66) and mtf_or_momentum_ok and (adx_val >= min_adx_req) and (learned_score >= -10.0)

            all_gates_ready = (
                (not in_cooldown)
                and (not peak_dd_paused)
                and (not conformal_abstain)
                and mtf_aligned
                and strat_ok
                and anti_chase_ok
                and meta_ok
                and grade_ok
                and (not funding_blocked)
                and (raw_prob >= effective_min_prob)
            )
            conv_pct = max(98, min(100, setup_score + 4)) if all_gates_ready else setup_score
            aligned_flag = bool(all_gates_ready)

            sig_entry = {
                "symbol": disp_sym,
                "direction": direction,
                "confidence_pct": conv_pct,
                "setup_score": setup_score,
                "mtf_aligned": aligned_flag,
                "side": direction,
                "conviction_pct": conv_pct,
                "aligned_4h": aligned_flag,
                "raw_prob": round(raw_prob, 4),
            }
            radar_signals.append(sig_entry)
            radar_by_disp[disp_sym] = sig_entry

            if is_quarantined:
                emit_layer_block_alert(bybit_sym, direction, "QUARANTINE_110126", f"{bybit_sym}: Auto-quarantined on Bybit [110126]")
                per_coin_summaries.append(f"{bybit_sym} {direction} ({conv_pct}%): Quarantined")
                continue

            if now_ms < eff_cd_until:
                rem_min = max(1, int((eff_cd_until - now_ms) / 60000))
                cd_lbl = "45m" if is_chop_gear else "4h"
                cd_msg = f"{bybit_sym} {direction} ({conv_pct}%): {cd_lbl} post-loss cooldown ({rem_min}m left)"
                emit_layer_block_alert(bybit_sym, direction, "LOSS_COOLDOWN", cd_msg)
                diagnostics.append(cd_msg)
                per_coin_summaries.append(f"{bybit_sym} {direction} ({conv_pct}%): Cooldown {rem_min}m")
                if top_wait_tuple is None or conv_pct > top_wait_tuple[0]:
                    top_wait_tuple = (conv_pct, cd_msg)
                continue

            if now_ms < _safe_int(agentic_veto_until_ms.get(bybit_sym), 0):
                emit_layer_block_alert(bybit_sym, direction, "AGENTIC_VETO", f"{bybit_sym}: Vetoed by Agentic AI Supervisor")
                per_coin_summaries.append(f"{bybit_sym}: AI Veto")
                continue

            if bybit_sym in open_syms:
                per_coin_summaries.append(f"{bybit_sym}: Position Open")
                continue

            quality_edge = (raw_prob - effective_min_prob) + (0.04 if raw_4h_ok else 0.0) + (0.02 if adx_val >= 19.0 else 0.0) + (learned_score * 0.002)

            if all_gates_ready:
                if conv_pct >= 99 and ((raw_4h_ok and adx_val >= 22.0) or (is_chop_gear and adx_val < 20.5)):
                    grade_tag = "A+"
                elif conv_pct >= 98 and (adx_val >= 18.5 or (is_chop_gear and adx_val < 23.0)):
                    grade_tag = "A"
                else:
                    grade_tag = "B+"

                per_coin_summaries.append(f"{bybit_sym}{gear_badge} {direction} ({conv_pct}% READY)")
                candidates.append({
                    "edge": quality_edge,
                    "conv_pct": conv_pct,
                    "grade": grade_tag,
                    "raw_prob": raw_prob,
                    "req_thresh": effective_min_prob,
                    "meta_prob": meta_prob,
                    "kelly_scale": kelly_scale,
                    "prior_wr_pct": _safe_float(entry.get("win_rate_pct"), 55.0),
                    "bybit_sym": bybit_sym,
                    "disp_sym": disp_sym,
                    "market_type": market_type,
                    "direction": direction,
                    "timeframe": tf_str.upper(),
                    "strategy_id": strat_id,
                    "conformal_regime": conformal_regime,
                    "cfg": cfg,
                    "closed_bar": closed_bar.copy(),
                    "live_bar": live_bar.copy(),
                    "latest_bar_ms": latest_bar_ms,
                    "candles_df": df.iloc[:-1].tail(72).copy(),
                    "adx_val": adx_val,
                    "fr_val": fr_val,
                    "oi_delta_4h": oi_delta_4h,
                    "is_chop_gear": is_chop_gear,
                    "ls_skew": ls_skew,
                })
            else:
                reasons = []
                primary_gate = "GENERAL"
                if peak_dd_paused:
                    reasons.append(f"Peak DD Guard ({peak_dd_pct:.1f}%>=10%)")
                    primary_gate = "PEAK_DD"
                if guard_blocked:
                    reasons.append("BTC 4H Guard")
                    primary_gate = "BTC_4H_GUARD"
                elif chop_4h_runaway:
                    reasons.append(f"4H Runaway ADX ({adx_4h:.1f}>=25)")
                    primary_gate = "CHOP_4H_RUNAWAY"
                elif not mtf_aligned:
                    reasons.append(f"4H MTF ({effective_mtf})")
                    primary_gate = "MTF_4H"
                if funding_blocked:
                    reasons.append(f"Crowded positioning (FR {fr_val * 100:+.3f}%, LS {ls_skew:+.0f})")
                    primary_gate = "FUNDING_OI_CROWDED"
                if not anti_chase_ok:
                    reasons.append(chase_reason)
                    primary_gate = "ANTI_CHASE"
                if not meta_ok:
                    reasons.append(f"Meta-Filter ({meta_prob:.2f})")
                    primary_gate = "META_LABEL"
                if not strat_ok:
                    reasons.append(f"Waiting {strat_id}")
                    primary_gate = f"STRATEGY_{strat_id}"
                if not grade_ok:
                    reasons.append(f"Below B floor (Setup {setup_score}%<66% or ADX {adx_val:.1f}<{min_adx_req:.1f})")
                    if primary_gate == "GENERAL":
                        primary_gate = "GRADE_FLOOR"
                if conformal_abstain or raw_prob < effective_min_prob:
                    tag_p = "Conformal Prob" if conformal_abstain else "Prob"
                    reasons.append(f"{tag_p} {raw_prob:.2f}<{effective_min_prob:.2f}")
                    primary_gate = "PROB_THRESH"

                # Run Layer 1 vs Gemini Shadow-Compare on high-setup candidates (setup_score >= 78, max once/10m per coin)
                if gem_shadow_on and gem_shadow_key and setup_score >= 78:
                    last_sh = _safe_int(last_shadow_eval_ms.get(bybit_sym), 0)
                    if (now_ms - last_sh) >= 10 * 60 * 1000:
                        last_shadow_eval_ms[bybit_sym] = now_ms
                        try:
                            import ai_regime_copilot
                            sh_res = ai_regime_copilot.evaluate_pre_trade_human_brain(
                                live_dir=self.live_dir,
                                symbol=bybit_sym,
                                side=direction,
                                candles_df=df.iloc[:-1].tail(72).copy(),
                                adx_val=adx_val,
                                funding_rate=fr_val,
                                session_name=curr_session,
                                gemini_enabled=True,
                                gemini_api_key=gem_shadow_key,
                                is_chop_regime=is_chop_gear,
                                timeframe=tf_str.upper(),
                                strategy_name=str(strat_id),
                            )
                            if sh_res.get("shadow_log"):
                                add_bot_log(sh_res["shadow_log"])
                        except Exception:
                            pass

                if setup_score >= 75:
                    try:
                        import ai_regime_copilot
                        if hasattr(ai_regime_copilot, "record_shadow_counterfactual"):
                            px_s = _safe_float(live_bar["Close"], 0.0)
                            atr_s = _safe_float(closed_bar["ATR"], 0.0)
                            if px_s > 0.0 and atr_s > 0.0:
                                r_dist = max(_safe_float(cfg.get("sl_atr"), 1.65 if is_chop_gear else 1.35) * atr_s, px_s * (0.0020 if market_type == "FOREX" else (0.0038 if is_chop_gear else 0.0155)))
                                sl_s = (px_s - r_dist) if direction == "LONG" else (px_s + r_dist)
                                tp_s = (px_s + r_dist * 1.75) if direction == "LONG" else (px_s - r_dist * 1.75)
                                ai_regime_copilot.record_shadow_counterfactual(
                                    self.live_dir, bybit_sym, direction, px_s, sl_s, tp_s, primary_gate, setup_score
                                )
                    except Exception:
                        pass

                short_why = reasons[0] if reasons else "Waiting"
                per_coin_summaries.append(f"{bybit_sym}{gear_badge} {direction} ({conv_pct}%): {short_why}")
                block_msg = f"{bybit_sym} {direction} ({conv_pct}%): {', '.join(reasons)} [ADX:{adx_val:.1f}]"
                diagnostics.append(block_msg)
                emit_layer_block_alert(bybit_sym, direction, primary_gate, block_msg)
                if top_wait_tuple is None or conv_pct > top_wait_tuple[0]:
                    top_wait_tuple = (conv_pct, block_msg)

        candidates.sort(key=lambda x: x["edge"], reverse=True)

        def _downgrade_radar(disp_key: str, max_cap_pct: int = 96):
            entry_r = radar_by_disp.get(disp_key)
            if entry_r:
                restored = min(int(entry_r.get("setup_score", entry_r["confidence_pct"])), max_cap_pct)
                entry_r["confidence_pct"] = restored
                entry_r["conviction_pct"] = restored
                entry_r["mtf_aligned"] = False
                entry_r["aligned_4h"] = False

        live_trades_this_cycle = 0

        for cand in candidates:
            bybit_sym = cand["bybit_sym"]
            disp_sym = cand["disp_sym"]
            if daily_drawdown_pct >= max_daily_loss_pct and max_daily_loss_pct > 0.0:
                _downgrade_radar(disp_sym)
                msg = f"{bybit_sym} {cand['direction']} blocked: Daily loss limit ({daily_drawdown_pct:.2f}% >= {max_daily_loss_pct:.2f}%)"
                diagnostics.append(msg)
                emit_layer_block_alert(bybit_sym, cand["direction"], "DAILY_LOSS_CAP", msg)
                order_rejections.append(msg)
                add_bot_log(msg)
                break
            if len(open_syms) >= effective_max_positions:
                _downgrade_radar(disp_sym)
                msg = f"{bybit_sym} {cand['direction']} skipped: Max open positions reached ({len(open_syms)}/{effective_max_positions})"
                diagnostics.append(msg)
                emit_layer_block_alert(bybit_sym, cand["direction"], "MAX_POSITIONS", msg)
                break

            sector_ok, sector_reason = self._passes_sector_correlation_cap(
                bybit_sym, cand["direction"], active_pos_records
            )
            if not sector_ok:
                _downgrade_radar(disp_sym)
                msg = f"{bybit_sym} {cand['direction']} skipped: {sector_reason}"
                diagnostics.append(msg)
                emit_layer_block_alert(bybit_sym, cand["direction"], "SECTOR_CAP", msg)
                add_bot_log(msg)
                continue

            corr_ok, corr_reason = True, "Corr OK"
            try:
                import ai_regime_copilot
                if hasattr(ai_regime_copilot, "check_rolling_correlation_guard"):
                    corr_ok, corr_reason = ai_regime_copilot.check_rolling_correlation_guard(
                        bybit_sym, cand["direction"], active_pos_records, returns_72h_map, max_corr=0.82
                    )
                else:
                    corr_ok, corr_reason = self._passes_rolling_correlation_guard(
                        bybit_sym, cand["direction"], active_pos_records, returns_72h_map, max_corr=0.82
                    )
            except Exception:
                corr_ok, corr_reason = self._passes_rolling_correlation_guard(
                    bybit_sym, cand["direction"], active_pos_records, returns_72h_map, max_corr=0.82
                )
            if not corr_ok:
                _downgrade_radar(disp_sym)
                msg = f"{bybit_sym} {cand['direction']} skipped: {corr_reason}"
                diagnostics.append(msg)
                emit_layer_block_alert(bybit_sym, cand["direction"], "CORRELATION_72H", msg)
                add_bot_log(msg)
                continue

            if live_orders_armed and avail_usdt < 2.0:
                _downgrade_radar(disp_sym)
                msg = f"{bybit_sym} {cand['direction']} blocked: Available USDT too low (${avail_usdt:.2f} < $2.00)"
                diagnostics.append(msg)
                emit_layer_block_alert(bybit_sym, cand["direction"], "LOW_BALANCE", msg)
                order_rejections.append(msg)
                add_bot_log(msg)
                break
            if traded_candle_ts.get(bybit_sym) == cand["latest_bar_ms"]:
                _downgrade_radar(disp_sym)
                msg = f"{bybit_sym} {cand['direction']} skipped: Already traded this candle"
                diagnostics.append(msg)
                add_bot_log(msg)
                continue
            if (now_ms - _safe_int(last_trade_epoch_ms.get(bybit_sym), 0)) < 30 * 60 * 1000:
                _downgrade_radar(disp_sym)
                rem_m = max(1, int((30 * 60 * 1000 - (now_ms - _safe_int(last_trade_epoch_ms.get(bybit_sym), 0))) / 60000))
                msg = f"{bybit_sym} {cand['direction']} skipped: 30m trade cooldown ({rem_m}m left)"
                diagnostics.append(msg)
                add_bot_log(msg)
                continue

            filters = self._get_instrument_filters(bybit_sym)
            if filters is None:
                _downgrade_radar(disp_sym)
                msg = f"{bybit_sym} skipped: Not tradable on Bybit Linear"
                diagnostics.append(msg)
                add_bot_log(msg)
                continue
            qty_step, min_qty, min_notional, tick_size = filters

            live_bar = cand["live_bar"]
            closed_bar = cand["closed_bar"]
            cfg = cand["cfg"]
            price = _safe_float(live_bar["Close"], 0.0)
            atr = _safe_float(closed_bar["ATR"], 0.0)
            if price <= 0.0 or atr <= 0.0:
                _downgrade_radar(disp_sym)
                continue

            is_chop_cand = bool(cand.get("is_chop_gear", False))
            if is_chop_cand:
                sl_atr = max(_safe_float(cfg.get("sl_atr"), 1.65), 1.55)
                tp_rr = min(max(_safe_float(cfg.get("tp1_rr"), 0.95), 0.88), 1.15)
                min_sl_floor = price * (0.0020 if cand["market_type"] == "FOREX" else 0.0038)
            else:
                sl_atr = max(_safe_float(cfg.get("sl_atr"), 1.55) * sl_mult, 1.50)
                tp_rr = min(max(_safe_float(cfg.get("tp1_rr"), 1.65) * tp_mult, 1.50), 1.85)
                min_sl_floor = price * (0.0020 if cand["market_type"] == "FOREX" else 0.0155)
            risk_dist = max(sl_atr * atr, min_sl_floor)

            sim_equity = avail_usdt if avail_usdt >= 2.0 else 10.0
            max_buying_power = sim_equity * leverage * 0.85
            min_required_notional = max(min_notional * 1.05, min_qty * price)
            if live_orders_armed and max_buying_power < min_required_notional:
                _downgrade_radar(disp_sym)
                msg = f"{bybit_sym} {cand['direction']} skipped: Min order ${min_required_notional:.2f} > buying power ${max_buying_power:.2f} (Avail ${avail_usdt:.2f})"
                diagnostics.append(msg)
                add_bot_log(msg)
                continue

            bayes_mult = 1.0
            try:
                import ai_regime_copilot
                if hasattr(ai_regime_copilot, "get_bayesian_risk_multiplier"):
                    bayes_mult = ai_regime_copilot.get_bayesian_risk_multiplier(
                        self.live_dir, bybit_sym, curr_session, _safe_float(cand.get("prior_wr_pct"), 55.0)
                    )
            except Exception:
                bayes_mult = 1.0

            conv_scale = self._conviction_risk_scale(_safe_float(cand.get("raw_prob"), 0.3), _safe_float(cand.get("req_thresh"), 0.3))
            combined_risk_mult = min(max(conv_scale * bayes_mult * _safe_float(cand.get("kelly_scale"), 1.0), 0.50), 1.35)
            max_risk_cap = max(effective_risk_pct * 1.25, 2.50)
            scaled_risk_pct = min(max(effective_risk_pct * combined_risk_mult, 0.50), max_risk_cap)
            risk_usdt = max(sim_equity * (scaled_risk_pct / 100.0), 0.15)
            notional = min(max(risk_usdt / (risk_dist / price), min_required_notional, 5.5), max(max_buying_power, min_required_notional))
            qty_str = self._format_step(max(notional / price, min_qty), qty_step, round_up=True)

            is_long = cand["direction"] == "LONG"
            sl_raw = (price - risk_dist) if is_long else (price + risk_dist)
            tp_raw = (price + risk_dist * tp_rr) if is_long else (price - risk_dist * tp_rr)
            sl_str = self._format_step(sl_raw, tick_size)
            tp_str = self._format_step(tp_raw, tick_size)

            try:
                import ai_regime_copilot
                last_shadow_eval_ms[bybit_sym] = now_ms
                pa_res = ai_regime_copilot.evaluate_pre_trade_human_brain(
                    live_dir=self.live_dir,
                    symbol=bybit_sym,
                    side=cand["direction"],
                    candles_df=cand.get("candles_df"),
                    adx_val=_safe_float(cand.get("adx_val"), 20.0),
                    funding_rate=_safe_float(cand.get("fr_val"), 0.0),
                    session_name=curr_session,
                    gemini_enabled=gem_shadow_on,
                    gemini_api_key=gem_shadow_key,
                    is_chop_regime=is_chop_cand,
                    timeframe=str(cand.get("timeframe", "1H")),
                    strategy_name=str(cand.get("strategy_id", "")),
                )
                if pa_res.get("shadow_log"):
                    add_bot_log(pa_res["shadow_log"])

                if pa_res.get("decision") == "REJECT":
                    _downgrade_radar(disp_sym)
                    if hasattr(ai_regime_copilot, "record_shadow_counterfactual"):
                        ai_regime_copilot.record_shadow_counterfactual(
                            self.live_dir, bybit_sym, cand["direction"], price, sl_raw, tp_raw, "LAYER1_BRAIN", cand["conv_pct"]
                        )
                    l1_why = "; ".join(pa_res.get("reasons", [])[:2])
                    msg = f"{bybit_sym} {cand['direction']} blocked by Layer 1 Human Brain ({pa_res.get('score', 0)}/100): {l1_why}"
                    diagnostics.append(msg)
                    emit_layer_block_alert(bybit_sym, cand["direction"], "LAYER1_BRAIN", msg)
                    add_bot_log(msg)
                    continue
                else:
                    add_bot_log(
                        f"[LAYER 1 APPROVED] {bybit_sym} {cand['direction']} ({pa_res.get('score', 75)}/100 • "
                        f"Meta {cand['meta_prob']:.2f} • Size {combined_risk_mult:.2f}x): {'; '.join(pa_res.get('reasons', [])[:1])}"
                    )
            except Exception as l1_err:
                _downgrade_radar(disp_sym)
                add_bot_log(f"[LAYER 1 ERROR] {bybit_sym} {cand['direction']} skipped: {l1_err}")
                continue

            spread_cap = 0.05 if cand["market_type"] == "FOREX" else 0.08
            ob_ok, ob_reason = self._check_orderbook_spread_and_depth(bybit_sym, notional, max_spread_pct=spread_cap)
            if not ob_ok:
                _downgrade_radar(disp_sym)
                msg = f"{bybit_sym} {cand['direction']} blocked by Spread/Depth Guard: {ob_reason}"
                diagnostics.append(msg)
                emit_layer_block_alert(bybit_sym, cand["direction"], "SPREAD_DEPTH_GUARD", msg)
                add_bot_log(msg)
                continue

            notified_block_layers.pop(f"{bybit_sym}_{cand['direction']}", None)

            if not live_orders_armed:
                v_list = state.setdefault("virtual_open_positions", [])
                v_pos = {
                    "symbol": bybit_sym,
                    "side": cand["direction"],
                    "qty": _safe_float(qty_str, 0.0),
                    "entry_price": price,
                    "mark_price": price,
                    "sl": _safe_float(sl_str, 0.0),
                    "tp": _safe_float(tp_str, 0.0),
                    "unrealized_pnl": 0.0,
                    "opened_ms": now_ms,
                    "raw_prob": cand["raw_prob"],
                    "is_chop_gear": is_chop_cand,
                    "conformal_regime": cand["conformal_regime"],
                    "oi_delta_4h": cand["oi_delta_4h"],
                    "ls_skew": cand["ls_skew"],
                }
                v_list.append(v_pos)
                active_pos_records.append(v_pos)
                open_syms.add(bybit_sym)
                traded_candle_ts[bybit_sym] = cand["latest_bar_ms"]
                last_trade_epoch_ms[bybit_sym] = now_ms
                daily_coin_trades[bybit_sym] = _safe_int(daily_coin_trades.get(bybit_sym), 0) + 1
                daily_total_trades += 1
                state["daily_total_trades"] = daily_total_trades
                live_trades_this_cycle += 1
                add_bot_log(
                    f"[DRY-RUN SIMULATED] (Grade {cand['grade']} • {cand['conv_pct']}% • {curr_session}): {bybit_sym} {cand['direction']} "
                    f"qty={qty_str} @ ~{price:.4f} | SL={sl_str} | TP={tp_str} ({tp_rr:.2f}R • No Exchange Order)"
                )
                continue

            self._bybit_request("POST", "/v5/position/set-leverage", body={
                "category": self.category,
                "symbol": bybit_sym,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage),
            })
            self._cancel_stale_entry_orders(bybit_sym, max_age_ms=600000)
            if is_chop_cand:
                best_bid, best_ask = self._get_top_of_book(bybit_sym)
                raw_maker_px = (best_bid if best_bid > 0 else price) if is_long else (best_ask if best_ask > 0 else price)
                maker_px_str = self._format_step(raw_maker_px, tick_size)
                res = self._bybit_request("POST", "/v5/order/create", body={
                    "category": self.category,
                    "symbol": bybit_sym,
                    "side": "Buy" if is_long else "Sell",
                    "orderType": "Limit",
                    "price": maker_px_str,
                    "qty": qty_str,
                    "timeInForce": "PostOnly",
                    "tpslMode": "Partial",
                    "tpOrderType": "Limit",
                    "tpLimitPrice": tp_str,
                    "slOrderType": "Market",
                    "stopLoss": sl_str,
                    "takeProfit": tp_str,
                    "tpTriggerBy": "LastPrice",
                    "slTriggerBy": "MarkPrice",
                    "positionIdx": 0,
                })
                if res.get("retCode") != 0:
                    res = self._bybit_request("POST", "/v5/order/create", body={
                        "category": self.category,
                        "symbol": bybit_sym,
                        "side": "Buy" if is_long else "Sell",
                        "orderType": "Limit",
                        "price": maker_px_str,
                        "qty": qty_str,
                        "timeInForce": "GTC",
                        "tpslMode": "Partial",
                        "tpOrderType": "Limit",
                        "tpLimitPrice": tp_str,
                        "slOrderType": "Market",
                        "stopLoss": sl_str,
                        "takeProfit": tp_str,
                        "tpTriggerBy": "LastPrice",
                        "slTriggerBy": "MarkPrice",
                        "positionIdx": 0,
                    })
                if res.get("retCode") != 0:
                    res = self._bybit_request("POST", "/v5/order/create", body={
                        "category": self.category,
                        "symbol": bybit_sym,
                        "side": "Buy" if is_long else "Sell",
                        "orderType": "Market",
                        "qty": qty_str,
                        "timeInForce": "IOC",
                        "stopLoss": sl_str,
                        "takeProfit": tp_str,
                        "slTriggerBy": "MarkPrice",
                        "positionIdx": 0,
                    })
            else:
                res = self._bybit_request("POST", "/v5/order/create", body={
                    "category": self.category,
                    "symbol": bybit_sym,
                    "side": "Buy" if is_long else "Sell",
                    "orderType": "Market",
                    "qty": qty_str,
                    "timeInForce": "IOC",
                    "tpslMode": "Partial",
                    "tpOrderType": "Limit",
                    "tpLimitPrice": tp_str,
                    "slOrderType": "Market",
                    "stopLoss": sl_str,
                    "takeProfit": tp_str,
                    "tpTriggerBy": "LastPrice",
                    "slTriggerBy": "MarkPrice",
                    "positionIdx": 0,
                })
            if res.get("retCode") == 0:
                open_syms.add(bybit_sym)
                active_pos_records.append({
                    "symbol": bybit_sym,
                    "side": cand["direction"],
                    "qty": qty_str,
                    "entry_price": price,
                    "mark_price": price,
                    "unrealized_pnl": 0.0,
                })
                traded_candle_ts[bybit_sym] = cand["latest_bar_ms"]
                last_trade_epoch_ms[bybit_sym] = now_ms
                pos_opened_map[bybit_sym] = now_ms
                entry_meta_map[bybit_sym] = {
                    "raw_prob": cand["raw_prob"],
                    "session": curr_session,
                    "regime": curr_regime,
                    "conformal_regime": cand["conformal_regime"],
                    "is_chop_gear": is_chop_cand,
                    "oi_delta_4h": cand["oi_delta_4h"],
                    "ls_skew": cand["ls_skew"],
                }
                daily_coin_trades[bybit_sym] = _safe_int(daily_coin_trades.get(bybit_sym), 0) + 1
                daily_total_trades += 1
                state["daily_total_trades"] = daily_total_trades
                live_trades_this_cycle += 1
                avail_usdt = max(0.0, avail_usdt - (notional / leverage))

                tp_dist = abs(tp_raw - price)
                native_ok = self._arm_native_trailing_stop(bybit_sym, cand["direction"], price, tp_dist, tick_size, trail_ratio)
                if native_ok:
                    native_trail_map[f"{bybit_sym}_{price:.6f}"] = True

                add_bot_log(
                    f"ORDER EXECUTED SUCCESS (Grade {cand['grade']} • {cand['conv_pct']}% • {curr_session}): {bybit_sym} {cand['direction']} "
                    f"qty={qty_str} @ ~{price:.4f} | SL={sl_str} (MarkPrice) | TP={tp_str} ({tp_rr:.2f}R, Trail@70%={'ON' if native_ok else 'SYNC'})"
                )
            else:
                _downgrade_radar(disp_sym)
                code = _safe_int(res.get("retCode"), -1)
                raw_msg = str(res.get("retMsg", "Unknown Bybit rejection"))
                hint = BYBIT_ERROR_HINTS.get(code, raw_msg)
                if code in (110126, 110132):
                    restricted_until_ms[bybit_sym] = now_ms + (30 * 24 * 60 * 60 * 1000)
                    try:
                        import ai_regime_copilot
                        if hasattr(ai_regime_copilot, "quarantine_symbol_in_memory"):
                            ai_regime_copilot.quarantine_symbol_in_memory(self.live_dir, bybit_sym, hint, days=30)
                    except Exception:
                        pass
                rej_msg = f"{bybit_sym} {cand['direction']} (qty {qty_str}) rejected by Bybit [{code}]: {hint}"
                diagnostics.append(rej_msg)
                order_rejections.append(rej_msg)
                add_bot_log(f"ORDER REJECTED: {rej_msg}")

        radar_signals.sort(key=lambda x: (x["mtf_aligned"], x["confidence_pct"]), reverse=True)
        summary_detail = top_wait_tuple[1] if top_wait_tuple else (diagnostics[0] if diagnostics else "All pairs scanned")
        dry_tag = " [DRY-RUN]" if not live_orders_armed else ""
        mode_tag = f"Active ({daily_total_trades} today • A+/A/B+/B){dry_tag}"
        sess_note = f"{curr_session}{' (Tight)' if (session_prob_penalty > 0 or any_conformal_tight) else ''}"
        state["top_status_summary"] = f"{mode_tag} | {sess_note} | {summary_detail}"

        if live_trades_this_cycle == 0:
            all_coins_line = " | ".join(per_coin_summaries) if per_coin_summaries else summary_detail
            dd_note = f" | PeakDD {peak_dd_pct:.1f}%" if peak_dd_pct >= 5.0 else ""
            add_bot_log(f"Scanning {coins_scanned_count} coins ({daily_total_trades} today • {sess_note}{dry_tag}, Avail ${avail_usdt:.2f}{dd_note}): {all_coins_line}")

        if not bot_logs_enabled:
            state["last_bot_logs"] = []
        else:
            state["last_bot_logs"] = bot_logs[-60:]

        for map_key in ("last_shadow_eval_ms", "traded_candle_ts", "last_trade_epoch_ms", "last_notified_block_layer", "tp1_partial_done", "native_trail_set"):
            m_obj = state.get(map_key)
            if isinstance(m_obj, dict) and len(m_obj) > 80:
                for k_old in list(m_obj.keys())[:-60]:
                    m_obj.pop(k_old, None)

        state["coins_scanned"] = coins_scanned_count
        state["radar_signals"] = radar_signals
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
    elif (api_key and api_secret) or (not bool(runtime_cfg.get("live_orders_armed", False))):
        out = bot.run_live_cycle(
            leverage=_safe_int(runtime_cfg.get("leverage"), 10),
            risk_per_trade_pct=_safe_float(runtime_cfg.get("risk_per_trade_pct"), 1.5),
            max_open_positions=_safe_int(runtime_cfg.get("max_open_positions"), 2),
            crypto_enabled=bool(runtime_cfg.get("crypto_enabled", True)),
            forex_enabled=bool(runtime_cfg.get("forex_enabled", False)),
            live_orders_armed=bool(runtime_cfg.get("live_orders_armed", False)),
            mtf_mode=str(runtime_cfg.get("mtf_mode", "STRICT")),
            btc_macro_guard=bool(runtime_cfg.get("btc_macro_guard", True)),
            daily_drawdown_pct=_safe_float(runtime_cfg.get("daily_drawdown_pct"), 0.0),
            max_daily_loss_pct=_safe_float(runtime_cfg.get("max_daily_loss_pct"), 3.0),
            bot_logs_enabled=bool(runtime_cfg.get("bot_logs_enabled", True)),
        )
        print(json.dumps(out))
