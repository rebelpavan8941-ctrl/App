import ast
import gc
import os

import threading as _at_threading
import uuid as _at_uuid
_ATOMIC_IO_LOCK = _at_threading.Lock()

def _locked_replace(src_tmp: str, dst_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    with _ATOMIC_IO_LOCK:
        _locked_replace(src_tmp, dst_path)

import glob
import json
import time
import hashlib
import requests
import numpy as np
import pandas as pd

DEFAULT_35_STRATEGIES = [
    "xgb_mtf_v2", "trendline_retest_pro", "xrp_donchian_breakout", "donchian_turtle_v2",
    "btc_vol_surge_adx", "supertrend_adx_filter", "bnb_ob_imbalance",
    "link_bb_expansion", "bb_keltner_squeeze", "eth_rsi_stoch_reversal",
    "stoch_rsi_trend_dip", "sol_ema_ribbon_atr", "ema_ribbon_pullback",
    "ada_macd_hist_accel", "macd_hist_zero_cross", "doge_supertrend_scalp",
    "avax_fib_ml_confluence", "near_vwap_profile", "vwap_reclaim_pro",
    "cvd_orderflow_div", "funding_arb_mom", "ict_fvg_retest",
    "liq_sweep_reversal", "volume_profile_poc", "ichimoku_cloud_mtf",
    "london_ny_orb", "parabolic_sar_trail", "anchored_vwap_reclaim_v3",
    "ict_liquidity_sweep_ob", "connors_rsi2_trend_pullback",
    "keltner_ttm_squeeze_pro", "cvd_orderflow_absorption",
    "g2_range_box_rejection", "g2_keltner_snapback", "g2_bb_rsi_exhaustion",
]


def _load_paper_bot_strategy_ids(paper_dir: str = "") -> list:
    import re as _re
    candidates = []
    if paper_dir:
        candidates.append(os.path.join(paper_dir, "paper_bot.py"))
    base_d = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(base_d, "bot_data", "PAPER", "paper_bot.py"))
    for p_bot in candidates:
        if os.path.exists(p_bot):
            try:
                with open(p_bot, "r", encoding="utf-8-sig") as pf:
                    ptxt = pf.read()
                m_cat = _re.search(r"STRATEGY_CATALOG_META\s*=\s*(\{.*?\n\})", ptxt, _re.DOTALL)
                if m_cat:
                    cat_obj = ast.literal_eval(m_cat.group(1))
                    if isinstance(cat_obj, dict) and len(cat_obj) >= 30:
                        return list(cat_obj.keys())
            except Exception:
                pass
    try:
        from paper_bot import STRATEGY_CATALOG_META
        if isinstance(STRATEGY_CATALOG_META, dict) and STRATEGY_CATALOG_META:
            return list(STRATEGY_CATALOG_META.keys())
    except Exception:
        pass
    return list(DEFAULT_35_STRATEGIES)


ALL_AVAILABLE_STRATEGIES = _load_paper_bot_strategy_ids()

FOREX_LINEAR_SYMBOLS = {"EURUSDUSDT", "GBPUSDUSDT", "USDJPYUSDT", "XAUUSDT", "XAGUSDT", "XAUTUSDT"}

FOREX_CANDIDATES = [
    ("EURUSD/USDT", "Bybit 24/7 EUR/USD USDT Perpetual liquidity & trend"),
    ("GBPUSD/USDT", "Bybit 24/7 GBP/USD London/NY session momentum"),
    ("XAU/USDT", "Gold (XAUUSDT) macro trend & safe-haven volatility"),
    ("XAG/USDT", "Silver (XAGUSDT) high-beta metals momentum"),
]

EXCLUDED_TICKERS = {
    "USDC", "FDUSD", "USDE", "DAI", "TUSD", "BUSD", "PYUSD",
    "SOXL", "SOXS", "SNXX", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "QQQ", "SPY", "MSTR", "COIN", "SNDK", "NBIS", "KORU", "PLTR", "AMD", "ARM", "SMCI", "HOOD", "MU", "INTC", "BABA", "NIO", "CRCL", "ANT"
}

SECTOR_MAJORS = {"BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "AVAX", "LINK", "DOT", "TON"}
SECTOR_MEME_HIGH_BETA = {"DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI", "MEME", "POPCAT", "TRUMP", "FARTCOIN"}

_SESSION = requests.Session()
_ADAPTER = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
_SESSION.mount("https://", _ADAPTER)

_REGIME_CACHE = {"ts": 0.0, "data": None}
_TICKERS_CACHE = {"ts": 0.0, "data": None}
CACHE_TTL_SEC = 1800.0


def _is_forex(pair_sym: str) -> bool:
    clean = pair_sym.replace("/", "").upper()
    if clean in {"EURUSDUSDT", "GBPUSDUSDT", "USDJPYUSDT", "XAUUSDT", "XAGUSDT", "XAUTUSDT"}:
        return True
    forex_bases = {"EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
    return len(clean) == 6 and clean[:3] in forex_bases and clean[3:] in (forex_bases | {"USD"})


def _classify_sector(pair_sym: str) -> str:
    if _is_forex(pair_sym):
        return "FOREX"
    base = pair_sym.replace("/USDT", "").replace("USDT", "").upper()
    if base in SECTOR_MAJORS:
        return "MAJORS"
    if base in SECTOR_MEME_HIGH_BETA or base.startswith("1000"):
        return "HIGH_BETA"
    return "ALT_L1_DEFI"


def _load_self_learning_memory(base_dir: str) -> dict:
    mem_path = os.path.join(base_dir, "ai_copilot_memory.json")
    if os.path.exists(mem_path):
        try:
            with open(mem_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "version": 2,
        "cycles_learned": 0,
        "coin_scores": {},
        "strategy_scores": {},
        "mtf_mode_scores": {"OFF": 1.0, "SOFT": 1.0, "STRICT": 1.0},
        "decayed_coins": [],
        "consecutive_live_losses": 0,
        "reported_upgrade_hashes": {},
    }


def _save_self_learning_memory(base_dir: str, mem: dict):
    os.makedirs(base_dir, exist_ok=True)
    mem_path = os.path.join(base_dir, "ai_copilot_memory.json")
    tmp_path = mem_path + f".tmp.{os.getpid()}_{_at_threading.get_ident()}_{_at_uuid.uuid4().hex[:8]}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2)
        _locked_replace(tmp_path, mem_path)
    except Exception:
        pass


# ==============================================================================
# UPGRADE 7: Proper Reports Handling, Deduplication & Attribution Sync
# ==============================================================================
def sanitize_and_sync_reports(live_dir: str, models_dir: str, reset_epoch_ms: int = 0) -> dict:
    os.makedirs(live_dir, exist_ok=True)
    history_path = os.path.join(live_dir, "live_reports_history.json")
    state_path = os.path.join(live_dir, "live_engine_state.json")

    raw_records = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    raw_records.extend(loaded)
                elif isinstance(loaded, dict):
                    raw_records.extend(loaded.get("trades", []))
        except Exception:
            raw_records = []

    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            closed_sum = st.get("closed_summary", [])
            if isinstance(closed_sum, list):
                raw_records.extend(closed_sum)
        except Exception:
            pass

    seen_keys = set()
    clean_trades = []
    by_symbol = {}

    for rec in raw_records:
        if not isinstance(rec, dict):
            continue
        sym = str(rec.get("symbol", "")).strip().upper()
        if not sym:
            continue
        if "/" not in sym and sym.endswith("USDT"):
            sym = f"{sym[:-4]}/USDT"

        try:
            pnl = float(rec.get("net_pnl", rec.get("closed_pnl", rec.get("pnl", 0.0))) or 0.0)
            if np.isnan(pnl) or np.isinf(pnl):
                continue
        except Exception:
            continue

        ts_ms = int(rec.get("closed_at_ms", rec.get("updated_time_ms", rec.get("timestamp_ms", 0))) or 0)
        if reset_epoch_ms > 0 and ts_ms > 0 and ts_ms < reset_epoch_ms:
            continue

        order_id = str(rec.get("order_id", rec.get("id", ""))).strip()
        dedup_key = order_id if order_id else f"{sym}_{ts_ms}_{round(pnl, 4)}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        clean_item = {
            "id": dedup_key,
            "symbol": sym,
            "side": str(rec.get("side", "LONG")).upper(),
            "net_pnl": round(pnl, 4),
            "closed_at_ms": ts_ms,
        }
        clean_trades.append(clean_item)

        side_u = clean_item["side"]
        prev_same = None
        for prev_t in reversed(clean_trades[:-1]):
            if prev_t["symbol"] == sym and prev_t["side"] == side_u:
                if ts_ms > 0 and prev_t["closed_at_ms"] > 0 and abs(ts_ms - prev_t["closed_at_ms"]) <= 7_200_000:
                    prev_same = prev_t
                break

        if prev_same is not None:
            old_pnl = prev_same["net_pnl"]
            new_pnl = round(old_pnl + pnl, 4)
            prev_same["net_pnl"] = new_pnl
            clean_trades.pop()
            agg = by_symbol[sym]
            agg["net_pnl"] = round(agg["net_pnl"] + pnl, 4)
            if old_pnl > 0:
                agg["wins"] = max(0, agg["wins"] - 1)
                agg["gross_win"] = max(0.0, agg["gross_win"] - old_pnl)
            elif old_pnl < 0:
                agg["gross_loss"] = max(0.0, agg["gross_loss"] - abs(old_pnl))
            if new_pnl > 0:
                agg["wins"] += 1
                agg["gross_win"] += new_pnl
            elif new_pnl < 0:
                agg["gross_loss"] += abs(new_pnl)
        else:
            agg = by_symbol.setdefault(sym, {"symbol": sym, "trades": 0, "wins": 0, "gross_win": 0.0, "gross_loss": 0.0, "net_pnl": 0.0})
            agg["trades"] += 1
            agg["net_pnl"] = round(agg["net_pnl"] + pnl, 4)
            if pnl > 0:
                agg["wins"] += 1
                agg["gross_win"] += pnl
            elif pnl < 0:
                agg["gross_loss"] += abs(pnl)

    clean_trades.sort(key=lambda x: x["closed_at_ms"], reverse=True)
    clean_trades = clean_trades[:300]

    attributions = []
    consec_losses = 0
    for t in clean_trades:
        if t["net_pnl"] < 0:
            consec_losses += 1
        elif t["net_pnl"] > 0:
            break

    for sym, agg in by_symbol.items():
        tr_cnt = agg["trades"]
        wr = round((agg["wins"] / tr_cnt) * 100.0, 2) if tr_cnt > 0 else 0.0
        pf = round(agg["gross_win"] / agg["gross_loss"], 2) if agg["gross_loss"] > 0 else (round(agg["gross_win"], 2) if agg["gross_win"] > 0 else 0.0)
        attributions.append({
            "symbol": sym,
            "trades": tr_cnt,
            "win_rate_pct": wr,
            "profit_factor": pf,
            "net_pnl": agg["net_pnl"],
        })

    try:
        tmp_h = history_path + f".tmp.{os.getpid()}_{_at_threading.get_ident()}_{_at_uuid.uuid4().hex[:8]}"
        with open(tmp_h, "w", encoding="utf-8") as wf:
            json.dump({"updated_at": int(time.time()), "trades": clean_trades, "attributions": attributions}, wf, indent=2)
        _locked_replace(tmp_h, history_path)
    except Exception:
        pass

    return {
        "total_clean_trades": len(clean_trades),
        "attributions": attributions,
        "consecutive_losses": consec_losses,
    }


# ==============================================================================
# UPGRADE 8A: Storage Detection & Safe Auto-Cleanup
# ==============================================================================
def cleanup_unwanted_storage(paper_dir: str, live_dir: str, models_dir: str) -> dict:
    removed_files = 0
    reclaimed_bytes = 0
    now = time.time()

    scan_roots = [paper_dir, live_dir, models_dir, os.path.dirname(os.path.abspath(__file__))]
    for root_dir in scan_roots:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    fsize = os.path.getsize(fpath)
                    age_sec = now - os.path.getmtime(fpath)
                except Exception:
                    continue

                is_temp = (
                    fname.startswith("_tmp_")
                    or fname.startswith("_radar_tmp_")
                    or fname.endswith(".tmp")
                    or fname.endswith(".bak")
                )
                is_empty_json = fname.endswith(".json") and fsize == 0 and age_sec > 120.0
                is_corrupt_tmp_json = (
                    fname.endswith(".json")
                    and fname not in ("UNIFIED_PORTFOLIO_MASTER.json", "live_engine_state.json", "ai_copilot_memory.json", "live_reports_history.json")
                    and fsize < 2048
                    and age_sec > 300.0
                )

                should_delete = (is_temp and age_sec > 300.0) or is_empty_json
                if not should_delete and is_corrupt_tmp_json:
                    try:
                        with open(fpath, "r", encoding="utf-8") as jf:
                            json.load(jf)
                    except Exception:
                        should_delete = True

                if should_delete:
                    try:
                        os.remove(fpath)
                        removed_files += 1
                        reclaimed_bytes += fsize
                    except Exception:
                        pass

    # Keep only the latest 2 Paper history model snapshots to save phone storage
    hist_dir = os.path.join(models_dir, "PAPER", "history")
    if os.path.isdir(hist_dir):
        try:
            snaps = sorted(glob.glob(os.path.join(hist_dir, "UNIFIED_PAPER_*.json")), key=os.path.getmtime, reverse=True)
            for old_snap in snaps[2:]:
                try:
                    sz = os.path.getsize(old_snap)
                    os.remove(old_snap)
                    removed_files += 1
                    reclaimed_bytes += sz
                except Exception:
                    pass
        except Exception:
            pass

    gc.collect()
    return {
        "removed_files": removed_files,
        "reclaimed_kb": round(reclaimed_bytes / 1024.0, 1),
    }


# ==============================================================================
# UPGRADE 8B: AST Static Code Analysis for paper_bot.py & live_bot.py
# ==============================================================================
def analyze_bot_code_ast(file_path: str, file_label: str) -> list:
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            source = f.read()
    except Exception:
        return []

    issues = []
    try:
        tree = ast.parse(source, filename=file_label)
    except SyntaxError as se:
        issues.append((
            file_label,
            f"Fix Python SyntaxError on line {se.lineno}: {se.msg}.",
            "Why: A syntax error prevents Chaquopy from loading the bot module during live/paper cycles."
        ))
        return issues

    seen_funcs = {}
    imported_names = set()
    used_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            key = f"{node.name}_{node.lineno // 200}"
            if node.name in seen_funcs and node.name not in ("__init__",):
                issues.append((
                    file_label,
                    f"Remove or merge duplicate function definition '{node.name}' (lines {seen_funcs[node.name]} & {node.lineno}).",
                    "Why: Duplicate function definitions shadow earlier logic and bloat script memory."
                ))
            seen_funcs[node.name] = node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    imported_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)

    unused = sorted(n for n in imported_names if n not in used_names and not n.startswith("_"))
    if len(unused) >= 3:
        issues.append((
            file_label,
            f"Remove unused imports ({', '.join(unused[:5])}) to clean up module namespace.",
            "Why: Cleaner imports speed up Chaquopy module reloads and reduce memory footprint."
        ))

    return issues


# ==============================================================================
# UPGRADES 1, 2, 5 & 6: Self-Learning Reinforcement + Decay Detector + Upgrade Advisor
# ==============================================================================
def _update_self_learning_from_outcomes(paper_dir: str, live_dir: str, models_dir: str, report_sync: dict) -> dict:
    mem = _load_self_learning_memory(live_dir)
    coin_scores = mem.setdefault("coin_scores", {})
    strat_scores = mem.setdefault("strategy_scores", {})
    mtf_scores = mem.setdefault("mtf_mode_scores", {"OFF": 1.0, "SOFT": 1.0, "STRICT": 1.0})

    paper_baselines = {}
    paper_res = os.path.join(paper_dir, "paper_trained_results.json")
    if os.path.exists(paper_res):
        try:
            with open(paper_res, "r", encoding="utf-8") as f:
                coins_map = json.load(f).get("coins", {})
            for sym, cdata in coins_map.items():
                if not isinstance(cdata, dict):
                    continue
                pnl = float(cdata.get("net_pnl_usdt", 0.0))
                pf = float(cdata.get("profit_factor", 0.0))
                wr = float(cdata.get("win_rate_pct", 0.0))
                sid = str(cdata.get("strategy_id", "xgb_mtf_v2"))
                mtf_m = str(cdata.get("mtf_alignment_mode", "STRICT")).upper()
                paper_baselines[sym] = {"wr": wr, "pf": pf, "sid": sid, "mtf": mtf_m}

                reward = (pnl * 0.05) + ((pf - 1.0) * 15.0) + ((wr - 50.0) * 0.25)
                prev_c = float(coin_scores.get(sym, 0.0))
                coin_scores[sym] = round((0.70 * prev_c) + (0.30 * reward), 3)

                prev_s = float(strat_scores.get(sid, 0.0))
                strat_scores[sid] = round((0.70 * prev_s) + (0.30 * reward), 3)

                if mtf_m in mtf_scores:
                    mtf_scores[mtf_m] = round(max(0.2, (0.75 * float(mtf_scores[mtf_m])) + (0.25 * (1.0 + reward / 40.0))), 3)
        except Exception:
            pass

    # Upgrade 1 & 2: Live Realized-PnL Reinforcement & Model Decay Detection
    decayed = []
    for attr in report_sync.get("attributions", []):
        sym = attr["symbol"]
        l_trades = int(attr["trades"])
        l_wr = float(attr["win_rate_pct"])
        l_pnl = float(attr["net_pnl"])
        if l_trades >= 3:
            live_reward = (l_pnl * 2.5) + ((l_wr - 50.0) * 0.6)
            prev_c = float(coin_scores.get(sym, 0.0))
            coin_scores[sym] = round((0.55 * prev_c) + (0.45 * live_reward), 3)

            base = paper_baselines.get(sym)
            if base:
                sid = base["sid"]
                strat_scores[sid] = round((0.60 * float(strat_scores.get(sid, 0.0))) + (0.40 * live_reward), 3)
                if l_wr < (base["wr"] - 15.0) and l_pnl < 0:
                    decayed.append(sym)

    mem["decayed_coins"] = decayed
    mem["consecutive_live_losses"] = int(report_sync.get("consecutive_losses", 0))
    mem["cycles_learned"] = int(mem.get("cycles_learned", 0)) + 1
    mem["last_learned_epoch"] = int(time.time())
    _save_self_learning_memory(live_dir, mem)
    return mem


def audit_bot_codes_for_upgrades(paper_dir: str, live_dir: str, models_dir: str, reset_epoch_ms: int = 0) -> str:
    # 1. Sync & sanitize reports first
    report_sync = sanitize_and_sync_reports(live_dir, models_dir, reset_epoch_ms=reset_epoch_ms)

    # 2. Clean unwanted temp/corrupt files & prune stale snapshots
    storage_stats = cleanup_unwanted_storage(paper_dir, live_dir, models_dir)

    # 3. Update self-learning reinforcement memory & decay flags
    mem = _update_self_learning_from_outcomes(paper_dir, live_dir, models_dir, report_sync)
    reported = mem.setdefault("reported_upgrade_hashes", {})
    now = time.time()

    paper_bot_path = os.path.join(paper_dir, "paper_bot.py")
    live_bot_path = os.path.join(live_dir, "live_bot.py")
    live_state_path = os.path.join(live_dir, "live_engine_state.json")
    paper_res_path = os.path.join(paper_dir, "paper_trained_results.json")

    findings = []
    # Run AST static code analysis on both bot files
    findings.extend(analyze_bot_code_ast(paper_bot_path, "paper_bot.py"))
    findings.extend(analyze_bot_code_ast(live_bot_path, "live_bot.py"))

    paper_code = ""
    live_code = ""
    if os.path.exists(paper_bot_path):
        try:
            with open(paper_bot_path, "r", encoding="utf-8") as f:
                paper_code = f.read()
        except Exception:
            pass
    if os.path.exists(live_bot_path):
        try:
            with open(live_bot_path, "r", encoding="utf-8") as f:
                live_code = f.read()
        except Exception:
            pass

    if paper_code:
        if "mtf_alignment_mode" not in paper_code or '"SOFT"' not in paper_code:
            findings.append((
                "paper_bot.py",
                "Add multi-mode 4H MTF backtesting ('OFF', 'SOFT', 'STRICT') inside _simulate_strategy_on_tf and save 'mtf_alignment_mode' in coin config.",
                "Why: Coins capture significantly higher Net PnL and Profit Factor when trained across all three 4H MTF modes."
            ))
        if "VWAP_Dev" not in paper_code:
            findings.append((
                "paper_bot.py",
                "Upgrade FEATURES list to 18 features including 'VWAP_Dev' and session VWAP deviation calculation.",
                "Why: Improves intraday mean-reversion and institutional session breakout accuracy."
            ))

    if live_code:
        if "daily_total_trades" not in live_code or "_passes_anti_chase_guard" not in live_code:
            findings.append((
                "live_bot.py",
                "Upgrade run_live_cycle with closed-candle inference (iloc[-2]), anti-chase guards, and 1 A+ trade/day discipline.",
                "Why: Prevents intrabar repainting and chasing exhausted moves while targeting achievable TP1 exits."
            ))
        if "mtf_alignment_mode" not in live_code:
            findings.append((
                "live_bot.py",
                "Read per-coin 'mtf_alignment_mode' ('OFF'/'SOFT'/'STRICT') from portfolio config in run_live_cycle.",
                "Why: Executes each Live coin using the exact 4H MTF mode proven most profitable during Paper training."
            ))

    if mem.get("decayed_coins"):
        d_list = ", ".join(mem["decayed_coins"][:4])
        findings.append((
            "paper_bot.py & live_bot.py",
            f"Concept Drift / Model Decay detected on {d_list} — retrain on latest 135d window and tighten SL/TP ATR ratios.",
            "Why: Realized Live win rate dropped >15% below Paper backtest accuracy."
        ))

    if os.path.exists(live_state_path):
        try:
            with open(live_state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            blocks = st.get("last_trade_blocks", []) or []
            rejects = st.get("last_order_rejections", []) or []
            coins_scanned = int(st.get("coins_scanned", 0))
            daily_trades = int(st.get("daily_total_trades", 0))

            if rejects:
                findings.append((
                    "live_bot.py",
                    f"Adjust order qty step / min-notional rounding in run_live_cycle (Bybit rejected: {rejects[-1][:110]}).",
                    "Why: Bybit rejected a valid ML entry due to lot-size, minimum notional, or leverage tier constraints."
                ))
            elif coins_scanned >= 4 and daily_trades == 0 and len(blocks) >= 6:
                sample_block = blocks[-1][:110]
                findings.append((
                    "live_bot.py & paper_bot.py",
                    f"Relax entry filter threshold or widen SOFT MTF tolerance (Repeatedly blocked: {sample_block}).",
                    "Why: Multiple Live coins are scanning actively but entry filters/thresholds are blocking trade execution."
                ))
        except Exception:
            pass

    if os.path.exists(paper_res_path):
        try:
            with open(paper_res_path, "r", encoding="utf-8") as f:
                coins_map = json.load(f).get("coins", {})
            if len(coins_map) >= 3:
                weak_coins = [
                    s for s, d in coins_map.items()
                    if isinstance(d, dict) and float(d.get("profit_factor", 0.0)) < 1.05 and int(d.get("trades", 0)) >= 5
                ]
                if len(weak_coins) >= max(2, len(coins_map) // 2):
                    findings.append((
                        "paper_bot.py",
                        f"Add tighter ATR stop-loss / adaptive quantile threshold sweep for choppy pairs ({', '.join(weak_coins[:4])}).",
                        "Why: Over half of trained Paper coins have Profit Factor < 1.05 in current market volatility."
                    ))
        except Exception:
            pass

    new_messages = []
    for target_file, upgrade_action, why_text in findings:
        sig_hash = hashlib.sha256(f"{target_file}:{upgrade_action}".encode("utf-8")).hexdigest()[:16]
        last_sent = float(reported.get(sig_hash, 0.0))
        if (now - last_sent) >= 43200.0:
            reported[sig_hash] = now
            new_messages.append(
                f"[AI Self-Learning Advisor -> {target_file}] Upgrade Needed: {upgrade_action} | {why_text}"
            )

    _save_self_learning_memory(live_dir, mem)
    gc.collect()
    return json.dumps({
        "ok": True,
        "cycles_learned": mem.get("cycles_learned", 1),
        "storage_cleaned": storage_stats,
        "clean_report_trades": report_sync.get("total_clean_trades", 0),
        "new_upgrades": new_messages,
    })


def _fetch_btc_4h_regime(force_refresh: bool = False) -> dict:
    now = time.time()
    if not force_refresh and _REGIME_CACHE["data"] is not None and (now - _REGIME_CACHE["ts"]) < CACHE_TTL_SEC:
        return dict(_REGIME_CACHE["data"])

    for base in ("https://api.bybit.com", "https://api.bytick.com", "https://api-demo.bybit.com"):
        try:
            resp = _SESSION.get(
                f"{base}/v5/market/kline",
                params={"category": "linear", "symbol": "BTCUSDT", "interval": "240", "limit": 80},
                timeout=6,
            )
            rows = resp.json().get("result", {}).get("list", [])
            if len(rows) < 40:
                continue
            df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Vol", "Turn"])
            df["ts"] = pd.to_numeric(df["ts"])
            df = df.sort_values("ts").reset_index(drop=True)
            for col in ("Open", "High", "Low", "Close"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

            c = df["Close"]
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - c.shift(1)).abs(),
                (df["Low"] - c.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            natr = float(((atr / c) * 100.0).iloc[-1])

            plus_dm = np.where((df["High"].diff() > -df["Low"].diff()) & (df["High"].diff() > 0), df["High"].diff(), 0.0)
            minus_dm = np.where((-df["Low"].diff() > df["High"].diff()) & (-df["Low"].diff() > 0), -df["Low"].diff(), 0.0)
            plus_di = 100.0 * (pd.Series(plus_dm).rolling(14).mean() / (atr + 1e-9))
            minus_di = 100.0 * (pd.Series(minus_dm).rolling(14).mean() / (atr + 1e-9))
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100.0
            adx = float(dx.rolling(14).mean().iloc[-1])

            hl2 = ((df["High"] + df["Low"]) * 0.5).to_numpy()
            atr_arr = atr.fillna(0.0).to_numpy()
            c_arr = c.to_numpy()
            upper = hl2 + 3.0 * atr_arr
            lower = hl2 - 3.0 * atr_arr
            uptrend = True
            for i in range(10, len(c_arr)):
                if uptrend and c_arr[i] < lower[i - 1]:
                    uptrend = False
                elif not uptrend and c_arr[i] > upper[i - 1]:
                    uptrend = True

            # Upgrade 3: Derivatives Open Interest & Squeeze Telemetry
            oi_surge = False
            try:
                oi_resp = _SESSION.get(
                    f"{base}/v5/market/open-interest",
                    params={"category": "linear", "symbol": "BTCUSDT", "intervalTime": "1h", "limit": 6},
                    timeout=4,
                )
                oi_list = oi_resp.json().get("result", {}).get("list", [])
                if len(oi_list) >= 2:
                    latest_oi = float(oi_list[0].get("openInterest", 0.0) or 0.0)
                    prev_oi = float(oi_list[-1].get("openInterest", 0.0) or 0.0)
                    if prev_oi > 0 and ((latest_oi - prev_oi) / prev_oi) >= 0.03:
                        oi_surge = True
            except Exception:
                pass

            state_tag = "CHOPPY" if (natr >= 2.8 or (adx < 14.0 and not uptrend)) else ("BULL" if uptrend else "BEAR")
            res = {
                "ok": True,
                "btc_bullish": bool(uptrend),
                "adx": round(adx, 2),
                "natr": round(natr, 2),
                "oi_surge": oi_surge,
                "btc_price": round(float(c.iloc[-1]), 2),
                "fingerprint": f"{'BULL' if uptrend else 'BEAR'}_{state_tag}",
            }
            del df
            gc.collect()
            _REGIME_CACHE["ts"] = now
            _REGIME_CACHE["data"] = res
            return dict(res)
        except Exception:
            continue
    return {"ok": False, "fingerprint": "UNKNOWN"}


def _scan_all_bybit_usdt_coins(force_refresh: bool = False, learned_coin_scores: dict = None) -> list:
    now = time.time()
    if not force_refresh and _TICKERS_CACHE["data"] is not None and (now - _TICKERS_CACHE["ts"]) < CACHE_TTL_SEC:
        return list(_TICKERS_CACHE["data"])

    learned_coin_scores = learned_coin_scores or {}

    for base in ("https://api.bybit.com", "https://api.bytick.com", "https://api-demo.bybit.com"):
        try:
            resp = _SESSION.get(
                f"{base}/v5/market/tickers",
                params={"category": "linear"},
                timeout=8,
            )
            tickers = resp.json().get("result", {}).get("list", [])
            if not tickers:
                continue

            eligible_linear = _fetch_eligible_linear_instruments(min_age_days=140)
            q_live_dir = os.environ.get("LIVE_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_data", "LIVE_ACCOUNT"))
            q_mem = _load_self_learning_memory(q_live_dir)
            quarantine_map = q_mem.get("auto_quarantine", {}) or {}
            now_ms_scan = int(time.time() * 1000)

            ranked = []
            for t in tickers:
                raw_sym = str(t.get("symbol", "")).upper()
                if now_ms_scan < int(quarantine_map.get(raw_sym, 0) or 0):
                    continue
                if eligible_linear and raw_sym not in eligible_linear:
                    continue
                if not raw_sym.endswith("USDT"):
                    continue
                base_coin = raw_sym[:-4]
                if (
                    not base_coin
                    or base_coin in EXCLUDED_TICKERS
                    or base_coin.startswith("1000000")
                    or any(ch.isdigit() for ch in base_coin[:2])
                ):
                    continue
                try:
                    last_px = float(t.get("lastPrice", 0.0) or 0.0)
                    high_24h = float(t.get("highPrice24h", 0.0) or 0.0)
                    low_24h = float(t.get("lowPrice24h", 0.0) or 0.0)
                    turnover_24h = float(t.get("turnover24h", 0.0) or 0.0)
                    pct_24h = abs(float(t.get("price24hPcnt", 0.0) or 0.0)) * 100.0
                    funding = float(t.get("fundingRate", 0.0) or 0.0) * 100.0
                except Exception:
                    continue

                if last_px <= 0.0 or turnover_24h < 25_000_000.0:
                    continue

                range_pct = ((high_24h - low_24h) / last_px) * 100.0 if high_24h > low_24h else 0.0
                if range_pct < 1.2 or range_pct > 20.0:
                    continue

                pair_sym = f"{base_coin}/USDT"
                vol_millions = turnover_24h / 1_000_000.0
                # Squeeze bonus for negative/extreme funding, penalty for overcrowded long tops
                squeeze_bonus = min(abs(funding) * 25.0, 12.0) if funding < 0 else min(abs(funding) * 10.0, 6.0)
                learned_bonus = max(-25.0, min(35.0, float(learned_coin_scores.get(pair_sym, 0.0))))
                score = (min(vol_millions, 1000.0) * 0.22) + (range_pct * 7.5) + (pct_24h * 3.5) + squeeze_bonus + learned_bonus
                reason = f"24h Vol ${vol_millions:.0f}M, Range {range_pct:.1f}%, Move {pct_24h:.1f}%"
                ranked.append((score, pair_sym, reason, len(tickers)))

            ranked.sort(key=lambda x: x[0], reverse=True)
            _TICKERS_CACHE["ts"] = now
            _TICKERS_CACHE["data"] = ranked
            gc.collect()
            return list(ranked)
        except Exception:
            continue
    return []



_ALLOWED_RULE_INDICATORS = {
    "RSI_14", "Stoch_K", "Stoch_D", "ADX", "DI_Spread", "MACD_Hist",
    "BB_PctB", "BB_Width", "NATR", "Dist_EMA20", "Dist_EMA50", "Dist_EMA200",
    "VWAP_Dev", "Vol_Surge", "Supertrend"
}
_ALLOWED_RULE_OPS = {">", "<", ">=", "<="}


def _load_adopted_and_discovered_strategies(paper_dir: str) -> dict:
    merged = {}
    for fname in ("ai_adopted_strategies.json", "ai_discovered_strategies.json"):
        fpath = os.path.join(paper_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for sid, sdata in (data.get("strategies", {}) or {}).items():
                    if isinstance(sdata, dict) and sdata.get("long_rules") and sdata.get("short_rules"):
                        merged[sid] = sdata
            except Exception:
                pass
    return merged


def _validate_strategy_rules(rules_list) -> list:
    if not isinstance(rules_list, list):
        return []
    clean = []
    for r in rules_list[:5]:
        if not isinstance(r, dict):
            continue
        ind = str(r.get("ind", "")).strip()
        op = str(r.get("op", "")).strip()
        try:
            val = float(r.get("val"))
        except Exception:
            continue
        if ind in _ALLOWED_RULE_INDICATORS and op in _ALLOWED_RULE_OPS and not (np.isnan(val) or np.isinf(val)):
            clean.append({"ind": ind, "op": op, "val": round(val, 4)})
    return clean


def _fetch_free_web_strategy_snippets() -> str:
    snippets = []
    queries = [
        "best crypto perpetual futures trading strategy RSI ADX VWAP 2026",
        "profitable algorithmic breakout mean reversion crypto indicators"
    ]
    for q in queries:
        try:
            resp = _SESSION.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": q},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=6,
            )
            if resp.status_code == 200 and resp.text:
                import re
                cleaned = re.sub(r"<[^>]+>", " ", resp.text)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                snippets.append(cleaned[:900])
        except Exception:
            continue
    return " | ".join(snippets)[:1800]


_DISCOVERED_MODELS_CACHE = {"ts": 0.0, "models": []}


def _get_active_gemini_models(api_key: str) -> list:
    """Dynamically queries Google AI Studio /v1beta/models so model renames/deprecations never break the app."""
    fallback_models = ["gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]
    key = str(api_key or "").strip()
    if not key:
        return fallback_models

    now = time.time()
    if _DISCOVERED_MODELS_CACHE["models"] and (now - _DISCOVERED_MODELS_CACHE["ts"]) < 86400.0:
        return list(_DISCOVERED_MODELS_CACHE["models"])

    try:
        resp = _SESSION.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": key},
            timeout=6,
        )
        if resp.status_code == 200:
            raw_models = resp.json().get("models", [])
            candidates = []
            blocked_tags = ("tts", "image", "live", "embedding", "transcribe", "robotics", "deep-research", "2.0", "2.5", "1.5")
            for m in raw_models:
                methods = m.get("supportedGenerationMethods") or []
                if "generateContent" not in methods:
                    continue
                short_name = str(m.get("name", "")).replace("models/", "").strip()
                low = short_name.lower()
                if not low.startswith("gemini-") or any(tag in low for tag in blocked_tags):
                    continue
                priority = 0
                if "hydrogen" in low or "flash" in low:
                    priority += 50
                if "4" in low:
                    priority += 30
                elif "3.8" in low:
                    priority += 25
                elif "3.6" in low or "3.5" in low:
                    priority += 20
                if "lite" in low:
                    priority += 5
                candidates.append((priority, short_name))

            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                ordered = list(dict.fromkeys([name for _, name in candidates] + fallback_models))[:5]
                _DISCOVERED_MODELS_CACHE["ts"] = now
                _DISCOVERED_MODELS_CACHE["models"] = ordered
                return list(ordered)
    except Exception:
        pass

    return fallback_models

def discover_and_sync_gemini_strategies(api_key: str, paper_dir: str, live_dir: str) -> str:
    key = str(api_key or "").strip()
    if not key:
        return json.dumps({
            "ok": False,
            "message": "Gemini API key is missing — AI Copilot will use the 35 built-in strategies from paper_bot.py."
        })

    regime = _fetch_btc_4h_regime(force_refresh=False)
    web_context = _fetch_free_web_strategy_snippets()
    regime_desc = f"BTC 4H {'Bullish' if regime.get('btc_bullish') else 'Bearish'}, ADX={regime.get('adx', 20)}, NATR={regime.get('natr', 1.8)}%"

    prompt = (
        "You are an algorithmic crypto/forex quant researcher building strategies for a Dual-Gear engine: "
        "Gear 1 (1h/4h directional trend when ADX >= 22) and Gear 2 (5m/15m choppy range mean-reversion when ADX < 22). "
        f"Based on the current market regime ({regime_desc}) and web trading insights ({web_context}), invent 2 distinct, "
        "high-probability quantitative entry strategies (include at least one 5m or 15m choppy range mean-reversion strategy "
        "using BB_PctB, RSI_14, Dist_EMA20, or VWAP_Dev with ADX < 22) using ONLY these exact indicator names: "
        "RSI_14, Stoch_K, Stoch_D, ADX, DI_Spread, MACD_Hist, BB_PctB, BB_Width, NATR, "
        "Dist_EMA20, Dist_EMA50, Dist_EMA200, VWAP_Dev, Vol_Surge, Supertrend. "
        "Allowed operators: '>', '<', '>=', '<='. Allowed timeframes: '5m', '15m', '30m', '1h', '4h'. "
        "Respond ONLY with valid JSON in this exact format: "
        '{"strategies":[{"id":"ai_gemini_strat_1","name":"Strategy Name","timeframes":["5m","15m"],'
        '"long_rules":[{"ind":"ADX","op":"<","val":22},{"ind":"BB_PctB","op":"<=","val":0.10}],'
        '"short_rules":[{"ind":"ADX","op":"<","val":22},{"ind":"BB_PctB","op":">=","val":0.90}]}]}'
    )

    models_to_try = _get_active_gemini_models(key)
    parsed_strategies = {}

    for model_name in models_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.35,
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "strategies": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "id": {"type": "STRING"},
                                        "name": {"type": "STRING"},
                                        "timeframes": {"type": "ARRAY", "items": {"type": "STRING"}},
                                        "long_rules": {
                                            "type": "ARRAY",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "ind": {"type": "STRING"},
                                                    "op": {"type": "STRING"},
                                                    "val": {"type": "NUMBER"},
                                                },
                                                "required": ["ind", "op", "val"],
                                            },
                                        },
                                        "short_rules": {
                                            "type": "ARRAY",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "ind": {"type": "STRING"},
                                                    "op": {"type": "STRING"},
                                                    "val": {"type": "NUMBER"},
                                                },
                                                "required": ["ind", "op", "val"],
                                            },
                                        },
                                    },
                                    "required": ["id", "name", "long_rules", "short_rules"],
                                },
                            }
                        },
                        "required": ["strategies"],
                    },
                },
            }
            resp = _SESSION.post(url, json=payload, timeout=14)
            if resp.status_code != 200:
                continue
            body = resp.json()
            cand_text = (
                body.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            if cand_text.startswith("```"):
                cand_text = cand_text.strip("`").replace("json\n", "", 1).strip()
            raw_json = json.loads(cand_text)
            for idx, item in enumerate(raw_json.get("strategies", [])[:3]):
                l_rules = _validate_strategy_rules(item.get("long_rules"))
                s_rules = _validate_strategy_rules(item.get("short_rules"))
                if len(l_rules) >= 2 and len(s_rules) >= 2:
                    raw_id = str(item.get("id") or f"ai_gemini_{int(time.time())}_{idx}").strip().lower()
                    safe_id = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in raw_id)
                    if not safe_id.startswith("ai_gemini_"):
                        safe_id = "ai_gemini_" + safe_id
                    parsed_strategies[safe_id] = {
                        "id": safe_id,
                        "name": str(item.get("name") or f"Gemini Web Strategy {idx + 1}")[:60],
                        "timeframes": [tf for tf in (item.get("timeframes") or ["1h"]) if tf in ("15m", "30m", "1h", "4h")] or ["1h"],
                        "long_rules": l_rules,
                        "short_rules": s_rules,
                        "discovered_at": int(time.time()),
                        "model": model_name,
                    }
            if parsed_strategies:
                break
        except Exception:
            continue

    if not parsed_strategies:
        return json.dumps({
            "ok": False,
            "message": "Gemini API key invalid or unreachable — AI Copilot will use the 35 built-in strategies from paper_bot.py."
        })

    os.makedirs(paper_dir, exist_ok=True)
    disc_path = os.path.join(paper_dir, "ai_discovered_strategies.json")
    tmp_p = disc_path + f".tmp.{os.getpid()}_{_at_threading.get_ident()}_{_at_uuid.uuid4().hex[:8]}"
    with open(tmp_p, "w", encoding="utf-8") as wf:
        json.dump({"updated_at": int(time.time()), "strategies": parsed_strategies}, wf, indent=2)
    _locked_replace(tmp_p, disc_path)

    names = ", ".join(v["name"] for v in parsed_strategies.values())
    return json.dumps({
        "ok": True,
        "count": len(parsed_strategies),
        "message": f"Gemini invented {len(parsed_strategies)} strategies ({names}) — queued for Paper coin testing & auto-adoption."
    })
def plan_daily_ai_portfolio(current_coins_json: str, max_coins: int = 8, crypto_enabled: bool = True, forex_enabled: bool = True) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    live_dir = os.environ.get("LIVE_DATA_DIR", os.path.join(base_dir, "bot_data", "LIVE_ACCOUNT"))
    mem = _load_self_learning_memory(live_dir)
    learned_coins = mem.get("coin_scores", {})
    learned_strats = mem.get("strategy_scores", {})

    regime = _fetch_btc_4h_regime(force_refresh=False)
    scanned = _scan_all_bybit_usdt_coins(force_refresh=False, learned_coin_scores=learned_coins) if crypto_enabled else []
    try:
        current_coins = json.loads(current_coins_json) if current_coins_json else []
    except Exception:
        current_coins = []

    cap = max(5, min(int(max_coins), 10))
    selected_coins = []
    coin_reasons = {}
    sector_counts = {"MAJORS": 0, "ALT_L1_DEFI": 0, "HIGH_BETA": 0, "FOREX": 0}

    # Preserve up to 4 proven profitable Paper coins so auto-retrain refreshes them instead of overwriting winners
    paper_dir = os.environ.get("PAPER_DATA_DIR", os.path.join(base_dir, "bot_data", "PAPER"))
    res_p = os.path.join(paper_dir, "paper_trained_results.json")
    if os.path.exists(res_p):
        try:
            with open(res_p, "r", encoding="utf-8-sig") as rf:
                p_coins = (json.load(rf).get("coins", {}) or {})
            proven = []
            for sym, cdata in p_coins.items():
                if not isinstance(cdata, dict):
                    continue
                is_fx = _is_forex(sym)
                if (is_fx and not forex_enabled) or ((not is_fx) and not crypto_enabled):
                    continue
                pf = float(cdata.get("profit_factor", 0.0))
                pnl = float(cdata.get("net_pnl_usdt", 0.0))
                if pf >= 1.25 and pnl > 0.0:
                    proven.append((pnl, sym, pf))
            proven.sort(reverse=True)
            for _, sym, pf in proven[:2]:
                if len(selected_coins) < cap and sym not in selected_coins:
                    selected_coins.append(sym)
                    coin_reasons[sym] = f"Anchored proven Paper winner (PF {pf:.2f}) ? refreshing 135d weights"
                    sec = "FOREX" if _is_forex(sym) else _classify_sector(sym)
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
        except Exception:
            pass

    # Upgrade 4: Sector & Correlation Diversity Cap (max 3 per crypto sector)
    if crypto_enabled:
        crypto_cap = cap - 2 if forex_enabled else cap
        if scanned:
            total_scanned = scanned[0][3]
            for _, sym, rsn, _ in scanned:
                if sym in selected_coins:
                    continue
                if len(selected_coins) >= crypto_cap:
                    break
                sec = _classify_sector(sym)
                if sector_counts.get(sec, 0) >= 3:
                    continue
                selected_coins.append(sym)
                coin_reasons[sym] = rsn
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
            scan_note = f"Scanned {total_scanned} Crypto markets -> top {len(selected_coins)} diversified pairs"
        else:
            fallback = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LINK/USDT", "AVAX/USDT"]
            for c in (current_coins + fallback):
                if c and "/" in c and c.endswith("USDT") and c not in selected_coins and len(selected_coins) < crypto_cap:
                    selected_coins.append(c)
                    coin_reasons[c] = "Core liquid Crypto pair"
            scan_note = f"Selected {len(selected_coins)} Crypto pairs"
    else:
        scan_note = "Crypto disabled"

    if forex_enabled:
        fx_count = 3 if not crypto_enabled else 2
        for fx_sym, fx_rsn in FOREX_CANDIDATES[:fx_count]:
            if fx_sym not in selected_coins:
                selected_coins.append(fx_sym)
                coin_reasons[fx_sym] = fx_rsn
        scan_note += f" + {fx_count} Forex pairs"

    if not selected_coins:
        for c in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LINK/USDT"]:
            selected_coins.append(c)
            coin_reasons[c] = "Core liquid fallback"

    base_dir_p = os.path.dirname(os.path.abspath(__file__))
    paper_dir_p = os.environ.get("PAPER_DATA_DIR", os.path.join(base_dir_p, "bot_data", "PAPER"))
    custom_strats = _load_adopted_and_discovered_strategies(paper_dir_p)
    live_code_strats = _load_paper_bot_strategy_ids(paper_dir_p)
    combined_pool = list(dict.fromkeys(list(custom_strats.keys()) + live_code_strats))
    selected_strats = sorted(
        combined_pool,
        key=lambda s: (100.0 if s in custom_strats else 0.0) + float(learned_strats.get(s, 0.0)),
        reverse=True,
    )
    learned_cycles = int(mem.get("cycles_learned", 0))
    if regime.get("ok"):
        reason = f"{scan_note} & {len(selected_strats)} self-ranked strategies (Learned {learned_cycles} cycles, BTC 4H {'Bullish' if regime['btc_bullish'] else 'Bearish'}, ADX {regime['adx']:.1f})."
    else:
        reason = f"{scan_note} & {len(selected_strats)} self-ranked strategies (Learned {learned_cycles} cycles)."

    gc.collect()
    return json.dumps({
        "ok": True,
        "coins": selected_coins,
        "coin_reasons": coin_reasons,
        "strategies": selected_strats,
        "fingerprint": regime.get("fingerprint", "UNKNOWN"),
        "reason": reason,
    })


def rank_coins_for_live_promotion(comparisons_json: str, max_promote: int = 5) -> str:
    try:
        items = json.loads(comparisons_json) if comparisons_json else []
    except Exception:
        items = []

    ranked = []
    for item in items:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip()
        if not sym:
            continue
        wr = float(item.get("paper_win_rate_pct", 0.0))
        pf = float(item.get("paper_profit_factor", 0.0))
        pnl = float(item.get("paper_net_pnl", 0.0))
        trades = int(item.get("paper_trades", 0))
        quality_bonus = 50.0 if (pnl > 0 and pf >= 1.05 and trades >= 2) else 0.0
        score = quality_bonus + (pnl * 0.4) + (pf * 20.0) + (wr * 0.5)
        ranked.append((score, sym))

    ranked.sort(key=lambda x: x[0], reverse=True)
    target_count = max(4, min(int(max_promote), len(ranked)))
    top_syms = [sym for _, sym in ranked[:target_count]]

    return json.dumps({
        "ok": True,
        "promote_coins": top_syms,
        "count": len(top_syms),
    })


def audit_live_coins(live_coins_json: str, max_live_coins: int = 8) -> str:
    try:
        coins = json.loads(live_coins_json) if live_coins_json else []
    except Exception:
        coins = []

    if len(coins) <= 4:
        return json.dumps({"ok": True, "remove": []})

    base_dir = os.path.dirname(os.path.abspath(__file__))
    live_dir = os.environ.get("LIVE_DATA_DIR", os.path.join(base_dir, "bot_data", "LIVE_ACCOUNT"))
    models_dir = os.environ.get("MODELS_DIR", os.path.join(base_dir, "ml_models"))
    mem = _load_self_learning_memory(live_dir)
    coin_scores = mem.get("coin_scores", {}) or {}
    decayed_set = set(mem.get("decayed_coins", []) or [])

    live_portfolio = {}
    live_master = os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json")
    if os.path.exists(live_master):
        try:
            with open(live_master, "r", encoding="utf-8-sig") as f:
                live_portfolio = json.load(f).get("portfolio", {}) or {}
        except Exception:
            pass

    max_removable = len(coins) - 4
    to_remove = []
    removed_syms = set()
    scored_survivors = []
    for sym in coins:
        if len(to_remove) >= max_removable:
            break
        clean = str(sym).replace("/", "").upper()
        if not clean or (len(clean) == 6 and not clean.endswith("USDT")):
            continue
        for base in ("https://api.bybit.com", "https://api.bytick.com", "https://api-demo.bybit.com"):
            try:
                resp = _SESSION.get(
                    f"{base}/v5/market/kline",
                    params={"category": "linear", "symbol": clean, "interval": "60", "limit": 50},
                    timeout=5,
                )
                rows = resp.json().get("result", {}).get("list", [])
                if len(rows) < 30:
                    continue
                df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Vol", "Turn"])
                df["ts"] = pd.to_numeric(df["ts"])
                df = df.sort_values("ts").reset_index(drop=True)
                for col in ("High", "Low", "Close", "Vol"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

                c = df["Close"]
                tr = pd.concat([
                    df["High"] - df["Low"],
                    (df["High"] - c.shift(1)).abs(),
                    (df["Low"] - c.shift(1)).abs(),
                ], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                natr = float(((atr / c) * 100.0).iloc[-1])

                plus_dm = np.where((df["High"].diff() > -df["Low"].diff()) & (df["High"].diff() > 0), df["High"].diff(), 0.0)
                minus_dm = np.where((-df["Low"].diff() > df["High"].diff()) & (-df["Low"].diff() > 0), -df["Low"].diff(), 0.0)
                plus_di = 100.0 * (pd.Series(plus_dm).rolling(14).mean() / (atr + 1e-9))
                minus_di = 100.0 * (pd.Series(minus_dm).rolling(14).mean() / (atr + 1e-9))
                dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100.0
                adx = float(dx.rolling(14).mean().iloc[-1])

                norm_sym = sym if "/" in sym else (f"{clean[:-4]}/USDT" if clean.endswith("USDT") else sym)
                base_tok = clean[:-4] if clean.endswith("USDT") else clean
                learned_sc = float(coin_scores.get(norm_sym, 0.0))
                p_entry = live_portfolio.get(norm_sym) or live_portfolio.get(sym) or {}
                pf_val = float(p_entry.get("profit_factor", 1.2))
                wr_val = float(p_entry.get("win_rate_pct", 55.0))

                if base_tok in EXCLUDED_TICKERS or "SNDK" in base_tok:
                    to_remove.append({"symbol": sym, "reason": "Unsigned/non-crypto contract on Bybit"})
                    removed_syms.add(sym)
                elif norm_sym in decayed_set or learned_sc <= -6.0:
                    to_remove.append({"symbol": sym, "reason": f"Negative Live performance score ({learned_sc:.1f})"})
                    removed_syms.add(sym)
                elif adx < 13.5:
                    to_remove.append({"symbol": sym, "reason": f"Weak/choppy 1H trend (ADX {adx:.1f} < 13.5)"})
                    removed_syms.add(sym)
                elif natr > 4.6:
                    to_remove.append({"symbol": sym, "reason": f"Extreme whipsaw volatility (NATR {natr:.2f}% > 4.6%)"})
                    removed_syms.add(sym)
                else:
                    rank_quality = (adx * 0.8) + (pf_val * 15.0) + (wr_val * 0.3) + learned_sc
                    ret_vec = df["Close"].pct_change().dropna().tail(36).to_numpy(dtype=float)
                    scored_survivors.append((rank_quality, sym, adx, pf_val, ret_vec))
                del df
                break
            except Exception:
                continue

    # Enforce Portfolio Correlation Diversity (corr > 0.88) + Top-8 Live cap
    if scored_survivors and len(to_remove) < max_removable:
        scored_survivors.sort(key=lambda x: x[0], reverse=True)
        kept = []
        for item in scored_survivors:
            _, sym, adx_v, pf_v, ret_vec = item
            is_dup_corr = False
            corr_peer = ""
            if len(ret_vec) >= 20:
                for _, k_sym, _, _, k_vec in kept:
                    if len(k_vec) >= 20:
                        m = min(len(ret_vec), len(k_vec))
                        c_mat = np.corrcoef(ret_vec[-m:], k_vec[-m:])
                        if not np.isnan(c_mat[0, 1]) and float(c_mat[0, 1]) > 0.88:
                            is_dup_corr = True
                            corr_peer = k_sym
                            break
            if is_dup_corr and len(to_remove) < max_removable and (len(kept) + len(scored_survivors) > 4):
                to_remove.append({"symbol": sym, "reason": f"High 1H correlation (>0.88) with stronger pair {corr_peer}"})
                removed_syms.add(sym)
            elif len(kept) < max_live_coins:
                kept.append(item)
            elif len(to_remove) < max_removable and sym not in removed_syms:
                to_remove.append({"symbol": sym, "reason": f"Trimmed to top {max_live_coins} diverse coins (ADX {adx_v:.1f}, PF {pf_v:.2f})"})
                removed_syms.add(sym)

    gc.collect()
    return json.dumps({"ok": True, "remove": to_remove})


def evaluate_and_tune_parameters(mode_name: str, current_cfg_json: str, state_dir: str) -> str:
    cfg = json.loads(current_cfg_json)
    regime = _fetch_btc_4h_regime(force_refresh=False)
    if not regime.get("ok"):
        return json.dumps({"changed": False, "fingerprint": "UNKNOWN", "reason": "Market telemetry unavailable"})

    mem = _load_self_learning_memory(state_dir)
    consec_losses = int(mem.get("consecutive_live_losses", 0))

    btc_bull = regime["btc_bullish"]
    adx = regime["adx"]
    natr = regime["natr"]
    fp = regime.get("fingerprint", "UNKNOWN")

    cur_mtf = str(cfg.get("mtf_mode", "STRICT")).upper()
    cur_guard = bool(cfg.get("btc_macro_guard", True))
    cur_max_pos = int(cfg.get("max_open_positions", 2))
    cur_risk = float(cfg.get("risk_per_trade_pct", 1.5))

    new_mtf = cur_mtf
    new_guard = cur_guard
    new_max_pos = cur_max_pos
    new_risk = cur_risk
    reasons = []

    if natr >= 2.8 or (adx < 14.0 and not btc_bull):
        new_mtf = "STRICT"
        new_guard = True
        new_max_pos = 1
        new_risk = min(cur_risk, 1.0)
        reasons.append(f"Extreme choppy/high-volatility regime (BTC 4H ADX {adx:.1f}, NATR {natr:.2f}%) -> STRICT MTF, max pos 1 & risk {new_risk:.2f}%")
    elif not btc_bull:
        new_mtf = "STRICT"
        new_guard = True
        new_max_pos = min(cur_max_pos, 2)
        new_risk = min(cur_risk, 1.25)
        reasons.append(f"BTC 4H bearish (${regime['btc_price']}) -> enabled BTC Macro Guard & STRICT MTF")
    elif btc_bull and adx >= 18.0 and natr < 2.6:
        new_mtf = "SOFT"
        new_guard = True
        new_max_pos = max(cur_max_pos, 2)
        new_risk = min(max(cur_risk, 1.5), 2.0)
        reasons.append(f"Strong BTC 4H uptrend (ADX {adx:.1f}) -> switched MTF to SOFT & risk {new_risk:.2f}%")
    else:
        zero_trades_today = True
        try:
            st_file = os.path.join(state_dir, "live_engine_state.json")
            if os.path.exists(st_file):
                with open(st_file, "r", encoding="utf-8") as sf:
                    st_data = json.load(sf)
                zero_trades_today = int(st_data.get("daily_total_trades", 0)) == 0
        except Exception:
            zero_trades_today = True

        if zero_trades_today:
            new_mtf = "SOFT"
            new_guard = True
            new_max_pos = max(cur_max_pos, 2)
            new_risk = min(max(cur_risk, 1.5), 1.75)
            reasons.append(f"0 trades executed today (BTC 4H ADX {adx:.1f}) -> switched MTF to SOFT to capture top daily setup")
        else:
            new_mtf = "STRICT"
            new_guard = True
            new_max_pos = max(cur_max_pos, 2)
            new_risk = 1.5
            reasons.append(f"Daily target met (BTC 4H ADX {adx:.1f}) -> STRICT MTF for additional A+ setups")

    # Upgrade 4: Drawdown streak risk scaling
    if consec_losses >= 2:
        new_risk = round(max(0.75, new_risk * 0.70), 2)
        new_max_pos = 1
        reasons[0] += f" (Scaled risk down to {new_risk:.2f}% after {consec_losses} consecutive Live losses)"

    changes = []
    if new_mtf != cur_mtf:
        changes.append(f"MTF Mode: {cur_mtf} -> {new_mtf}")
    if new_guard != cur_guard:
        changes.append(f"BTC Macro Guard: {cur_guard} -> {new_guard}")
    if new_max_pos != cur_max_pos:
        changes.append(f"Max Positions: {cur_max_pos} -> {new_max_pos}")
    if abs(new_risk - cur_risk) >= 0.05:
        changes.append(f"Risk/Trade: {cur_risk:.2f}% -> {new_risk:.2f}%")

    if not changes:
        return json.dumps({
            "changed": False,
            "fingerprint": fp,
            "regime": regime,
            "summary": f"No parameter change needed ({fp}, ADX {adx:.1f})",
        })

    summary = f"AI updated {', '.join(changes)}. Reason: {reasons[0]}"
    if natr >= 2.8 or (adx < 14.0 and not btc_bull):
        sl_mult, tp_mult, trail_mult = 1.15, 0.85, 0.80
    elif btc_bull and adx >= 18.0 and natr < 2.6:
        sl_mult, tp_mult, trail_mult = 1.05, 1.20, 1.10
    else:
        sl_mult, tp_mult, trail_mult = 1.00, 1.00, 1.00

    return json.dumps({
        "changed": True,
        "fingerprint": fp,
        "mode": mode_name,
        "mtf_mode": new_mtf,
        "btc_macro_guard": new_guard,
        "max_open_positions": new_max_pos,
        "risk_per_trade_pct": round(new_risk, 2),
        "sl_atr_mult": sl_mult,
        "tp_rr_mult": tp_mult,
        "trail_atr_mult": trail_mult,
        "changes": changes,
        "reason": reasons[0],
        "summary": summary,
    })
def cleanup_storage_and_artifacts(paper_dir: str, live_dir: str, models_dir: str) -> str:
    removed_files = 0
    freed_bytes = 0
    scan_dirs = [
        paper_dir,
        live_dir,
        models_dir,
        os.path.join(models_dir, "PAPER"),
        os.path.join(models_dir, "LIVE"),
    ]
    for d in scan_dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            for name in os.listdir(d):
                fpath = os.path.join(d, name)
                if not os.path.isfile(fpath):
                    continue
                is_temp = (
                    name.startswith("_tmp_")
                    or name.endswith(".tmp")
                    or name.endswith(".bak")
                    or name.endswith(".pyc")
                )
                if is_temp:
                    try:
                        sz = os.path.getsize(fpath)
                        os.remove(fpath)
                        removed_files += 1
                        freed_bytes += sz
                    except Exception:
                        pass
        except Exception:
            pass

    # Keep only the latest 2 Paper history snapshots to prevent storage bloat
    hist_dir = os.path.join(models_dir, "PAPER", "history")
    if os.path.isdir(hist_dir):
        try:
            snaps = sorted(
                [os.path.join(hist_dir, f) for f in os.listdir(hist_dir) if f.startswith("UNIFIED_PAPER_") and f.endswith(".json")],
                key=os.path.getmtime,
                reverse=True,
            )
            for old_snap in snaps[2:]:
                try:
                    sz = os.path.getsize(old_snap)
                    os.remove(old_snap)
                    removed_files += 1
                    freed_bytes += sz
                except Exception:
                    pass
        except Exception:
            pass

    # Trim oversized state diagnostics/logs if > 512 KB
    state_file = os.path.join(live_dir, "live_engine_state.json")
    if os.path.exists(state_file) and os.path.getsize(state_file) > 524288:
        try:
            with open(state_file, "r", encoding="utf-8-sig") as f:
                st = json.load(f)
            st["last_diagnostics"] = (st.get("last_diagnostics") or [])[-15:]
            st["last_order_rejections"] = (st.get("last_order_rejections") or [])[-10:]
            st["last_trade_blocks"] = (st.get("last_trade_blocks") or [])[-10:]
            with open(state_file, "w", encoding="utf-8") as wf:
                json.dump(st, wf, indent=2)
        except Exception:
            pass

    return json.dumps({
        "ok": True,
        "removed_files": removed_files,
        "freed_kb": round(freed_bytes / 1024.0, 2),
    })


# ==============================================================================
# AGENTIC AI SUPERVISOR: Observe -> Reason -> Decide -> Act (with Guardrails)
# ==============================================================================
def run_agentic_ai_supervisor(
    api_key: str,
    paper_dir: str,
    live_dir: str,
    models_dir: str,
    current_cfg_json: str,
    force: bool = False,
) -> str:
    os.makedirs(live_dir, exist_ok=True)
    state_file = os.path.join(live_dir, "live_engine_state.json")
    mem = _load_self_learning_memory(live_dir)
    now_ms = int(time.time() * 1000)

    # Throttle LLM calls to once every 15 mins unless forced or a new loss occurred
    last_agent_ms = int(mem.get("last_agentic_run_ms", 0))
    if not force and (now_ms - last_agent_ms) < 15 * 60 * 1000:
        return json.dumps({
            "ok": True,
            "skipped": True,
            "reasoning": mem.get("last_agentic_reasoning", "Agentic AI monitoring active."),
        })

    cfg = {}
    try:
        cfg = json.loads(current_cfg_json) if current_cfg_json else {}
    except Exception:
        cfg = {}

    regime = _fetch_btc_4h_regime(force_refresh=force)
    st = {}
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8-sig") as f:
                st = json.load(f)
        except Exception:
            st = {}

    live_coins = []
    live_master = os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json")
    if os.path.exists(live_master):
        try:
            with open(live_master, "r", encoding="utf-8-sig") as f:
                live_coins = list((json.load(f).get("portfolio") or {}).keys())
        except Exception:
            pass

    recent_blocks = (st.get("last_trade_blocks") or [])[-6:]
    recent_rejects = (st.get("last_order_rejections") or [])[-4:]
    daily_trades = int(st.get("daily_total_trades", 0))
    consec_losses = int(mem.get("consecutive_live_losses", 0))
    decayed_coins = mem.get("decayed_coins", [])

    decision = None
    key = str(api_key or "").strip()

    if key:
        context_snapshot = {
            "btc_4h_regime": regime,
            "current_live_settings": cfg,
            "active_live_coins": live_coins,
            "daily_trades_today": daily_trades,
            "consecutive_losses": consec_losses,
            "decayed_coins": decayed_coins,
            "recent_blocks": recent_blocks,
            "recent_rejections": recent_rejects,
        }
        prompt = (
            "You are an autonomous Agentic AI Trading Supervisor managing a live Bybit crypto/forex bot "
            "with a small $6-$15 USDT account. Analyze the live telemetry snapshot below and decide what "
            "executive action to take right now to maximize win rate and protect capital from whipsaws.\n\n"
            f"TELEMETRY:\n{json.dumps(context_snapshot)}\n\n"
            "Available actions:\n"
            "- HOLD_AND_MONITOR: Keep current settings if healthy.\n"
            "- TIGHTEN_RISK: Switch mtf_mode to STRICT, max_open_positions to 1, and reduce risk_per_trade_pct (0.75-1.25) when choppy or after losses.\n"
            "- AGGRESSIVE_TREND: Switch mtf_mode to SOFT, max_open_positions to 2, risk_per_trade_pct 1.5 when BTC trend is strong and clean.\n"
            "- VETO_SYMBOLS: Temporarily block specific symbols (e.g. ['ETH/USDT']) for 2 hours if overextended or whipsawing.\n"
            "- REMOVE_WEAK_COINS: Permanently remove unsigned/stock contracts (like SNDK/USDT) or chronically failing coins.\n\n"
            "Respond ONLY with valid JSON matching this exact schema:\n"
            '{"action":"TIGHTEN_RISK","confidence":88,"mtf_mode":"STRICT","max_open_positions":1,'
            '"risk_per_trade_pct":1.0,"veto_symbols":[],"remove_coins":[],"veto_hours":2,'
            '"reasoning":"Concise 1-2 sentence human-like explanation of why you made this decision."}'
        )

        for model_name in _get_active_gemini_models(key):
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.25,
                        "responseMimeType": "application/json",
                        "responseSchema": {
                            "type": "OBJECT",
                            "properties": {
                                "action": {"type": "STRING"},
                                "confidence": {"type": "INTEGER"},
                                "mtf_mode": {"type": "STRING"},
                                "max_open_positions": {"type": "INTEGER"},
                                "risk_per_trade_pct": {"type": "NUMBER"},
                                "veto_symbols": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "remove_coins": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "veto_hours": {"type": "INTEGER"},
                                "reasoning": {"type": "STRING"},
                            },
                            "required": [
                                "action", "confidence", "mtf_mode", "max_open_positions",
                                "risk_per_trade_pct", "veto_symbols", "remove_coins",
                                "veto_hours", "reasoning"
                            ],
                        },
                    },
                }
                resp = _SESSION.post(url, json=payload, timeout=12)
                if resp.status_code != 200:
                    continue
                txt = (
                    resp.json()
                    .get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
                )
                if txt.startswith("```"):
                    txt = txt.strip("`").replace("json\n", "", 1).strip()
                parsed = json.loads(txt)
                if isinstance(parsed, dict) and parsed.get("reasoning"):
                    decision = parsed
                    decision["engine"] = f"Gemini Agent ({model_name})"
                    break
            except Exception:
                continue

    # Deterministic quantitative agent fallback if offline or no Gemini key
    if decision is None:
        adx = float(regime.get("adx", 20.0))
        natr = float(regime.get("natr", 1.8))
        btc_bull = bool(regime.get("btc_bullish", True))
        auto_remove = [c for c in live_coins if any(tok in c.upper() for tok in ("SNDK", "NVDA", "TSLA", "AAPL", "MSTR"))]
        if consec_losses >= 2 or natr >= 2.7 or adx < 15.0:
            decision = {
                "action": "TIGHTEN_RISK",
                "confidence": 85,
                "mtf_mode": "STRICT",
                "max_open_positions": 1,
                "risk_per_trade_pct": 1.0,
                "veto_symbols": [],
                "remove_coins": auto_remove,
                "veto_hours": 2,
                "reasoning": f"Detected choppy/loss streak conditions (ADX {adx:.1f}, NATR {natr:.2f}%, losses {consec_losses}) -> capped exposure to 1 position & STRICT 4H alignment.",
                "engine": "Autonomous Quant Agent",
            }
        else:
            decision = {
                "action": "AGGRESSIVE_TREND" if (btc_bull and adx >= 20.0) else "HOLD_AND_MONITOR",
                "confidence": 80,
                "mtf_mode": "SOFT" if daily_trades == 0 else "STRICT",
                "max_open_positions": 1 if daily_trades == 0 else 2,
                "risk_per_trade_pct": 1.25,
                "veto_symbols": [],
                "remove_coins": auto_remove,
                "veto_hours": 2,
                "reasoning": f"BTC 4H {'Bullish' if btc_bull else 'Bearish'} (ADX {adx:.1f}) -> enforcing single-position anti-chase discipline with {('SOFT' if daily_trades == 0 else 'STRICT')} MTF.",
                "engine": "Autonomous Quant Agent",
            }

    # Apply bounded guardrails & persist agentic vetoes + thought logs into live_engine_state.json
    action = str(decision.get("action", "HOLD_AND_MONITOR")).upper()
    mtf_out = str(decision.get("mtf_mode", cfg.get("mtf_mode", "STRICT"))).upper()
    if mtf_out not in ("OFF", "SOFT", "STRICT"):
        mtf_out = "STRICT"
    max_pos_out = max(1, min(int(decision.get("max_open_positions", 1)), 2))
    risk_out = round(max(0.75, min(float(decision.get("risk_per_trade_pct", 1.25)), 2.0)), 2)
    veto_syms = [str(s).replace("/", "").upper() for s in (decision.get("veto_symbols") or []) if str(s).strip()]
    remove_syms = [str(s).strip().upper() for s in (decision.get("remove_coins") or []) if str(s).strip()]
    veto_hours = max(1, min(int(decision.get("veto_hours", 2)), 12))
    reasoning = str(decision.get("reasoning", "Monitoring live market structure.")).strip()
    engine_label = str(decision.get("engine", "Agentic AI"))

    veto_map = st.setdefault("agentic_veto_until_ms", {})
    for vs in veto_syms:
        veto_map[vs] = now_ms + (veto_hours * 3600 * 1000)

    clock_utc = time.strftime("%H:%M:%S UTC", time.gmtime())
    bot_logs = st.setdefault("last_bot_logs", [])
    bot_logs.append(f"[{clock_utc}] [AGENTIC AI • {action}] {reasoning}")
    st["last_bot_logs"] = bot_logs[-60:]

    try:
        with open(state_file, "w", encoding="utf-8") as wf:
            json.dump(st, wf, indent=2)
    except Exception:
        pass

    mem["last_agentic_run_ms"] = now_ms
    mem["last_agentic_reasoning"] = reasoning
    _save_self_learning_memory(live_dir, mem)

    return json.dumps({
        "ok": True,
        "skipped": False,
        "engine": engine_label,
        "action": action,
        "mtf_mode": mtf_out,
        "max_open_positions": max_pos_out,
        "risk_per_trade_pct": risk_out,
        "veto_symbols": veto_syms,
        "remove_coins": remove_syms,
        "reasoning": reasoning,
    })

def record_post_trade_reflection(live_dir: str, symbol: str, side: str, pnl: float, entry_px: float, exit_px: float, adx_val: float = 20.0) -> dict:
    """Updates self-learning memory & IST trading session attribution when a live trade closes."""
    import datetime as _dt
    mem = _load_self_learning_memory(live_dir)
    coin_scores = mem.setdefault("coin_scores", {})
    sess_stats = mem.setdefault("session_stats", {})
    norm_sym = symbol if "/" in symbol else (f"{symbol[:-4]}/USDT" if symbol.endswith("USDT") else symbol)

    ist_now = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=5, minutes=30)))
    hm = ist_now.hour * 60 + ist_now.minute
    sess_name = "ASIA" if (330 <= hm < 810) else ("LONDON" if (810 <= hm < 1110) else "NEW_YORK")
    s_rec = sess_stats.setdefault(sess_name, {"wins": 0, "losses": 0, "net_pnl": 0.0})

    prev_score = float(coin_scores.get(norm_sym, 0.0))
    if pnl > 0:
        coin_scores[norm_sym] = round(min(40.0, prev_score + 6.0), 3)
        mem["consecutive_live_losses"] = 0
        s_rec["wins"] = int(s_rec.get("wins", 0)) + 1
        s_rec["net_pnl"] = round(float(s_rec.get("net_pnl", 0.0)) + pnl, 4)
        lesson = f"WIN [{sess_name}] {norm_sym} ({side}): +${pnl:.4f} -> boosted symbol priority."
    else:
        move_pct = abs(exit_px - entry_px) / max(entry_px, 1e-9) * 100.0
        cause = "low_adx_chop" if adx_val < 16.0 else ("tight_sl_wick" if move_pct < 1.35 else "trend_reversal")
        coin_scores[norm_sym] = round(max(-35.0, prev_score - 8.5), 3)
        mem["consecutive_live_losses"] = int(mem.get("consecutive_live_losses", 0)) + 1
        s_rec["losses"] = int(s_rec.get("losses", 0)) + 1
        s_rec["net_pnl"] = round(float(s_rec.get("net_pnl", 0.0)) + pnl, 4)
        lesson = f"LOSS [{sess_name}] {norm_sym} ({side}): -${abs(pnl):.4f} ({cause}, {move_pct:.2f}%) -> penalized & 3h cooldown."

    mem["last_agentic_reasoning"] = lesson
    mem["last_layer1_learning_event"] = {
        "id": f"L1_TRADE_{int(time.time() * 1000)}_{norm_sym}",
        "timestamp_ms": int(time.time() * 1000),
        "title": f"Layer 1 Self-Learned ({norm_sym})",
        "message": (
            f"Trade Outcome: {'WIN' if pnl > 0 else 'LOSS'} ({'+' if pnl >= 0 else ''}${pnl:.4f} USDT)\n"
            f"Updated Coin Priority: {coin_scores[norm_sym]:+.1f} ({sess_name} session)\n\n"
            f"What Layer 1 Learned:\n{lesson}"
        ),
    }
    _save_self_learning_memory(live_dir, mem)
    return {"ok": True, "lesson": lesson, "coin_score": coin_scores[norm_sym], "session": sess_name}



def _analyze_human_trendline(df, side_u: str, atr: float) -> tuple:
    """
    Non-repainting 2-pivot & 3-pivot Swing Trendline detector (bounces, rejections, and breakout-retests).
    """
    if df is None or len(df) < 25 or atr <= 0:
        return 0, None, None
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    opens = df["Open"].to_numpy(dtype=float)
    n = len(closes)
    c_now = closes[-1]
    o_now = opens[-1]

    swing_lows = []
    swing_highs = []
    for i in range(3, n - 3):
        if lows[i] == min(lows[i - 3:i + 4]):
            swing_lows.append((i, lows[i]))
        if highs[i] == max(highs[i - 3:i + 4]):
            swing_highs.append((i, highs[i]))

    if side_u == "LONG":
        # 1. Rising Support Trendline Bounce (2-pivot or 3-pivot validated)
        if len(swing_lows) >= 2:
            (i1, p1), (i2, p2) = swing_lows[-2], swing_lows[-1]
            if i2 > i1 and p2 > p1:
                slope = (p2 - p1) / float(i2 - i1)
                tl_now = p2 + slope * float((n - 1) - i2)
                dist_atr = (c_now - tl_now) / atr
                three_touch = False
                if len(swing_lows) >= 3:
                    i0, p0 = swing_lows[-3]
                    proj0 = p1 - slope * float(i1 - i0)
                    three_touch = abs(p0 - proj0) <= (0.35 * atr)
                if -0.25 <= dist_atr <= 0.65 and c_now > o_now:
                    bonus = 13 if three_touch else 9
                    lbl = "3-pivot validated bullish trendline bounce" if three_touch else "Bullish trendline retest/bounce"
                    return bonus, f"{lbl} ({dist_atr:+.2f} ATR)", "tl_bounce_long"
                elif dist_atr < -0.50:
                    return -11, f"Broke below rising trendline ({dist_atr:.2f} ATR)", "tl_broken_long"

        # 2. Breakout -> Pullback Retest above Falling Resistance Trendline
        if len(swing_highs) >= 2:
            (i1, p1), (i2, p2) = swing_highs[-2], swing_highs[-1]
            if i2 > i1 and p2 < p1:
                slope = (p2 - p1) / float(i2 - i1)
                tl_now = p2 + slope * float((n - 1) - i2)
                dist_atr = (c_now - tl_now) / atr
                if 0.05 <= dist_atr <= 0.65 and lows[-1] <= tl_now + (0.20 * atr) and c_now > o_now:
                    return 11, f"Bullish trendline breakout-retest hold ({dist_atr:+.2f} ATR)", "tl_break_retest_long"

    elif side_u == "SHORT":
        # 1. Falling Resistance Trendline Rejection (2-pivot or 3-pivot validated)
        if len(swing_highs) >= 2:
            (i1, p1), (i2, p2) = swing_highs[-2], swing_highs[-1]
            if i2 > i1 and p2 < p1:
                slope = (p2 - p1) / float(i2 - i1)
                tl_now = p2 + slope * float((n - 1) - i2)
                dist_atr = (tl_now - c_now) / atr
                three_touch = False
                if len(swing_highs) >= 3:
                    i0, p0 = swing_highs[-3]
                    proj0 = p1 - slope * float(i1 - i0)
                    three_touch = abs(p0 - proj0) <= (0.35 * atr)
                if -0.25 <= dist_atr <= 0.65 and c_now < o_now:
                    bonus = 13 if three_touch else 9
                    lbl = "3-pivot validated bearish trendline rejection" if three_touch else "Bearish trendline rejection"
                    return bonus, f"{lbl} ({dist_atr:+.2f} ATR)", "tl_reject_short"
                elif dist_atr < -0.50:
                    return -11, f"Broke above falling trendline ({dist_atr:.2f} ATR)", "tl_broken_short"

        # 2. Breakdown -> Pullback Retest below Rising Support Trendline
        if len(swing_lows) >= 2:
            (i1, p1), (i2, p2) = swing_lows[-2], swing_lows[-1]
            if i2 > i1 and p2 > p1:
                slope = (p2 - p1) / float(i2 - i1)
                tl_now = p2 + slope * float((n - 1) - i2)
                dist_atr = (tl_now - c_now) / atr
                if 0.05 <= dist_atr <= 0.65 and highs[-1] >= tl_now - (0.20 * atr) and c_now < o_now:
                    return 11, f"Bearish trendline breakdown-retest ({dist_atr:+.2f} ATR)", "tl_break_retest_short"

    return 0, None, None


def _detect_layer1_micro_patterns(df, side_u: str, adx_val: float, atr: float) -> dict:
    """Deterministic offline OHLCV detectors that Layer 1 can auto-activate from Gemini feedback."""
    hits = {}
    if df is None or len(df) < 8 or atr <= 0:
        return hits

    recent8 = df.tail(8)
    first_half = recent8.iloc[:4]
    second_half = recent8.iloc[4:]
    bull_closes = int((recent8["Close"] > recent8["Open"]).sum())
    bear_closes = int((recent8["Close"] < recent8["Open"]).sum())

    # 1. Lower-highs / Higher-lows with mixed closes (counter-momentum)
    if side_u == "LONG":
        if float(second_half["High"].max()) < float(first_half["High"].max()) and bull_closes <= 3:
            hits["LOWER_HIGHS_MIXED_CLOSES"] = f"Lower highs & mixed closes ({bull_closes}/8 bullish) despite ADX {adx_val:.1f}"
    elif side_u == "SHORT":
        if float(second_half["Low"].min()) > float(first_half["Low"].min()) and bear_closes <= 3:
            hits["LOWER_HIGHS_MIXED_CLOSES"] = f"Higher lows & mixed closes ({bear_closes}/8 bearish) despite ADX {adx_val:.1f}"

    # 2. Overlapping sideways chop (small bodies relative to full candle range)
    last6 = df.tail(6)
    ranges6 = (last6["High"] - last6["Low"]).replace(0, 1e-9)
    body_ratios6 = (last6["Close"] - last6["Open"]).abs() / ranges6
    if float(body_ratios6.mean()) < 0.32:
        hits["OVERLAPPING_CHOP_BODIES"] = f"Choppy small-body candles (avg body {float(body_ratios6.mean())*100:.0f}% of range)"

    # 3. Late chasing spike (> 2.3 ATR move already completed over last 3 bars)
    last3 = df.tail(3)
    move3 = float(last3["Close"].iloc[-1]) - float(last3["Open"].iloc[0])
    if (side_u == "LONG" and move3 > 2.3 * atr) or (side_u == "SHORT" and -move3 > 2.3 * atr):
        hits["LATE_EXTENDED_SPIKE"] = f"Extended 3-bar chase ({abs(move3)/atr:.2f} ATR move already spent)"

    # 4. Weak entry candle close position inside its own high-low range
    last_bar = df.iloc[-1]
    bar_rng = max(float(last_bar["High"]) - float(last_bar["Low"]), 1e-9)
    close_pos = (float(last_bar["Close"]) - float(last_bar["Low"])) / bar_rng
    if (side_u == "LONG" and close_pos < 0.45) or (side_u == "SHORT" and close_pos > 0.55):
        hits["WEAK_CANDLE_CLOSE_POS"] = f"Weak entry candle close position ({close_pos*100:.0f}% of bar range)"

    # 5. Short-term 5-bar slope opposing trade direction
    last5 = df.tail(5)
    slope5 = float(last5["Close"].iloc[-1]) - float(last5["Close"].iloc[0])
    if (side_u == "LONG" and slope5 <= 0) or (side_u == "SHORT" and slope5 >= 0):
        hits["OPPOSING_5BAR_SLOPE"] = "5-bar price slope flat or opposing trade direction"

    return hits


def evaluate_layer1_human_price_action(
    candles_df,
    side: str,
    adx_val: float = 20.0,
    funding_rate: float = 0.0,
    session_name: str = "NEW_YORK",
    mem: dict = None,
    is_chop_regime: bool = False,
) -> dict:
    """
    Layer 1 Local Human Price-Action Brain v3 (100% offline, $0 API cost, self-tuning):
    Evaluates trend strength, liquidity-trap wicks, S/R walls, stop-hunt sweeps,
    volume climax exhaustion, and 2/3-pivot swing trendline structure.
    """
    side_u = str(side or "LONG").upper()
    score = 72
    reasons = []
    tags = []
    mem_dict = mem if isinstance(mem, dict) else {}
    tag_weights = mem_dict.get("layer1_tag_weights", {}) or {}

    if candles_df is None or len(candles_df) < 20:
        return {"decision": "REJECT", "score": 30, "reasons": ["Insufficient closed candles for Layer 1"], "tags": ["no_data"]}

    df = candles_df.copy()
    last = df.iloc[-1]
    o = float(last["Open"])
    h = float(last["High"])
    l = float(last["Low"])
    c = float(last["Close"])
    rng = max(h - l, 1e-9)

    atr = float(last["ATR"]) if "ATR" in df.columns and float(last["ATR"]) > 0 else float((df["High"] - df["Low"]).tail(14).mean())
    atr = max(atr, 1e-9)

    # 1. Regime-Aware ADX Filter (Trend Gear vs Gear 2 Chop Mean-Reversion)
    if is_chop_regime:
        if adx_val >= 25.0:
            score -= 22
            reasons.append(f"Runaway trend breakout (ADX {adx_val:.1f} >= 25.0) unsafe for chop fade")
            tags.append("chop_breakout_risk")
        elif adx_val < 22.0:
            score += 8
            reasons.append(f"Confirmed range-bound chop (ADX {adx_val:.1f}) ideal for mean-reversion")
            tags.append("chop_range_ok")
    else:
        if adx_val < 16.5:
            score -= 22
            reasons.append(f"Choppy low-trend market (ADX {adx_val:.1f} < 16.5)")
            tags.append("chop_adx")
        elif adx_val >= 22.0:
            score += 6
            reasons.append(f"Healthy directional trend (ADX {adx_val:.1f})")

    # 2. Liquidity-Trap Wick Filter (>= 46% opposing wick)
    upper_wick_ratio = (h - max(o, c)) / rng
    lower_wick_ratio = (min(o, c) - l) / rng
    if side_u == "LONG" and upper_wick_ratio >= 0.46:
        score -= 18
        reasons.append(f"Upper liquidity-trap rejection wick ({upper_wick_ratio*100:.0f}%)")
        tags.append("wick_trap_long")
    elif side_u == "SHORT" and lower_wick_ratio >= 0.46:
        score -= 18
        reasons.append(f"Lower absorption/trap wick ({lower_wick_ratio*100:.0f}%)")
        tags.append("wick_trap_short")

    # 3. Nearby S/R Wall Filter (< 0.85 ATR room to swing barrier)
    prior_20 = df.iloc[-21:-1]
    swing_high_20 = float(prior_20["High"].max())
    swing_low_20 = float(prior_20["Low"].min())
    if side_u == "LONG":
        room_atr = (swing_high_20 - c) / atr
        if 0.0 < room_atr < 0.85:
            score -= 15
            reasons.append(f"Overhead resistance wall ({room_atr:.2f} ATR away)")
            tags.append("sr_wall_long")
        elif l < swing_low_20 and c > swing_low_20 and c > o:
            score += 10
            reasons.append("Bullish stop-hunt liquidity sweep & reclaim")
            tags.append("liq_sweep_long")
    else:
        room_atr = (c - swing_low_20) / atr
        if 0.0 < room_atr < 0.85:
            score -= 15
            reasons.append(f"Nearby support floor ({room_atr:.2f} ATR away)")
            tags.append("sr_wall_short")
        elif h > swing_high_20 and c < swing_high_20 and c < o:
            score += 10
            reasons.append("Bearish stop-hunt liquidity sweep & rejection")
            tags.append("liq_sweep_short")

    # 4. Candle Direction & Volume Climax Exhaustion
    vol_surge = float(last["Vol_Surge"]) if "Vol_Surge" in df.columns else (
        float(last["Volume"]) / max(float(df["Volume"].tail(20).mean()), 1e-9)
    )
    if (side_u == "LONG" and c < o) or (side_u == "SHORT" and c > o):
        score -= 10
        reasons.append("Entry candle closed against trade direction")
        tags.append("opposing_candle")
    else:
        score += 5

    if vol_surge >= 2.8:
        score -= 12
        reasons.append(f"Volume climax exhaustion ({vol_surge:.1f}x avg)")
        tags.append("vol_climax")
    elif 1.10 <= vol_surge < 2.4:
        score += 4

    # 5. Swing-Pivot Trendline Bounce / Rejection / Breakout-Retest
    tl_delta, tl_reason, tl_tag = _analyze_human_trendline(df, side_u, atr)
    if tl_delta != 0:
        score += tl_delta
        if tl_reason:
            reasons.append(tl_reason)
        if tl_tag:
            tags.append(tl_tag)

    # 5b. Auto-Learned Permanent Price-Action Rules (Gear-Isolated G1 vs G2 buckets)
    detected_patterns = _detect_layer1_micro_patterns(df, side_u, adx_val, atr)
    raw_learned = mem.get("layer1_learned_rules", {}) if isinstance(mem, dict) else {}
    gear_bucket = "g2_rules" if is_chop_regime else "g1_rules"
    if isinstance(raw_learned, dict) and isinstance(raw_learned.get(gear_bucket), dict):
        learned_rules = raw_learned.get(gear_bucket, {})
    elif isinstance(raw_learned, dict):
        learned_rules = {k: v for k, v in raw_learned.items() if k not in ("g1_rules", "g2_rules")}
    else:
        learned_rules = {}

    for pat_code, pat_desc in detected_patterns.items():
        if is_chop_regime and pat_code in ("OVERLAPPING_CHOP_BODIES", "OPPOSING_5BAR_SLOPE"):
            continue
        tags.append(pat_code)
        rule_cfg = learned_rules.get(pat_code)
        if not is_chop_regime and pat_code == "LOWER_HIGHS_MIXED_CLOSES":
            pen = int(rule_cfg.get("penalty", 24)) if isinstance(rule_cfg, dict) else 24
            score -= abs(pen)
            reasons.append(pat_desc)
        elif isinstance(rule_cfg, dict) and int(rule_cfg.get("flags", 0)) >= 1:
            pen = int(rule_cfg.get("penalty", 20 if rule_cfg.get("active") else 8))
            score -= abs(pen)
            status_lbl = "Permanent L1 Rule" if rule_cfg.get("active") else "L1 Caution"
            reasons.append(f"[{status_lbl}] {pat_desc}")

    # 5c. Gear 1 Trend Anti-Chase, Counter-Spike & High-ADX Exhaustion Guards (from Gemini Shadow-Compare)
    if not is_chop_regime:
        dist_e20 = float(last["Dist_EMA20"]) if "Dist_EMA20" in df.columns else 0.0
        if (
            "LATE_EXTENDED_SPIKE" in detected_patterns
            or (side_u == "LONG" and dist_e20 > 1.75)
            or (side_u == "SHORT" and dist_e20 < -1.75)
        ):
            score -= 22
            reasons.append("Extended multi-bar spike ? waiting for constructive pullback")
            tags.append("g1_extended_spike_chase")

        if side_u == "SHORT" and len(df) >= 5:
            spike_3b = (float(df["High"].tail(3).max()) - float(df["Low"].iloc[-4])) / atr
            ema20_now = float(df["EMA_20"].iloc[-1]) if "EMA_20" in df.columns else c
            if dist_e20 > 1.35 or (spike_3b > 2.2 and c > ema20_now):
                score -= 22
                reasons.append("Recent vertical upward spike ? wait for structure breakdown before shorting")
                tags.append("g1_counter_spike_short")

        if adx_val >= 48.0 and "OVERLAPPING_CHOP_BODIES" in detected_patterns:
            score -= 20
            reasons.append(f"Extreme ADX ({adx_val:.1f}) offset by overlapping chop bodies")
            tags.append("g1_extreme_adx_chop")

    # 6. Apply Agentic AI Learned Tag Adjustments (from shadow_compare & post-trade outcomes)
    for t in tags:
        adj = int(tag_weights.get(t, 0))
        if adj != 0:
            score += adj

    score = int(max(10, min(99, score)))
    decision = "APPROVE" if score >= 56 else "REJECT"
    if not reasons:
        reasons.append("Clean human price-action structure & trendline alignment")
    return {"decision": decision, "score": score, "reasons": reasons, "tags": tags}


def evaluate_pre_trade_human_brain(
    live_dir: str,
    symbol: str,
    side: str,
    candles_df=None,
    adx_val: float = 20.0,
    funding_rate: float = 0.0,
    session_name: str = "NEW_YORK",
    gemini_enabled: bool = False,
    gemini_api_key: str = "",
    is_chop_regime: bool = False,
    timeframe: str = "1H",
    strategy_name: str = "",
) -> dict:
    """
    Runs Layer 1 Local Human Price-Action Brain (gatekeeper) + optional Gemini Shadow-Compare (test only).
    Persists agreement/disagreement metrics to ai_copilot_memory.json for Settings -> Layer 1 vs Gemini AI.
    """
    mem = _load_self_learning_memory(live_dir)
    l1 = evaluate_layer1_human_price_action(
        candles_df=candles_df,
        side=side,
        adx_val=adx_val,
        funding_rate=funding_rate,
        session_name=session_name,
        mem=mem,
        is_chop_regime=is_chop_regime,
    )
    shadow_log = ""

    key = str(gemini_api_key or "").strip()
    if not key:
        for cfg_name in ("ai_shadow_config.json", "ai_copilot_config.json", "live_runtime_config.json"):
            c_path = os.path.join(live_dir, cfg_name)
            if os.path.exists(c_path):
                try:
                    with open(c_path, "r", encoding="utf-8-sig") as cf:
                        c_obj = json.load(cf)
                    cand_k = str(c_obj.get("gemini_api_key") or c_obj.get("api_key_gemini") or "").strip()
                    if cand_k:
                        key = cand_k
                        break
                except Exception:
                    pass

    if gemini_enabled and key and candles_df is not None and len(candles_df) >= 12:
        try:
            tail12 = candles_df.tail(12)
            bars_txt = []
            for idx, r in tail12.iterrows():
                bars_txt.append(
                    f"O={float(r['Open']):.4f} H={float(r['High']):.4f} L={float(r['Low']):.4f} C={float(r['Close']):.4f}"
                )
            tf_lbl = str(timeframe or ("15M" if is_chop_regime else "1H")).upper()
            gear_ctx = (
                f"GEAR 2 CHOP/RANGE MEAN-REVERSION on {tf_lbl} (strategy={strategy_name or 'G2_CHOP'}): "
                "fading range extremes/Bollinger/Keltner extensions back toward the mean midline in low-ADX chop; "
                "do NOT reject merely for sideways range or counter-5bar pullback if price is rejecting a range boundary."
                if is_chop_regime
                else f"GEAR 1 DIRECTIONAL TREND on {tf_lbl} (strategy={strategy_name or 'G1_TREND'}): "
                "requires clean trend continuation/breakout structure without chasing extended spikes."
            )
            prompt = (
                f"You are a senior institutional trader reviewing a {side} setup on {symbol} ({tf_lbl}) "
                f"(ADX={adx_val:.1f}, Funding={funding_rate*100:+.4f}%, Session={session_name}). "
                f"Mode: {gear_ctx} "
                f"Last 12 closed {tf_lbl} candles (oldest to newest): {' | '.join(bars_txt)}. "
                'Return ONLY raw JSON: {"decision": "APPROVE" or "REJECT", "confidence": 0-100, '
                '"pattern_code": "LOWER_HIGHS_MIXED_CLOSES"|"OVERLAPPING_CHOP_BODIES"|"LATE_EXTENDED_SPIKE"|"WEAK_CANDLE_CLOSE_POS"|"OPPOSING_5BAR_SLOPE"|"NONE", '
                '"reason": "short string"}.'
            )
            gem_dec = None
            gem_conf = 0
            gem_reason = ""
            gem_pattern = "NONE"
            last_http_err = ""
            model_candidates = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
            for m_discovered in (_get_active_gemini_models(key) or []):
                if m_discovered not in model_candidates and "2.5" not in m_discovered:
                    model_candidates.append(m_discovered)

            for model_name in model_candidates[:4]:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                        },
                    }
                    resp = _SESSION.post(url, json=payload, timeout=7.0)
                    if resp.status_code == 200:
                        txt = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if txt.startswith("```"):
                            txt = txt.strip("`").replace("json" + chr(10), "", 1).strip()
                        parsed = json.loads(txt)
                        gem_dec = str(parsed.get("decision", "REJECT")).upper()
                        if gem_dec not in ("APPROVE", "REJECT"):
                            gem_dec = "REJECT"
                        gem_conf = int(max(0, min(100, int(parsed.get("confidence", 50)))))
                        gem_reason = str(parsed.get("reason", ""))[:160]
                        gem_pattern = str(parsed.get("pattern_code", "NONE")).strip().upper()
                        break
                    else:
                        last_http_err = f"{model_name} HTTP {resp.status_code}"
                except Exception as call_ex:
                    last_http_err = f"{model_name} {call_ex}"
                    continue
            if not gem_dec and last_http_err:
                shadow_log = f"[GEMINI SHADOW WARN] {symbol}: {last_http_err}"

            if gem_dec:
                sc = mem.setdefault("shadow_compare", {
                    "total": 0, "agreements": 0, "disagreements": 0,
                    "agreement_rate_pct": 100.0, "last_summary": "", "recent_diffs": []
                })
                sc["total"] = int(sc.get("total", 0)) + 1
                matched = (l1["decision"] == gem_dec)
                if matched:
                    sc["agreements"] = int(sc.get("agreements", 0)) + 1
                else:
                    sc["disagreements"] = int(sc.get("disagreements", 0)) + 1
                    tip = f"Layer 1 said {l1['decision']} ({l1['score']}/100) while Gemini said {gem_dec} ({gem_conf}%): {gem_reason}"
                    tw = mem.setdefault("layer1_tag_weights", {})
                    for tg in l1.get("tags", []):
                        cur_w = int(tw.get(tg, 0))
                        tw[tg] = max(-10, min(10, cur_w + (1 if gem_dec == "APPROVE" else -1)))
                    diffs = sc.setdefault("recent_diffs", [])
                    diffs.insert(0, {
                        "symbol": symbol,
                        "side": side,
                        "layer1": l1["decision"],
                        "layer1_score": l1["score"],
                        "gemini": gem_dec,
                        "gemini_conf": gem_conf,
                        "why_different": gem_reason,
                        "improvement_tip": tip,
                    })
                    sc["recent_diffs"] = diffs[:20]
                    valid_codes = {
                        "LOWER_HIGHS_MIXED_CLOSES",
                        "OVERLAPPING_CHOP_BODIES",
                        "LATE_EXTENDED_SPIKE",
                        "WEAK_CANDLE_CLOSE_POS",
                        "OPPOSING_5BAR_SLOPE",
                    }
                    r_low = gem_reason.lower()
                    if gem_pattern not in valid_codes:
                        if "lower high" in r_low or "higher low" in r_low or "mixed close" in r_low or "sustain" in r_low:
                            gem_pattern = "LOWER_HIGHS_MIXED_CLOSES"
                        elif "chop" in r_low or "consolidation" in r_low or "sideways" in r_low or "indecis" in r_low:
                            gem_pattern = "OVERLAPPING_CHOP_BODIES"
                        elif "extended" in r_low or "overbought" in r_low or "oversold" in r_low or "exhaust" in r_low:
                            gem_pattern = "LATE_EXTENDED_SPIKE"
                        elif "close" in r_low or "wick" in r_low or "rejection" in r_low:
                            gem_pattern = "WEAK_CANDLE_CLOSE_POS"
                        elif "momentum" in r_low or "slope" in r_low or "trend" in r_low:
                            gem_pattern = "OPPOSING_5BAR_SLOPE"

                    rule_status_line = ""
                    if gem_dec == "REJECT" and gem_pattern in valid_codes:
                        root_rules = mem.setdefault("layer1_learned_rules", {})
                        g_key = "g2_rules" if is_chop_regime else "g1_rules"
                        l_rules = root_rules.setdefault(g_key, {})
                        r_entry = l_rules.setdefault(gem_pattern, {"flags": 0, "active": False, "penalty": 8, "learned_from": ""})
                        r_entry["flags"] = int(r_entry.get("flags", 0)) + 1
                        r_entry["learned_from"] = gem_reason
                        if r_entry["flags"] >= 2:
                            r_entry["active"] = True
                            r_entry["penalty"] = 22
                            rule_status_line = f"PERMANENT RULE ACTIVATED: {gem_pattern} (-22 pts offline)\n"
                        else:
                            r_entry["penalty"] = 8
                            rule_status_line = f"New Pattern Tracked (1/2 flags): {gem_pattern} (-8 pts caution)\n"

                    adj_tags = ", ".join(l1.get("tags", [])) or "setup_score"
                    delta_txt = "+1" if gem_dec == "APPROVE" else "-1"
                    mem["last_layer1_learning_event"] = {
                        "id": f"L1_SHADOW_{int(time.time() * 1000)}_{symbol}",
                        "timestamp_ms": int(time.time() * 1000),
                        "title": f"Layer 1 Self-Learned ({symbol} {side})",
                        "message": (
                            f"Shadow Audit: Layer 1={l1['decision']} ({l1['score']}/100) vs Gemini={gem_dec} ({gem_conf}%)\n"
                            f"{rule_status_line}"
                            f"Updated Weights: [{adj_tags}] ({delta_txt})\n\n"
                            f"What Layer 1 Learned:\n{gem_reason}"
                        ),
                    }

                sc["agreement_rate_pct"] = round((sc["agreements"] / max(sc["total"], 1)) * 100.0, 1)
                sc["last_summary"] = f"Last: {symbol} {side} | L1={l1['decision']} ({l1['score']}) vs Gemini={gem_dec} ({gem_conf}%)"
                _save_self_learning_memory(live_dir, mem)
                shadow_log = f"[L1 vs GEMINI] {symbol} {side}: Layer1={l1['decision']}({l1['score']}) vs Gemini={gem_dec}({gem_conf}%) | {gem_reason}"
        except Exception as e:
            shadow_log = f"[GEMINI SHADOW WARN] {symbol}: {e}"

    return {
        "decision": l1["decision"],
        "score": l1["score"],
        "reasons": l1["reasons"],
        "shadow_log": shadow_log,
    }


def prune_underperforming_gemini_strategies(paper_dir: str, live_dir: str, models_dir: str = None) -> dict:
    """
    Auto-removes Gemini-invented or Gemini-adopted strategies that fail out-of-sample Paper backtests
    (PF < 1.15, net_pnl <= 0, or < 4 trades) or accumulate >= 2 net losses in Live reflection memory.
    """
    removed_ids = []
    kept_ids = []

    res_path = os.path.join(paper_dir, "paper_trained_results.json")
    trained_coins = {}
    if os.path.exists(res_path):
        try:
            with open(res_path, "r", encoding="utf-8-sig") as rf:
                trained_coins = (json.load(rf).get("coins", {}) or {})
        except Exception:
            trained_coins = {}

    live_mem = _load_self_learning_memory(live_dir)
    live_strat_stats = live_mem.get("strategy_stats", {}) or {}

    for fname in ("ai_discovered_strategies.json", "ai_adopted_strategies.json"):
        fpath = os.path.join(paper_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8-sig") as f:
                payload = json.load(f)
            strats = payload.get("strategies", {}) or {}
            if not strats:
                continue

            new_strats = {}
            for sid, spec in strats.items():
                ls = live_strat_stats.get(sid, {})
                l_wins = int(ls.get("wins", 0))
                l_losses = int(ls.get("losses", 0))
                l_pnl = float(ls.get("net_pnl", 0.0))
                if (l_losses >= 2 and l_wins == 0) or (l_losses >= 3 and l_pnl < 0.0):
                    if sid not in removed_ids:
                        removed_ids.append(sid)
                    continue

                if trained_coins:
                    matching_runs = []
                    for cdata in trained_coins.values():
                        if not isinstance(cdata, dict):
                            continue
                        if str(cdata.get("strategy_id", "")) == sid:
                            matching_runs.append(cdata)
                        for mrun in (cdata.get("tested_matrix", []) or []):
                            if isinstance(mrun, dict) and str(mrun.get("strategy_id", "")) == sid:
                                matching_runs.append(mrun)

                    if matching_runs:
                        best_run = max(
                            matching_runs,
                            key=lambda r: (float(r.get("net_pnl_usdt", 0.0)), float(r.get("profit_factor", 0.0)))
                        )
                        b_pnl = float(best_run.get("net_pnl_usdt", 0.0))
                        b_pf = float(best_run.get("profit_factor", 0.0))
                        b_tr = int(best_run.get("trades", 0))
                        if b_tr < 4 or b_pnl <= 0.0 or b_pf < 1.15:
                            if sid not in removed_ids:
                                removed_ids.append(sid)
                            continue

                new_strats[sid] = spec
                if sid not in kept_ids:
                    kept_ids.append(sid)

            if len(new_strats) != len(strats):
                payload["strategies"] = new_strats
                payload["updated_at"] = int(time.time())
                tmp_f = fpath + f".tmp.{os.getpid()}_{_at_threading.get_ident()}_{_at_uuid.uuid4().hex[:8]}"
                with open(tmp_f, "w", encoding="utf-8") as wf:
                    json.dump(payload, wf, indent=2)
                _locked_replace(tmp_f, fpath)
        except Exception:
            continue

    if removed_ids and models_dir:
        live_master_p = os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json")
        if os.path.exists(live_master_p):
            try:
                with open(live_master_p, "r", encoding="utf-8-sig") as lf:
                    lm = json.load(lf)
                port = lm.get("portfolio", {}) or {}
                changed = False
                for sym, entry in port.items():
                    if isinstance(entry, dict) and str(entry.get("strategy_id", "")) in removed_ids:
                        entry["strategy_id"] = "xgb_mtf_v2"
                        entry["strategy"] = f"XGBoost 1H/4H Momentum ({entry.get('timeframe', '1H')})"
                        cfg = entry.get("config", {})
                        if isinstance(cfg, dict):
                            cfg["strategy_id"] = "xgb_mtf_v2"
                            cfg.pop("custom_rules", None)
                        changed = True
                if changed:
                    with open(live_master_p, "w", encoding="utf-8") as wf:
                        json.dump(lm, wf, indent=2)
            except Exception:
                pass

    return {"ok": True, "removed": removed_ids, "kept": kept_ids}


def export_full_state_bundle_json(paper_dir: str, live_dir: str, models_dir: str) -> str:
    """Packages Master ML Models, Agentic memory, Gemini adopted strategies & reports into one JSON for Google Drive."""
    bundle = {
        "schema": "AUTONOMOUS_AI_TRADING_DRIVE_BACKUP_V1",
        "exported_at": int(time.time()),
        "files": {}
    }
    rel_targets = [
        ("ml_models/LIVE/UNIFIED_PORTFOLIO_MASTER.json", os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json")),
        ("ml_models/PAPER/UNIFIED_PORTFOLIO_MASTER.json", os.path.join(models_dir, "PAPER", "UNIFIED_PORTFOLIO_MASTER.json")),
        ("bot_data/LIVE_ACCOUNT/ai_copilot_memory.json", os.path.join(live_dir, "ai_copilot_memory.json")),
        ("bot_data/LIVE_ACCOUNT/live_reports_history.json", os.path.join(live_dir, "live_reports_history.json")),
        ("bot_data/PAPER/ai_adopted_strategies.json", os.path.join(paper_dir, "ai_adopted_strategies.json")),
        ("bot_data/PAPER/ai_discovered_strategies.json", os.path.join(paper_dir, "ai_discovered_strategies.json")),
    ]
    for rel_key, abs_path in rel_targets:
        if os.path.exists(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8-sig") as f:
                    bundle["files"][rel_key] = json.load(f)
            except Exception:
                pass
    return json.dumps(bundle)


def import_full_state_bundle_json(base_files_dir: str, bundle_json: str) -> str:
    """Restores Master ML Models, Agentic memory, and strategies from a Google Drive backup JSON."""
    try:
        data = json.loads(bundle_json)
        if data.get("schema") != "AUTONOMOUS_AI_TRADING_DRIVE_BACKUP_V1":
            return json.dumps({"ok": False, "error": "Invalid Google Drive backup file format."})
        files_map = data.get("files", {}) or {}
        restored = 0
        for rel_key, content in files_map.items():
            safe_rel = rel_key.replace(chr(92), "/").lstrip("/")
            if ".." in safe_rel:
                continue
            target_path = os.path.join(base_files_dir, *safe_rel.split("/"))
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as wf:
                json.dump(content, wf, indent=2)
            restored += 1
        return json.dumps({"ok": True, "restored_files": restored})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


# ==============================================================================
# SELF-EVOLVING 4-LAYER AGENTIC AI ENGINES (ZERO-MAINTENANCE AUTONOMY)
# ==============================================================================

_ELIGIBLE_INSTRUMENTS_CACHE = {"ts": 0.0, "symbols": set()}


def _fetch_eligible_linear_instruments(min_age_days: int = 140) -> set:
    """Queries Bybit V5 instruments-info (cached 6h) to filter out <140d listings, dated futures, and pre-listings."""
    now_s = time.time()
    if _ELIGIBLE_INSTRUMENTS_CACHE["symbols"] and (now_s - _ELIGIBLE_INSTRUMENTS_CACHE["ts"]) < 21600.0:
        return set(_ELIGIBLE_INSTRUMENTS_CACHE["symbols"])

    min_launch_ms = int((now_s - (min_age_days * 86400.0)) * 1000.0)
    eligible = set()
    for base in ("https://api.bybit.com", "https://api.bytick.com"):
        try:
            cursor = ""
            pages = 0
            while pages < 4:
                params = {"category": "linear", "limit": 1000}
                if cursor:
                    params["cursor"] = cursor
                resp = _SESSION.get(f"{base}/v5/market/instruments-info", params=params, timeout=7)
                res = resp.json().get("result", {}) or {}
                items = res.get("list", []) or []
                for it in items:
                    sym = str(it.get("symbol", "")).upper()
                    if not sym.endswith("USDT"):
                        continue
                    if str(it.get("status", "")).lower() != "trading":
                        continue
                    if bool(it.get("isPreListing", False)):
                        continue
                    deliv = str(it.get("deliveryTime", "0") or "0").strip()
                    if deliv not in ("", "0"):
                        continue
                    launch_ms = int(float(it.get("launchTime", 0) or 0))
                    if launch_ms > 0 and launch_ms > min_launch_ms:
                        continue
                    eligible.add(sym)
                cursor = str(res.get("nextPageCursor", "") or "").strip()
                pages += 1
                if not cursor:
                    break
            if eligible:
                _ELIGIBLE_INSTRUMENTS_CACHE["ts"] = now_s
                _ELIGIBLE_INSTRUMENTS_CACHE["symbols"] = eligible
                return set(eligible)
        except Exception:
            continue
    return set(_ELIGIBLE_INSTRUMENTS_CACHE["symbols"])


# _save_self_learning_memory unified at top of module


def quarantine_symbol_in_memory(live_dir: str, symbol: str, reason: str = "", days: int = 30):
    """Automatically quarantines restricted/unsigned Bybit symbols for 30 days without code edits."""
    try:
        clean = symbol.replace("/", "").upper()
        mem = _load_self_learning_memory(live_dir)
        q_map = mem.setdefault("auto_quarantine", {})
        q_reasons = mem.setdefault("auto_quarantine_reasons", {})
        q_map[clean] = int(time.time() * 1000) + (days * 86400 * 1000)
        q_reasons[clean] = reason[:120]
        _save_self_learning_memory(live_dir, mem)
    except Exception:
        pass


def record_shadow_counterfactual(
    live_dir: str,
    symbol: str,
    side: str,
    entry_px: float,
    sl_px: float,
    tp_px: float,
    blocking_gate: str,
    setup_score: int,
):
    """Logs 75%-97% blocked setups so the AI can verify whether each gate saved money or missed winners."""
    if entry_px <= 0.0 or sl_px <= 0.0 or tp_px <= 0.0 or setup_score < 75:
        return
    try:
        clean = symbol.replace("/", "").upper()
        mem = _load_self_learning_memory(live_dir)
        shadows = mem.setdefault("shadow_counterfactuals", [])
        now_ms = int(time.time() * 1000)
        if any(s.get("symbol") == clean and not s.get("resolved", False) for s in shadows):
            return
        shadows.append({
            "symbol": clean,
            "side": side.upper(),
            "entry": round(float(entry_px), 6),
            "sl": round(float(sl_px), 6),
            "tp": round(float(tp_px), 6),
            "gate": str(blocking_gate),
            "setup_score": int(setup_score),
            "created_ms": now_ms,
            "expires_ms": now_ms + (8 * 3600 * 1000),
            "resolved": False,
        })
        mem["shadow_counterfactuals"] = shadows[-60:]
        _save_self_learning_memory(live_dir, mem)
    except Exception:
        pass


def evaluate_shadow_counterfactuals(live_dir: str, mark_prices: dict) -> dict:
    """Evaluates pending blocked setups against live mark prices and auto-tunes gate strictness."""
    try:
        mem = _load_self_learning_memory(live_dir)
        shadows = mem.get("shadow_counterfactuals", []) or []
        if not shadows:
            return mem.get("gate_adjustments", {}) or {}

        now_ms = int(time.time() * 1000)
        gate_stats = mem.setdefault("gate_counterfactual_stats", {})
        gate_adj = mem.setdefault("gate_adjustments", {})
        changed = False

        for s in shadows:
            if s.get("resolved", False):
                continue
            sym = str(s.get("symbol", "")).upper()
            mark = float(mark_prices.get(sym, 0.0) or 0.0)
            if mark <= 0.0:
                continue
            side = str(s.get("side", "LONG")).upper()
            sl = float(s.get("sl", 0.0))
            tp = float(s.get("tp", 0.0))
            gate = str(s.get("gate", "GENERAL"))

            hit_tp = (mark >= tp) if side == "LONG" else (mark <= tp)
            hit_sl = (mark <= sl) if side == "LONG" else (mark >= sl)
            expired = now_ms >= int(s.get("expires_ms", 0))

            if hit_tp or hit_sl or expired:
                s["resolved"] = True
                changed = True
                if hit_tp or hit_sl:
                    gs = gate_stats.setdefault(gate, {"would_win": 0, "would_lose": 0})
                    if hit_tp:
                        gs["would_win"] = int(gs.get("would_win", 0)) + 1
                    else:
                        gs["would_lose"] = int(gs.get("would_lose", 0)) + 1
                    tot = gs["would_win"] + gs["would_lose"]
                    if tot >= 8:
                        missed_win_rate = gs["would_win"] / float(tot)
                        if missed_win_rate > 0.65:
                            gate_adj[gate] = round(max(-0.12, float(gate_adj.get(gate, 0.0)) - 0.04), 3)
                        elif missed_win_rate < 0.40:
                            gate_adj[gate] = round(min(0.10, float(gate_adj.get(gate, 0.0)) + 0.03), 3)

        if changed:
            mem["shadow_counterfactuals"] = [x for x in shadows if not x.get("resolved", False)][-40:]
            _save_self_learning_memory(live_dir, mem)
        return gate_adj
    except Exception:
        return {}


def get_bayesian_risk_multiplier(live_dir: str, symbol: str, session_name: str, prior_wr_pct: float = 55.0) -> float:
    """Thompson-inspired Beta posterior risk scaler (0.60x to 1.25x) per (coin, session)."""
    try:
        clean = symbol.replace("/", "").upper()
        key = f"{clean}_{session_name.upper()}"
        mem = _load_self_learning_memory(live_dir)
        buckets = mem.get("bayesian_buckets", {}) or {}
        b = buckets.get(key)
        if not isinstance(b, dict):
            p0 = min(max(float(prior_wr_pct) / 100.0, 0.40), 0.70)
            alpha, beta = 4.0 * p0, 4.0 * (1.0 - p0)
        else:
            alpha = max(1.0, float(b.get("alpha", 2.2)))
            beta = max(1.0, float(b.get("beta", 1.8)))
        mean_p = alpha / (alpha + beta)
        # Neutral at 0.55 win probability -> 1.00x; scales smoothly between 0.60x and 1.25x
        mult = 1.0 + ((mean_p - 0.55) * 1.65)
        return round(min(max(mult, 0.60), 1.25), 3)
    except Exception:
        return 1.0


def update_bayesian_bucket(live_dir: str, symbol: str, session_name: str, net_pnl: float):
    """Updates Beta(alpha, beta) posterior after every closed trade."""
    try:
        clean = symbol.replace("/", "").upper()
        key = f"{clean}_{session_name.upper()}"
        mem = _load_self_learning_memory(live_dir)
        buckets = mem.setdefault("bayesian_buckets", {})
        b = buckets.setdefault(key, {"alpha": 2.2, "beta": 1.8})
        if net_pnl > 0.0:
            b["alpha"] = round(min(40.0, float(b.get("alpha", 2.2)) + 1.0), 3)
        elif net_pnl < 0.0:
            b["beta"] = round(min(40.0, float(b.get("beta", 1.8)) + 1.0), 3)
        _save_self_learning_memory(live_dir, mem)
    except Exception:
        pass


def check_coin_feature_drift(
    live_dir: str,
    symbol: str,
    live_row: dict,
    features_list: list,
    feature_means: list,
    feature_stds: list,
) -> tuple:
    """Detects regime/volatility drift between live bars and stored training baselines."""
    try:
        if not features_list or not feature_means or not feature_stds:
            return False, 0.0
        watch_feats = ("NATR", "ADX", "BB_Width", "Dist_EMA200")
        z_scores = []
        for idx, fname in enumerate(features_list):
            if fname not in watch_feats or idx >= len(feature_means) or idx >= len(feature_stds):
                continue
            val = float(live_row.get(fname, 0.0) or 0.0)
            mu = float(feature_means[idx] or 0.0)
            sd = max(float(feature_stds[idx] or 1.0), 1e-5)
            z_scores.append(abs(val - mu) / sd)
        if not z_scores:
            return False, 0.0
        avg_z = sum(z_scores) / len(z_scores)
        drifted = bool(avg_z > 2.65)
        if drifted:
            clean = symbol.replace("/", "").upper()
            mem = _load_self_learning_memory(live_dir)
            dw = mem.setdefault("drift_watchlist", {})
            dw[clean] = {"avg_z": round(avg_z, 2), "flagged_at": int(time.time())}
            _save_self_learning_memory(live_dir, mem)
        return drifted, round(avg_z, 2)
    except Exception:
        return False, 0.0


def passes_2x_fee_stress_test(trades_raw_returns: list, leverage: int = 10, double_round_trip_fee: float = 0.0022) -> bool:
    """Stress-tests Gemini-invented strategies under 2x taker fees & slippage."""
    if not trades_raw_returns or len(trades_raw_returns) < 14:
        return False
    gross_wins = 0.0
    gross_losses = 0.0
    for r in trades_raw_returns:
        net_r = (float(r) - double_round_trip_fee) * leverage
        if net_r > 0.0:
            gross_wins += net_r
        elif net_r < 0.0:
            gross_losses += abs(net_r)
    if gross_losses <= 1e-9:
        return gross_wins > 0.0
    return (gross_wins / gross_losses) >= 1.35


# ==============================================================================
# WAVE 2: EPISTEMIC CALIBRATION, ORDERFLOW POSITIONING & FAST REPORTS ENGINE
# ==============================================================================

_POSITIONING_CACHE = {}
_REPORTS_FAST_CACHE = {"ts": 0.0, "payload": ""}


def fetch_bybit_positioning_metrics(bybit_sym: str) -> dict:
    """Fetches Bybit V5 1H Open Interest delta & Long/Short account skew (cached 15m; 0.0 for FX/Metals)."""
    clean = str(bybit_sym or "").replace("/", "").upper()
    if not clean.endswith("USDT") or clean in ("EURUSDUSDT", "GBPUSDUSDT", "USDJPYUSDT", "XAUUSDT", "XAGUSDT", "XAUTUSDT"):
        return {"oi_delta_4h": 0.0, "ls_skew": 0.0, "buy_ratio": 0.5}

    now_s = time.time()
    cached = _POSITIONING_CACHE.get(clean)
    if cached and (now_s - cached[0]) < 900.0:
        return dict(cached[1])

    out = {"oi_delta_4h": 0.0, "ls_skew": 0.0, "buy_ratio": 0.5}
    for base in ("https://api.bybit.com", "https://api.bytick.com"):
        try:
            r_oi = _SESSION.get(
                f"{base}/v5/market/open-interest",
                params={"category": "linear", "symbol": clean, "intervalTime": "1h", "limit": 6},
                timeout=3.5,
            )
            oi_list = r_oi.json().get("result", {}).get("list", []) or []
            if len(oi_list) >= 5:
                oi_now = float(oi_list[0].get("openInterest", 0.0) or 0.0)
                oi_4h = float(oi_list[4].get("openInterest", 0.0) or 0.0)
                if oi_now > 0.0 and oi_4h > 0.0:
                    out["oi_delta_4h"] = round(((oi_now - oi_4h) / oi_4h) * 100.0, 3)

            r_ls = _SESSION.get(
                f"{base}/v5/market/account-ratio",
                params={"category": "linear", "symbol": clean, "period": "1h", "limit": 6},
                timeout=3.5,
            )
            ls_list = r_ls.json().get("result", {}).get("list", []) or []
            if ls_list:
                buy_r = float(ls_list[0].get("buyRatio", 0.5) or 0.5)
                out["buy_ratio"] = round(buy_r, 4)
                out["ls_skew"] = round((buy_r - 0.50) * 200.0, 2)  # -100 (crowded short) to +100 (crowded long)

            _POSITIONING_CACHE[clean] = (now_s, out)
            return dict(out)
        except Exception:
            continue
    return out


def update_conformal_calibration(live_dir: str, regime: str, predicted_prob: float, realized_win: bool):
    """Tracks nonconformity residuals |y - p_hat| per regime for adaptive abstention & Fractional-Kelly sizing."""
    try:
        mem = _load_self_learning_memory(live_dir)
        conf_map = mem.setdefault("conformal_calibration", {})
        reg_key = str(regime or "DEFAULT").upper()
        rec = conf_map.setdefault(reg_key, {"residuals": [], "wins": 0, "total": 0})
        y_val = 1.0 if realized_win else 0.0
        p_val = min(max(float(predicted_prob), 0.05), 0.95)
        res_list = rec.setdefault("residuals", [])
        res_list.append(round(abs(y_val - p_val), 4))
        rec["residuals"] = res_list[-40:]
        rec["wins"] = int(rec.get("wins", 0)) + (1 if realized_win else 0)
        rec["total"] = int(rec.get("total", 0)) + 1
        _save_self_learning_memory(live_dir, mem)
    except Exception:
        pass


def get_conformal_calibration_adjustments(live_dir: str, regime: str) -> tuple:
    """Returns (prob_penalty, kelly_risk_multiplier) from per-regime conformal uncertainty intervals."""
    try:
        mem = _load_self_learning_memory(live_dir)
        conf_map = mem.get("conformal_calibration", {}) or {}
        rec = conf_map.get(str(regime or "DEFAULT").upper()) or conf_map.get("DEFAULT")
        if not isinstance(rec, dict):
            return 0.0, 1.0
        res_list = rec.get("residuals", []) or []
        total = int(rec.get("total", 0))
        wins = int(rec.get("wins", 0))
        if len(res_list) < 6 or total < 6:
            return 0.0, 1.0

        sorted_r = sorted(float(x) for x in res_list)
        q80 = sorted_r[min(len(sorted_r) - 1, int(len(sorted_r) * 0.80))]
        emp_wr = wins / float(max(total, 1))

        # High conformal uncertainty (q80 > 0.68) raises abstention threshold & trims Kelly fraction
        prob_penalty = round(max(0.0, (q80 - 0.65) * 0.14), 3)
        kelly_raw = 0.70 + (emp_wr - 0.45) * 1.35 - max(0.0, q80 - 0.55) * 0.65
        kelly_mult = round(min(max(kelly_raw, 0.55), 1.25), 3)
        return prob_penalty, kelly_mult
    except Exception:
        return 0.0, 1.0


def synthesize_failure_playbook(
    live_dir: str,
    symbol: str,
    side: str,
    net_pnl: float,
    session_name: str,
    regime: str,
    oi_delta: float = 0.0,
    ls_skew: float = 0.0,
):
    """Distills losing trades into structured failure-mode rules injected into Layer 1 & Gemini research."""
    if net_pnl >= 0.0:
        return
    try:
        mem = _load_self_learning_memory(live_dir)
        pb = mem.setdefault("failure_playbook", [])
        clean = symbol.replace("/", "").upper()
        tag = f"{clean}_{side.upper()}_{session_name.upper()}_{regime.upper()}"
        existing = next((x for x in pb if x.get("tag") == tag), None)
        if existing:
            existing["losses"] = int(existing.get("losses", 1)) + 1
            existing["last_ms"] = int(time.time() * 1000)
        else:
            crowd_note = ""
            if side.upper() == "LONG" and ls_skew > 28.0:
                crowd_note = " into crowded longs"
            elif side.upper() == "SHORT" and ls_skew < -28.0:
                crowd_note = " into crowded shorts"
            pb.append({
                "tag": tag,
                "symbol": clean,
                "side": side.upper(),
                "session": session_name.upper(),
                "regime": regime.upper(),
                "losses": 1,
                "rule": f"Caution on {clean} {side.upper()} during {session_name.upper()} ({regime.upper()}){crowd_note}",
                "last_ms": int(time.time() * 1000),
            })
        mem["failure_playbook"] = sorted(pb, key=lambda x: int(x.get("losses", 1)), reverse=True)[:25]
        _save_self_learning_memory(live_dir, mem)
    except Exception:
        pass


def get_active_failure_playbook(live_dir: str) -> list:
    try:
        mem = _load_self_learning_memory(live_dir)
        return list(mem.get("failure_playbook", []) or [])
    except Exception:
        return []


def build_lightweight_portfolio_summary(master_path: str, summary_path: str):
    """Writes a <15 KB tree-free portfolio summary so Android Reports open in <50 ms."""
    if not os.path.exists(master_path):
        return
    try:
        with open(master_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        port = data.get("portfolio", {}) or {}
        lite_port = {}
        for sym, entry in port.items():
            if not isinstance(entry, dict):
                continue
            lite_entry = {k: v for k, v in entry.items() if k not in ("models", "tested_matrix")}
            if isinstance(lite_entry.get("stats"), dict):
                lite_entry["stats"] = {k: v for k, v in lite_entry["stats"].items() if k != "raw_trade_returns"}
            lite_port[sym] = lite_entry
        lite_doc = {
            "version": data.get("version", ""),
            "exported_at": data.get("exported_at", 0),
            "training_days": data.get("training_days", 135),
            "model_accuracy_pct": data.get("model_accuracy_pct", 0.0),
            "total_net_pnl": data.get("total_net_pnl", 0.0),
            "portfolio": lite_port,
        }
        tmp_s = summary_path + f".tmp.{os.getpid()}_{_at_threading.get_ident()}_{_at_uuid.uuid4().hex[:8]}"
        with open(tmp_s, "w", encoding="utf-8") as wf:
            json.dump(lite_doc, wf, indent=2)
        _locked_replace(tmp_s, summary_path)
    except Exception:
        pass


def get_fast_reports_json(paper_dir: str, live_dir: str, models_dir: str) -> str:
    """15s-cached, tree-free Reports payload eliminating multi-megabyte JSON parsing lag."""
    now_s = time.time()
    if _REPORTS_FAST_CACHE["payload"] and (now_s - _REPORTS_FAST_CACHE["ts"]) < 15.0:
        return _REPORTS_FAST_CACHE["payload"]

    paper_sum = os.path.join(paper_dir, "paper_reports_summary.json")
    live_sum = os.path.join(live_dir, "live_reports_summary.json")
    if not os.path.exists(paper_sum):
        build_lightweight_portfolio_summary(os.path.join(models_dir, "PAPER", "UNIFIED_PORTFOLIO_MASTER.json"), paper_sum)
    if not os.path.exists(live_sum):
        build_lightweight_portfolio_summary(os.path.join(models_dir, "LIVE", "UNIFIED_PORTFOLIO_MASTER.json"), live_sum)

    def _read_safe(fp, default_val):
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8-sig") as rf:
                    return json.load(rf)
            except Exception:
                pass
        return default_val

    payload = json.dumps({
        "ok": True,
        "paper_summary": _read_safe(paper_sum, {"portfolio": {}}),
        "live_summary": _read_safe(live_sum, {"portfolio": {}}),
        "live_history": _read_safe(os.path.join(live_dir, "live_reports_history.json"), {"trades": []}),
        "comparison": _read_safe(os.path.join(paper_dir, "paper_vs_live_comparison.json"), {"comparisons": []}),
        "agentic_audit": _read_safe(os.path.join(paper_dir, "agentic_model_analysis.json"), {}),
    })
    _REPORTS_FAST_CACHE["ts"] = now_s
    _REPORTS_FAST_CACHE["payload"] = payload
    return payload
