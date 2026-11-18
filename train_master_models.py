#!/usr/bin/env python3
"""
Standalone Master ML Trainer for AutonomousAITrading (PC / GitHub Release Builder)
Trains the exact same 135-day, 18-feature XGBoost multi-timeframe models as paper_bot.py
with live per-coin progress logging and incremental saving.
"""

import os
import json
import time
import argparse
from paper_bot import PaperStrategyTrainerAndTester, FEATURES, TRAINING_WINDOW_DAYS

DEFAULT_MASTER_COINS = [
    "SOL/USDT",
    "XRP/USDT",
    "NEAR/USDT",
    "INJ/USDT",
    "OP/USDT",
    "APT/USDT",
    "LINK/USDT",
    "DOGE/USDT",
    "SUI/USDT",
    "AVAX/USDT",
    "ETH/USDT",
    "BTC/USDT",
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
]

DEFAULT_STRATEGIES = [
    "xgb_mtf_v2",
    "xrp_donchian_breakout",
    "btc_vol_surge_adx",
    "eth_rsi_stoch_reversal",
    "sol_ema_ribbon_atr",
    "near_vwap_profile",
]


def save_master_bundle(out_dir: str, trained_map: dict):
    win_rates = [v["win_rate_pct"] for v in trained_map.values()]
    total_pnl = sum(v["net_pnl_usdt"] for v in trained_map.values())
    master_payload = {
        "version": f"UNIFIED_MASTER_{len(trained_map)}C_{TRAINING_WINDOW_DAYS}D",
        "exported_at": int(time.time()),
        "training_days": TRAINING_WINDOW_DAYS,
        "model_accuracy_pct": round(sum(win_rates) / len(win_rates), 2) if win_rates else 0.0,
        "total_net_pnl": round(total_pnl, 2),
        "features": FEATURES,
        "portfolio": trained_map,
    }
    root_master_path = os.path.join(out_dir, "UNIFIED_PORTFOLIO_MASTER.json")
    with open(root_master_path, "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2)
    return root_master_path


def main():
    parser = argparse.ArgumentParser(description="Train Master XGBoost Portfolio JSON for GitHub Sync")
    parser.add_argument("--coins", nargs="+", default=DEFAULT_MASTER_COINS, help="List of symbols to train")
    parser.add_argument("--leverage", type=int, default=10, help="Target leverage (default: 10)")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory for UNIFIED_PORTFOLIO_MASTER.json")
    parser.add_argument("--forex", action="store_true", default=True, help="Include Forex pairs")
    args = parser.parse_args()

    paper_dir = os.path.join(args.out_dir, "bot_data", "PAPER")
    models_dir = os.path.join(args.out_dir, "ml_models")
    os.makedirs(paper_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    total = len(args.coins)
    print(f"=== Training Master ML Portfolio ({total} symbols • {TRAINING_WINDOW_DAYS} days • 18 features) ===", flush=True)
    t0 = time.time()

    trainer = PaperStrategyTrainerAndTester(paper_dir=paper_dir, models_dir=models_dir)
    all_trained = {}
    root_master_path = os.path.join(args.out_dir, "UNIFIED_PORTFOLIO_MASTER.json")

    for idx, sym in enumerate(args.coins, start=1):
        c_start = time.time()
        print(f"[{idx}/{total}] Fetching 135d candles & training {sym}...", end=" ", flush=True)
        res = trainer.train_and_evaluate_paper_coins(
            added_coins=[sym],
            active_strategies=DEFAULT_STRATEGIES,
            timeframes=["15m", "30m", "1h", "4h"],
            leverage=args.leverage,
            crypto_enabled=True,
            forex_enabled=args.forex,
        )
        c_sec = round(time.time() - c_start, 1)
        if sym in res:
            all_trained[sym] = res[sym]
            info = res[sym]
            root_master_path = save_master_bundle(args.out_dir, all_trained)
            print(
                f"DONE ({c_sec}s) | {info['strategy']} | WR: {info['win_rate_pct']:.1f}% | "
                f"PF: {info['profit_factor']:.2f} | Trades: {info['trades']} | PnL: ${info['net_pnl_usdt']:.2f}",
                flush=True,
            )
        else:
            print(f"SKIPPED ({c_sec}s - insufficient candle data or network timeout)", flush=True)

    # Also write complete portfolio into PAPER & LIVE master paths
    if all_trained:
        master_json = json.load(open(root_master_path, "r", encoding="utf-8"))
        for sub in ("PAPER", "LIVE"):
            sub_dir = os.path.join(models_dir, sub)
            os.makedirs(sub_dir, exist_ok=True)
            with open(os.path.join(sub_dir, "UNIFIED_PORTFOLIO_MASTER.json"), "w", encoding="utf-8") as sf:
                json.dump(master_json, sf, indent=2)

    elapsed = round(time.time() - t0, 1)
    print(f"\nCompleted in {elapsed}s! Trained {len(all_trained)}/{total} symbols.", flush=True)
    print(f"Saved GitHub-ready master model to: {os.path.abspath(root_master_path)}", flush=True)


if __name__ == "__main__":
    main()