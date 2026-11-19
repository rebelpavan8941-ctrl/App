#!/usr/bin/env python3
"""
Standalone Master ML Trainer for AutonomousAITrading (PC / GitHub Release Builder)
Trains the exact same 135-day, 18-feature XGBoost multi-timeframe models as paper_bot.py
for all specified coins in a single batch call and outputs a clean, compact
UNIFIED_PORTFOLIO_MASTER.json ready for GitHub upload & instant new-user sync.
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
    
    # Train all coins in one batch call so all 15 symbols are correctly evaluated and saved together
    trained = trainer.train_and_evaluate_paper_coins(
        added_coins=args.coins,
        active_strategies=DEFAULT_STRATEGIES,
        timeframes=["15m", "30m", "1h", "4h"],
        leverage=args.leverage,
        crypto_enabled=True,
        forex_enabled=args.forex,
    )

    if not trained:
        print("Error: No coins were successfully trained.", flush=True)
        return

    win_rates = [v["win_rate_pct"] for v in trained.values()]
    total_pnl = sum(v["net_pnl_usdt"] for v in trained.values())
    master_payload = {
        "version": f"UNIFIED_MASTER_{len(trained)}C_{TRAINING_WINDOW_DAYS}D",
        "exported_at": int(time.time()),
        "training_days": TRAINING_WINDOW_DAYS,
        "model_accuracy_pct": round(sum(win_rates) / len(win_rates), 2) if win_rates else 0.0,
        "total_net_pnl": round(total_pnl, 2),
        "features": FEATURES,
        "portfolio": trained,
    }

    # Save compact JSON (separators without extra whitespace) so file size is ~2.5 MB instead of 14 MB
    root_master_path = os.path.join(args.out_dir, "UNIFIED_PORTFOLIO_MASTER.json")
    with open(root_master_path, "w", encoding="utf-8") as f:
        json.dump(master_payload, f, separators=(",", ":"))

    # Copy into PAPER & LIVE subdirectories as well
    for sub in ("PAPER", "LIVE"):
        sub_dir = os.path.join(models_dir, sub)
        os.makedirs(sub_dir, exist_ok=True)
        with open(os.path.join(sub_dir, "UNIFIED_PORTFOLIO_MASTER.json"), "w", encoding="utf-8") as sf:
            json.dump(master_payload, sf, separators=(",", ":"))

    elapsed = round(time.time() - t0, 1)
    print(f"\nCompleted in {elapsed}s! Successfully trained {len(trained)}/{total} symbols:", flush=True)
    for sym, info in sorted(trained.items(), key=lambda x: x[1]["net_pnl_usdt"], reverse=True):
        print(
            f"  - {sym:<10} | {info['strategy']:<38} | WR: {info['win_rate_pct']:>5.1f}% | "
            f"PF: {info['profit_factor']:>4.2f} | Trades: {info['trades']:>3} | PnL: ${info['net_pnl_usdt']:>8.2f}",
            flush=True,
        )
    print(f"\nSaved GitHub-ready master model to: {os.path.abspath(root_master_path)}", flush=True)


if __name__ == "__main__":
    main()