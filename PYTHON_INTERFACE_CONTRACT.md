# Kotlin <-> Python Interface Contract (Do Not Rename These Entry Points)

The compiled Android APK (`BotRuntimeBridge.kt` & `PythonBotExecutor.kt`) invokes these exact Python module and function names. Keep their function names and parameter order intact when editing in VS Code:

## 1. `dynamic_bot_runner.py`
- `execute_live_cycle(live_dir: str, models_dir: str, config_json: str) -> str`
- `execute_live_kill_switch(live_dir: str, models_dir: str, config_json: str) -> str`
- `close_single_live_position(live_dir: str, models_dir: str, config_json: str, symbol: str) -> str`
- `execute_paper_training(paper_dir: str, models_dir: str, config_json: str) -> str`
- `execute_single_paper_coin_training(paper_dir: str, models_dir: str, config_json: str) -> str`
- `execute_paper_simulation_tick(paper_dir: str, models_dir: str, config_json: str) -> str`
- `promote_coin_with_paper_bot(paper_dir: str, models_dir: str, symbol: str) -> str`
- `remove_coin_with_paper_bot(paper_dir: str, models_dir: str, symbol: str) -> str`
- `test_bybit_wallet(api_key: str, api_secret: str, endpoint: str) -> str`
- `fetch_live_snapshot(live_dir: str, models_dir: str, api_key: str, api_secret: str, endpoint: str) -> str`
- `sync_github_file(target_dir: str, repo_url: str, branch: str, secret_key: str, target_filename: str) -> str`
- `analyze_market_symbols(symbols_json: str) -> str`

## 2. `ai_regime_copilot.py`
- `run_agentic_ai_supervisor(...) -> str`
- `discover_and_sync_gemini_strategies(api_key: str, paper_dir: str, live_dir: str) -> str`
- `discover_and_rank_online_strategies_json(existing_catalog_json: str) -> str`
- `evaluate_active_strategies_health_json(strategies_json: str) -> str`
- `audit_bot_codes_for_upgrades(paper_dir: str, live_dir: str, models_dir: str, reset_epoch_ms: int = 0) -> str`
- `cleanup_storage_and_artifacts(paper_dir: str, live_dir: str, models_dir: str) -> str`
- `export_full_state_bundle_json(paper_dir: str, live_dir: str, models_dir: str) -> str`
- `import_full_state_bundle_json(base_files_dir: str, bundle_json: str) -> str`
- `get_fast_reports_json(paper_dir: str, live_dir: str, models_dir: str) -> str`
- `evaluate_pre_trade_human_brain(...) -> dict` (called by `live_bot.py`)
- `prune_underperforming_gemini_strategies(...) -> dict` (called by `paper_bot.py`)

## 3. `radar_engine.py`
- `get_fast_radar_signals(...) -> list` (called by `dynamic_bot_runner.py`)
- `reset_network_pools()`

## 4. `live_bot.py` & `paper_bot.py`
- Loaded dynamically by `dynamic_bot_runner.py` when executing Live and Paper cycles.
