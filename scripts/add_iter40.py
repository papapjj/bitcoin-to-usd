from pathlib import Path

p = Path("scripts/run_mt5_observation_optimization_loop.py")
text = p.read_text(encoding="utf-8")

iter40_code = '''    elif args.iteration == 40:
        logger.info(
            f"[Iter 40] Running 2-Day Anti-Cascading + Precision Breakeven Lock 1.8 ATR Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.0R (5.0 ATR) / 1.2R (6.0 ATR) | Trailing 1.8 ATR | Breakeven Lock (+0.65 ATR at >= 1.80 ATR) | "
            f"Anti-Falling-Knife Spacing (8 bars) | Long Only | Max 5 Positions..."
        )
        safe_cppi_40 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_40 = TwoTierConfig(
            breakeven_trigger_atr=1.80,
            breakeven_lock_atr=0.65,
            partial_tp_atr=5.0,
            trailing_atr=1.80,
        )
        base_router_40 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_40,
            two_tier_config=two_tier_40,
        )
        policy_40 = MythosMT5Policy(base_policy=base_router_40)
        policy_40.long_only = True
        policy_40.short_only = False
        policy_40.prm_threshold = 0.50
        policy_40.min_confidence = 0.0
        policy_40.max_atr_percentile = 1.0
        policy_40.require_margin_alignment = False
        policy_40.loss_cooldown_bars = 4
        policy_40.max_consecutive_losses = 2
        policy_40.min_spacing_bars = 8
        policy_40.max_positions = 5
        policy_40.scalp_sl_atr = 5.0
        policy_40.sniper_sl_atr = 5.0
        policy_40.scalp_tp_rr = 1.0
        policy_40.sniper_tp_rr = 1.2
        policy_40.trailing_atr = 1.80

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_40.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 40] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_40.pipeline, "world_model"):
                policy_40.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 40] Successfully loaded Latent World Model from %s", champ_pt)

        res_40 = run_single_iteration(
            iter_name="Iter_40_2Days_AntiCascading_PrecisionBreakeven_Champion",
            policy=policy_40,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_40_2Days_AntiCascading_PrecisionBreakeven_Champion"]
        results.append(res_40)
'''

target_marker = "    # Save summary json"
if target_marker in text:
    text = text.replace(target_marker, iter40_code + "\n" + target_marker)
    p.write_text(text, encoding="utf-8")
    print("Successfully added Iteration 40 to run_mt5_observation_optimization_loop.py")
else:
    print("Could not find target marker!")
