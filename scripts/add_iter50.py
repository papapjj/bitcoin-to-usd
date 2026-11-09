with open('scripts/run_mt5_observation_optimization_loop.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '        results.append(res_49)'
addition = '''        results.append(res_49)

    elif args.iteration == 50:
        logger.info(
            f"[Iter 50] Running 2-Day Terminal-Flat Golden Master Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.5R (7.5 ATR) / 2.0R (10.0 ATR) | Trailing 2.5 ATR | Breakeven Lock (+1.0 ATR at >= 2.80 ATR) | Partial TP (4.5 ATR) | "
            f"Cognitive Margin (0.12 <= margin <= 0.88) & Min Conf (>=0.14) & PRM (>=0.35) & Min ATR (>=42.0 USD) | Zero Artificial Gating Locks | Long Only | Max 5 Positions | Terminal Flat Safety..."
        )
        safe_cppi_50 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_50 = TwoTierConfig(
            initial_sl_atr=5.0,
            breakeven_trigger_atr=2.80,
            breakeven_lock_atr=1.00,
            partial_tp_atr=4.50,
            trailing_atr=2.50,
        )
        base_router_50 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=False,
            momentum_gate_enabled=False,
            circuit_breaker_enabled=False,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_50,
            two_tier_config=two_tier_50,
        )
        policy_50 = MythosMT5Policy(base_policy=base_router_50)
        policy_50.long_only = True
        policy_50.short_only = False
        policy_50.min_margin_threshold = 0.12
        policy_50.max_margin_threshold = 0.88
        policy_50.min_confidence = 0.14
        policy_50.prm_threshold = 0.35
        policy_50.min_atr_threshold = 42.0
        policy_50.max_atr_percentile = 1.0
        policy_50.require_margin_alignment = False
        policy_50.loss_cooldown_bars = 0
        policy_50.max_consecutive_losses = 999
        policy_50.min_spacing_bars = 4
        policy_50.max_positions = 5
        policy_50.max_packets = 1660  # Terminal flat safety before simulation ends
        policy_50.scalp_sl_atr = 5.0
        policy_50.sniper_sl_atr = 5.0
        policy_50.scalp_tp_rr = 1.50
        policy_50.sniper_tp_rr = 2.00
        policy_50.trailing_atr = 2.50

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_50.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 50] Successfully loaded Retrained Anti-Exhaustion Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_50.pipeline, "world_model"):
                policy_50.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 50] Successfully loaded Latent World Model from %s", champ_pt)

        res_50 = run_single_iteration(
            iter_name="Iter_50_2Days_TerminalFlat_GoldenMaster",
            policy=policy_50,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_50_2Days_TerminalFlat_GoldenMaster"]
        results.append(res_50)'''

if target in text:
    new_text = text.replace(target, addition, 1)
    with open('scripts/run_mt5_observation_optimization_loop.py', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print("SUCCESS: Added Iteration 50 to loop script.")
else:
    print("ERROR: Target not found.")
