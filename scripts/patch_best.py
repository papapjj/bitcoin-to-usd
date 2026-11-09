with open('scripts/train_high_precision_champion.py', 'r') as f:
    code = f.read()

old_block = """            if val_pf > best_val_pf and len(val_trades) >= 50:
                best_val_pf = val_pf
                best_val_wr = val_wr
                best_checkpoint = engine.get_state_dict()
                logger.info("  *** New Best OOS Checkpoint Promoted (PF: %.2f, WR: %.2f%%) ***", val_pf, val_wr)

    # Save best promoted checkpoint
    save_path = Path("outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt")
    engine.save_checkpoint(save_path, metadata={'best_val_pf': best_val_pf, 'best_val_wr': best_val_wr})
    logger.info('Saved High-Precision Champion to %s', save_path)"""

new_block = """            save_path = Path("outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt")
            if val_pf > best_val_pf and len(val_trades) >= 30:
                best_val_pf = val_pf
                best_val_wr = val_wr
                engine.save_checkpoint(save_path, metadata={'best_val_pf': best_val_pf, 'best_val_wr': best_val_wr, 'epoch': epoch})
                logger.info("  *** New Best OOS Checkpoint Saved (PF: %.2f, WR: %.2f%%) ***", val_pf, val_wr)

    logger.info("Training complete. Best OOS Validation PF: %.2f, WR: %.2f%%", best_val_pf, best_val_wr)"""

code = code.replace(old_block, new_block)
with open('scripts/train_high_precision_champion.py', 'w') as f:
    f.write(code)

print("Updated train_high_precision_champion.py successfully")
