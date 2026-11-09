with open('scripts/train_high_precision_champion.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "if best_checkpoint is not None:" in line:
        skip = True
        new_lines.append("    engine.save_checkpoint(save_path, metadata={'best_val_pf': best_val_pf, 'best_val_wr': best_val_wr})\n")
        new_lines.append("    logger.info('Saved High-Precision Champion to %s', save_path)\n")
        continue
    if skip and "logger.info(\"Saved Final High-Precision" in line:
        skip = False
        continue
    if not skip:
        new_lines.append(line)

with open('scripts/train_high_precision_champion.py', 'w') as f:
    f.writelines(new_lines)

print("Patch applied successfully")
