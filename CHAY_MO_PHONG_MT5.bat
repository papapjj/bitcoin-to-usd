@echo off
title MT5 Visual Strategy Tester - Frontier Bilateral Dual-CoT Champion
color 0A
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\Admin\Desktop\new-chat"
echo =======================================================================
echo   KHOI DONG MT5 STRATEGY TESTER VISUAL MODE (7 NGAY OOS)
echo   Policy : FRONTIER BILATERAL DUAL-CoT CHAMPION (OPTION 2)
echo   Model  : mythos_bilateral_dual_cot_champion.pt (Bull & Bear Independent)
echo   Target : WIN RATE >= 80 PERCENT - PROFIT FACTOR > 3.0
echo   Feature: TWO-TIER RUNNER + BREAKEVEN RATCHET (+0.30 ATR) + COUNTER-PUNCH
echo =======================================================================
"C:\Python314\python.exe" scripts/launch_7day_visual_master_champion.py --from-date 2025.02.20 --to-date 2025.02.27
echo =======================================================================
echo   MO PHONG KET THUC.
echo =======================================================================
pause
