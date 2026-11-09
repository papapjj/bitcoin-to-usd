@echo off
title DAU TRUONG METATRADER 5 - CHAMPION VS CHALLENGER SELF-PLAY
color 0B
cd /d "C:\Users\Admin\Desktop\new-chat"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo ==============================================================================
echo        DAU TRUONG METATRADER 5 - CHAMPION VS CHALLENGER (SELF-PLAY)
echo ==============================================================================
echo  [1] Chay tran dau MT5 ngam (Fast Tournament - Champion vs Challenger)
echo  [2] Mo phong do thi MT5 truc quan (Visual Chart Tester - Giu cua so MT5)
echo  [3] Tao ung vien Challenger moi (Mo khoa Reasoning 10%%)
echo  [4] Kiem tra nhanh bang chung & 5 Cong (Fast Dry-Run Mode)
echo  [5] Dau truong Champion vs Challenger co cua so Visual (Full Match)
echo ==============================================================================
set "opt="
set /p opt="Nhap lua chon cua ban [1, 2, 3, 4, 5] (Enter mac dinh [2]): "
if "%opt%"=="" set opt=2

if "%opt%"=="1" (
    echo Dang chay tran dau MT5 Strategy Tester giua Champion va Challenger...
    "C:\Python314\python.exe" scripts/run_mt5_champion_vs_challenger.py --promote-on-win
) else if "%opt%"=="2" (
    echo Dang khoi chay MT5 Strategy Tester o che do TRUC QUAN (Visual Chart Mode)...
    "C:\Python314\python.exe" scripts/launch_visual_simulation.py --phase 3 --from-date 2025.01.14 --to-date 2025.01.16
) else if "%opt%"=="3" (
    echo Dang tao ung vien Challenger moi voi co che suy luan closed-loop...
    "C:\Python314\python.exe" scripts/train_challenger_candidate.py
) else if "%opt%"=="4" (
    echo Dang chay kiem tra nhanh trong tai MT5 & 5 Cong Thang Hang...
    "C:\Python314\python.exe" scripts/run_mt5_champion_vs_challenger.py --dry-run --promote-on-win
) else if "%opt%"=="5" (
    echo Dang khoi chay Dau truong Champion vs Challenger o che do Visual...
    "C:\Python314\python.exe" scripts/run_mt5_champion_vs_challenger.py --visual --promote-on-win
) else (
    echo Lua chon khong hop le, mac dinh khoi chay che do TRUC QUAN [2]...
    "C:\Python314\python.exe" scripts/launch_visual_simulation.py --phase 3 --from-date 2025.01.14 --to-date 2025.01.16
)

echo.
echo Tien trinh MT5 da hoan tat!
pause
