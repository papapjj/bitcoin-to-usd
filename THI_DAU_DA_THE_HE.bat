@echo off
title GIAI DAU DA THE HE - MULTI-VERSION LEAGUE ARENA
color 0A
cd /d "C:\Users\Admin\Desktop\new-chat"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo ==============================================================================
echo        GIAI DAU METATRADER 5 DA THE HE (MULTI-VERSION LEAGUE ARENA)
echo ==============================================================================
echo  [1] Chay giai dau 3 the he: Gen-1 (Defensive) vs Gen-2 (Bear) vs Challenger
echo  [2] Chay giai dau day du 5 the he (Gen-1, Gen-2, Gen-3, Router, Challenger)
echo  [3] Chay giai dau che do Truc Quan (Visual Mode MT5 Window)
echo ==============================================================================
set /p opt="Nhap lua chon cua ban [1, 2, 3] (Enter mac dinh [1]): "

if "%opt%"=="2" (
    echo Dang khoi chay Giai dau day du 5 the he...
    "C:\Python314\python.exe" scripts/run_multi_version_tournament.py --versions gen1 gen2 gen3 dual_router challenger
) else if "%opt%"=="3" (
    echo Dang khoi chay Giai dau MT5 che do Truc Quan (Visual Mode)...
    "C:\Python314\python.exe" scripts/run_multi_version_tournament.py --visual --versions gen1 gen2 challenger
) else (
    echo Dang khoi chay Giai dau 3 the he tieu bieu...
    "C:\Python314\python.exe" scripts/run_multi_version_tournament.py --versions gen1 gen2 challenger
)

echo.
echo Giai dau da ket thuc! Bang tong sap da duoc xuat ra outputs/tournaments/
pause
