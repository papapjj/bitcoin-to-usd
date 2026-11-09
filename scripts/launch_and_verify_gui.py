import ctypes
from ctypes import wintypes
import os
import subprocess
import time
from pathlib import Path

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

ROOT = Path(r"C:\Users\Admin\Desktop\new-chat\mt5_v2_runner")
terminal_exe = ROOT / "terminal64.exe"
ini_path = ROOT / "tester.ini"

# Kill existing terminal64
subprocess.run("taskkill /f /im terminal64.exe /im metatester64.exe", shell=True, capture_output=True)
time.sleep(1)

# Ensure tester.ini has Visual=1 and ShutdownTerminal=0
ini_content = """[Tester]
Expert=SelfGrowing_MT5_Trainer_EA
ExpertParameters=SelfGrowing_MT5_Trainer_EA.set
Symbol=XAUUSDm
Period=M1
Model=1
ExecutionMode=0
Optimization=0
FromDate=2024.11.18
ToDate=2024.11.25
Visual=1
ShutdownTerminal=0
ReplaceReport=1
"""
ini_path.write_text(ini_content, encoding="utf-8")

cmd = f'"{terminal_exe}" /portable /config:"{ini_path}"'
print(f"Launching command: {cmd}")

# Launch process
proc = subprocess.Popen(cmd, cwd=str(ROOT), shell=True)
print(f"Launched process PID: {proc.pid}")

# Wait for MT5 window to initialize
time.sleep(5)

# Enumerate all windows and force display
found_windows = []

def enum_cb(hwnd, l):
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    # Check if PID is related or window title contains metatrader / tester
    length = user32.GetWindowTextLengthW(hwnd)
    buff = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buff, length + 1)
    title = buff.value
    
    cls_buff = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, cls_buff, 256)
    cls_name = cls_buff.value

    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top

    if any(k in title.lower() for k in ["metatrader", "tester", "xauusd", "exness"]) or any(k in cls_name.lower() for k in ["metatrader", "tester"]):
        found_windows.append({
            "hwnd": hwnd,
            "pid": pid.value,
            "title": title,
            "class": cls_name,
            "visible": bool(user32.IsWindowVisible(hwnd)),
            "rect": (rect.left, rect.top, rect.right, rect.bottom),
            "width": w,
            "height": h
        })
        # Force Show and Maximize to screen
        user32.ShowWindow(hwnd, 9) # SW_RESTORE
        user32.ShowWindow(hwnd, 3) # SW_MAXIMIZE
        user32.SetForegroundWindow(hwnd)
    return True

EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows(EnumWindowsProc(enum_cb), 0)

print(f"\nFound {len(found_windows)} MetaTrader / Tester window(s):")
for w in found_windows:
    print(f"  -> HWND {w['hwnd']} | PID {w['pid']} | Visible: {w['visible']} | Size: {w['width']}x{w['height']} | Title: '{w['title']}' | Class: '{w['class']}'")
