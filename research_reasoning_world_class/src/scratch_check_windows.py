import ctypes
from ctypes import wintypes
import subprocess

user32 = ctypes.windll.user32
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

# Get PIDs of terminal64 and metatester64
out = subprocess.check_output(["powershell", "-Command", "Get-Process terminal64, metatester64 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"]).decode()
target_pids = set(int(line.strip()) for line in out.splitlines() if line.strip().isdigit())
print("Target PIDs:", target_pids)

windows_found = []

def enum_cb(hwnd, lparam):
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value in target_pids:
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        visible = user32.IsWindowVisible(hwnd)
        windows_found.append((hwnd, pid.value, bool(visible), buff.value))
        print(f"HWND: {hwnd} | PID: {pid.value} | Visible: {bool(visible)} | Title: '{buff.value}'")
    return True

user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
if not windows_found:
    print("NO WINDOWS FOUND for target PIDs!")
