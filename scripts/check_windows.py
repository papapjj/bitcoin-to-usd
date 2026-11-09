import ctypes
from ctypes import wintypes
import subprocess

user32 = ctypes.windll.user32

output = subprocess.check_output("tasklist /fi \"imagename eq terminal64.exe\" /fo csv /nh", shell=True).decode()
pids = []
for line in output.strip().split("\n"):
    if "terminal64" in line:
        parts = line.replace('"', '').split(",")
        if len(parts) > 1 and parts[1].strip().isdigit():
            pids.append(int(parts[1].strip()))

print(f"Terminal PIDs: {pids}")

hwnds = []
def enum_cb(hwnd, l):
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value in pids:
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        is_vis = user32.IsWindowVisible(hwnd)
        hwnds.append((hwnd, buff.value, is_vis))
    return True

EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows(EnumWindowsProc(enum_cb), 0)

for h, title, vis in hwnds:
    if title:
        print(f"HWND {h} | Visible: {vis} | Title: {title}")
