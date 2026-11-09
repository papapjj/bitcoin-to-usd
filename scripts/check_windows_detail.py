import ctypes
from ctypes import wintypes
import subprocess

user32 = ctypes.windll.user32

output = subprocess.check_output("tasklist /fi \"imagename eq terminal64.exe\" /fo csv /nh", shell=True).decode()
pids = [int(line.replace('"', '').split(",")[1].strip()) for line in output.strip().split("\n") if "terminal64" in line and len(line.split(",")) > 1]
print("PIDs:", pids)

hwnds = []
def enum_cb(hwnd, l):
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value in pids:
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        cls_buff = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls_buff, 256)
        is_vis = user32.IsWindowVisible(hwnd)
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        hwnds.append((hwnd, buff.value, cls_buff.value, is_vis, rect.left, rect.top, rect.right, rect.bottom))
    return True

EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows(EnumWindowsProc(enum_cb), 0)

for h, title, cls_name, vis, l, t, r, b in hwnds:
    print(f"HWND {h} | Class: {cls_name} | Vis: {vis} | Rect: ({l},{t},{r},{b}) | Title: {title}")
