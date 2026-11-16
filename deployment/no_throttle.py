"""2026-11-16 關掉 Windows 對這個行程的 power throttling(EcoQoS)。

背景: 在背景(最小化 / 非前景)視窗跑超過約 4–8 分鐘後,Windows 會把行程改成節能排程,筆電 i5-13500H 上第一階段
(12 執行緒,含 E 核心)從約 210 ms 變成約 330 ms(1.5 倍),U-Net(4 執行緒)不受影響;明確關掉節流後在背景 32 分鐘
都維持約 190–230 ms(EXPERIMENTS.md「2026-11-16 筆電延遲變異的真正原因」)。
用法: import no_throttle; no_throttle.disable()  — 在載入模型前呼叫;不是 Windows 時什麼都不做。回傳是否成功。
只影響執行速度,不影響結果。
"""
import sys


def disable():
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes as w

    class State(ctypes.Structure):
        _fields_ = [("Version", w.ULONG), ("ControlMask", w.ULONG), ("StateMask", w.ULONG)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = w.HANDLE
    k.SetProcessInformation.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    k.SetProcessInformation.restype = w.BOOL
    # ProcessPowerThrottling = 4;ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED(1),StateMask = 0 → 明確關閉
    s = State(1, 1, 0)
    return bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s)))
