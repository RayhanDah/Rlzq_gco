"""Mémoire du système et du processus, stdlib seule (ADR-009 D5 : RAM obligatoire pour un run avec SIL, KS1).

Windows : `GlobalMemoryStatusEx` (RAM physique disponible) et `K32GetProcessMemoryInfo` (`PrivateUsage`), même
méthode que `scripts/bench_run.py` (`_win_memory`, mesures ADR-001 M6–M8). Linux (Kaggle) : `/proc/meminfo`
(`MemAvailable`) et `/proc/self/status` (`RssAnon`, approximation de la mémoire privée). Ailleurs, ou en cas d'échec :
None (le journal l'écrit comme absent, la garde KS1 ne compte rien).
"""

from __future__ import annotations

import os
from pathlib import Path

MB = 1024 * 1024


def _win_memory_mb() -> tuple[float | None, float | None]:
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class MSX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    msx = MSX()
    msx.dwLength = ctypes.sizeof(MSX)
    ok_proc = k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    private = float(pmc.PrivateUsage) / MB if ok_proc else None
    avail = float(msx.ullAvailPhys) / MB if k32.GlobalMemoryStatusEx(ctypes.byref(msx)) else None
    return avail, private


def _proc_kb(path: Path, key: str) -> float | None:
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if line.startswith(key + ":"):
            return float(line.split()[1]) / 1024.0
    return None


def memory_mb() -> tuple[float | None, float | None]:
    """(RAM physique disponible, mémoire privée du processus), en Mo ; None pour une valeur non mesurable."""
    try:
        if os.name == "nt":
            return _win_memory_mb()
        meminfo, status = Path("/proc/meminfo"), Path("/proc/self/status")
        if meminfo.exists() and status.exists():
            return _proc_kb(meminfo, "MemAvailable"), _proc_kb(status, "RssAnon")
    except (OSError, ValueError, AttributeError):
        pass
    return None, None
