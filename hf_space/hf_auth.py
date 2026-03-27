from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Optional

from huggingface_hub import HfFolder


_HF_WIN_TARGET = "git:https://huggingface.co"


def _read_win_credential(target: str) -> Optional[str]:
    if os.name != "nt":
        return None

    class CREDENTIAL_ATTRIBUTEW(ctypes.Structure):
        _fields_ = [
            ("Keyword", wintypes.LPWSTR),
            ("Flags", wintypes.DWORD),
            ("ValueSize", wintypes.DWORD),
            ("Value", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    class CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.POINTER(CREDENTIAL_ATTRIBUTEW)),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    PCREDENTIALW = ctypes.POINTER(CREDENTIALW)
    cred_ptr = PCREDENTIALW()
    cred_read = ctypes.windll.advapi32.CredReadW
    cred_read.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(PCREDENTIALW)]
    cred_read.restype = wintypes.BOOL
    cred_free = ctypes.windll.advapi32.CredFree
    cred_free.argtypes = [ctypes.c_void_p]
    cred_free.restype = None

    if not cred_read(target, 1, 0, ctypes.byref(cred_ptr)):
        return None

    try:
        cred = cred_ptr.contents
        blob = ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
        if not blob:
            return None
        if b"\x00" in blob:
            return blob.decode("utf-16-le").rstrip("\x00")
        try:
            return blob.decode("utf-8")
        except UnicodeDecodeError:
            return blob.decode("latin-1")
    finally:
        cred_free(cred_ptr)


def resolve_hf_token() -> Optional[str]:
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = str(os.environ.get(name, "")).strip()
        if value:
            return value

    cached = HfFolder.get_token()
    if cached:
        return cached

    return _read_win_credential(_HF_WIN_TARGET)
