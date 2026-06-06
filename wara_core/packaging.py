from __future__ import annotations


WARA_RUNTIME_EXCLUDES: list[str] = []


def collect_wara_runtime_datas() -> list[tuple[str, str]]:
    return []


def collect_wara_runtime_hiddenimports() -> list[str]:
    from PyInstaller.utils.hooks import collect_submodules

    return collect_submodules("wara_core")
