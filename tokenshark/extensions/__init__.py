"""Auto-discovery for Tokenshark Feature Plugins & Extensions (`tokenshark.extensions`).

Extensions can be loaded from three sources:
1. Built-in extensions in `tokenshark/extensions/<name>/` (or `.py` modules)
2. User / community git-cloned folders in `~/.tokenshark/extensions/` (or `TOKENSHARK_EXTENSIONS_DIR`)
3. Installed Python packages declaring `entry_points(group="tokenshark.extensions")`
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import pkgutil
import sys
from pathlib import Path

from ..sdk import _EXTENSIONS, FeaturePlugin

_DISCOVERED = False


def _load_external_dir(ext_dir: Path):
    if not ext_dir.exists() or not ext_dir.is_dir():
        return
    for child in sorted(ext_dir.iterdir()):
        if child.name.startswith(("_", ".")):
            continue
        mod_name = f"tokenshark_ext_{child.stem}"
        if mod_name in sys.modules:
            continue
        target = None
        if child.is_dir():
            for cand in ("__init__.py", "plugin.py"):
                if (child / cand).exists():
                    target = child / cand
                    break
        elif child.suffix == ".py":
            target = child
        if target is None:
            continue
        try:
            spec = importlib.util.spec_from_file_location(mod_name, target)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[tokenshark] Failed to load external extension {child}: {e}", file=sys.stderr)


def _load_entry_points():
    try:
        eps = importlib.metadata.entry_points(group="tokenshark.extensions")
    except TypeError:
        eps = importlib.metadata.entry_points().get("tokenshark.extensions", [])
    for ep in eps:
        try:
            ep.load()
        except Exception as e:
            print(f"[tokenshark] Failed to load entry_point extension {ep.name}: {e}", file=sys.stderr)


def discover_extensions(force: bool = False) -> dict[str, type[FeaturePlugin]]:
    """Discover and import all built-in, user-directory, and pip entry_point extensions."""
    global _DISCOVERED
    if _DISCOVERED and not force:
        return _EXTENSIONS

    # 1. Built-in extensions inside tokenshark/extensions/
    pkg_dir = Path(__file__).parent
    for mod in pkgutil.iter_modules([str(pkg_dir)]):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")

    # 2. User / community extensions directory (~/.tokenshark/extensions or TOKENSHARK_EXTENSIONS_DIR)
    env_dir = os.environ.get("TOKENSHARK_EXTENSIONS_DIR")
    if env_dir:
        for part in env_dir.split(os.pathsep):
            if part.strip():
                _load_external_dir(Path(part.strip()))
    user_dir = Path.home() / ".tokenshark" / "extensions"
    _load_external_dir(user_dir)

    # 3. Pip-installed entry_points(group="tokenshark.extensions")
    _load_entry_points()

    _DISCOVERED = True
    return _EXTENSIONS
