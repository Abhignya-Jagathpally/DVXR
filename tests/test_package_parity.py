"""Guard against silent drift between the dvxr package and its goal1_pipeline shims.

Every public function/class DEFINED in a dvxr.<mod> must be re-exported as the *same
object* by goal1_pipeline.<mod>. Catches the case where dvxr gains/changes a symbol but
the compatibility shim quietly falls out of sync (review finding m3).
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import goal1_pipeline  # noqa: E402


def _shim_modules():
    pkgdir = Path(goal1_pipeline.__file__).parent
    return sorted(m.name for m in pkgutil.iter_modules([str(pkgdir)])
                  if not m.ispkg and not m.name.startswith("_"))


class PackageParityTest(unittest.TestCase):
    def test_shims_reexport_dvxr_public_symbols(self):
        checked = 0
        for mod in _shim_modules():
            try:
                d = importlib.import_module(f"dvxr.{mod}")
            except Exception:
                continue                         # shim with no dvxr counterpart
            g = importlib.import_module(f"goal1_pipeline.{mod}")
            for name, obj in vars(d).items():
                if name.startswith("_"):
                    continue
                is_defhere = (isinstance(obj, (types.FunctionType, type))
                              and getattr(obj, "__module__", "").startswith("dvxr"))
                if not is_defhere:
                    continue
                self.assertIs(getattr(g, name, None), obj,
                              f"goal1_pipeline.{mod}.{name} drifted from dvxr.{mod}.{name}")
                checked += 1
        self.assertGreater(checked, 0, "no shared symbols checked — discovery broken")


if __name__ == "__main__":
    unittest.main()
