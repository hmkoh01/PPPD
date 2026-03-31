import importlib.util
import sys
from pathlib import Path

_page = Path(__file__).parent / "pages" / "1_Admin.py"
_spec = importlib.util.spec_from_file_location("admin_page", _page)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["admin_page"] = _mod
_spec.loader.exec_module(_mod)
