"""Allow tests to run from a checkout with any directory name."""
import importlib.util
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
name = "maibot_qwen_bilibili"
if name not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
