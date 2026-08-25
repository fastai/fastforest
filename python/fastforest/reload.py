"""Reload a maturin package's compiled extension and Python modules in a running kernel, without restarting it.

Usage, from a kernel with fastforest already imported:

    from fastforest.reload import reloadlib
    # edit the Rust source, then:
    reloadlib()

`reloadlib` rebuilds the crate when its source changed, swaps in the fresh extension, and re-executes the package's Python modules so every name in the package namespace refers to the new generation. A no-change call is cheap: the hook only checks freshness. Requires the `maturin-import-hook` package; `reloadlib` installs the hook into the running interpreter on first use, and the hook's first import of the package does one full rebuild into its own cache (later cycles are incremental).

Usage notes:

- Objects created before a reload belong to the previous generation. They keep working, but they fail `isinstance` checks against current classes and must not be mixed into new-generation calls. Recreate models after reloading.
- Each reload keeps the previous extension image mapped: memory grows by the library size per cycle, and a used generation keeps its own rayon thread pool. Restart the kernel occasionally on long sessions.
- The hook builds into its own cache directory, and the reloaded kernel runs that binary. The in-tree `_core.so` and the `target/` executables do not update until the next `maturin develop` or `cargo build`.
- Reloading extension modules is unsupported on Windows.
"""
import importlib,re,sys
from pathlib import Path

def _ensure_hook():
    try: import maturin_import_hook
    except ImportError as error: raise ImportError("reloadlib needs the maturin-import-hook package (pip install maturin-import-hook)") from error
    if not any(type(finder).__module__.startswith("maturin_import_hook") for finder in sys.meta_path):
        from maturin_import_hook.settings import MaturinSettings
        maturin_import_hook.install(settings=MaturinSettings(profile="play", quiet=True))

def _dep_order(pkg):
    "Names of `pkg`'s imported Python submodules, each after every submodule it imports."
    prefix = pkg.__name__+"."
    sources = {name:Path(module.__file__).read_text() for name,module in sys.modules.items()
        if name.startswith(prefix) and getattr(module, "__file__", "").endswith(".py")}
    pattern = re.compile(rf"from +(?:\.|{pkg.__name__}\.)(\w+) +import|import +{pkg.__name__}\.(\w+)")
    deps = {name:{prefix+(a or b) for a,b in pattern.findall(source)} & sources.keys() for name,source in sources.items()}
    order,todo = [],sorted(sources)
    while todo:
        name = next((n for n in todo if not (deps[n]-set(order))), todo[0])
        todo.remove(name)
        order.append(name)
    return order

def reloadlib(pkg=None):
    "Rebuild and reload `pkg` (default: fastforest); see this module's docstring for important usage details."
    _ensure_hook()
    if pkg is None: import fastforest as pkg
    if isinstance(pkg, str): pkg = importlib.import_module(pkg)
    pkg = sys.modules[pkg.__name__]
    importlib.reload(pkg)
    for name in _dep_order(pkg): importlib.reload(sys.modules[name])
    del sys.modules[pkg.__name__]
    fresh = importlib.import_module(pkg.__name__)
    for name in getattr(fresh, "__all__", ()): pkg.__dict__.pop(name, None)
    pkg.__dict__.update(fresh.__dict__)
    sys.modules[pkg.__name__] = pkg
    return pkg
