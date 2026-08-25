"""Fast approximate-forest models in Rust with Python bindings

Modules:

- `fastforest.reload`: Reload a maturin package's compiled extension and Python modules in a running kernel, without restarting it."""

from typing import TYPE_CHECKING

if TYPE_CHECKING: from .core import *
from .reload import reloadlib

__all__ = ["FastForest", "FastForestClassifier", "load", "ColumnInfo", "Importance", "Explanation", "PartialDependence",
    "FeatureRelations", "FeatureDependence", "permutation_importance", "drop_column_importance", "partial_dependence",
    "feature_relations", "feature_dependence", "sklearn_preprocessor", "sklearn_hist_preprocessor", "__version__"]

def __getattr__(name):
    "Load `.core`, and with it the compiled extension, on first use: plain `import fastforest` must stay binary-free."
    if name not in __all__: raise AttributeError(f"module 'fastforest' has no attribute {name!r}")
    from . import core
    value = getattr(core, name)
    globals()[name] = value
    return value

def __dir__(): return sorted({*globals(), *__all__})
