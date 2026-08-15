from dataclasses import dataclass

import numpy as np

from ._core import Encoder as _NativeEncoder

@dataclass(frozen=True)
class ColumnInfo:
    "Fitted interpretation and encoding metadata for one input column."
    name: str
    kind: str
    cardinality: int
    all_int: bool
    had_missing: bool
    median: object
    encoded_features: tuple

def _column_arrays(X, expected_names=None):
    names = getattr(X, "columns", None)
    if names is not None:
        names = tuple(str(name) for name in names)
        series = [X.iloc[:,i] for i in range(len(names))]
        columns = [(np.asarray(column.cat.codes), np.asarray(column.cat.categories, dtype=object))
            if hasattr(column, "cat") else np.asarray(column) for column in series]
        rows = len(X)
    else:
        values = np.asarray(X)
        if values.ndim != 2: raise ValueError("X must be a two-dimensional array")
        rows = len(values)
        names = tuple(f"x{i}" for i in range(values.shape[1]))
        columns = [values[:,i] for i in range(values.shape[1])]
    if not rows: raise ValueError("X must contain at least one row")
    if len(set(names)) != len(names): raise ValueError("feature names must be unique")
    if expected_names is not None:
        if len(columns) != len(expected_names): raise ValueError(f"expected {len(expected_names)} features, got {len(columns)}")
        if getattr(X, "columns", None) is not None and names != expected_names: raise ValueError("prediction columns must match training columns")
        names = expected_names
    columns = [column if isinstance(column, tuple) else column.astype(str).astype(object) if column.dtype.kind == "S"
        else column.astype(object) if column.dtype.kind == "U" else column for column in columns]
    return columns,names,rows

def _table(X, expected_names=None):
    columns,names,_ = _column_arrays(X, expected_names)
    return np.column_stack(columns),names

def _is_nan(value):
    try: return bool(np.isscalar(value) and np.isnan(value))
    except TypeError: return False

def _missing_mask(values, marker):
    values = np.asarray(values)
    if marker is None: return values == None
    if _is_nan(marker):
        if values.dtype.kind in "biufc": return np.isnan(values)
        return np.fromiter((_is_nan(value) for value in values), bool, len(values))
    if marker == "" and values.dtype.kind in "biufc": return np.zeros(len(values), dtype=bool)
    return np.asarray(values == marker, dtype=bool)

def _markers(spec, names):
    result = [""]*len(names)
    if spec is None: return result
    if isinstance(spec, dict):
        for key,value in spec.items():
            idx = names.index(key) if isinstance(key, str) else int(key)
            if idx < 0 or idx >= len(names): raise ValueError(f"missing-value column {key!r} is out of range")
            result[idx] = value
        return result
    if len(spec) != len(names): raise ValueError("missing_values must have one value per column")
    return list(spec)

def _numeric(values, name):
    try: result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError): return None
    if not np.isfinite(result).all(): raise ValueError(f"column {name!r} contains a non-finite numeric value")
    return result

def _strings(values): return np.asarray(values, dtype=str)

@dataclass
class _Column:
    name: str
    marker: object
    numeric: bool
    all_int: bool
    values: np.ndarray
    median: object
    had_missing: bool
    encoded: tuple

    def canonical(self, raw, missing, inference=False):
        if inference and missing.any() and not self.had_missing:
            row = int(np.flatnonzero(missing)[0])
            raise ValueError(f"column {self.name!r} has a missing value at row {row}, but had none during training")
        observed = raw[~missing] if missing.any() else raw
        if self.numeric:
            values = _numeric(observed, self.name)
            if values is None: raise ValueError(f"column {self.name!r} was numeric during training")
        else: values = _strings(observed)
        if not missing.any(): return values
        result = np.empty(len(raw), dtype=np.float32 if self.numeric else self.values.dtype)
        result[~missing] = values
        if missing.any(): result[missing] = self.median
        return result

class _Encoder:
    def __init__(self, missing_values=None, max_dummy_cardinality=4):
        if not isinstance(max_dummy_cardinality, int) or max_dummy_cardinality < 1:
            raise ValueError("max_dummy_cardinality must be a positive integer")
        self.missing_values,self.max_dummy_cardinality = missing_values,max_dummy_cardinality

    def fit_transform(self, X):
        columns,self.names,_ = _column_arrays(X)
        markers = _markers(self.missing_values, self.names)
        self._native,ranked = _NativeEncoder.fit(columns, self.names, markers, self.max_dummy_cardinality)
        fitted,encoded_names = [],[]
        for col,(name,marker) in enumerate(zip(self.names, markers)):
            numeric,all_int,had_missing,median_num,median_text,numeric_values,text_values,raw_encoded = self._native.metadata(col)
            values = np.asarray(numeric_values, dtype=np.float32) if numeric else np.asarray(text_values, dtype=str)
            median = median_num if numeric else median_text
            encoded = tuple(("ordered", None) if kind == 0 else ("dummy", category) if kind == 1 else ("missing", None)
                for kind,category in raw_encoded)
            for kind,category in encoded:
                if kind == "ordered": encoded_names.append(name)
                elif kind == "dummy": encoded_names.append(f"{name}={values[category]}")
                else: encoded_names.append(f"{name}_missing")
            fitted.append(_Column(name, marker, numeric, all_int, values, median, had_missing, encoded))
        self.columns = tuple(fitted)
        self.encoded_names = tuple(encoded_names)
        self.encoded_to_raw = np.asarray(self._native.encoded_to_raw)
        self.feature_group_ids = np.asarray(self._native.feature_group_ids)
        self.cutoff_offsets = np.asarray(self._native.cutoff_offsets)
        self.cutoff_values = np.asarray(self._native.cutoff_values)
        self.column_info = tuple(ColumnInfo(column.name, "discarded" if not len(column.values) else "numeric" if column.numeric else "lexical",
            len(column.values), column.all_int, column.had_missing, int(column.median) if column.all_int and column.median is not None else column.median,
            tuple(self.encoded_names[i] for i in np.flatnonzero(self.encoded_to_raw == col)))
            for col,column in enumerate(self.columns))
        return np.asarray(ranked)

    def transform(self, X):
        columns,_,_ = _column_arrays(X, self.names)
        markers = [column.marker for column in self.columns]
        return np.asarray(self._native.transform(columns, markers))

    def display(self, X):
        values,_ = _table(X, self.names)
        result = np.empty(values.shape, dtype=object)
        for col,(column,raw) in enumerate(zip(self.columns, values.T)):
            missing = _missing_mask(raw, column.marker)
            if not len(column.values): result[:,col] = raw
            else:
                displayed = column.canonical(raw, missing)[~missing]
                result[~missing,col] = displayed.astype(np.int64) if column.all_int else displayed
                result[missing,col] = raw[missing]
        return result

    def aggregate(self, values):
        result = np.zeros(values.shape[:-1]+(len(self.names),), dtype=values.dtype)
        for encoded,original in enumerate(self.encoded_to_raw): result[...,original] += values[...,encoded]
        return result
