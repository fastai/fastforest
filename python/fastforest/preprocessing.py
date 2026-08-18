from dataclasses import dataclass

import numpy as np
import pyarrow as pa

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

def _arrow_batch(X, expected_names=None):
    "Convert a supported table to the single native Arrow boundary."
    if isinstance(X, pa.RecordBatch): batch = X
    elif isinstance(X, pa.Table):
        table = X.combine_chunks()
        batch = table.to_batches(max_chunksize=max(1, len(table)))[0]
    elif getattr(X, "columns", None) is not None:
        batch = pa.RecordBatch.from_pandas(X, preserve_index=False)
    else:
        values = np.asarray(X)
        if values.ndim != 2: raise ValueError("X must be a two-dimensional array")
        names = expected_names or tuple(f"x{i}" for i in range(values.shape[1]))
        arrays = []
        for column in values.T:
            try: arrays.append(pa.array(column))
            except (pa.ArrowInvalid, pa.ArrowTypeError): arrays.append(pa.array([None if value is None else str(value) for value in column]))
        batch = pa.RecordBatch.from_arrays(arrays, names=names)
    if not batch.num_rows: raise ValueError("X must contain at least one row")
    names = tuple(str(name) for name in batch.schema.names)
    if len(set(names)) != len(names): raise ValueError("feature names must be unique")
    if expected_names is not None:
        if batch.num_columns != len(expected_names): raise ValueError(f"expected {len(expected_names)} features, got {batch.num_columns}")
        if names != expected_names: raise ValueError("prediction columns must match training columns")
    return batch,names

def _table(X, expected_names=None):
    batch,names = _arrow_batch(X, expected_names)
    return batch.to_pandas().to_numpy(dtype=object),names

def _take_rows(X, indices):
    if indices is None: return X
    if hasattr(X, "iloc"): return X.iloc[indices]
    if isinstance(X, (pa.Table, pa.RecordBatch)): return X.take(pa.array(indices))
    return np.asarray(X)[indices]

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

def _saved_scalar(value):
    if value is None: return 0,""
    if _is_nan(value): return 1,""
    if isinstance(value, (bool,np.bool_)): return 2,"1" if value else "0"
    if isinstance(value, (int,np.integer)): return 3,str(int(value))
    if isinstance(value, (float,np.floating)): return 4,repr(float(value))
    if isinstance(value, str): return 5,value
    raise TypeError(f"model metadata value {value!r} is not a portable scalar")

def _one_hot_layout(spec, names):
    if spec is None: spec = {}
    if not isinstance(spec, dict): raise TypeError("one_hot_groups must be a dict")
    groups,grouped = [],set()
    for group,selectors in spec.items():
        if not isinstance(group, str): raise TypeError("one-hot group names must be strings")
        if isinstance(selectors, (str, bytes)) or not hasattr(selectors, "__iter__"):
            raise TypeError(f"one-hot group {group!r} must contain a sequence of columns")
        indices = []
        for selector in selectors:
            try: index = names.index(selector) if isinstance(selector, str) else int(selector)
            except ValueError as error: raise ValueError(f"unknown one-hot column {selector!r}") from error
            if index < 0 or index >= len(names): raise ValueError(f"one-hot column {selector!r} is out of range")
            if index in grouped: raise ValueError(f"column {names[index]!r} belongs to more than one one-hot group")
            grouped.add(index)
            indices.append(index)
        if len(indices) < 2: raise ValueError(f"one-hot group {group!r} must contain at least two columns")
        groups.append((group, indices))
    direct = [index for index in range(len(names)) if index not in grouped]
    logical_names = tuple(names[index] for index in direct)+tuple(group for group,_ in groups)
    if len(set(logical_names)) != len(logical_names): raise ValueError("one-hot group and feature names must be unique")
    return groups,direct,logical_names

def _date_columns(spec, names, direct):
    if not isinstance(spec, dict): raise TypeError("date_columns must be a dict of columns to formats")
    dates = []
    for selector,format in spec.items():
        try: index = names.index(selector) if isinstance(selector, str) else int(selector)
        except ValueError as error: raise ValueError(f"unknown date column {selector!r}") from error
        if index < 0 or index >= len(names): raise ValueError(f"date column {selector!r} is out of range")
        if index not in direct: raise ValueError(f"date column {names[index]!r} is already grouped or configured")
        if not isinstance(format, str) or not format: raise TypeError(f"date column {names[index]!r} must have a non-empty format")
        direct.remove(index)
        dates.append((index, format))
    return dates,direct

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
    def __init__(self, missing_values=None, max_dummy_cardinality=1, one_hot_groups=None, date_columns=None,
        allow_new_missing=False, seed=None):
        if not isinstance(max_dummy_cardinality, int) or max_dummy_cardinality < 1:
            raise ValueError("max_dummy_cardinality must be a positive integer")
        self.missing_values,self.max_dummy_cardinality = missing_values,max_dummy_cardinality
        self.one_hot_groups,self.date_columns = one_hot_groups,date_columns
        self.allow_new_missing,self.seed = allow_new_missing,seed

    def fit_transform(self, X, indices=None):
        if indices is not None:
            X = _take_rows(X, indices)
        batch,self.input_names = _arrow_batch(X)
        markers = _markers(self.missing_values, self.input_names)
        self._groups,self._direct,_ = _one_hot_layout(self.one_hot_groups, self.input_names)
        if self.date_columns is None:
            detected = _NativeEncoder.detect_dates(batch, [_saved_scalar(marker) for marker in markers], self._groups, self.seed)
            self.date_columns = {self.input_names[index]:format for index,format in detected}
        dates,self._direct = _date_columns(self.date_columns, self.input_names, self._direct)
        native,ranked = _NativeEncoder.fit(batch, [_saved_scalar(marker) for marker in markers], self.max_dummy_cardinality,
            self.allow_new_missing, self._groups, dates)
        self.names,self._dates = tuple(native.logical_names),tuple((index,format,tuple(parts)) for index,format,parts in native.date_layout)
        logical_markers = [markers[index] for index in self._direct]+[""]*len(self._groups)+[np.nan]*(16*len(self._dates))
        self._set_native(native, logical_markers)
        return np.asarray(ranked)

    def _set_native(self, native, logical_markers):
        "Attach fitted native state and reconstruct its small Python display metadata."
        self._native = native
        fitted,encoded_names = [],[]
        for col,(name,marker) in enumerate(zip(self.names, logical_markers)):
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

    @classmethod
    def from_native(cls, native, markers, one_hot_groups, date_columns):
        "Reconstruct the Python adapter around a loaded native schema."
        names = tuple(native.input_names)
        group_spec = {group:[names[index] for index in indices] for group,indices in one_hot_groups}
        date_spec = {names[index]:format for index,format in date_columns}
        result = cls(markers, one_hot_groups=group_spec, date_columns=date_spec)
        result.input_names = names
        result._groups,result._direct,_ = _one_hot_layout(group_spec, names)
        _,result._direct = _date_columns(date_spec, names, result._direct)
        result.names,result._dates = tuple(native.logical_names),tuple((index,format,tuple(parts)) for index,format,parts in native.date_layout)
        logical_markers = [markers[index] for index in result._direct]+[""]*len(result._groups)+[np.nan]*(16*len(result._dates))
        result._set_native(native, logical_markers)
        return result

    def transform(self, X):
        batch,_ = _arrow_batch(X, self.input_names)
        markers = _markers(self.missing_values, self.input_names)
        return np.asarray(self._native.transform(batch, [_saved_scalar(marker) for marker in markers]))

    def predict(self, model, X, proba=False):
        "Transform and predict in bounded native row blocks."
        batch,_ = _arrow_batch(X, self.input_names)
        markers = _markers(self.missing_values, self.input_names)
        method = model.predict_proba_encoded if proba else model.predict_encoded
        return np.asarray(method(self._native, batch, [_saved_scalar(marker) for marker in markers]))

    def display(self, X):
        batch,_ = _arrow_batch(X, self.input_names)
        source = batch.to_pandas().to_numpy(dtype=object)
        logical = [source[:,index] for index in self._direct]
        for _,indices in self._groups:
            active = np.asarray(source[:,indices], dtype=np.float32).argmax(axis=1)
            logical.append(np.asarray(self.input_names, dtype=object)[np.asarray(indices)[active]])
        markers = _markers(self.missing_values, self.input_names)
        logical.extend(np.asarray(self._native.date_values(batch, [_saved_scalar(marker) for marker in markers])).T)
        values = np.column_stack(logical)
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

    @property
    def analysis_groups(self):
        groups = [(self.names[i], [raw]) for i,raw in enumerate(self._direct)]
        groups += [(name, indices) for name,indices in self._groups]
        return dict(groups)
