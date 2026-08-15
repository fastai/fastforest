from dataclasses import dataclass
from datetime import datetime,timezone
import calendar,re

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

_date_parts = ("Year", "Month", "Week", "Day", "Dayofweek", "Dayofyear", "Is_month_end", "Is_month_start",
    "Is_quarter_end", "Is_quarter_start", "Is_year_end", "Is_year_start", "Hour", "Minute", "Second", "Elapsed")

def _date_layout(spec, names, groups, direct):
    if spec is None: spec = {}
    if not isinstance(spec, dict): raise TypeError("date_columns must be a dict of columns to formats")
    dates = []
    for selector,format in spec.items():
        try: index = names.index(selector) if isinstance(selector, str) else int(selector)
        except ValueError as error: raise ValueError(f"unknown date column {selector!r}") from error
        if index < 0 or index >= len(names): raise ValueError(f"date column {selector!r} is out of range")
        if index not in direct: raise ValueError(f"date column {names[index]!r} is already grouped or configured")
        if not isinstance(format, str) or not format: raise TypeError(f"date column {names[index]!r} must have a non-empty format")
        direct.remove(index)
        prefix = re.sub("[Dd]ate$", "", names[index])
        dates.append((index, format, tuple(prefix+part for part in _date_parts)))
    logical_names = tuple(names[index] for index in direct)+tuple(group for group,_ in groups)
    logical_names += tuple(name for _,_,parts in dates for name in parts)
    if len(set(logical_names)) != len(logical_names): raise ValueError("generated and input feature names must be unique")
    native = [(index, format, part, name) for index,format,names in dates for part,name in enumerate(names)]
    return dates,direct,logical_names,native

def _date_value(value, part):
    month_end = value.day == calendar.monthrange(value.year, value.month)[1]
    values = (value.year, value.month, value.isocalendar().week, value.day, value.weekday(), value.timetuple().tm_yday,
        month_end, value.day == 1, month_end and value.month%3 == 0, value.day == 1 and value.month%3 == 1,
        month_end and value.month == 12, value.day == 1 and value.month == 1, value.hour, value.minute, value.second,
        value.replace(tzinfo=timezone.utc).timestamp())
    return values[part]

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
    def __init__(self, missing_values=None, max_dummy_cardinality=4, one_hot_groups=None, date_columns=None):
        if not isinstance(max_dummy_cardinality, int) or max_dummy_cardinality < 1:
            raise ValueError("max_dummy_cardinality must be a positive integer")
        self.missing_values,self.max_dummy_cardinality = missing_values,max_dummy_cardinality
        self.one_hot_groups,self.date_columns = one_hot_groups,date_columns

    def fit_transform(self, X):
        columns,self.input_names,_ = _column_arrays(X)
        markers = _markers(self.missing_values, self.input_names)
        self._groups,self._direct,_ = _one_hot_layout(self.one_hot_groups, self.input_names)
        self._dates,self._direct,self.names,date_parts = _date_layout(self.date_columns, self.input_names, self._groups, self._direct)
        logical_markers = [markers[index] for index in self._direct]+[""]*len(self._groups)
        logical_markers += [np.nan for _,_,parts in self._dates for _ in parts]
        self._native,ranked = _NativeEncoder.fit(columns, self.input_names, markers, self.max_dummy_cardinality, self._groups, date_parts)
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
        direct_numeric = all(self.columns[col].numeric and not self.columns[col].had_missing and markers[raw] == ""
            for col,raw in enumerate(self._direct))
        grouped_numeric = all(markers[index] == "" and not isinstance(columns[index], tuple)
            and np.asarray(columns[index]).dtype.kind in "biuf" for _,indices in self._groups for index in indices)
        self._numeric_matrix = direct_numeric and grouped_numeric and not self._dates
        return np.asarray(ranked)

    def transform(self, X):
        values = np.asarray(X)
        if self._numeric_matrix and getattr(X, "columns", None) is None and values.dtype.kind in "biuf":
            if values.ndim != 2: raise ValueError("X must be a two-dimensional array")
            if not len(values): raise ValueError("X must contain at least one row")
            if values.shape[1] != len(self.input_names): raise ValueError(f"expected {len(self.input_names)} features, got {values.shape[1]}")
            if values.dtype not in (np.dtype("float32"), np.dtype("float64")): values = values.astype(np.float32)
            return np.asarray(self._native.transform_numeric(np.ascontiguousarray(values)))
        columns,_,_ = _column_arrays(X, self.input_names)
        markers = _markers(self.missing_values, self.input_names)
        return np.asarray(self._native.transform(columns, markers))

    def display(self, X):
        _column_arrays(X, self.input_names)
        source = np.asarray(X, dtype=object)
        logical = [source[:,index] for index in self._direct]
        for _,indices in self._groups:
            active = np.asarray(source[:,indices], dtype=np.float32).argmax(axis=1)
            logical.append(np.asarray(self.input_names, dtype=object)[np.asarray(indices)[active]])
        for index,format,parts in self._dates:
            raw = source[:,index]
            missing = _missing_mask(raw, _markers(self.missing_values, self.input_names)[index])
            parsed = [None if is_missing else datetime.strptime(str(value), format) for value,is_missing in zip(raw, missing)]
            for part in range(len(parts)): logical.append(np.asarray([np.nan if value is None else _date_value(value, part) for value in parsed]))
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
