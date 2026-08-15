"Helpers for reproducible sklearn comparisons on mixed data."

def _marker_map(columns, missing_values):
    markers = {name:"" for name in columns}
    if missing_values is None: return markers
    if isinstance(missing_values, dict):
        for key,value in missing_values.items(): markers[columns[int(key)] if not isinstance(key, str) else key] = value
    else:
        if len(missing_values) != len(columns): raise ValueError("missing_values must have one value per column")
        markers.update(zip(columns, missing_values))
    return markers

def _clean_frame(df, markers, numeric=()):
    import numpy as np,pandas as pd

    result = df.copy()
    for name,marker in markers.items():
        if marker is None or np.isscalar(marker) and pd.isna(marker): result[name] = result[name].mask(result[name].isna())
        else: result[name] = result[name].mask(result[name] == marker)
    for name in numeric: result[name] = pd.to_numeric(result[name], errors="raise")
    return result

def _schema(df, missing_values):
    columns = list(df.columns)
    markers = _marker_map(columns, missing_values)
    cleaned = _clean_frame(df, markers)
    numeric,categorical = [],[]
    import pandas as pd
    for name in columns:
        observed = cleaned[name].dropna()
        if len(observed):
            try: pd.to_numeric(observed, errors="raise")
            except (TypeError, ValueError): categorical.append(name)
            else: numeric.append(name)
    return cleaned,markers,tuple(numeric),tuple(categorical)

def _adapter(markers, numeric):
    from sklearn.preprocessing import FunctionTransformer
    return FunctionTransformer(_clean_frame, kw_args=dict(markers=markers, numeric=numeric), feature_names_out="one-to-one")

def _target_encoder(random_state):
    import sklearn
    from sklearn.preprocessing import TargetEncoder
    if tuple(map(int, sklearn.__version__.split(".")[:2])) < (1, 9):
        return TargetEncoder(target_type="continuous", cv=5, shuffle=True, random_state=random_state)
    from sklearn.model_selection import KFold
    return TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=random_state))

def sklearn_preprocessor(df, missing_values=None, onehot_max=20, random_state=42):
    "Build sklearn's documented one-hot/target preprocessing policy from a training dataframe."
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    if onehot_max < 1: raise ValueError("onehot_max must be positive")
    cleaned,markers,numeric,categorical = _schema(df, missing_values)
    cardinality = cleaned[list(categorical)].nunique() if categorical else {}
    onehot = tuple(name for name in categorical if cardinality[name] <= onehot_max)
    target = tuple(name for name in categorical if cardinality[name] > onehot_max)
    transformers = []
    if numeric: transformers.append(("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric))
    if onehot: transformers.append(("onehot", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), onehot))
    if target: transformers.append(("target", _target_encoder(random_state), target))
    if not transformers: raise ValueError("dataframe has no usable columns")
    columns = ColumnTransformer(transformers, verbose_feature_names_out=False)
    return make_pipeline(_adapter(markers, numeric), columns)

def sklearn_hist_preprocessor(df, missing_values=None, native_max=255, random_state=42):
    "Build sklearn's documented native/target preprocessing policy for HistGradientBoosting."
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OrdinalEncoder

    if native_max < 1: raise ValueError("native_max must be positive")
    cleaned,markers,numeric,categorical = _schema(df, missing_values)
    cardinality = cleaned[list(categorical)].nunique() if categorical else {}
    native = tuple(name for name in categorical if cardinality[name] <= native_max)
    target = tuple(name for name in categorical if cardinality[name] > native_max)
    transformers = []
    if numeric: transformers.append(("numeric", "passthrough", numeric))
    if target: transformers.append(("target", _target_encoder(random_state), target))
    if native:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformers.append(("categorical", encoder, native))
    if not transformers: raise ValueError("dataframe has no usable columns")
    columns = ColumnTransformer(transformers, verbose_feature_names_out=False).set_output(transform="pandas")
    return make_pipeline(_adapter(markers, numeric), columns),native
