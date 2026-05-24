import pandas as pd
import numpy as np
import json
import warnings
import os
import urllib.request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

QUESTION = '''{{#1776584414076.analysis_question#}}'''
QUESTION_LOWER = QUESTION.lower()


def parse_floor_area(val):
    try:
        parts = str(val).replace(',', '').split(' to ')
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        return float(parts[0])
    except Exception:
        return np.nan


def download_url_to_csv(url, output_path):
    request = urllib.request.Request(url.strip(), headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    with open(output_path, 'wb') as f:
        f.write(data)
    return output_path


def load_dataframe():
    data_dir = '/home/user/data/'
    upload_dirs = [data_dir, '/home/user/upload/', '/home/user/uploads/']
    for directory in upload_dirs:
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.lower().endswith('.csv')]
            if files:
                return pd.read_csv(os.path.join(directory, files[0]), on_bad_lines='skip')
    if os.path.exists(data_dir):
        url_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.url')]
        if url_files:
            with open(os.path.join(data_dir, url_files[0]), 'r', encoding='utf-8') as f:
                url = f.read().strip()
            if not url:
                raise ValueError('CSV URL file is empty.')
            csv_path = os.path.join(data_dir, 'input.csv')
            download_url_to_csv(url, csv_path)
            return pd.read_csv(csv_path, on_bad_lines='skip')
    raise ValueError('No CSV file found. Upload a CSV file or provide a CSV URL.')


def clean_dataframe(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    dirty_nulls = {'-', '–', '—', 'na', 'n/a', 'null', 'none', ''}
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map(lambda x: np.nan if str(x).strip().lower() in dirty_nulls else x)

    if 'Floor Area (sq ft)' in df.columns:
        df['Floor Area (sq ft)'] = df['Floor Area (sq ft)'].apply(parse_floor_area)
    if 'No. of Bedroom(for Non-Landed Only)' in df.columns:
        df['No. of Bedroom(for Non-Landed Only)'] = pd.to_numeric(df['No. of Bedroom(for Non-Landed Only)'], errors='coerce')
    if 'Lease Commencement Date' in df.columns:
        df['Lease Year'] = pd.to_datetime(df['Lease Commencement Date'], format='%b-%Y', errors='coerce').dt.year

    for col in df.columns:
        if df[col].dtype == 'object':
            s = df[col].astype(str).str.strip()
            cleaned = s.str.replace(',', '', regex=False).str.replace('$', '', regex=False).str.replace('%', '', regex=False)
            numeric = pd.to_numeric(cleaned, errors='coerce')
            non_null_original = df[col].notna().sum()
            if non_null_original and numeric.notna().sum() / non_null_original >= 0.70:
                df[col] = numeric
            else:
                df[col] = s.replace({'nan': np.nan, 'None': np.nan})

    if 'room_type' in df.columns and df['room_type'].dtype == 'object':
        df['room_type'] = df['room_type'].str.strip()

    if {'min_selling_price', 'max_selling_price'}.issubset(df.columns):
        df['avg_selling_price'] = df[['min_selling_price', 'max_selling_price']].mean(axis=1)
        df['selling_price_range'] = df['max_selling_price'] - df['min_selling_price']
    if {'min_selling_price_less_ahg_shg', 'max_selling_price_less_ahg_shg'}.issubset(df.columns):
        df['avg_selling_price_less_ahg_shg'] = df[['min_selling_price_less_ahg_shg', 'max_selling_price_less_ahg_shg']].mean(axis=1)
    return df


def json_safe_dict(d):
    return json.loads(json.dumps(d, default=str))


def choose_target_column(df):
    numeric_names = list(df.select_dtypes(include='number').columns)
    if not numeric_names:
        return None
    for col in numeric_names:
        cl = col.lower()
        if f'target: {cl}' in QUESTION_LOWER or f'target column {cl}' in QUESTION_LOWER:
            return col
        if f'predict {cl}' in QUESTION_LOWER or f'predicting {cl}' in QUESTION_LOWER:
            return col
        if f'use {cl} as' in QUESTION_LOWER and 'target' in QUESTION_LOWER:
            return col

    priority = []
    if 'max_selling_price' in QUESTION_LOWER:
        priority += ['max_selling_price']
    if 'min_selling_price' in QUESTION_LOWER:
        priority += ['min_selling_price']
    if 'average' in QUESTION_LOWER or 'avg' in QUESTION_LOWER:
        priority += ['avg_selling_price', 'avg_selling_price_less_ahg_shg']
    if 'price' in QUESTION_LOWER or 'affordability' in QUESTION_LOWER or 'cost' in QUESTION_LOWER:
        priority += ['max_selling_price', 'avg_selling_price', 'min_selling_price']
    if 'rent' in QUESTION_LOWER or 'rental' in QUESTION_LOWER:
        priority += ['Monthly Gross Rent ($)', 'monthly_gross_rent', '1room_rental', '2room_rental', '3room_rental']
    if 'dwelling' in QUESTION_LOWER or 'stock' in QUESTION_LOWER or 'supply' in QUESTION_LOWER:
        priority += ['total_dwelling_units']

    for wanted in priority:
        for col in numeric_names:
            if col.lower() == wanted.lower():
                return col
    for wanted in priority:
        matches = [c for c in numeric_names if wanted.lower() in c.lower()]
        if matches:
            return matches[0]
    non_time_cols = [c for c in numeric_names if c not in ['S/N'] and 'year' not in c.lower() and 'date' not in c.lower()]
    return non_time_cols[-1] if non_time_cols else numeric_names[-1]


def build_ml_frame(df, target_col):
    df_ml = df.copy()
    numeric_features = [c for c in df_ml.select_dtypes(include='number').columns if c != target_col and c != 'S/N']
    leakage_groups = {
        'max_selling_price': ['min_selling_price', 'avg_selling_price', 'selling_price_range', 'min_selling_price_less_ahg_shg', 'max_selling_price_less_ahg_shg', 'avg_selling_price_less_ahg_shg'],
        'min_selling_price': ['max_selling_price', 'avg_selling_price', 'selling_price_range', 'min_selling_price_less_ahg_shg', 'max_selling_price_less_ahg_shg', 'avg_selling_price_less_ahg_shg'],
        'avg_selling_price': ['min_selling_price', 'max_selling_price', 'selling_price_range', 'min_selling_price_less_ahg_shg', 'max_selling_price_less_ahg_shg', 'avg_selling_price_less_ahg_shg'],
    }
    numeric_features = [c for c in numeric_features if c not in leakage_groups.get(target_col, [])]

    cat_features = []
    for col in df_ml.select_dtypes(include='object').columns:
        nunique = df_ml[col].nunique(dropna=True)
        if 2 <= nunique <= 50:
            cat_features.append(col)

    keep = numeric_features + cat_features + [target_col]
    model_df = df_ml[keep].dropna(subset=[target_col]).copy()
    for col in numeric_features:
        model_df[col] = model_df[col].fillna(model_df[col].median())
    for col in cat_features:
        model_df[col] = model_df[col].fillna('Unknown')
    if cat_features:
        model_df = pd.get_dummies(model_df, columns=cat_features, drop_first=True)
    feature_cols = [c for c in model_df.columns if c != target_col]
    return model_df, feature_cols, numeric_features, cat_features


try:
    df_raw = load_dataframe()
    df = clean_dataframe(df_raw)
    result = {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'column_names': list(df.columns),
        'question': QUESTION,
    }

    numeric_cols = df.select_dtypes(include='number')
    cat_cols = df.select_dtypes(include='object')
    if not numeric_cols.empty:
        result['descriptive_stats'] = json_safe_dict(numeric_cols.describe().round(2).to_dict())
    if len(numeric_cols.columns) >= 2:
        result['correlation'] = json_safe_dict(numeric_cols.corr(numeric_only=True).round(3).to_dict())

    missing = df.isnull().sum()
    result['missing_values'] = json_safe_dict(missing[missing > 0].to_dict())

    cat_summary = {}
    for col in list(cat_cols.columns)[:8]:
        cat_summary[col] = df[col].value_counts(dropna=False).head(10).to_dict()
    result['categorical_summary'] = json_safe_dict(cat_summary)

    grouped = {}
    if 'financial_year' in df.columns:
        price_cols = [c for c in ['min_selling_price', 'max_selling_price', 'avg_selling_price'] if c in df.columns]
        if price_cols:
            grouped['price_by_year'] = json_safe_dict(df.groupby('financial_year')[price_cols].mean().round(2).tail(10).to_dict())
    if 'town' in df.columns:
        cols = [c for c in ['max_selling_price', 'avg_selling_price', 'total_dwelling_units'] if c in df.columns]
        if cols:
            grouped['top_towns'] = json_safe_dict(df.groupby('town')[cols].mean().round(2).sort_values(cols[0], ascending=False).head(10).to_dict())
    if 'room_type' in df.columns:
        cols = [c for c in ['max_selling_price', 'avg_selling_price'] if c in df.columns]
        if cols:
            grouped['price_by_room_type'] = json_safe_dict(df.groupby('room_type')[cols].mean().round(2).to_dict())
    if grouped:
        result['grouped_insights'] = grouped

    try:
        if len(numeric_cols.columns) > 0:
            target_for_chart = choose_target_column(df) or numeric_cols.columns[-1]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(df[target_for_chart].dropna(), bins=40, color='steelblue', edgecolor='white')
            axes[0].set_title(f'{target_for_chart} Distribution')
            axes[0].set_xlabel(target_for_chart)
            axes[0].set_ylabel('Frequency')
            if 'financial_year' in df.columns:
                trend = df.groupby('financial_year')[target_for_chart].mean().dropna()
                axes[1].plot(trend.index, trend.values, marker='o', color='coral')
                axes[1].set_title(f'Average {target_for_chart} by Year')
                axes[1].tick_params(axis='x', rotation=45)
            elif len(cat_cols.columns) > 0:
                group_col = 'room_type' if 'room_type' in df.columns else cat_cols.columns[0]
                group_data = df.groupby(group_col)[target_for_chart].median().sort_values(ascending=False).head(15)
                axes[1].bar(range(len(group_data)), group_data.values, color='coral', edgecolor='white')
                axes[1].set_xticks(range(len(group_data)))
                axes[1].set_xticklabels(group_data.index.astype(str), rotation=45, ha='right', fontsize=8)
                axes[1].set_title(f'Median {target_for_chart} by {group_col}')
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            result['chart_base64'] = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
    except Exception as e:
        result['chart_error'] = str(e)

    try:
        if numeric_cols.empty:
            result['ml_error'] = 'No numeric columns available for modeling.'
        else:
            df_sample = df.sample(min(5000, len(df)), random_state=42).copy()
            target_col = choose_target_column(df_sample)
            if target_col is None:
                result['ml_error'] = 'No target column could be selected.'
            else:
                model_df, feature_cols, numeric_features, cat_features = build_ml_frame(df_sample, target_col)
                model_df = model_df.dropna()
                if len(model_df) < 20 or len(feature_cols) == 0:
                    result['ml_error'] = 'Insufficient rows or features after cleaning.'
                else:
                    X = model_df[feature_cols]
                    y = model_df[target_col]
                    task = 'classification' if y.nunique() <= 10 and not pd.api.types.is_float_dtype(y) else 'regression'
                    if task == 'classification':
                        y = y.astype(str)
                    result['ml_sample_size'] = int(len(X))
                    result['target_column'] = target_col
                    result['task_type'] = task
                    result['numeric_features_used'] = numeric_features
                    result['categorical_features_encoded'] = cat_features
                    result['features_used_count'] = int(len(feature_cols))
                    result['features_used_preview'] = feature_cols[:30]

                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    try:
                        from lazypredict.Supervised import LazyClassifier, LazyRegressor
                        lz = LazyClassifier(verbose=0, ignore_warnings=True) if task == 'classification' else LazyRegressor(verbose=0, ignore_warnings=True)
                        models, _ = lz.fit(X_train, X_test, y_train, y_test)
                        top_models = models.head(10).round(3)
                        result['lazypredict_leaderboard'] = {
                            'target': target_col,
                            'task': task,
                            'metric_columns': list(top_models.columns),
                            'best_model': str(top_models.index[0]) if len(top_models) else None,
                            'best_model_metrics': top_models.iloc[0].to_dict() if len(top_models) else {},
                            'top_10': json_safe_dict(top_models.to_dict())
                        }
                    except Exception as e:
                        result['lazypredict_leaderboard'] = {'error': str(e)}

                    try:
                        from flaml import AutoML
                        automl = AutoML()
                        automl.fit(X_train, y_train, task=task, time_budget=30, verbose=0)
                        result['flaml_best_model'] = {
                            'target': target_col,
                            'task': task,
                            'winner': automl.best_estimator,
                            'score': round(float(automl.best_loss), 4)
                        }
                    except Exception as e:
                        result['flaml_best_model'] = {'error': str(e)}
    except Exception as e:
        result['ml_error'] = str(e)

    print(json.dumps(result, indent=2, default=str))
except Exception as e:
    print(json.dumps({'error': str(e)}, indent=2))