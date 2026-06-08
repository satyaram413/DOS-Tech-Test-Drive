python3 - << 'PYEOF'
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

    print(json.dumps(result, indent=2, default=str))
except Exception as e:
    print(json.dumps({'error': str(e)}, indent=2))
PYEOF
