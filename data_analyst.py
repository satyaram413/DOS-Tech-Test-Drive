python3 << 'EOF'
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

try:
    df = pd.read_csv('/home/user/data/URA 21 26 (1).csv')

    result = {}
    # Force numeric conversion
    df['Floor Area (sq ft)'] = pd.to_numeric(df['Floor Area (sq ft)'], errors='coerce')
    df['No. of Bedroom(for Non-Landed Only)'] = pd.to_numeric(df['No. of Bedroom(for Non-Landed Only)'], errors='coerce')
    # Basic shape
    result['rows'] = int(df.shape[0])
    result['columns'] = int(df.shape[1])
    result['column_names'] = list(df.columns)


    # Descriptive stats
    numeric_cols = df.select_dtypes(include='number')
    if not numeric_cols.empty:
        desc = numeric_cols.describe().round(2)
        result['descriptive_stats'] = desc.to_dict()

    # Correlation
    if len(numeric_cols.columns) >= 2:
        corr = numeric_cols.corr().round(3)
        result['correlation'] = corr.to_dict()

    # Missing values
    missing = df.isnull().sum()
    result['missing_values'] = missing[missing > 0].to_dict()

    # Categorical summary
    cat_cols = df.select_dtypes(include='object')
    cat_summary = {}
    for col in cat_cols.columns[:3]:
        cat_summary[col] = df[col].value_counts().head(5).to_dict()
    result['categorical_summary'] = cat_summary

    # LazyPredict — auto ML model comparison
    if len(numeric_cols.columns) >= 2:
        try:
            from lazypredict.Supervised import LazyClassifier, LazyRegressor
            from sklearn.model_selection import train_test_split

            # Use last numeric column as target
            target_col = numeric_cols.columns[-1]
            X = numeric_cols.drop(columns=[target_col]).dropna()
            y = numeric_cols[target_col].loc[X.index]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Decide classifier or regressor
            unique_vals = y.nunique()
            if unique_vals <= 10:
                model = LazyClassifier(verbose=0, ignore_warnings=True)
                models, _ = model.fit(X_train, X_test, y_train, y_test)
                model_type = 'classification'
            else:
                model = LazyRegressor(verbose=0, ignore_warnings=True)
                models, _ = model.fit(X_train, X_test, y_train, y_test)
                model_type = 'regression'

            top5 = models.head(5).round(3)
            result['lazypredict'] = {
                'target_column': target_col,
                'model_type': model_type,
                'top_5_models': top5.to_dict()
            }

        except Exception as lp_err:
            result['lazypredict'] = {'error': str(lp_err)}

    print(json.dumps(result, indent=2, default=str))

except Exception as e:
    print(json.dumps({"error": str(e)}))
EOF
