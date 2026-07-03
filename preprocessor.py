from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

DROP_COLS = ['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan']
DIRTY_NUMERIC_COLS = [
    'Age', 'Annual_Income', 'Outstanding_Debt',
    'Num_of_Loan', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Amount_invested_monthly'
]

class CreditFeaturePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.num_cols_ = None
        self.cat_cols_ = None
        self.num_imputer_ = None
        self.cat_imputer_ = None
        self.encoder_ = None
        self.scaler_ = None

    @staticmethod
    def _parse_credit_history(val):
        if pd.isna(val):
            return np.nan
        match = re.search(r'(\d+)\s+Years?\s+and\s+(\d+)\s+Months?', str(val))
        if match:
            return int(match.group(1)) * 12 + int(match.group(2))
        return np.nan

    @staticmethod
    def _clean_numeric(series):
        return pd.to_numeric(
            series.astype(str).str.replace(r'[^0-9.\-]', '', regex=True),
            errors='coerce'
        )

    def _clean(self, X):
        X = X.copy()
        for col in DIRTY_NUMERIC_COLS:
            if col in X.columns:
                X[col] = self._clean_numeric(X[col])
        if 'Credit_History_Age' in X.columns:
            X['Credit_History_Age'] = X['Credit_History_Age'].apply(self._parse_credit_history)
        if 'Payment_Behaviour' in X.columns:
            X['Payment_Behaviour'] = X['Payment_Behaviour'].replace('!@9#%8', np.nan)
        if 'Credit_Mix' in X.columns:
            X['Credit_Mix'] = X['Credit_Mix'].replace('_', np.nan)
        if 'Payment_of_Min_Amount' in X.columns:
            X['Payment_of_Min_Amount'] = X['Payment_of_Min_Amount'].replace('NM', np.nan)
        return X

    def fit(self, X, y=None):
        X = self._clean(X)
        self.num_cols_ = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols_ = X.select_dtypes(include='object').columns.tolist()
        self.num_imputer_ = SimpleImputer(strategy='median').fit(X[self.num_cols_])
        self.cat_imputer_ = SimpleImputer(strategy='most_frequent').fit(X[self.cat_cols_])
        X_cat_imputed = pd.DataFrame(
            self.cat_imputer_.transform(X[self.cat_cols_]), columns=self.cat_cols_
        )
        self.encoder_ = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1
        ).fit(X_cat_imputed)
        self.scaler_ = StandardScaler().fit(self.num_imputer_.transform(X[self.num_cols_]))
        return self

    def transform(self, X):
        X = self._clean(X)
        X_num = self.scaler_.transform(self.num_imputer_.transform(X[self.num_cols_]))
        X_cat = self.encoder_.transform(self.cat_imputer_.transform(X[self.cat_cols_]))
        processed = np.hstack([X_num, X_cat])
        return pd.DataFrame(processed, columns=self.num_cols_ + self.cat_cols_, index=X.index)