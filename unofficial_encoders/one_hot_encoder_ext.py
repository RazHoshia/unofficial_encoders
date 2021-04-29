from collections.abc import Iterable

import numpy as np
import pandas as pd
import scipy
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from .label_encoder_ext import LabelEncoderExt


class OneHotEncoderExt(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()
        # using regular label encoder because its illegal to give the model labels that it didnt trained on

    def fit(self, X, y=None):
        self._x_label_encoder_dict = dict()
        self._x_one_hot_encoder_dict = dict()
        X = self.prepare_X(X)
        self._orig_cols = list(X.columns)

        for c in X.columns:
            # TODO raise warning when there are to many unique values
            self._x_label_encoder_dict[c] = LabelEncoderExt()
            label_encoded_col = self._x_label_encoder_dict[c].fit_transform(X[c])
            self._x_one_hot_encoder_dict[c] = OneHotEncoder(handle_unknown='ignore')
            self._x_one_hot_encoder_dict[c].fit(label_encoded_col.reshape(-1, 1))

        if y is not None:
            # TODO raise warning that y will not be encoded
            pass

        return self

    def transform(self, X):
        if not hasattr(self, '_x_label_encoder_dict') or not hasattr(self, '_x_one_hot_encoder_dict') or not \
                hasattr(self, '_orig_cols'):
            raise NotFittedError

        X = self.prepare_X(X)

        if list(X.columns) != self._orig_cols:
            raise ValueError('Features during train and test are different.')

        for c in self._orig_cols:
            # TODO raise warning when there are to many unique values
            label_encoded_col = self._x_label_encoder_dict[c].transform(X[c])
            one_hot_cols = self._x_one_hot_encoder_dict[c].transform(label_encoded_col.reshape(-1, 1))
            one_hot_df = pd.DataFrame(one_hot_cols.toarray()).astype('uint8').add_prefix(f'{c}_').reset_index(drop=True)
            X = pd.concat([X, one_hot_df], axis=1)

        X.drop(self._orig_cols, axis=1, inplace=True)
        return X

    def _more_tags(self):
        return {
            'preserves_dtype': [np.uint8],
            'allow_nan': True,
            '_xfail_checks': {'check_complex_data': 'test',
                              'check_dtype_object': 'test',
                              'check_transformer_data_not_an_array': 'test',
                              'check_transformer_preserve_dtypes': 'test does not support DataFrame output',
                              'check_methods_sample_order_invariance': 'One Hot Encoder changes columns',
                              'check_fit_idempotent': 'test does not support DataFrame output'},
        }

    @staticmethod
    def prepare_X(X) -> pd.DataFrame:  # noqa
        if hasattr(X, 'shape') and len(X.shape) == 1:
            raise ValueError(
                f'Encoder expects 2D array. Reshape your data either using array. '
                f'reshape(-1, 1) if your data has a single feature or array. '
                f'reshape(1, -1) if it contains a single sample')
        elif hasattr(X, 'shape') and X.shape[1] == 0:
            raise ValueError(f'0 feature(s) (shape={X.shape}) while a minimum of 1 is required.')
        elif scipy.sparse.issparse(X):
            raise ValueError('Sparse matrix is not supported.')
        elif isinstance(X, pd.DataFrame):
            X = X.copy()
        elif isinstance(X, np.ndarray) or isinstance(X, Iterable):
            X = pd.DataFrame(X)
        else:
            raise ValueError('Encoder support only 2D numpy.ndarray and pandas.DataFrame and python Iterables. '
                             'sparse data is not supported')

        X.fillna(value=np.nan, inplace=True)
        return X
