"""
NOT FULLY TESTED AND FOR INTERNAL USE ONLY!
based on https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
"""
from collections.abc import Iterable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class LabelEncoderExt(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """

    def fit(self, X, y=None):
        """
        This will fit the unofficial_encoders for all the unique values and introduce unknown value
        :param X: A list of string
        :return: self
        """
        self.label_encoder = LabelEncoder()
        X = self.prepare_X(X)
        self.label_encoder = self.label_encoder.fit(list(X) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, X):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param X:
        :return:
        """
        X = self.prepare_X(X)
        new_data_list = list(X)
        for unique_item in np.unique(X):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x == unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)  # TODO change to pd.series

    def _more_tags(self):
        return {
            'preserves_dtype': [np.int],
            'allow_nan': True,
            'X_types': ['1dlabels'],
            '_xfail_checks': {'check_complex_data': 'test',
                              },
        }

    @staticmethod
    def prepare_X(X):  # noqa
        if isinstance(X, pd.Series):
            return X.astype(str)  # astype(str) creates a class for nan values
        elif isinstance(X, Iterable):
            return pd.Series(X).astype(str)
        elif hasattr(X, 'shape') and X.shape[1] != 1:
            raise ValueError('must be 1D array')
        else:
            raise ValueError(f'X is not a 1D iterable')
