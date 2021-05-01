"""
LabelEncoderExt
It differs from sklearn's LabelEncoder by handling new classes and providing a value for it [Unknown]
Unknown values will be added in fit and transform will take care of new item. It gives unknown class id.

based on https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class LabelEncoderExt(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        """
        :param X:  1D Iterable of data to be encoded
        :type X: Iterable/pandas.Series/numpy.ndarray
        :return: Fitted LabelEncoderExt for all the unique values in X + unknown value
        :rtype: unofficial_encoders.label_encoder_ext.LabelEncoderExt
        """
        self.label_encoder = LabelEncoder()
        X = self.prepare_X(X)
        self.label_encoder = self.label_encoder.fit(list(X) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, X):
        """
        :param X:  1D Iterable of data to be encoded
        :type X: Iterable/pandas.Series/numpy.ndarray
        :return: Transformed pandas.Series. X transformed encoded ids where the new values get assigned to Unknown class
        :rtype: numpy.ndarray
        """
        X = self.prepare_X(X)
        new_data_list = list(X)
        for unique_item in np.unique(X):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x == unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)  # TODO change to pd.series

    def _more_tags(self):
        """
        :return: dict with skelarn test configuration.
        see https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
        :rtype: dict
        """
        return {
            'preserves_dtype': [np.int],
            'allow_nan': True,
            'X_types': ['1dlabels'],
            '_xfail_checks': {'check_complex_data': 'test'},  # TODO replace test with a meaningful reason.
        }

    @staticmethod
    def prepare_X(X):  # noqa
        """
        :param X: 1D Iterable that will be transformed into pandas.Series
        :type X: Iterable/pandas.Series/numpy.ndarray
        :return: X as pandas.Series
        :rtype: pandas.Series
        """
        if isinstance(X, pd.Series):
            return X.astype(str)  # astype(str) creates a class for nan values
        elif isinstance(X, Iterable):
            return pd.Series(X).astype(str)
        elif hasattr(X, 'shape') and X.shape[1] != 1:
            raise ValueError('must be 1D array')
        else:
            raise ValueError('X is not a 1D iterable')
