from copy import copy
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from unofficial_encoders.label_encoder_ext import LabelEncoderExt


class OneHotEncoderExt(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_values=None):
        super().__init__()
        self.categoricals = categorical_values
        self.x_label_encoder_dict = dict()
        self.x_one_hot_encoder_dict = dict()
        # using regular label encoder because its illegal to give the model labels that it didnt trained on

    def fit(self, X, y=None):

        if not self.categoricals:
            self.categoricals = [x for x in X.columns if X[x].dtype == 'object']

        if isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        else:
            X = X.copy()

        # fit label + one hot encoder
        for c in self.categoricals:
            # TODO raise warning when there are to many unique values
            self.x_label_encoder_dict[c] = LabelEncoderExt()
            print(c)
            label_encoded_col = self.x_label_encoder_dict[c].fit_transform(
                X[c].astype(str))  # astype creates a class for nan values
            self.x_one_hot_encoder_dict[c] = OneHotEncoder(handle_unknown='ignore')
            one_hot_cols = self.x_one_hot_encoder_dict[c].fit_transform(label_encoded_col.reshape(-1, 1))
            one_hot_df = pd.DataFrame(one_hot_cols.toarray()).astype('uint8').add_prefix(f'{c}_').reset_index(drop=True)
            X = pd.concat([X, one_hot_df], axis=1)
            pass

        X.drop(self.categoricals, axis=1, inplace=True)
        if y:
            # TODO raise warning that y will not be encoded
            pass

        return self

    def transform(self, X):
        X = X.copy()
        return X
