import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from .label_encoder_ext import LabelEncoderExt


class OneHotEncoderExt(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()
        self.x_label_encoder_dict = dict()
        self.x_one_hot_encoder_dict = dict()
        # using regular label encoder because its illegal to give the model labels that it didnt trained on

    def fit(self, X, y=None):

        X = self.prepare_X(X)

        for c in X.columns:
            # TODO raise warning when there are to many unique values
            self.x_label_encoder_dict[c] = LabelEncoderExt()
            label_encoded_col = self.x_label_encoder_dict[c].fit_transform(X[c])
            self.x_one_hot_encoder_dict[c] = OneHotEncoder(handle_unknown='ignore')
            self.x_one_hot_encoder_dict[c].fit(label_encoded_col.reshape(-1, 1))

        if y:
            # TODO raise warning that y will not be encoded
            pass

        return self

    def transform(self, X):
        X = self.prepare_X(X)
        orig_cols = X.columns
        for c in orig_cols:
            # TODO raise warning when there are to many unique values
            label_encoded_col = self.x_label_encoder_dict[c].transform(X[c])
            one_hot_cols = self.x_one_hot_encoder_dict[c].transform(label_encoded_col.reshape(-1, 1))
            one_hot_df = pd.DataFrame(one_hot_cols.toarray()).astype('uint8').add_prefix(f'{c}_').reset_index(drop=True)
            X = pd.concat([X, one_hot_df], axis=1)

        X.drop(orig_cols, axis=1, inplace=True)
        return X

    @staticmethod
    def prepare_X(X) -> pd.DataFrame: # noqa
        if isinstance(X, pd.DataFrame):
            return X.copy()
        else:
            return pd.DataFrame(X)
