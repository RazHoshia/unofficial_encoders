"""
Example file for one hot encoding using unofficial_encoder package.
"""

import pandas
from unofficial_encoders.one_hot_encoder_ext import OneHotEncoderExt  # noqa: I201

# using the classic titanic dataset from https://www.kaggle.com/c/titanic
df = pandas.read_csv('titanic.csv')

categorical_columns = ['Name', 'Sex', 'Ticket', 'Cabin']
categorical_data = df[categorical_columns]

one_hot_encoder = OneHotEncoderExt()
encoded_data = one_hot_encoder.fit_transform(categorical_data)
print(encoded_data.head(3))
