"""
Example file for label encoding using unofficial_encoder package.
"""

import pandas
from unofficial_encoders.label_encoder_ext import LabelEncoderExt  # noqa: I201

# using the classic titanic dataset from https://www.kaggle.com/c/titanic
df = pandas.read_csv('titanic.csv')

cabin_col = df['Cabin']

label_encoder = LabelEncoderExt()
encoded_data = label_encoder.fit_transform(cabin_col)
print(encoded_data[:20])  # prints the encoded column's first 50 values.
