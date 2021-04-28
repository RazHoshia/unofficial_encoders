import pandas
import unofficial_encoders

df = pandas.read_csv('titanic.csv')

one_hot_encoder = unofficial_encoders.OneHotEncoderExt()
one_hot_encoder.fit(df)