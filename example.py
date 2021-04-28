import pandas
import unofficial_encoders

df = pandas.read_csv('titanic.csv')
categorical_data = df[['Name', 'Sex', 'Ticket', 'Cabin']]

one_hot_encoder = unofficial_encoders.OneHotEncoderExt()
encoded_data = one_hot_encoder.fit_transform(categorical_data)
print(encoded_data)
