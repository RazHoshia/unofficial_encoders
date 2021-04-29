import pytest
import numpy as np
import pandas as pd
from hypothesis import given, example, strategies
from hypothesis.extra.pandas import data_frames
from hypothesis.extra.numpy import from_dtype

from unofficial_encoders import OneHotEncoderExt


@given(
    data_frames(
        rows=strategies.lists(elements=strategies.one_of(
            from_dtype(np.dtype('uint32')),
            from_dtype(np.dtype('str')),
            strategies.just(float('nan'))
            ),
        )
    )
)
@example(pd.DataFrame([None, np.nan, np.NaN, 'rak bibi']))  # check that None and np.nan are considered as same class
def test_fuzz_pandas(df):
    encoder = OneHotEncoderExt()
    if df.empty:
        with pytest.raises(ValueError):
            encoder.fit(encoder)
    else:
        encoded_df = encoder.fit_transform(df)
        for c in df.columns:
            unique_values = len(df[c].fillna(value=np.nan).unique())
            encoded_cols = [col for col in encoded_df if col.startswith(f'{c}_')]
            assert unique_values == len(encoded_cols)
