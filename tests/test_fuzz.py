import numpy as np
from hypothesis import given
from hypothesis import strategies
from hypothesis.extra.pandas import column, columns, data_frames
from hypothesis.extra.numpy import from_dtype


@given(
    data_frames(
        columns=columns(5,
                        elements=from_dtype(np.dtype('uint32'), allow_nan=True)
                        ) +
                columns(5,
                        elements=from_dtype(np.dtype('str'), allow_nan=True)
                        )
    )
)
def test_one_hot_encoder(df):
    print(df)
