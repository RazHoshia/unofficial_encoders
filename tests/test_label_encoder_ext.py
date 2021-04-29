import pandas as pd
import numpy as np
from hypothesis import given, strategies, example
from hypothesis.extra.pandas import series
from hypothesis.extra.numpy import from_dtype

from unofficial_encoders import LabelEncoderExt


@given(
    series(
        elements=strategies.one_of(
            from_dtype(np.dtype('uint32')),
            from_dtype(np.dtype('str')),
            strategies.just(float('nan'))
        ),
    )
)
@example(pd.Series([np.nan, 1, 2]))
@example(pd.Series(['0', 0, '0', '0', '0', '0', '0',
                    '0']))  # by default number and str with the same value are considered the same class
def test_fuzz_pandas(s):
    encoder = LabelEncoderExt()
    encoded_s = encoder.fit_transform(s)
    assert len(s.astype(str).fillna(value=np.nan).unique()) == len(np.unique(encoded_s)) \
           == len(encoder.classes_) - 1  # sub 1 because there is always unknown class
