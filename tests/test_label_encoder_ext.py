from hypothesis import example, given, strategies
from hypothesis.extra.numpy import from_dtype
from hypothesis.extra.pandas import series
import numpy as np
import pandas as pd

from unofficial_encoders.label_encoder_ext import LabelEncoderExt  # noqa: I202


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
