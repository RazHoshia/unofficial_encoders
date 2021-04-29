# -*- coding: utf-8 -*-
from sklearn.utils import estimator_checks
from unofficial_encoders import LabelEncoderExt, OneHotEncoderExt


@estimator_checks.parametrize_with_checks([LabelEncoderExt(), OneHotEncoderExt()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
