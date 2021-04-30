# -*- coding: utf-8 -*-
from sklearn.utils import estimator_checks
from unofficial_encoders.label_encoder_ext import LabelEncoderExt
from unofficial_encoders.one_hot_encoder_ext import OneHotEncoderExt


@estimator_checks.parametrize_with_checks([LabelEncoderExt(), OneHotEncoderExt()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
