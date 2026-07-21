# examples/specific_examples/make_explain_func.py
import pandas as pd

from xai_metrics.base import ExplainerContext
from xai_metrics.explainers.lime import LIMEExplainer


def make_lime_explain_func(background, y_background=None, params=None):
    if not isinstance(background, pd.DataFrame):
        background = pd.DataFrame(background)

    context = ExplainerContext(
        X_background=background,
        y_background=y_background,
    )

    return LIMEExplainer(context=context, params=params).as_explain_func()
