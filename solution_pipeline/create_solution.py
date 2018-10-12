class BaseTask(object):
    def __init__(self):
        pass


class ClassificationTask(BaseTask):
    def __init__(self):
        super(BaseTask, self).__init__()


class RegressionTask(BaseTask):
    def __init__(self):
        super(BaseTask, self).__init__()


def get_wrapped_estimator(model, model_init_params):
    if model == 'lightgbm':
        from lightgbm import LGBMClassifier
        from modeling.model_wrappers import LightGBMWrapper
        return LightGBMWrapper(model=LGBMClassifier, params=model_init_params)
    elif model == 'xgboost':
        from xgboost import XGBClassifier
        from modeling.model_wrappers import XGBWrapper
        return XGBWrapper(model=XGBClassifier, params=model_init_params)
    elif model == 'et':
        from sklearn.ensemble import ExtraTreesClassifier
        from modeling.model_wrappers import SklearnWrapper
        SklearnWrapper(model=ExtraTreesClassifier, params=model_init_params, name='et')
    elif model == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        from modeling.model_wrappers import SklearnWrapper
        return SklearnWrapper(model=LogisticRegression, params=model_init_params, name='logreg')
    elif model == 'linear_regression':
        from sklearn.linear_model import LinearRegression
        from modeling.model_wrappers import SklearnWrapper
        return SklearnWrapper(model=LinearRegression, params=model_init_params, name='linreg')
    else:
        raise NotImplemented()


def load_feature_selector_class(feature_selector_method):
    if feature_selector_method == 'target_permutation':
        from modeling.feature_selection import FeatureSelectorByTargetPermutation
        return FeatureSelectorByTargetPermutation
    else:
        raise NotImplemented()


def load_hp_optimization_class(hpo_method):
    if hpo_method == 'bayes':
        from modeling.hyper_parameters_optimization import BayesHyperParamsOptimization
        return BayesHyperParamsOptimization
    else:
        raise NotImplemented()