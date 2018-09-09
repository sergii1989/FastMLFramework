class ModelWrapper(object):
    def __init__(self, model, params=None, seed=27, model_name=None):
        self.model = model
        self.model_name = model_name
        self._verify_model_name_is_ok()
        self._add_seed_to_params(params, seed)
        self.params = params
        self.estimator = model(**params)

    def _verify_model_name_is_ok(self):
        if self.model_name is None:
            try:
                self.model_name = self.model.__name__
            except Exception as e:
                print('Please provide name of the model. Error %s encountered when trying self.model.__name__' % e)

    def _add_seed_to_params(self, params, seed):  # type: (dict, int) -> dict
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def reinit_model_with_new_params(self, new_params):  # type: (dict) -> None
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def reset_models_seed(self, seed):  # type: (int) -> None
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def fit_model(self, train_x, train_y, valid_x, valid_y, eval_metric, verbose, early_stopping_rounds):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def predict_probability(self, x, num_iteration):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def get_model_name(self):
        return self.model_name


class LightGBMWrapper(ModelWrapper):
    """Class that wraps LightGBM"""
    HP_DATATYPES = {
        'learning_rate': lambda x: round(x, 3),
        'max_depth': lambda x: int(round(x, 0)),
        'num_leaves': lambda x: int(round(x, 0)),
        'reg_alpha': lambda x: round(max(x, 0), 2),
        'reg_lambda': lambda x: round(max(x, 0), 2),
        'min_split_gain': lambda x: round(max(x, 0), 3),
        'subsample': lambda x: round(max(min(x, 1), 0), 2),
        'colsample_bytree': lambda x: round(max(min(x, 1), 0), 2),
        'min_child_weight': lambda x: int(round(x, 0))
    }

    def __init__(self, model, params=None, seed=27, name='lgbm'):
        super(LightGBMWrapper, self).__init__(model, params, seed, name)

    def _add_seed_to_params(self, params, seed):  # type: (dict, int) -> dict
        params['seed'] = seed
        return params

    def reinit_model_with_new_params(self, new_params):  # type: (dict) -> None
        self.params = new_params
        self.estimator = self.model(**self.params)

    def reset_models_seed(self, seed):  # type: (int) -> None
        self.params['seed'] = seed
        self.estimator = self.model(**self.params)

    def fit_model(self, train_x, train_y, valid_x, valid_y, eval_metric, verbose, early_stopping_rounds):
        self.estimator.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric=eval_metric)#, verbose=verbose, early_stopping_rounds=early_stopping_rounds)

    def predict_probability(self, x, num_iteration=1000):
        return self.estimator.predict_proba(x, num_iteration=num_iteration)[:, 1]

    def get_best_iteration(self):
        return self.estimator.best_iteration_


class XGBWrapper(ModelWrapper):
    """Class that wraps XGBoost"""

    def __init__(self, model, params=None, seed=27, name='xgb'):
        super(XGBWrapper, self).__init__(model, params, seed, name)

    def _add_seed_to_params(self, params, seed):  # type: (dict, int) -> dict
        params['seed'] = seed
        return params

    def reinit_model_with_new_params(self, new_params):  # type: (dict) -> None
        self.params = new_params
        self.estimator = self.model(**self.params)

    def reset_models_seed(self, seed):  # type: (int) -> None
        self.params['seed'] = seed
        self.estimator = self.model(**self.params)

    def fit_model(self, train_x, train_y, valid_x, valid_y, eval_metric, verbose, early_stopping_rounds):
        self.estimator.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric=eval_metric, verbose=verbose, early_stopping_rounds=early_stopping_rounds)

    def predict_probability(self, x, num_iteration=1000):
        return self.estimator.predict_proba(x, ntree_limit=num_iteration)[:, 1]

    def get_best_iteration(self):
        return self.estimator.best_iteration


class SklearnWrapper(ModelWrapper):
    """Class that wraps Sklearn ML algorithms (ET, LinerRegression, LogisticRegression, etc."""

    def __init__(self, model, params=None, seed=27, name=None):
        super(SklearnWrapper, self).__init__(model, params, seed, name)

    def _add_seed_to_params(self, params, seed):  # type: (dict, int) -> dict
        params['random_state'] = seed
        return params

    def reinit_model_with_new_params(self, new_params):  # type: (dict) -> None
        self.params = new_params
        self.estimator = self.model(**self.params)

    def reset_models_seed(self, seed):  # type: (int) -> None
        self.params['random_state'] = seed
        self.estimator = self.model(**self.params)

    def fit_model(self, train_x, train_y, valid_x, valid_y, eval_metric, verbose, early_stopping_rounds):
        self.estimator.fit(train_x, train_y)

    def predict_probability(self, x, num_iteration=1000):
        return self.estimator.predict_proba(x)[:, 1]


# TODO: Implement CatBoost wrapper
class CBWrapper(ModelWrapper):
    """Class that wraps CatBoost algorithm"""

    def __init__(self, model, params=None, seed=27, name='cb'):
        super(CBWrapper, self).__init__(model, params, seed, name)

    def _add_seed_to_params(self, params, seed):  # type: (dict, int) -> dict
        params['random_state'] = seed
        return params

    def reinit_model_with_new_params(self, new_params):  # type: (dict) -> None
        self.params = new_params
        self.estimator = self.model(**self.params)

    def reset_models_seed(self, seed):  # type: (int) -> None
        self.params['random_state'] = seed
        self.estimator = self.model(**self.params)

    def fit_model(self, train_x, train_y, valid_x, valid_y, eval_metric, verbose, early_stopping_rounds):
        self.estimator.fit(train_x, train_y)

    def predict_probability(self, x, num_iteration=1000):
        return self.estimator.predict_proba(x)[:, 1]

