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

    def fit_model(self, train_x, train_y, valid_x, valid_y, eval_metric, cv_verbosity):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def predict_probability(self, x, num_iteration):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def get_features_importance(self):
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

    def fit_model(self, train_x, train_y, valid_x, valid_y, eval_metric, cv_verbosity):
        """
        This method is responsible for fitting of LGBM estimator.
        :param train_x: train dataset with features (not including the target)
        :param train_y: train dataset with target column
        :param valid_x: validation dataset with features (not including the target)
        :param valid_y: validation dataset with target column
        :param eval_metric: 'rmse', 'mae', 'logloss', 'auc', etc.
        :param cv_verbosity: print info about CV training and validation errors every x iterations (e.g. 1000)
        :return: fitted LGBM estimator
        """
        self.estimator.fit(train_x, train_y,
                           eval_set=[(train_x, train_y), (valid_x, valid_y)],
                           eval_metric=eval_metric,
                           verbose=cv_verbosity)

    def predict_probability(self, x, num_iteration=1000):
        return self.estimator.predict_proba(x, num_iteration=num_iteration)[:, 1]

    def get_best_iteration(self):
        return self.estimator.booster_.best_iteration

    def get_features_importance(self):
        """
        In LightGBM there are two types of features importance:
            - 'split': result contains numbers of times the feature is used in a model.
            - 'gain': result contains total gains of splits which use the feature.
        'Gain' is slightly more informative than the 'split' (thus it is proposed to be used in here by default).
        Note: self.estimator.feature_importances_ is not used here because it reflects 'split' importance by default.
        :return: features_names, features_importances
        """
        features_names = self.estimator.booster_.feature_name()
        features_importances = self.estimator.booster_.feature_importance(importance_type='gain')
        return features_names, features_importances


class XGBWrapper(ModelWrapper):
    """Class that wraps XGBoost"""
    HP_DATATYPES = {
        'learning_rate': lambda x: round(x, 3),
        'max_depth': lambda x: int(round(x, 0)),
        'reg_alpha': lambda x: round(max(x, 0), 2),
        'reg_lambda': lambda x: round(max(x, 0), 2),
        'subsample': lambda x: round(max(min(x, 1), 0), 2),
        'colsample_bytree': lambda x: round(max(min(x, 1), 0), 2),
        'min_child_weight': lambda x: int(round(x, 0)),
        'gamma': lambda x: round(x, 3),
    }

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

    def fit_model(self, train_x, train_y, valid_x, valid_y, eval_metric, cv_verbosity):
        """
        This method is responsible for fitting of XGB estimator. It should be noted that in xgboost v0.72 (the latest
        at the moment of creation of this code), there is no support of early_stopping_rounds directly from general dict
        of parameters (unlike in the case of LightGBM). Therefore, an explicit extraction of this parameter is
        implemented. If the early_stopping_rounds parameter is missing in the dict of parameters, None is used.
        :param train_x: train dataset with features (not including the target)
        :param train_y: train dataset with target column
        :param valid_x: validation dataset with features (not including the target)
        :param valid_y: validation dataset with target column
        :param eval_metric: 'rmse', 'mae', 'logloss', 'auc', etc.
        :param cv_verbosity: print info about CV training and validation errors every x iterations (e.g. 1000)
        :return: fitted XGB estimator
        """
        self.estimator.fit(train_x, train_y,
                           eval_set=[(train_x, train_y), (valid_x, valid_y)],
                           eval_metric=eval_metric,
                           early_stopping_rounds=self.params.get("early_stopping_rounds", None),
                           verbose=cv_verbosity)

    def predict_probability(self, x, num_iteration=1000):
        return self.estimator.predict_proba(x, ntree_limit=num_iteration)[:, 1]

    def get_best_iteration(self):
        return self.estimator.best_iteration

    def get_features_importance(self):
        """
        The default value of self.estimator.feature_importances_ in XGBoost is 'weight' (not 'gain').
        The following options are available for XGBoost:
            - 'weight': the number of times a feature is used to split the data across all trees.
            - 'gain': the average gain across all splits the feature is used in.
            - 'cover': the average coverage across all splits the feature is used in.
            - 'total_gain': the total gain across all splits the feature is used in.
            - 'total_cover': the total coverage across all splits the feature is used in.
        'Gain' is slightly more informative than the 'split' (thus it is proposed to be used in here by default).
        :return: features_names, features_importances
        """
        features_names = self.estimator.get_booster().get_score(importance_type='gain').keys()
        features_importances = self.estimator.get_booster().get_score(importance_type='gain').values()
        return features_names, features_importances


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

    def fit_model(self, train_x, train_y, valid_x, valid_y, eval_metric, cv_verbosity):
        """
        This method is responsible for fitting of Sklearn estimator. It should be noted that fit method in Sklearn API
        does not support valid_x, valid_y, eval_metric, cv_verbosity arguments (thus they are not used here).
        :param train_x: train dataset with features (not including the target)
        :param train_y: train dataset with target column
        :param valid_x: validation dataset with features (not including the target)
        :param valid_y: validation dataset with target column
        :param eval_metric: 'rmse', 'mae', 'logloss', 'auc', etc.
        :param cv_verbosity: print info about CV training and validation errors every x iterations (e.g. 1000)
        :return: fitted Sklearn estimator
        """
        self.estimator.fit(train_x, train_y)

    def predict_probability(self, x, num_iteration=1000):
        return self.estimator.predict_proba(x)[:, 1]

    def get_features_importance(self):
        """
        In Sklearn algorithms, which do have feature_importances_ attribute in base_estimator (e.g. AdaBoostClassifier,
        ExtraTreesClassifier, etc.), features_names are not available and are assigned explicitly from the column names.
        :return: features_names, features_importances
        """
        features_names = None
        features_importances = self.estimator.feature_importances_
        return features_names, features_importances


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

    def fit_model(self, train_x, train_y, valid_x, valid_y, eval_metric, cv_verbosity):
        self.estimator.fit(train_x, train_y)

    def predict_probability(self, x, num_iteration=1000):
        return self.estimator.predict_proba(x)[:, 1]

    def get_features_importance(self):
        return
