## FastMLFramework

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/minerva-ml/open-solution-home-credit/blob/master/LICENSE)

The main goal of this ML Framework is to put all re-usable code needed for typical ML tasks in a single library so to avoid re-writing the code for each unique ML task but rather to plug & play with the harness available in the library. This library comprises methods for the following ML steps:
- Automatic feature selection in cross-validation fashion. For now only the target-permutation method is available, but Boruta and sequential forward feature selection method are coming shortly.
- Automatic hyperparameter optimization. For now, only the Bayes optimization (bayes_opt library) is available, but Random Selection (of Sklearn) is coming shortly. Hyperparameter optimization is re-usable for all-levels models (including models stackers).
- Cross-validation and test prediction (for now stratified and not-stratified KFold are implemented). Implemented code allows also to perform bagging using series of different seeds and to persist a corresponding out-of-fold and test results which can then be re-used when building a stacker supermodel.
- Selection of the ML models. The framework has necessary wrappers for various algorithms such as LightGBM, XGBoost, Sklearn algorithms, etc. 
- Methods for EDA are also available: visualization of categorical and numerical features, plotting of the binned numerical data, density distributions, automatic detection of the differences between train and test data sets, methods for plotting a comparison between train and test features, etc.

## Usage
Will add the indications shortly.

## Other flags
So far it is Python 2.7 compatible, but it was coded in a way to be easily adapted to Python 3.5+ (though this should be accurately tested and any possible inconsistencies should be fixed).