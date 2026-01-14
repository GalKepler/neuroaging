"""Brain age prediction utilities including bias correction."""

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone, is_classifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import check_cv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


# -----------------------------------------------------------------------------
# Bias Correction Methods
# -----------------------------------------------------------------------------


def beheshti_correction(y_train, y_tr_pred, y_te_pred, y_te_true):
    delta = y_tr_pred - y_train
    model = LinearRegression().fit(y_train.reshape(-1, 1), delta)
    alpha = model.coef_[0]
    beta = model.intercept_
    bias = alpha * y_te_true + beta
    y_te_corrected = y_te_pred - bias
    return y_te_corrected


# -----------------------------------------------------------------------------
# Cross-validation with Bias Correction
# -----------------------------------------------------------------------------


def _fit_predict_correct(estimator, X, y, train, test, fit_params, method, correction_func):
    """Fit on train, predict on test, apply bias correction."""
    est = clone(estimator)
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    # Subset fit_params for train indices
    fit_params_fold = {}
    for k, v in fit_params.items():
        if v is not None and hasattr(v, "__len__") and len(v) == len(y):
            fit_params_fold[k] = v[train]
        else:
            fit_params_fold[k] = v

    est.fit(X_train, y_train, **fit_params_fold)
    y_train_pred = getattr(est, method)(X_train)
    y_test_pred = getattr(est, method)(X_test)

    if correction_func is not None:
        y_test_pred = correction_func(y_train, y_train_pred, y_test_pred, y_test)

    return y_test_pred


def corrected_cross_val_predict(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    params=None,
    pre_dispatch="2*n_jobs",
    method="predict",
    correction_func=None,
):
    """
    Cross-validated predictions with optional bias correction applied per fold.

    This function works like sklearn's cross_val_predict, but applies the supplied
    correction_func after each split's model is fit.

    Parameters
    ----------
    estimator : estimator object
        The estimator to fit.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values.
    groups : array-like, optional
        Group labels for GroupKFold.
    cv : int or cross-validator
        Cross-validation splitting strategy.
    n_jobs : int, optional
        Number of jobs for parallel execution.
    verbose : int
        Verbosity level.
    params : dict, optional
        Parameters to pass to fit (e.g., sample_weight).
    pre_dispatch : str
        Controls parallel dispatch.
    method : str
        Method to call on estimator (default: 'predict').
    correction_func : callable, optional
        Function(y_train, y_train_pred, y_test_pred, y_test) -> corrected predictions.

    Returns
    -------
    predictions : ndarray
        Cross-validated predictions with bias correction.
    """
    X, y, groups = indexable(X, y, groups)
    if params is None:
        params = {}

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    splits = list(cv.split(X, y, groups))

    test_indices = np.concatenate([test for _, test in splits])
    if np.unique(test_indices).size != _num_samples(X):
        raise ValueError(
            "cross_val_predict only works for partition splits "
            "(all samples must appear in exactly one test set)"
        )

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)

    predictions = parallel(
        delayed(_fit_predict_correct)(
            estimator, X, y, train, test, params, method, correction_func
        )
        for train, test in splits
    )

    # Reconstruct predictions in original order
    inv_test_indices = np.argsort(test_indices)
    predictions = np.concatenate(predictions)[inv_test_indices]

    return predictions
