import numpy as np
from scipy.sparse import issparse


def _make_matmat_ATsA(X):
    """Создаёт matmat_ATsA для разреженной или плотной матрицы X."""
    if issparse(X):
        def matmat_ATsA(s):
            # X - sparse (m x n), s - array (m,)
            # X.T @ diag(s) @ X  ->  (n x n) dense
            X_scaled = X.multiply(s[:, None])          # поэлементное масштабирование
            return (X.T @ X_scaled).toarray()
        return matmat_ATsA
    else:
        def matmat_ATsA(s):
            return X.T @ (s[:, None] * X)
        return matmat_ATsA

def make_nonsmooth_oracle(data_kind, X, y, regcoef):
    matmat_ATsA = _make_matmat_ATsA(X)
    if data_kind == 'binary':
        from oracles import ClassificationNonsmoothOracle
        return ClassificationNonsmoothOracle(
            matvec_Ax=lambda x: X @ x,
            matvec_ATx=lambda x: X.T @ x,
            matmat_ATsA=matmat_ATsA,
            b=y,
            regcoef=regcoef
        )
    else:
        from oracles import RegressionNonsmoothOracle
        return RegressionNonsmoothOracle(
            matvec_Ax=lambda x: X @ x,
            matvec_ATx=lambda x: X.T @ x,
            matmat_ATsA=matmat_ATsA,
            b=y,
            regcoef=regcoef
        )


def make_prox_oracle(data_kind, X, y, regcoef):
    matmat_ATsA = _make_matmat_ATsA(X)
    if data_kind == 'binary':
        from oracles import ClassificationProxOracle
        return ClassificationProxOracle(
            matvec_Ax=lambda x: X @ x,
            matvec_ATx=lambda x: X.T @ x,
            matmat_ATsA=matmat_ATsA,
            b=y,
            regcoef=regcoef
        )
    else:
        from oracles import RegressionProxOracle
        return RegressionProxOracle(
            matvec_Ax=lambda x: X @ x,
            matvec_ATx=lambda x: X.T @ x,
            matmat_ATsA=matmat_ATsA,
            b=y,
            regcoef=regcoef
        )


def make_smooth_oracle(data_kind, X, y):
    matmat_ATsA = _make_matmat_ATsA(X)
    if data_kind == 'binary':
        from oracles import ClassificationSmoothOracle
        return ClassificationSmoothOracle(
            matvec_Ax=lambda x: X @ x,
            matvec_ATx=lambda x: X.T @ x,
            matmat_ATsA=matmat_ATsA,
            b=y
        )
    else:
        from oracles import RegressionSmoothOracle
        return RegressionSmoothOracle(
            matvec_Ax=lambda x: X @ x,
            matvec_ATx=lambda x: X.T @ x,
            matmat_ATsA=matmat_ATsA,
            b=y
        )


def subgradient_wrapper(data_kind, X, y, lambda_reg, tolerance, max_iter, trace, **kwargs):
    oracle = make_nonsmooth_oracle(data_kind, X, y, lambda_reg)
    x0 = np.zeros(X.shape[1])
    from optimization import subgradient_method
    x_star, msg, hist = subgradient_method(oracle, x0,
                                          tolerance=tolerance,
                                          max_iter=max_iter,
                                          alpha_0=1.0,
                                          trace=trace)
    return x_star, msg, hist


def ista_wrapper(data_kind, X, y, lambda_reg, tolerance, max_iter, trace, **kwargs):
    oracle = make_prox_oracle(data_kind, X, y, lambda_reg)
    x0 = np.zeros(X.shape[1])
    from optimization import proximal_gradient_method
    x_star, msg, hist = proximal_gradient_method(oracle, x0,
                                                L_0=1.0,
                                                tolerance=tolerance,
                                                max_iter=max_iter,
                                                trace=trace)
    return x_star, msg, hist


def fista_wrapper(data_kind, X, y, lambda_reg, tolerance, max_iter, trace, **kwargs):
    oracle = make_prox_oracle(data_kind, X, y, lambda_reg)
    x0 = np.zeros(X.shape[1])
    from optimization import proximal_fast_gradient_method
    x_star, msg, hist = proximal_fast_gradient_method(oracle, x0,
                                                     L_0=1.0,
                                                     tolerance=tolerance,
                                                     max_iter=max_iter,
                                                     trace=trace)
    return x_star, msg, hist


def frank_wolfe_wrapper(data_kind, X, y, R, tolerance, max_iter, trace,
                        step_size_strategy='standard', **kwargs):
    oracle = make_smooth_oracle(data_kind, X, y)
    x0 = np.zeros(X.shape[1])
    from optimization import frank_wolfe_method
    x_star, msg, hist = frank_wolfe_method(oracle, x0, R,
                                          tolerance=tolerance,
                                          max_iter=max_iter,
                                          step_size_strategy=step_size_strategy,
                                          trace=trace)
    return x_star, msg, hist


def barrier_wrapper(data_kind, X, y, lambda_reg, tolerance_inner, tolerance_outer,
                    max_outer_iter, max_inner_iter, trace, **kwargs):
    oracle = make_smooth_oracle(data_kind, X, y)
    n = X.shape[1]
    x0 = np.zeros(n)
    u0 = np.ones(n) * 1.0
    from optimization import barrier_method
    x_star, u_star, msg, hist = barrier_method(oracle, x0, u0, lambda_reg,
                                              t_0=1.0, mu=10.0,
                                              tolerance_inner=tolerance_inner,
                                              tolerance_outer=tolerance_outer,
                                              max_iter=max_outer_iter,
                                              max_inner_iter=max_inner_iter,
                                              trace=trace)
    return x_star, msg, hist