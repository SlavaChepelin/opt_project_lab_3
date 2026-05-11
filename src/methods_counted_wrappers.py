from optimization_counted import *
from methods_wrappers import *
def subgradient_wrapper_counted(data_kind, X, y, lambda_reg, tolerance, max_iter, trace, **kwargs):
    oracle = make_nonsmooth_oracle(data_kind, X, y, lambda_reg)
    x0 = np.zeros(X.shape[1])
    x_star, msg, hist = subgradient_method_counted(oracle, x0,
                                                   tolerance=tolerance,
                                                   max_iter=max_iter,
                                                   alpha_0=1.0,
                                                   trace=trace)
    return x_star, msg, hist

def ista_wrapper_counted(data_kind, X, y, lambda_reg, tolerance, max_iter, trace, **kwargs):
    oracle = make_prox_oracle(data_kind, X, y, lambda_reg)
    x0 = np.zeros(X.shape[1])
    x_star, msg, hist = proximal_gradient_method_counted(oracle, x0,
                                                         L_0=1.0,
                                                         tolerance=tolerance,
                                                         max_iter=max_iter,
                                                         trace=trace)
    return x_star, msg, hist

def fista_wrapper_counted(data_kind, X, y, lambda_reg, tolerance, max_iter, trace, **kwargs):
    oracle = make_prox_oracle(data_kind, X, y, lambda_reg)
    x0 = np.zeros(X.shape[1])
    x_star, msg, hist = proximal_fast_gradient_method_counted(oracle, x0,
                                                              L_0=1.0,
                                                              tolerance=tolerance,
                                                              max_iter=max_iter,
                                                              trace=trace)
    return x_star, msg, hist

def frank_wolfe_wrapper_counted(data_kind, X, y, R, tolerance, max_iter, trace,
                                step_size_strategy='standard', **kwargs):
    oracle = make_smooth_oracle(data_kind, X, y)
    x0 = np.zeros(X.shape[1])
    x_star, msg, hist = frank_wolfe_method_counted(oracle, x0, R,
                                                   tolerance=tolerance,
                                                   max_iter=max_iter,
                                                   step_size_strategy=step_size_strategy,
                                                   trace=trace)
    return x_star, msg, hist

def barrier_wrapper_counted(data_kind, X, y, lambda_reg, tolerance_inner, tolerance_outer,
                            max_outer_iter, max_inner_iter, trace, **kwargs):
    oracle = make_smooth_oracle(data_kind, X, y)
    n = X.shape[1]
    x0 = np.zeros(n)
    u0 = np.ones(n) * 1.0
    x_star, u_star, msg, hist = barrier_method_counted(oracle, x0, u0, lambda_reg,
                                                       t_0=1.0, mu=10.0,
                                                       tolerance_inner=tolerance_inner,
                                                       tolerance_outer=tolerance_outer,
                                                       max_iter=max_outer_iter,
                                                       max_inner_iter=max_inner_iter,
                                                       trace=trace)
    return x_star, msg, hist