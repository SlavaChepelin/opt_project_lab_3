def cg_wrapper_quadratic(oracle, x0, tolerance, max_iter, trace, **kwargs):
    A = oracle.hess(x0)
    b = oracle.b
    from optimization import linear_conjugate_gradients
    x_star, message, history = linear_conjugate_gradients(
        matvec=lambda v: A @ v, b=b, x_0=x0,
        tolerance=tolerance, max_iter=max_iter, trace=trace
    )
    return x_star, message, history

def lbfgs_wrapper(oracle, x0, tolerance, max_iter, trace, **kwargs):
    from optimization import lbfgs
    return lbfgs(oracle, x0, tolerance=tolerance, max_iter=max_iter,
                 memory_size=10,
                 line_search_options={"method": "Wolfe", "c1": 1e-4, "c2": 0.9},
                 trace=trace)

def ncg_wrapper(oracle, x0, tolerance, max_iter, trace, **kwargs):
    from optimization import nonlinear_conjugate_gradients
    return nonlinear_conjugate_gradients(oracle, x0, tolerance=tolerance,
                                         max_iter=max_iter,
                                         line_search_options={"method": "Wolfe", "c1": 1e-4, "c2": 0.9},
                                         trace=trace)

def hfn_wrapper(oracle, x0, tolerance, max_iter, trace, **kwargs):
    from optimization import hessian_free_newton
    return hessian_free_newton(oracle, x0, tolerance=tolerance,
                               max_iter=max_iter,
                               line_search_options={"method": "Wolfe", "c1": 1e-4, "c2": 0.9},
                               trace=trace)