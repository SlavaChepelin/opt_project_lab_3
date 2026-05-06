import numpy as np
import pytest
from scipy.linalg import LinAlgError
from src.oracles import QuadraticOracle
from src.optimization_first import newton

# ----------------------------------------------------------------------
# Test data
# ----------------------------------------------------------------------
def quad_oracle_simple():
    """Quadratic: f(x) = 0.5 * x^T A x, A = 2I, minimum at 0."""
    A = 2.0 * np.eye(2)
    b = np.zeros(2)
    return QuadraticOracle(A, b)

def quad_oracle_ill_conditioned():
    """Quadratic with condition number 100: A = diag([100, 1])"""
    A = np.diag([100.0, 1.0])
    b = np.zeros(2)
    return QuadraticOracle(A, b)

def quad_oracle_with_linear():
    """Quadratic: f(x) = 0.5 x^T A x - b^T x, A = [[2, 1],[1, 2]], b = [1, 1]."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 1.0])
    return QuadraticOracle(A, b)

class NonConvexOracle:
    """Simple non-convex function: f(x) = x[0]**2 - x[1]**2, Hessian indefinite."""
    def func(self, x):
        return x[0]**2 - x[1]**2
    def grad(self, x):
        return np.array([2*x[0], -2*x[1]])
    def hess(self, x):
        return np.array([[2.0, 0.0], [0.0, -2.0]])

# ----------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------
NEWTON_CASES = [
    ("simple", quad_oracle_simple(), np.array([1.0, 1.0]), 1e-8, 10,
     {'method': 'Constant', 'c': 1.0}, 'success'),
    ("ill_conditioned", quad_oracle_ill_conditioned(), np.array([1.0, 1.0]), 1e-8, 10,
     {'method': 'Constant', 'c': 1.0}, 'success'),
    ("linear_term", quad_oracle_with_linear(), np.array([0.0, 0.0]), 1e-8, 10,
     {'method': 'Constant', 'c': 1.0}, 'success'),
    ("armijo", quad_oracle_simple(), np.array([1.0, 1.0]), 1e-8, 10,
     {'method': 'Armijo', 'c1': 1e-4, 'alpha_0': 1.0}, 'success'),
    ("wolfe", quad_oracle_simple(), np.array([1.0, 1.0]), 1e-8, 10,
     {'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9, 'alpha_0': 1.0}, 'success'),
]

MAX_ITER_CASES = [
    ("max_iter_exceeded", quad_oracle_simple(), np.array([1.0, 1.0]), 1e-12, 1,
     {'method': 'Constant', 'c': 1.0}, 'iterations_exceeded'),
]

NON_POSITIVE_DEFINITE_CASES = [
    ("indefinite_hessian", NonConvexOracle(), np.array([1.0, 1.0]), 1e-8, 10,
     {'method': 'Constant', 'c': 1.0}, 'newton_direction_error'),
]

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name,oracle,x0,tol,max_iter,ls_options,expected_msg",
                         NEWTON_CASES)
def test_newton_convergence(name, oracle, x0, tol, max_iter, ls_options, expected_msg):
    x_star, message, _ = newton(oracle, x0, tolerance=tol, max_iter=max_iter,
                                 line_search_options=ls_options, trace=False)
    assert message == expected_msg
    if isinstance(oracle, QuadraticOracle):
        true_min = np.linalg.solve(oracle.A, oracle.b)
        # Newton with exact step should converge in 1 iteration to machine precision.
        # But due to line search fallback, allow a small tolerance.
        np.testing.assert_allclose(x_star, true_min, atol=tol)

@pytest.mark.parametrize("name,oracle,x0,tol,max_iter,ls_options,expected_msg",
                         MAX_ITER_CASES)
def test_newton_max_iter(name, oracle, x0, tol, max_iter, ls_options, expected_msg):
    x_star, message, _ = newton(oracle, x0, tolerance=tol, max_iter=max_iter,
                                 line_search_options=ls_options, trace=False)
    assert message == expected_msg

@pytest.mark.parametrize("name,oracle,x0,tol,max_iter,ls_options,expected_msg",
                         NON_POSITIVE_DEFINITE_CASES)
def test_newton_non_positive_definite(name, oracle, x0, tol, max_iter, ls_options, expected_msg):
    x_star, message, _ = newton(oracle, x0, tolerance=tol, max_iter=max_iter,
                                 line_search_options=ls_options, trace=False)
    assert message == expected_msg
