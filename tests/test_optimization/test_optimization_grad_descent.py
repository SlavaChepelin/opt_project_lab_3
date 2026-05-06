import numpy as np
import pytest
from src.oracles import QuadraticOracle
from src.optimization_first import gradient_descent

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

# ----------------------------------------------------------------------
# Test cases - adjusted tolerances and max_iter for convergence
# ----------------------------------------------------------------------
CONSTANT_CASES = [
    ("constant_simple", quad_oracle_simple(), np.array([1.0, 1.0]), 1e-5, 200,
     {'method': 'Constant', 'c': 0.25}, 'success'),
    ("constant_ill", quad_oracle_ill_conditioned(), np.array([1.0, 1.0]), 1e-5, 10000,
     {'method': 'Constant', 'c': 0.01}, 'success'),
]

ARMIJO_CASES = [
    ("armijo_simple", quad_oracle_simple(), np.array([1.0, 1.0]), 1e-5, 200,
     {'method': 'Armijo', 'c1': 1e-4, 'alpha_0': 1.0}, 'success'),
    ("armijo_ill", quad_oracle_ill_conditioned(), np.array([1.0, 1.0]), 1e-5, 10000,
     {'method': 'Armijo', 'c1': 1e-4, 'alpha_0': 1.0}, 'success'),
    ("armijo_linear", quad_oracle_with_linear(), np.array([0.0, 0.0]), 1e-5, 500,
     {'method': 'Armijo', 'c1': 1e-4, 'alpha_0': 1.0}, 'success'),
]

WOLFE_CASES = [
    ("wolfe_simple", quad_oracle_simple(), np.array([1.0, 1.0]), 1e-5, 200,
     {'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9, 'alpha_0': 1.0}, 'success'),
]

MAX_ITER_CASES = [
    ("max_iter_exceeded", quad_oracle_simple(), np.array([1.0, 1.0]), 1e-12, 3,
     {'method': 'Constant', 'c': 0.001}, 'iterations_exceeded'),
]

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name,oracle,x0,tol,max_iter,ls_options,expected_msg",
                         CONSTANT_CASES + ARMIJO_CASES + WOLFE_CASES)
def test_gradient_descent_convergence(name, oracle, x0, tol, max_iter, ls_options, expected_msg):
    x_star, message, _ = gradient_descent(oracle, x0, tolerance=tol, max_iter=max_iter,
                                          line_search_options=ls_options, trace=False)
    assert message == expected_msg

    grad_norm = np.linalg.norm(oracle.grad(x_star))
    assert grad_norm**2 <= tol * np.linalg.norm(oracle.grad(x0))**2

@pytest.mark.parametrize("name,oracle,x0,tol,max_iter,ls_options,expected_msg",
                         MAX_ITER_CASES)
def test_gradient_descent_max_iter(name, oracle, x0, tol, max_iter, ls_options, expected_msg):
    """Check that max_iter termination works."""
    x_star, message, _ = gradient_descent(oracle, x0, tolerance=tol, max_iter=max_iter,
                                          line_search_options=ls_options, trace=False)
    assert message == expected_msg

def test_gradient_descent_computational_error():
    """Simulate a line search failure leading to computational_error."""
    from src.optimization_first import LineSearchTool
    original_line_search = LineSearchTool.line_search

    def mock_line_search(self, *args, **kwargs):
        return None

    try:
        LineSearchTool.line_search = mock_line_search
        oracle = quad_oracle_simple()
        x0 = np.array([1.0, 1.0])
        _, message, _ = gradient_descent(oracle, x0, tolerance=1e-8, max_iter=100,
                                         line_search_options={'method': 'Armijo'})
        assert message == 'computational_error'
    finally:
        LineSearchTool.line_search = original_line_search