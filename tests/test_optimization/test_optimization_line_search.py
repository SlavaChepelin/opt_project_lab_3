import numpy as np
import pytest
from src.oracles import QuadraticOracle, NonConvexOracle
from src.optimization_first import LineSearchTool

class Case:
    def __init__(self, name, x, d, method, alpha_0=None, c1=None, c2=None,
                 previous_alpha=None, expected_alpha_range=None,
                 expected_armijo=True, expected_wolfe=None):
        self.name = name
        self.x = x
        self.d = d
        self.method = method
        self.alpha_0 = alpha_0
        self.c1 = c1
        self.c2 = c2
        self.previous_alpha = previous_alpha
        self.expected_alpha_range = expected_alpha_range  # (min, max)
        self.expected_armijo = expected_armijo
        self.expected_wolfe = expected_wolfe  # True/False/None

    def __str__(self):
        return f"test_{self.name}"

# ----------------------------------------------------------------------
# Common test data
# ----------------------------------------------------------------------
def quad_oracle():
    A = np.array([[2.0, 0.5], [0.5, 1.0]])
    b = np.array([1.0, 2.0])
    return QuadraticOracle(A, b)

def x0_quad():
    return np.array([3.0, 1.0])

def d_quad(oracle, x):
    return -oracle.grad(x)   # descent direction

# ----------------------------------------------------------------------
# Test cases for constant method
# ----------------------------------------------------------------------
CONSTANT_CASES = [
    Case("constant_default", x0_quad(), d_quad(quad_oracle(), x0_quad()),
         method="Constant", alpha_0=0.5,
         expected_alpha_range=(0.5, 0.5)),
    Case("constant_with_previous", x0_quad(), d_quad(quad_oracle(), x0_quad()),
         method="Constant", alpha_0=1.0, previous_alpha=0.3,
         expected_alpha_range=(0.3, 0.3)),
]
# ----------------------------------------------------------------------
# Test cases for Armijo method
# ----------------------------------------------------------------------
def check_armijo(oracle, x, d, alpha, c1):
    phi0 = oracle.func_directional(x, d, 0.0)
    phi0_prime = oracle.grad_directional(x, d, 0.0)
    phi_alpha = oracle.func_directional(x, d, alpha)
    return phi_alpha <= phi0 + c1 * alpha * phi0_prime

ARMijo_CASES = [
    Case("armijo_default", x0_quad(), d_quad(quad_oracle(), x0_quad()),
         method="Armijo", alpha_0=1.0, c1=1e-4,
         expected_alpha_range=(0.0, 2.0), expected_armijo=True),
    Case("armijo_small_c1", x0_quad(), d_quad(quad_oracle(), x0_quad()),
         method="Armijo", alpha_0=1.0, c1=0.5,
         expected_alpha_range=(0.0, 2.0), expected_armijo=True),
    Case("armijo_with_previous", x0_quad(), d_quad(quad_oracle(), x0_quad()),
         method="Armijo", alpha_0=1.0, c1=1e-4, previous_alpha=0.1,
         expected_alpha_range=(0.0, 2.0), expected_armijo=True),
]

# ----------------------------------------------------------------------
# Test cases for Wolfe method (strong Wolfe)
# ----------------------------------------------------------------------
def check_wolfe(oracle, x, d, alpha, c1, c2):
    phi0 = oracle.func_directional(x, d, 0.0)
    phi0_prime = oracle.grad_directional(x, d, 0.0)
    phi_alpha = oracle.func_directional(x, d, alpha)
    phi_alpha_prime = oracle.grad_directional(x, d, alpha)
    armijo_ok = phi_alpha <= phi0 + c1 * alpha * phi0_prime
    curvature_ok = np.abs(phi_alpha_prime) <= c2 * np.abs(phi0_prime)
    return armijo_ok and curvature_ok

WOLFE_CASES = [
    Case("wolfe_default", x0_quad(), d_quad(quad_oracle(), x0_quad()),
         method="Wolfe", alpha_0=1.0, c1=1e-4, c2=0.9,
         expected_alpha_range=(0.0, 2.0), expected_armijo=True, expected_wolfe=True),
    Case("wolfe_small_c2", x0_quad(), d_quad(quad_oracle(), x0_quad()),
         method="Wolfe", alpha_0=1.0, c1=1e-4, c2=0.1,
         expected_alpha_range=(0.0, 2.0), expected_armijo=True, expected_wolfe=True),
]

# ----------------------------------------------------------------------
# Test case for non-descent direction
# ----------------------------------------------------------------------
def x0_nonconvex():
    return np.array([0.0, 0.0])

def d_ascent(oracle, x):
    # use gradient as direction (ascent)
    return oracle.grad(x)

NONDESCENT_CASE = Case("non_descent_direction", x0_nonconvex(), d_ascent(NonConvexOracle(), x0_nonconvex()),
                       method="Armijo", alpha_0=1.0, c1=1e-4,
                       expected_alpha_range=None)   # should not raise, but could return something

# ----------------------------------------------------------------------
# Test case for unsupported method
# ----------------------------------------------------------------------
UNSUPPORTED_CASE = Case("unsupported_method", x0_quad(), d_quad(quad_oracle(), x0_quad()),
                        method="FakeMethod")

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
@pytest.mark.parametrize("case", CONSTANT_CASES, ids=str)
def test_constant_step(case):
    oracle = quad_oracle()
    if case.method == "Constant":
        tool = LineSearchTool(method=case.method, c=case.alpha_0)
    else:
        tool = LineSearchTool(method=case.method, alpha_0=case.alpha_0,
                              c1=case.c1, c2=case.c2)
    alpha = tool.line_search(oracle, case.x, case.d, previous_alpha=case.previous_alpha)
    assert alpha is not None
    if case.expected_alpha_range is not None:
        lo, hi = case.expected_alpha_range
        assert lo <= alpha <= hi

@pytest.mark.parametrize("case", ARMijo_CASES, ids=str)
def test_armijo(case):
    oracle = quad_oracle()
    tool = LineSearchTool(method=case.method, alpha_0=case.alpha_0,
                          c1=case.c1 if case.c1 is not None else 1e-4,
                          c2=case.c2 if case.c2 is not None else 0.9)
    alpha = tool.line_search(oracle, case.x, case.d, previous_alpha=case.previous_alpha)
    assert alpha is not None
    if case.expected_alpha_range is not None:
        lo, hi = case.expected_alpha_range
        assert lo <= alpha <= hi
    if case.expected_armijo:
        assert check_armijo(oracle, case.x, case.d, alpha, tool.c1)

@pytest.mark.parametrize("case", WOLFE_CASES, ids=str)
def test_wolfe(case):
    oracle = quad_oracle()
    tool = LineSearchTool(method=case.method, alpha_0=case.alpha_0,
                          c1=case.c1 if case.c1 is not None else 1e-4,
                          c2=case.c2 if case.c2 is not None else 0.9)
    alpha = tool.line_search(oracle, case.x, case.d, previous_alpha=case.previous_alpha)
    assert alpha is not None
    if case.expected_alpha_range is not None:
        lo, hi = case.expected_alpha_range
        assert lo <= alpha <= hi
    if case.expected_armijo and case.expected_wolfe:
        assert check_wolfe(oracle, case.x, case.d, alpha, tool.c1, tool.c2)

def test_non_descent_direction():
    """Test that line_search handles non-descent direction gracefully."""
    oracle = NonConvexOracle()
    x = x0_nonconvex()
    d = d_ascent(oracle, x)  # this is ascent direction
    tool = LineSearchTool(method="Armijo", alpha_0=1.0)
    # Should not raise, returns something (alpha_0 or previous_alpha)
    alpha = tool.line_search(oracle, x, d)
    assert alpha is not None

def test_unsupported_method():
    """Test that unsupported method raises NotImplementedError."""
    oracle = quad_oracle()
    x = x0_quad()
    d = d_quad(oracle, x)
    with pytest.raises(ValueError):
        tool = LineSearchTool(method="FakeMethod")
        tool.line_search(oracle, x, d)

def test_very_small_initial_alpha():
    """Test that Armijo does not loop forever with tiny starting alpha."""
    oracle = quad_oracle()
    x = x0_quad()
    d = d_quad(oracle, x)
    tool = LineSearchTool(method="Armijo", alpha_0=1e-15, c1=1e-4)
    alpha = tool.line_search(oracle, x, d)
    assert alpha is not None
    assert alpha > 0
    # Should still produce a valid step (might be small but positive)
    assert check_armijo(oracle, x, d, alpha, tool.c1)