import numpy as np
import pytest

from src.oracles import NonConvexOracle, grad_finite_diff, hess_finite_diff

class Case:
    def __init__(self, name: str, x: np.ndarray, expected_func=None,
                 expected_grad=None, expected_hess=None):
        self.name = name
        self.x = x
        self.expected_func = expected_func
        self.expected_grad = expected_grad
        self.expected_hess = expected_hess

    def __str__(self) -> str:
        return f"test_{self.name}"

TEST_POINTS = [
    (np.array([3.0, 2.0]), 0.0),
    (np.array([-2.805118, 3.131312]), 0.0),
    (np.array([-3.779310, -3.283186]), 0.0),
    (np.array([3.584428, -1.848126]), 0.0),
]

NUMERICAL_POINTS = [
    np.array([0.0, 0.0]),
    np.array([1.0, 1.0]),
    np.array([-1.0, 2.0]),
    np.array([0.5, -1.5]),
]

def _numerical_grad(oracle, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (oracle.func(x_plus) - oracle.func(x_minus)) / (2 * eps)
    return grad


def _numerical_hess(oracle, x, eps=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_pp = x.copy()
            x_pp[i] += eps
            x_pp[j] += eps
            x_pm = x.copy()
            x_pm[i] += eps
            x_pm[j] -= eps
            x_mp = x.copy()
            x_mp[i] -= eps
            x_mp[j] += eps
            x_mm = x.copy()
            x_mm[i] -= eps
            x_mm[j] -= eps
            hess[i, j] = (oracle.func(x_pp) - oracle.func(x_pm) -
                          oracle.func(x_mp) + oracle.func(x_mm)) / (4 * eps * eps)
    return hess


# ----------------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------------

FUNC_CASES = [
    Case(name=f"min_{i+1}", x=x, expected_func=val)
    for i, (x, val) in enumerate(TEST_POINTS)
] + [
    Case(name="origin", x=np.array([0.0, 0.0]), expected_func=170.0),
    Case(name="ones", x=np.array([1.0, 1.0]), expected_func=106.0),
]

GRAD_CASES = []
for x in NUMERICAL_POINTS:
    oracle_tmp = NonConvexOracle()
    expected_grad = _numerical_grad(oracle_tmp, x)
    GRAD_CASES.append(Case(name=f"num_grad_{x[0]}_{x[1]}",
                           x=x, expected_grad=expected_grad))

HESS_CASES = []
for x in NUMERICAL_POINTS:
    oracle_tmp = NonConvexOracle()
    expected_hess = _numerical_hess(oracle_tmp, x)
    HESS_CASES.append(Case(name=f"num_hess_{x[0]}_{x[1]}",
                           x=x, expected_hess=expected_hess))

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("test_case", FUNC_CASES, ids=str)
def test_func(test_case: Case) -> None:
    oracle = NonConvexOracle()
    result = oracle.func(test_case.x)
    np.testing.assert_allclose(result, test_case.expected_func, rtol=1e-6,  atol=1e-10)


@pytest.mark.parametrize("test_case", GRAD_CASES, ids=str)
def test_grad(test_case: Case) -> None:
    oracle = NonConvexOracle()
    result = oracle.grad(test_case.x)
    np.testing.assert_allclose(result, test_case.expected_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("test_case", HESS_CASES, ids=str)
def test_hess(test_case: Case) -> None:
    oracle = NonConvexOracle()
    result = oracle.hess(test_case.x)
    np.testing.assert_allclose(result, test_case.expected_hess, rtol=1e-4, atol=1e-4)


# ----------------------------------------------------------------------------
# Tests for finite difference approximations
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("x", NUMERICAL_POINTS)
def test_grad_finite_diff(x: np.ndarray) -> None:
    oracle = NonConvexOracle()
    grad_analytical = oracle.grad(x)
    grad_fd = grad_finite_diff(oracle.func, x, eps=1e-8)
    np.testing.assert_allclose(grad_fd, grad_analytical, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("x", NUMERICAL_POINTS)
def test_hess_finite_diff(x: np.ndarray) -> None:
    oracle = NonConvexOracle()
    hess_analytical = oracle.hess(x)
    hess_fd = hess_finite_diff(oracle.func, x, eps=1e-5)
    np.testing.assert_allclose(hess_fd, hess_analytical, rtol=1e-4, atol=1e-4)


import numpy as np
import pytest
import scipy.sparse as sp
from src.oracles import LogCoshL2Oracle, ExpLossL2Oracle, grad_finite_diff, hess_finite_diff


# Помощник для создания matvec функций из обычной матрицы
def create_matvecs(A):
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    def matmat_ATsA(s):
        if sp.issparse(A):
            return A.T.dot(sp.diags(s) @ A)
        else:
            return A.T.dot(s[:, np.newaxis] * A)

    return matvec_Ax, matvec_ATx, matmat_ATsA


class OracleCase:
    def __init__(self, oracle_cls, m=20, n=10, reg=0.1, sparse=False):
        self.oracle_cls = oracle_cls
        self.m = m
        self.n = n
        self.reg = reg

        # Генерируем случайные данные
        A = np.random.randn(m, n)
        if sparse:
            A = sp.csr_matrix(A)

        # Для ExpLoss нужны метки {-1, 1}, для Log-Cosh сойдут любые
        if oracle_cls.__name__ == 'ExpLossL2Oracle':
            self.b = np.random.choice([-1, 1], size=m)
        else:
            self.b = np.random.randn(m)

        self.x = np.random.randn(n)

        # Создаем оракула
        mv_Ax, mv_ATx, mm_ATsA = create_matvecs(A)
        self.oracle = oracle_cls(mv_Ax, mv_ATx, mm_ATsA, self.b, self.reg)

    def __str__(self) -> str:
        return f"{self.oracle_cls.__name__}_s{int(sp.issparse(self.oracle.matmat_ATsA.__self__ if hasattr(self.oracle.matmat_ATsA, '__self__') else False))}"


# ----------------------------------------------------------------------------
# Параметризация: проверяем обоих оракулов на плотных и разреженных матрицах
# ----------------------------------------------------------------------------
ORACLE_TYPES = [LogCoshL2Oracle, ExpLossL2Oracle]
SPARSITY = [False, True]


@pytest.fixture
def random_oracle(request):
    oracle_cls, is_sparse = request.param
    return OracleCase(oracle_cls, sparse=is_sparse)


@pytest.mark.parametrize("random_oracle",
                         [(cls, s) for cls in ORACLE_TYPES for s in SPARSITY],
                         ids=lambda x: f"{x[0].__name__}_sparse={x[1]}",
                         indirect=True)
class TestNewOracles:

    def test_grad_against_finite_diff(self, random_oracle):
        """Проверка аналитического градиента через конечные разности"""
        case = random_oracle
        grad_analytical = case.oracle.grad(case.x)
        # Используем твою функцию из oracles.py
        grad_fd = grad_finite_diff(case.oracle.func, case.x, eps=1e-8)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-5, atol=1e-5)

    def test_hess_against_finite_diff(self, random_oracle):
        """Проверка аналитического гессиана через конечные разности"""
        case = random_oracle
        hess_analytical = case.oracle.hess(case.x)
        # Используем твою функцию из oracles.py
        hess_fd = hess_finite_diff(case.oracle.func, case.x, eps=1e-5)

        if isinstance(case.oracle, ExpLossL2Oracle):
            np.testing.assert_allclose(hess_analytical, hess_fd, rtol=1e-3, atol=1e-3)
        else:
            np.testing.assert_allclose(hess_analytical, hess_fd, rtol=1e-4, atol=1e-4)

    def test_numerical_stability(self, random_oracle):
        """Проверка, что оракул не выдает inf на больших весах (проверка клиппинга/трюков)"""
        case = random_oracle
        big_x = case.x * 100.0  # Очень большие веса

        f_val = case.oracle.func(big_x)
        g_val = case.oracle.grad(big_x)
        h_val = case.oracle.hess(big_x)

        assert np.all(np.isfinite(f_val)), "Value is NaN or Inf!"
        assert np.all(np.isfinite(g_val)), "Gradient is NaN or Inf!"
        assert np.all(np.isfinite(h_val)), "Hessian is NaN or Inf!"