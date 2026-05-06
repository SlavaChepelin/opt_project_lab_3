import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class NonConvexOracle(BaseSmoothOracle):
    """
    Oracle for test function from your assignment.
    """

    def __init__(self):
        return

    def func(self, x):
        x, y = x[0], x[1]
        term1 = x ** 2 + y - 11
        term2 = x + y ** 2 - 7
        return term1 ** 2 + term2 ** 2

    def grad(self, x):
        x, y = x[0], x[1]
        dx = 4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7)
        dy = 2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)
        return np.array([dx, dy])

    def hess(self, x):
        x, y = x[0], x[1]
        term1 = x ** 2 + y - 11
        term2 = x + y ** 2 - 7

        d2f_dx1x1 = 4 * term1 + 8 * x ** 2 + 2
        d2f_dx1x2 = 4 * x + 4 * y
        d2f_dx2x1 = 4 * x + 4 * y
        d2f_dx2x2 = 2 + 4 * term2 + 8 * y ** 2

        return np.array([[d2f_dx1x1, d2f_dx1x2],
                         [d2f_dx2x1, d2f_dx2x2]])


class LogCoshL2Oracle(BaseSmoothOracle):
    """
    Oracle for regression loss  function with l2 regularization:
         check your individual assignment

    Let A and b be parameters of the model (feature matrix
    and labels vector respectively).   

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        Ax = self.matvec_Ax(x)
        error = Ax - self.b

        # ln (cosh (z)) = ln ((e^z - e^-z) / 2) = ln(e^z - e^-z) - ln(2) = |z| + ln(1 - e^-2z) - ln(2)
        loss = np.abs(error) + np.log1p(np.exp(-2 * np.abs(error))) - np.log(2)

        return np.mean(loss) + (self.regcoef / 2) * np.linalg.norm(x) ** 2

    def grad(self, x):
        z = self.matvec_Ax(x) - self.b
        m = len(self.b)

        return self.matvec_ATx(np.tanh(z)) / m + self.regcoef * x

    def hess(self, x):
        z = self.matvec_Ax(x) - self.b
        m = len(self.b)

        return self.matmat_ATsA((1.0 - np.tanh(z) ** 2) / m) + self.regcoef * np.eye(len(x))

    def hess_vec(self, x, v):
        z = self.matvec_Ax(x) - self.b
        m = len(self.b)

        # D = (1 - tanh^2(z)) / m
        diag = (1.0 - np.tanh(z) ** 2) / m

        # H * v = A^T * D * A * v + regcoef * v
        return self.matvec_ATx(diag * self.matvec_Ax(v)) + self.regcoef * v


class ExpLossL2Oracle(BaseSmoothOracle):
    """
    Oracle for classification loss  function with l2 regularization:
         check your individual assignment

    Let A and b be parameters of the model (feature matrix
    and labels vector respectively).   

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        margins = self.b * self.matvec_Ax(x)

        return np.mean(np.exp(np.clip(-margins, None, 100))) + (self.regcoef / 2) * np.linalg.norm(x) ** 2

    def grad(self, x):
        m = len(self.b)
        margins = self.b * self.matvec_Ax(x)

        exp_val = np.exp(np.clip(-margins, None, 100))

        return (self.matvec_ATx(-(self.b * exp_val)) / m) + self.regcoef * x

    def hess(self, x):
        m = len(self.b)
        margins = self.b * self.matvec_Ax(x)

        exp_val = np.exp(np.clip(-margins, None, 100))

        hess_matrix = self.matmat_ATsA(np.asarray(exp_val).reshape(-1) / m)

        return hess_matrix + self.regcoef * np.eye(len(x))

    def hess_vec(self, x, v):
        m = len(self.b)
        margins = self.b * self.matvec_Ax(x)
        exp_val = np.exp(np.clip(-margins, None, 100))

        # D = exp_val / m
        diag = exp_val / m

        # H * v = A^T * diag(b) * D * diag(b) * A * v + regcoef * v
        return self.matvec_ATx(self.b * (diag * self.b * self.matvec_Ax(v))) + self.regcoef * v


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    grad = np.zeros(n)
    fx = func(x)

    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (func(x_plus) - fx) / eps

    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    hess = np.zeros((n, n))
    fx = func(x)

    f_plus = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus[i] = func(x_plus)

    for i in range(n):
        for j in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_plus[j] += eps
            hess[i][j] = (func(x_plus) - f_plus[i] - f_plus[j] + fx) / (eps ** 2)

    return hess


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Verification of the correctness of the Hessian-vector product calculation for a given vector $v$ using finite differences.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(x)
    hess_vec = np.zeros(n)

    fx = func(x)
    fx_eps_v = func(x + eps * v)

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0

        term1 = func(x + eps * v + eps * e_i)
        term2 = fx_eps_v
        term3 = func(x + eps * e_i)

        hess_vec[i] = (term1 - term2 - term3 + fx) / (eps ** 2)

    return hess_vec
