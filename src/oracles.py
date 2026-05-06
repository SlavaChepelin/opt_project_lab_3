import numpy as np
from scipy.special import expit
from scipy.sparse import isspmatrix_dia

class BaseSmoothOracle:
    def func(self, x):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    def __init__(self, A, b):
        if not isspmatrix_dia(A) and not np.allclose(A, A.T):
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
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        Ax = self.matvec_Ax(x)
        error = Ax - self.b
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
        diag = (1.0 - np.tanh(z) ** 2) / m
        return self.matvec_ATx(diag * self.matvec_Ax(v)) + self.regcoef * v


class ExpLossL2Oracle(BaseSmoothOracle):
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
        diag = exp_val / m
        return self.matvec_ATx(self.b * (diag * self.b * self.matvec_Ax(v))) + self.regcoef * v


def grad_finite_diff(func, x, eps=1e-8):
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


class BaseProxOracle:
    """Базовый класс для прокс-части h(x) в f(x) + h(x)."""
    def func(self, x):
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle:
    """phi(x) := f(x) + h(x), где f – гладкая, h – простая."""
    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        return self._f.grad(x)

    def prox(self, x, alpha):
        return self._h.prox(x, alpha)


class BaseNonsmoothConvexOracle:
    """Базовый класс для негладкой выпуклой функции."""
    def func(self, x):
        raise NotImplementedError('Func is not implemented.')

    def subgrad(self, x):
        raise NotImplementedError('Subgrad is not implemented.')


class L1RegOracle(BaseProxOracle):
    """Оракул для L1-регуляризатора h(x) = regcoef * ||x||_1."""
    def __init__(self, regcoef):
        self.regcoef = regcoef

    def func(self, x):
        return self.regcoef * np.linalg.norm(x, 1)

    def prox(self, x, alpha):
        # soft-thresholding
        threshold = alpha * self.regcoef
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

    def lmo(self, grad, radius):
        """
        Линейный минимизационный оракул для L1-шара радиуса radius.
        Находит y = argmin_{||y||_1 <= radius} <grad, y>.
        """
        y = np.zeros_like(grad)
        if radius <= 0:
            return y
        idx = np.argmax(np.abs(grad))
        y[idx] = -radius * np.sign(grad[idx])
        return y


class BarrierL1Oracle:
    """
    Оракул для барьерной функции во внутреннем цикле барьерного метода:
    F_t(x, u) = t * (f(x) + lambda * sum(u)) - sum(log(u - x)) - sum(log(u + x)).
    Состояние z = [x, u].
    """
    def __init__(self, smooth_oracle, lambda_reg, t):
        self.smooth_oracle = smooth_oracle
        self.lambda_reg = lambda_reg
        self.t = t

    def _split(self, z):
        n = len(z) // 2
        return z[:n], z[n:]

    def func(self, z):
        x, u = self._split(z)
        # f(x) + lambda * sum(u)
        f_val = self.smooth_oracle.func(x)
        barrier = -np.sum(np.log(u - x)) - np.sum(np.log(u + x))
        return self.t * (f_val + self.lambda_reg * np.sum(u)) + barrier

    def grad(self, z):
        x, u = self._split(z)
        grad_f = self.smooth_oracle.grad(x)
        inv_diff = 1.0 / (u - x)
        inv_sum = 1.0 / (u + x)

        grad_x = self.t * grad_f + inv_diff - inv_sum
        grad_u = self.t * self.lambda_reg - inv_diff - inv_sum
        return np.concatenate([grad_x, grad_u])

    def hess(self, z):
        x, u = self._split(z)
        hess_f = self.smooth_oracle.hess(x)
        inv_diff2 = 1.0 / (u - x) ** 2
        inv_sum2 = 1.0 / (u + x) ** 2

        # Блоки гессиана
        hess_xx = self.t * hess_f + np.diag(inv_diff2 + inv_sum2)
        hess_xu = np.diag(-inv_diff2 + inv_sum2)
        hess_uu = np.diag(inv_diff2 + inv_sum2)

        top = np.hstack([hess_xx, hess_xu])
        bottom = np.hstack([hess_xu, hess_uu])  # hess_ux = hess_xu^T, но диагональ симметрична
        return np.vstack([top, bottom])


class RegressionSmoothOracle(BaseSmoothOracle):
    """
    Гладкая функция потерь для регрессии: logcosh-потери без регуляризации.
    Обёртка над LogCoshL2Oracle с regcoef=0.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b):
        self._oracle = LogCoshL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=0.0)

    def func(self, x):
        return self._oracle.func(x)

    def grad(self, x):
        return self._oracle.grad(x)

    def hess(self, x):
        return self._oracle.hess(x)


class ClassificationSmoothOracle(BaseSmoothOracle):
    """
    Гладкая функция потерь для классификации: exp-потери без регуляризации.
    Обёртка над ExpLossL2Oracle с regcoef=0.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b):
        self._oracle = ExpLossL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=0.0)

    def func(self, x):
        return self._oracle.func(x)

    def grad(self, x):
        return self._oracle.grad(x)

    def hess(self, x):
        return self._oracle.hess(x)


class RegressionNonsmoothOracle(BaseNonsmoothConvexOracle):
    """regression_loss (logcosh) + regcoef * ||x||_1."""
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self._smooth = LogCoshL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=0.0)
        self.regcoef = regcoef

    def func(self, x):
        return self._smooth.func(x) + self.regcoef * np.linalg.norm(x, 1)

    def subgrad(self, x):
        g = self._smooth.grad(x)
        # Субградиент L1-нормы: sign(x), при x_i=0 берём 0
        return g + self.regcoef * np.sign(x)


class ClassificationNonsmoothOracle(BaseNonsmoothConvexOracle):
    """classification_loss (exp) + regcoef * ||x||_1."""
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self._smooth = ExpLossL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=0.0)
        self.regcoef = regcoef

    def func(self, x):
        return self._smooth.func(x) + self.regcoef * np.linalg.norm(x, 1)

    def subgrad(self, x):
        g = self._smooth.grad(x)
        return g + self.regcoef * np.sign(x)


class RegressionProxOracle(BaseCompositeOracle):
    """
    Прокс-оракул для regression_loss + regcoef * ||x||_1.
    Гладкая часть: logcosh-потери; прокс-часть: L1-норма.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=1.0):
        f = RegressionSmoothOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b)
        h = L1RegOracle(regcoef)
        super().__init__(f, h)


class ClassificationProxOracle(BaseCompositeOracle):
    """
    Прокс-оракул для classification_loss + regcoef * ||x||_1.
    Гладкая часть: exp-потери; прокс-часть: L1-норма.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=1.0):
        f = ClassificationSmoothOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b)
        h = L1RegOracle(regcoef)
        super().__init__(f, h)