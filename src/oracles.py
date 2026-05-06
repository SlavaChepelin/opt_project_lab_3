import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """Base class for smooth function."""
    def func(self, x):
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        raise NotImplementedError('Grad is not implemented.')

    def hess(self, x):
        raise NotImplementedError('Hess is not implemented.')


class BaseProxOracle(object):
    """Base class for proximal h(x)-part in a composite function f(x) + h(x)."""
    def func(self, x):
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle(object):
    """phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part."""
    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        return self._f.grad(x)

    def prox(self, x, alpha):
        return self._h.prox(x, alpha)


class BaseNonsmoothConvexOracle(object):
    """Base class for implementation of oracle for nonsmooth convex function."""
    def func(self, x):
        raise NotImplementedError('Func is not implemented.')

    def subgrad(self, x):
        raise NotImplementedError('Subgrad is not implemented.')


class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer.
        h(x) = regcoef * ||x||_1.
    """
    def __init__(self, regcoef):
        self.regcoef = regcoef

    def func(self, x):
        return self.regcoef * np.linalg.norm(x, 1)

    def prox(self, x, alpha):
        # TODO: Implement Soft-Thresholding
        pass

    def lmo(self, grad, radius):
        """
        Linear Minimization Oracle for L1-ball.
        Returns y_k = argmin_{||y||_1 <= R} <grad, y>
        (Required for Frank-Wolfe method)
        """
        # TODO: Implement
        pass


class BarrierL1Oracle(object):
    """
    Oracle for the barrier function in the inner loop of the barrier method:
    F_t(x, u) = t * (f(x) + lambda * sum(u_i)) - sum(ln(u_i - x_i)) - sum(ln(u_i + x_i))
    
    The state vector z is the concatenation of x and u: z = [x, u].
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
        # TODO: Implement F_t(x, u)
        pass

    def grad(self, z):
        x, u = self._split(z)
        # TODO: Implement gradient of F_t(x, u)
        pass

    def hess(self, z):
        x, u = self._split(z)
        # TODO: Implement Hessian of F_t(x, u)
        pass


class RegressionSmoothOracle(BaseSmoothOracle):
    """Smooth Oracle for your specific regression loss function."""
    # TODO: implement using your lab1 code.
    pass


class ClassificationSmoothOracle(BaseSmoothOracle):
    """Smooth Oracle for your specific classification loss function."""
    # TODO: implement using your lab1 code.
    pass


class RegressionNonsmoothOracle(BaseNonsmoothConvexOracle):
    """regression_loss + regcoef * ||x||_1."""
    # TODO: implement func and subgrad.
    pass


class ClassificationNonsmoothOracle(BaseNonsmoothConvexOracle):
    """classification_loss + regcoef * ||x||_1."""
    # TODO: implement func and subgrad.
    pass


class RegressionProxOracle(BaseCompositeOracle):
    """
    Oracle for regression_loss + regcoef * ||x||_1.
        f(x) = regression_loss is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """
    def __init__(self, *args, regcoef=1.0, **kwargs):
        # TODO: Instantiate your RegressionSmoothOracle here and pass to parent
        # f = RegressionSmoothOracle(*args, **kwargs)
        # h = L1RegOracle(regcoef)
        # super().__init__(f, h)
        pass


class ClassificationProxOracle(BaseCompositeOracle):
    """
    Oracle for classification_loss + regcoef * ||x||_1.
        f(x) = classification_loss is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """
    def __init__(self, *args, regcoef=1.0, **kwargs):
        # TODO: Instantiate your ClassificationSmoothOracle here and pass to parent
        # f = ClassificationSmoothOracle(*args, **kwargs)
        # h = L1RegOracle(regcoef)
        # super().__init__(f, h)
        pass