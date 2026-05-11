import numpy as np
from collections import defaultdict
from time import time
from numpy.linalg import norm
from oracles import L1RegOracle, BarrierL1Oracle


def subgradient_method_counted(oracle, x_0, tolerance=1e-5, max_iter=1000,
                               alpha_0=1.0, trace=False):
    x = x_0.copy()
    best_x = x.copy()
    best_f = oracle.func(x)
    history = defaultdict(list) if trace else None
    start_time = time()
    n_calls = 1                         # func(x0)

    for k in range(max_iter):
        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(best_f)
            history['oracle_calls'].append(n_calls)
            if x.size <= 2:
                history['x'].append(x.copy())

        g = oracle.subgrad(x)
        n_calls += 1                    # subgrad
        alpha = alpha_0 / np.sqrt(k + 1)
        x_new = x - alpha * g

        if norm(x_new - x) / max(1.0, norm(x)) <= tolerance:
            if trace:
                history['time'].append(time() - start_time)
                f_new = oracle.func(x_new)
                n_calls += 1
                history['func'].append(f_new)
                history['oracle_calls'].append(n_calls)
                if x_new.size <= 2:
                    history['x'].append(x_new.copy())
            return best_x, 'success', history

        x = x_new
        f_new = oracle.func(x)
        n_calls += 1
        if f_new < best_f:
            best_f = f_new
            best_x = x.copy()

    return best_x, 'iterations_exceeded', history


def proximal_gradient_method_counted(oracle, x_0, L_0=1.0, tolerance=1e-5,
                                     max_iter=1000, trace=False):
    x = x_0.copy()
    L = L_0
    history = defaultdict(list) if trace else None
    start_time = time()
    n_calls = 0

    for k in range(max_iter):
        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(oracle.func(x))
            n_calls += 1
            history['oracle_calls'].append(n_calls)
            if x.size <= 2:
                history['x'].append(x.copy())

        while True:
            grad_f = oracle.grad(x)
            n_calls += 1
            y = x - grad_f / L
            x_new = oracle.prox(y, 1.0 / L)

            f_x = oracle.f(x)
            n_calls += 1
            f_new = oracle.f(x_new)
            n_calls += 1
            diff = x_new - x
            rhs = f_x + np.dot(grad_f, diff) + (L / 2) * np.dot(diff, diff)
            if f_new <= rhs:
                break
            L *= 2.0
        L /= 2.0

        if norm(x_new - x) / max(1.0, norm(x)) <= tolerance:
            if trace:
                history['time'].append(time() - start_time)
                history['func'].append(oracle.func(x_new))
                n_calls += 1
                history['oracle_calls'].append(n_calls)
                if x_new.size <= 2:
                    history['x'].append(x_new.copy())
            return x_new, 'success', history

        x = x_new

    return x, 'iterations_exceeded', history


def proximal_fast_gradient_method_counted(oracle, x_0, L_0=1.0, tolerance=1e-5,
                                          max_iter=1000, trace=False):
    x = x_0.copy()
    v = x.copy()
    A = 0.0
    L = L_0
    best_x = x.copy()
    best_f = oracle.func(x)
    history = defaultdict(list) if trace else None
    start_time = time()
    n_calls = 1                         # initial func

    for k in range(max_iter):
        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(best_f)
            history['oracle_calls'].append(n_calls)
            if x.size <= 2:
                history['x'].append(x.copy())

        while True:
            discriminant = 1 + 4 * L * A
            a = (1 + np.sqrt(discriminant)) / (2.0 * L)
            A_new = A + a
            y = (A * x + a * v) / A_new

            grad_f = oracle.grad(y)
            n_calls += 1
            x_new = oracle.prox(y - grad_f / L, 1.0 / L)

            f_y = oracle.f(y)
            n_calls += 1
            f_new = oracle.f(x_new)
            n_calls += 1
            diff = x_new - y
            rhs = f_y + np.dot(grad_f, diff) + (L / 2) * np.dot(diff, diff)
            if f_new <= rhs:
                break
            L *= 2.0

        v_new = v + (A_new / a) * (x_new - y)
        x_prev = x.copy()
        x = x_new
        v = v_new
        A = A_new
        L /= 2.0

        f_current = oracle.func(x)
        n_calls += 1
        if f_current < best_f:
            best_f = f_current
            best_x = x.copy()

        f_y_total = oracle.func(y)
        n_calls += 1
        if f_y_total < best_f:
            best_f = f_y_total
            best_x = y.copy()

        if k > 0 and norm(x - x_prev) / max(1.0, norm(x_prev)) <= tolerance:
            if trace:
                history['time'].append(time() - start_time)
                history['func'].append(best_f)
                history['oracle_calls'].append(n_calls)
                if best_x.size <= 2:
                    history['x'].append(best_x.copy())
            return best_x, 'success', history

    return best_x, 'iterations_exceeded', history


def frank_wolfe_method_counted(oracle, x_0, R, tolerance=1e-5, max_iter=1000,
                               step_size_strategy='standard', trace=False):
    x = x_0.copy()
    l1_oracle = L1RegOracle(regcoef=0.0)
    history = defaultdict(list) if trace else None
    start_time = time()
    n_calls = 0

    for k in range(1, max_iter + 1):
        grad = oracle.grad(x)
        n_calls += 1
        y = l1_oracle.lmo(grad, R)
        fw_gap = np.dot(grad, x - y)

        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(oracle.func(x))
            n_calls += 1
            history['fw_gap'].append(fw_gap)
            history['oracle_calls'].append(n_calls)
            history['x'].append(x.copy())          # сохраняем всегда

        if fw_gap <= tolerance:
            return x, 'success', history

        if step_size_strategy == 'standard':
            gamma = 2 / (k + 1)
        elif step_size_strategy == 'armijo':
            beta = 0.5
            c = 1e-4
            d = y - x
            f_x = oracle.func(x)
            n_calls += 1
            grad_dot = np.dot(grad, d)
            gamma = 1.0
            while True:
                x_try = x + gamma * d
                f_try = oracle.func(x_try)
                n_calls += 1
                if f_try <= f_x + c * gamma * grad_dot:
                    break
                gamma *= beta
                if gamma < 1e-12:
                    break
        else:
            raise ValueError("Unknown step_size_strategy: " + step_size_strategy)

        x = x + gamma * (y - x)

    return x, 'iterations_exceeded', history


def barrier_method_counted(oracle, x_0, u_0, lambda_reg, t_0=1.0, mu=10.0,
                           tolerance_inner=1e-6, tolerance_outer=1e-5,
                           max_iter=100, max_inner_iter=100, trace=False):
    # Оборачиваем гладкий оракул счётчиком
    class CountedOracle:
        def __init__(self, oracle):
            self._oracle = oracle
            self.calls_func = 0
            self.calls_grad = 0
            self.calls_hess = 0

        def func(self, x):
            self.calls_func += 1
            return self._oracle.func(x)

        def grad(self, x):
            self.calls_grad += 1
            return self._oracle.grad(x)

        def hess(self, x):
            self.calls_hess += 1
            return self._oracle.hess(x)

        @property
        def total_calls(self):
            return self.calls_func + self.calls_grad + self.calls_hess

    counted = CountedOracle(oracle)
    x = x_0.copy()
    u = u_0.copy()
    n = x.size
    t = t_0
    history = defaultdict(list) if trace else None
    start_time = time()

    for outer_iter in range(max_iter):
        inner_oracle = BarrierL1Oracle(counted, lambda_reg, t)
        z = np.concatenate([x, u])

        for inner_iter in range(max_inner_iter):
            grad_z = inner_oracle.grad(z)
            if norm(grad_z) <= tolerance_inner:
                break

            hess_z = inner_oracle.hess(z)
            try:
                delta_z = np.linalg.solve(hess_z, -grad_z)
            except np.linalg.LinAlgError:
                delta_z = -grad_z

            alpha = 1.0
            beta = 0.5
            c = 1e-4
            while True:
                z_new = z + alpha * delta_z
                x_new = z_new[:n]
                u_new = z_new[n:]

                if np.any(u_new <= np.abs(x_new) + 1e-14):
                    alpha *= beta
                    continue

                f_old = inner_oracle.func(z)
                f_new = inner_oracle.func(z_new)
                if f_new <= f_old + c * alpha * np.dot(grad_z, delta_z):
                    break
                alpha *= beta
                if alpha < 1e-12:
                    break

            z = z_new
            x = z[:n]
            u = z[n:]

        if trace:
            history['time'].append(time() - start_time)
            outer_obj = oracle.func(x) + lambda_reg * np.linalg.norm(x, 1)
            history['func'].append(outer_obj)
            history['oracle_calls'].append(counted.total_calls)
            if x.size <= 2:
                history['x'].append(x.copy())

        if 2 * n / t <= tolerance_outer:
            return x, u, 'success', history

        t *= mu

    return x, u, 'iterations_exceeded', history