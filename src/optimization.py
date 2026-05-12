from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from time import time

from oracles import L1RegOracle, BarrierL1Oracle


def subgradient_method(oracle, x_0, tolerance=1e-5, max_iter=1000, alpha_0=1.0,
                       display=False, trace=False):
    x = x_0.copy()
    best_x = x.copy()
    best_f = oracle.func(x)
    history = defaultdict(list) if trace else None
    start_time = time()

    for k in range(max_iter):
        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(best_f)
            history['x'].append(x.copy())

        g = oracle.subgrad(x)
        alpha = alpha_0 / np.sqrt(k + 1)
        x_new = x - alpha * g

        if np.linalg.norm(x_new - x) / max(1.0, np.linalg.norm(x)) <= tolerance:
            if trace:
                history['time'].append(time() - start_time)
                history['func'].append(oracle.func(x_new))
                history['x'].append(x_new.copy())
            return best_x, 'success', history

        x = x_new
        f_new = oracle.func(x)
        if f_new < best_f:
            best_f = f_new
            best_x = x.copy()

    return best_x, 'iterations_exceeded', history


def proximal_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                             max_iter=1000, trace=False, display=False):
    x = x_0.copy()
    L = L_0
    history = defaultdict(list) if trace else None
    start_time = time()

    for k in range(max_iter):
        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(oracle.func(x))
            history['x'].append(x.copy())

        while True:
            grad_f = oracle.grad(x)
            y = x - grad_f / L
            x_new = oracle.prox(y, 1.0 / L)

            f_x = oracle.f(x)
            f_new = oracle.f(x_new)
            diff = x_new - x
            rhs = f_x + np.dot(grad_f, diff) + (L / 2) * np.dot(diff, diff)
            if f_new <= rhs:
                break
            else:
                L *= 2.0
        L /= 2.0

        if np.linalg.norm(x_new - x) / max(1.0, np.linalg.norm(x)) <= tolerance:
            if trace:
                history['time'].append(time() - start_time)
                history['func'].append(oracle.func(x_new))
                history['x'].append(x_new.copy())
            return x_new, 'success', history

        x = x_new

    return x, 'iterations_exceeded', history


def proximal_fast_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                                  max_iter=1000, trace=False, display=False):
    x = x_0.copy()
    v = x.copy()
    A = 0.0
    L = L_0
    best_x = x.copy()
    best_f = oracle.func(x)
    history = defaultdict(list) if trace else None
    start_time = time()

    for k in range(max_iter):
        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(best_f)
            history['x'].append(x.copy())

        while True:
            discriminant = 1 + 4 * L * A
            a = (1 + np.sqrt(discriminant)) / (2.0 * L)
            A_new = A + a

            y = (A * x + a * v) / A_new

            grad_f = oracle.grad(y)
            x_new = oracle.prox(y - grad_f / L, 1.0 / L)

            f_y = oracle.f(y)
            f_new = oracle.f(x_new)
            diff = x_new - y
            rhs = f_y + np.dot(grad_f, diff) + (L / 2) * np.dot(diff, diff)
            if f_new <= rhs:
                break
            else:
                L *= 2.0

        v_new = v + (A_new / a) * (x_new - y)

        x_prev = x.copy()
        x = x_new
        v = v_new
        A = A_new
        L /= 2.0

        f_current = oracle.func(x)
        if f_current < best_f:
            best_f = f_current
            best_x = x.copy()

        f_y_total = oracle.func(y)
        if f_y_total < best_f:
            best_f = f_y_total
            best_x = y.copy()

        if k > 0 and np.linalg.norm(x - x_prev) / max(1.0, np.linalg.norm(x_prev)) <= tolerance:
            if trace:
                history['time'].append(time() - start_time)
                history['func'].append(best_f)
                history['x'].append(best_x.copy())
            return best_x, 'success', history

    return best_x, 'iterations_exceeded', history


def frank_wolfe_method(oracle, x_0, R, tolerance=1e-5, max_iter=1000,
                       step_size_strategy='standard', trace=False, display=False):
    x = x_0.copy()
    l1_oracle = L1RegOracle(regcoef=0.0)
    history = defaultdict(list) if trace else None
    start_time = time()

    for k in range(1, max_iter + 1):
        grad = oracle.grad(x)
        y = l1_oracle.lmo(grad, R)
        fw_gap = np.dot(grad, x - y)
        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(oracle.func(x))
            history['fw_gap'].append(fw_gap)
            history['x'].append(x.copy())
        if fw_gap <= tolerance:
            return x, 'success', history

        if step_size_strategy == 'standard':
            gamma = 2 / (k + 1)
        elif step_size_strategy == 'armijo':
            beta = 0.5
            c = 1e-4
            d = y - x
            f_x = oracle.func(x)
            grad_dot = np.dot(grad, d)
            gamma = 1.0
            while True:
                x_try = x + gamma * d
                if oracle.func(x_try) <= f_x + c * gamma * grad_dot:
                    break
                gamma *= beta
                if gamma < 1e-12:
                    break
        else:
            raise ValueError("Unknown step_size_strategy: " + step_size_strategy)

        x = x + gamma * (y - x)

    return x, 'iterations_exceeded', history


def barrier_method(oracle, x_0, u_0, lambda_reg, t_0=1.0, mu=10.0,
                   tolerance_inner=1e-6, tolerance_outer=1e-5,
                   max_iter=100, max_inner_iter=100,
                   trace=False, display=False):
    x = x_0.copy()
    u = u_0.copy()
    n = x.size
    t = t_0
    history = defaultdict(list) if trace else None
    start_time = time()
    inner_iters_count = []

    for outer_iter in range(max_iter):
        inner_oracle = BarrierL1Oracle(oracle, lambda_reg, t)
        z = np.concatenate([x, u])
        inner_iter = 0

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
            history['x'].append(x.copy())
            inner_iters_count.append(inner_iter + 1)
            history['inner_iters'] = inner_iters_count

        if 2 * n / t <= tolerance_outer:
            return x, u, 'success', history
        t *= mu

    return x, u, 'iterations_exceeded', history


def _soft_threshold(z, threshold):
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)


def _cd_column_stats_logcosh(Ax, col_rows, col_vals, b, m):
    z = Ax - b
    t = np.tanh(z)
    w = 1.0 - t ** 2
    g_i = np.sum(col_vals * t[col_rows]) / m
    hess_i = np.sum((col_vals ** 2) * w[col_rows]) / m
    L_i = max(hess_i, np.sum(col_vals ** 2) / m)
    return g_i, max(L_i, 1e-12)


def _cd_column_stats_exp(Ax, col_rows, col_vals, b, m):
    margins = b * Ax
    exp_val = np.exp(np.clip(-margins, None, 100))
    s = -b * exp_val
    w = exp_val * (b ** 2)
    g_i = np.sum(col_vals * s[col_rows]) / m
    hess_i = np.sum((col_vals ** 2) * w[col_rows]) / m
    L_i = max(hess_i, np.sum(col_vals ** 2) / m)
    return g_i, max(L_i, 1e-12)


def cyclic_coordinate_descent(X, y, data_kind, lambda_reg, x_0=None,
                              tolerance=1e-5, max_epochs=500, trace=False,
                              use_residual=True):
    from scipy.sparse import issparse, csc_matrix

    X = csc_matrix(X) if issparse(X) else np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, n = X.shape
    x = np.zeros(n) if x_0 is None else np.asarray(x_0, dtype=float).copy()

    if data_kind == 'binary':
        from libsvm_utils import ensure_binary_labels
        b = ensure_binary_labels(y)
        col_stats = _cd_column_stats_exp
    else:
        b = y
        col_stats = _cd_column_stats_logcosh

    from methods_wrappers import make_prox_oracle
    oracle = make_prox_oracle(data_kind, X, y, lambda_reg)

    Ax = X.dot(x) if use_residual else None
    history = defaultdict(list) if trace else None
    start_time = time()

    if trace:
        history['time'].append(time() - start_time)
        history['func'].append(oracle.func(x))

    for epoch in range(max_epochs):
        x_prev = x.copy()

        for i in range(n):
            if issparse(X):
                col = X.getcol(i)
                col_rows = col.nonzero()[0]
                col_vals = col.data
            else:
                col_vals = X[:, i]
                col_rows = np.arange(m)

            if use_residual:
                g_i, L_i = col_stats(Ax, col_rows, col_vals, b, m)
            else:
                Ax = X.dot(x)
                g_i, L_i = col_stats(Ax, col_rows, col_vals, b, m)

            v = x[i] - g_i / L_i
            x_i_new = _soft_threshold(v, lambda_reg / L_i)
            delta = x_i_new - x[i]
            x[i] = x_i_new

            if use_residual and delta != 0.0:
                Ax[col_rows] += delta * col_vals

        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(oracle.func(x))

        rel_change = np.linalg.norm(x - x_prev) / max(1.0, np.linalg.norm(x_prev))
        if rel_change <= tolerance:
            return x, 'success', history

    return x, 'iterations_exceeded', history
