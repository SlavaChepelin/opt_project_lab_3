from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from time import time

from oracles import L1RegOracle, BarrierL1Oracle


def subgradient_method(oracle, x_0, tolerance=1e-5, max_iter=1000, alpha_0=1.0,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for the stopping criterion:
        ||x_{k+1} - x_k||_2 / max(1, ||x_k||_2) <= tolerance
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes (e.g., alpha_k = alpha_0 / sqrt(k+1)).
    display : bool
        If True, debug information is displayed during optimization.
    trace:  bool
        If True, the progress information is appended into history dictionary.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or 'iterations_exceeded'
    history : dictionary of lists or None
        - history['func'] : list of function values f(x_k)
        - history['time'] : list of floats, containing time in seconds passed from the start
        - history['x'] : list of np.arrays, containing the trajectory (ONLY STORE IF x.size <= 2)
    """
    x = x_0.copy()
    best_x = x.copy()
    best_f = oracle.func(x)
    history = defaultdict(list) if trace else None
    start_time = time()

    for k in range(max_iter):
        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(best_f)
            if x.size <= 2:
                history['x'].append(x.copy())

        g = oracle.subgrad(x)
        # alpha_k = alpha_0 / sqrt(k+1)
        alpha = alpha_0 / np.sqrt(k + 1)
        x_new = x - alpha * g

        # Stopping criterion
        if np.linalg.norm(x_new - x) / max(1.0, np.linalg.norm(x)) <= tolerance:
            if trace:
                history['time'].append(time() - start_time)
                history['func'].append(oracle.func(x_new))
                if x_new.size <= 2:
                    history['x'].append(x_new.copy())
            return best_x, 'success', history

        x = x_new
        # Best pointo
        f_new = oracle.func(x)
        if f_new < best_f:
            best_f = f_new
            best_x = x.copy()

    return best_x, 'iterations_exceeded', history


def proximal_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                             max_iter=1000, trace=False, display=False):
    """
    Proximal Gradient Method (ISTA) for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func(), .grad(), and .prox() methods implemented.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search (backtracking for Lipschitz constant).
    tolerance : float
        Epsilon value for the stopping criterion:
        ||x_{k+1} - x_k||_2 / max(1, ||x_k||_2) <= tolerance
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
    trace:  bool
        If True, the progress information is appended into history dictionary.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or 'iterations_exceeded'
    history : dictionary of lists or None
    """
    x = x_0.copy()
    L = L_0
    history = defaultdict(list) if trace else None
    start_time = time()

    for k in range(max_iter):
        # Save current history
        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(oracle.func(x))
            if x.size <= 2:
                history['x'].append(x.copy())

        # Backtracking for Lipschitz constant
        while True:
            grad_f = oracle.grad(x)
            y = x - grad_f / L
            x_new = oracle.prox(y, 1.0 / L)

            # f(x_new) <= f(x) + <grad f(x), x_new - x> + (L/2)||x_new - x||^2
            f_x = oracle.f(x)  # We assume oracle has _f() for smooth part
            f_new = oracle.f(x_new)
            diff = x_new - x
            rhs = f_x + np.dot(grad_f, diff) + (L / 2) * np.dot(diff, diff)
            if f_new <= rhs:
                break   # acceptable L
            else:
                L *= 2.0   # increase L and recompute
        L /= 2.0

        if np.linalg.norm(x_new - x) / max(1.0, np.linalg.norm(x)) <= tolerance:
            if trace:
                history['time'].append(time() - start_time)
                history['func'].append(oracle.func(x_new))
                if x_new.size <= 2:
                    history['x'].append(x_new.copy())
            return x_new, 'success', history

        x = x_new

    return x, 'iterations_exceeded', history

def proximal_fast_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                                  max_iter=1000, trace=False, display=False):
    """
    Fast gradient method (FISTA) for composite minimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func(), .grad(), and .prox() methods implemented.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search (backtracking for Lipschitz constant).
    tolerance : float
        Epsilon value for the stopping criterion:
        ||x_{k+1} - x_k||_2 / max(1, ||x_k||_2) <= tolerance
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
    trace:  bool
        If True, the progress information is appended into history dictionary.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or 'iterations_exceeded'
    history : dictionary of lists or None
        - history['func'] : list of objective function values phi(x_k)
        - history['time'] : list of floats, containing time in seconds passed from the start
    """
    x = x_0.copy()
    v = x.copy()          # inertial sequence
    A = 0.0               # accumulated weight
    L = L_0
    best_x = x.copy()
    best_f = oracle.func(x)
    history = defaultdict(list) if trace else None
    start_time = time()

    for k in range(max_iter):
        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(best_f)
            if x.size <= 2:
                history['x'].append(x.copy())

        while True:
            # a^2 = 1/L * (A + a)
            # La^2 -a-A
            # a_(k+1) = 1 + sqrt(1 + 4LA)/2L
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
                if best_x.size <= 2:
                    history['x'].append(best_x.copy())
            return best_x, 'success', history



    return best_x, 'iterations_exceeded', history


def frank_wolfe_method(oracle, x_0, R, tolerance=1e-5, max_iter=1000,
                       step_size_strategy='standard', trace=False, display=False):
    """
    Frank-Wolfe (Conditional Gradient) method for constrained optimization:
    min f(x) s.t. ||x||_1 <= R

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm (usually np.zeros).
    R : float
        Radius of the L1-ball constraint.
    tolerance : float
        Epsilon value for the Frank-Wolfe gap stopping criterion:
        <grad f(x_k), x_k - y_k> <= tolerance
    max_iter : int
        Maximum number of iterations.
    step_size_strategy : str
        'standard' (gamma_k = (k-1)/(k+1)) or 'armijo' (line search).
    display : bool
        If True, debug information is displayed during optimization.
    trace:  bool
        If True, the progress information is appended into history dictionary.

    Returns
    -------
    x_star : np.array
        The point found by the      # Also track the point y where we computed gradient (oracle was called)optimization procedure
    message : string
        'success' or 'iterations_exceeded'
    history : dictionary of lists or None
        - history['func'] : list of objective function values f(x_k)
        - history['time'] : list of floats, containing time in seconds passed from the start
        - history['fw_gap'] : list of Frank-Wolfe gaps <grad f(x_k), x_k - y_k>
    """
    x = x_0.copy()
    l1_oracle = L1RegOracle(regcoef=0.0)  # regcoef not used for lmo
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
            if x.size <= 2:
                history['x'].append(x.copy())
        if fw_gap <= tolerance:
            return x, 'success', history

        if step_size_strategy == 'standard':
            gamma = 2 / (k + 1)   # starts at 0 for k=1
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

        x =  x + (gamma) * (y-x)

    return x, 'iterations_exceeded', history


def barrier_method(oracle, x_0, u_0, lambda_reg, t_0=1.0, mu=10.0,
                   tolerance_inner=1e-6, tolerance_outer=1e-5,
                   max_iter=100, max_inner_iter=100,
                   trace=False, display=False):
    """
    Logarithmic barrier method for L1-regularized optimization.
    min f(x) + lambda * sum(u_i)  s.t.  -u_i <= x_i <= u_i

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        The SMOOTH oracle for f(x). Must have .func(), .grad(), and .hess().
    x_0 : 1-dimensional np.array
        Starting point for x.
    u_0 : 1-dimensional np.array
        Starting point for u. MUST satisfy strict feasibility: u_0_i > |x_0_i| for all i.
    lambda_reg : float
        L1 regularization coefficient.
    t_0 : float
        Initial value of the barrier parameter.
    mu : float
        Multiplication factor for t on each outer iteration.
    tolerance_inner : float
        Stopping criterion for the inner Newton method (norm of the gradient of F_t).
    tolerance_outer : float
        Stopping criterion for the outer loop: 2 * n / t <= tolerance_outer.
    max_iter : int
        Maximum number of outer iterations.
    max_inner_iter : int
        Maximum number of inner Newton iterations per outer step.
    trace : bool
        If True, the progress information is appended into history dictionary.

    Returns
    -------
    x_star : np.array
        The optimal x.
    u_star : np.array
        The optimal u.
    message : string
        'success' or 'iterations_exceeded'
    history : dictionary of lists or None
    """
    x = x_0.copy()
    u = u_0.copy()
    n = x.size
    t = t_0
    history = defaultdict(list) if trace else None
    start_time = time()

    for outer_iter in range(max_iter):
        inner_oracle = BarrierL1Oracle(oracle, lambda_reg, t)

        z = np.concatenate([x, u])

        for inner_iter in range(max_inner_iter):
            grad_z = inner_oracle.grad(z)
            # Inner stopping criterion: norm of gradient
            if norm(grad_z) <= tolerance_inner:
                break

            hess_z = inner_oracle.hess(z)
            try:
                delta_z = np.linalg.solve(hess_z, -grad_z)
            except np.linalg.LinAlgError:
                delta_z = -grad_z   # fallback to gradient step

            # Backtracking line search with feasibility guard
            alpha = 1.0
            beta = 0.5
            c = 1e-4
            while True:
                z_new = z + alpha * delta_z
                x_new = z_new[:n]
                u_new = z_new[n:]

                # Check strict feasibility: u_i > |x_i|
                if np.any(u_new <= np.abs(x_new) + 1e-14):
                    alpha *= beta
                    continue

                    # Armijo sufficient decrease
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
            if x.size <= 2:
                history['x'].append(x.copy())

        if 2 * n / t <= tolerance_outer:
            return x, u, 'success', history
        t *= mu

    return x, u, 'iterations_exceeded', history
