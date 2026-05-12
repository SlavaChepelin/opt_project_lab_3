import matplotlib.pyplot as plt
import numpy as np
from time import time
from save_graphic import save_current_plot
from optimization import *
from methods_counted_wrappers import *

def find_lambda_for_sparsity(data_kind, X, y, target_sparsity=(0.5, 0.8), max_trials=200):
    best_lambda = None
    best_sparsity = 0
    for _ in range(max_trials):
        lam = 10 ** np.random.uniform(-4, 1)
        oracle = make_prox_oracle(data_kind, X, y, lam)
        x0 = np.zeros(X.shape[1])
        from optimization import proximal_fast_gradient_method
        x_star, _, _ = proximal_fast_gradient_method(oracle, x0, L_0=1.0,
                                                     tolerance=1e-7,
                                                     max_iter=2000, trace=False)
        sparsity = np.mean(np.abs(x_star) < 1e-8)
        if target_sparsity[0] <= sparsity <= target_sparsity[1]:
            return lam, sparsity
        if sparsity > best_sparsity:
            best_sparsity = sparsity
            best_lambda = lam
    return best_lambda, best_sparsity

def run_experiment_1(data_kind, data_name, prefix = "3_2"):
    from libsvm_utils import load_libsvm_scaled
    X, y, info = load_libsvm_scaled(data_kind, data_name)
    m, n = info['m'], info['n']

    lam, sparsity = find_lambda_for_sparsity(data_kind, X, y)

    oracle_ref = make_prox_oracle(data_kind, X, y, lam)
    x_ref, _, _ = proximal_fast_gradient_method(oracle_ref, np.zeros(n),
                                                L_0=1.0, tolerance=1e-7,
                                                max_iter=10000, trace=False)
    F_star = oracle_ref.func(x_ref)
    R_fw = np.linalg.norm(x_ref, 1)

    methods = {
        'Subgradient': lambda: subgradient_wrapper(data_kind, X, y, lam,
                                                   tolerance=1e-5, max_iter=500,
                                                   trace=True),
        'ISTA':        lambda: ista_wrapper(data_kind, X, y, lam,
                                            tolerance=1e-5, max_iter=500,
                                            trace=True),
        'FISTA':       lambda: fista_wrapper(data_kind, X, y, lam,
                                             tolerance=1e-5, max_iter=500,
                                             trace=True),
        'Frank-Wolfe': lambda: frank_wolfe_wrapper(data_kind, X, y, R=R_fw,
                                                   tolerance=1e-5, max_iter=500,
                                                   step_size_strategy='standard',
                                                   trace=True),
        'Barrier':     lambda: barrier_wrapper(data_kind, X, y, lam,
                                               tolerance_inner=1e-6,
                                               tolerance_outer=1e-5,
                                               max_outer_iter=50,
                                               max_inner_iter=100,
                                               trace=True),
    }

    histories = {}
    times = {}

    for name, method_func in methods.items():
        t_start = time()
        x_star, msg, hist = method_func()
        elapsed = time() - t_start
        histories[name] = hist
        times[name] = elapsed

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for name, hist in histories.items():
        x_history = np.array(hist['x'])

        if x_history.ndim == 1:
            final_sparsity = np.mean(np.abs(x_history) < 1e-8) * 100
            num_iters = len(hist.get('time', [1]))
            sparsity_percent = np.full(num_iters, final_sparsity)
        else:
            sparsity_percent = np.mean(np.abs(x_history) < 1e-8, axis=1) * 100

        ax1.plot(sparsity_percent, label=name, lw=2)

        t = np.array(hist['time'])
        ax2.plot(t, sparsity_percent, label=name, lw=2)

    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Доля нулевых весов, %')
    ax1.set_title(f'{data_name}: динамика разреженности')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 200)

    ax2.set_xlabel('Время, с')
    ax2.set_ylabel('Доля нулевых весов, %')
    ax2.set_title(f'{data_name}: динамика разреженности по времени')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    save_current_plot(prefix + "_" + data_name + "_1", fig1)
    save_current_plot(prefix + "_" + data_name + "_2", fig2)
    plt.show()

    return histories, lam, F_star

def run_experiment_2(data_kind, data_name, lam, prefix="3_3"):
    from libsvm_utils import load_libsvm_scaled
    X, y, info = load_libsvm_scaled(data_kind, data_name)
    m, n = info['m'], info['n']

    oracle_ref = make_prox_oracle(data_kind, X, y, lam)
    x_ref, _, _ = proximal_fast_gradient_method(oracle_ref, np.zeros(n),
                                                L_0=1.0, tolerance=1e-7,
                                                max_iter=10000, trace=False)
    F_star = oracle_ref.func(x_ref)
    R_fw = np.linalg.norm(x_ref, 1)

    methods = {
        'Subgradient': subgradient_wrapper_counted(data_kind, X, y, lam,
                                                   tolerance=1e-5, max_iter=500, trace=True),
        'ISTA':        ista_wrapper_counted(data_kind, X, y, lam,
                                            tolerance=1e-5, max_iter=500, trace=True),
        'FISTA':       fista_wrapper_counted(data_kind, X, y, lam,
                                             tolerance=1e-5, max_iter=500, trace=True),
        'Frank-Wolfe': frank_wolfe_wrapper_counted(data_kind, X, y, R=R_fw,
                                                   tolerance=1e-5, max_iter=500,
                                                   step_size_strategy='standard', trace=True),
        'Barrier':     barrier_wrapper_counted(data_kind, X, y, lam,
                                               tolerance_inner=1e-6, tolerance_outer=1e-5,
                                               max_outer_iter=50, max_inner_iter=100, trace=True)
    }

    histories = {}

    for name, method_func in methods.items():
        x_star, msg, hist = method_func
        histories[name] = hist

    fig1, ax1 = plt.subplots(figsize=(10,6))
    fig2, ax2 = plt.subplots(figsize=(10,6))

    for name, hist in histories.items():
        F_vals = np.array(hist['func'])
        err = F_vals - F_star
        err[err <= 0] = 1e-16

        calls = np.array(hist['oracle_calls'])
        ax1.semilogy(calls, err, label=name, lw=2)
        t = np.array(hist['time'])
        ax2.semilogy(t, err, label=name, lw=2)

    ax1.set_xlabel('Число вызовов оракула')
    ax1.set_ylabel('F(x) - F*')
    ax1.set_title(f'{data_name}: сходимость по вызовам оракула: lambda = {lam}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Время, с')
    ax2.set_ylabel('F(x) - F*')
    ax2.set_title(f'{data_name}: сходимость по времени: lambda = {lam}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.show()

    save_current_plot(prefix + "_" + data_name + "_" + str(lam) + "_1", fig1)
    save_current_plot(prefix + "_" + data_name +  "_" + str(lam) + "_2", fig2)
    return histories, lam, F_star

def run_experiment_3(data_kind, data_name, lam = 0.5, prefix="3_4"):
    from libsvm_utils import load_libsvm_scaled

    X, y, info = load_libsvm_scaled(data_kind, data_name)
    m, n = info['m'], info['n']

    oracle_ref = make_prox_oracle(data_kind, X, y, lam)
    x_ref, _, _ = proximal_fast_gradient_method(oracle_ref, np.zeros(n),
                                                L_0=1.0, tolerance=1e-7,
                                                max_iter=10000, trace=False)
    F_star = oracle_ref.func(x_ref)
    R_fw = np.linalg.norm(x_ref, 1)

    _, _, hist_ista = ista_wrapper_counted(data_kind, X, y, lam,
                                           tolerance=1e-5, max_iter=5000, trace=True)
    _, _, hist_fista = fista_wrapper_counted(data_kind, X, y, lam,
                                             tolerance=1e-5, max_iter=5000, trace=True)

    fig1 = plt.figure(figsize=(10,6))
    for name, hist in [('ISTA', hist_ista), ('FISTA', hist_fista)]:
        F_vals = np.array(hist['func'])
        err = F_vals - F_star
        err[err <= 0] = 1e-16
        plt.semilogy(np.arange(len(err)), err, label=name, lw=2)
    plt.xlabel('Итерация')
    plt.ylabel('F(x) - F*')
    plt.title(f'{data_name}: ISTA vs FISTA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    save_current_plot(prefix + "_" + data_name + "_" + str(lam) + "_1", fig1)

    _, _, hist_fw_std = frank_wolfe_wrapper_counted(data_kind, X, y, R=R_fw,
                                                    tolerance=1e-5, max_iter=5000,
                                                    step_size_strategy='standard', trace=True)
    _, _, hist_fw_armijo = frank_wolfe_wrapper_counted(data_kind, X, y, R=R_fw,
                                                       tolerance=1e-5, max_iter=5000,
                                                       step_size_strategy='armijo', trace=True)

    hist_fw_std['func'] = [oracle_ref.func(xi) for xi in hist_fw_std['x']]
    hist_fw_armijo['func'] = [oracle_ref.func(xi) for xi in hist_fw_armijo['x']]

    fig2 = plt.figure(figsize=(10,6))
    for name, hist in [('Standard', hist_fw_std), ('Armijo', hist_fw_armijo)]:
        F_vals = np.array(hist['func'])
        err = F_vals - F_star
        err[err <= 0] = 1e-16
        plt.semilogy(np.arange(len(err)), err, label=name, lw=2)
    plt.xlabel('Итерация')
    plt.ylabel('F(x) - F*')
    plt.title(f'{data_name}: Frank-Wolfe с разным шагом')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    save_current_plot(prefix + "_" + data_name + "_" + str(lam) + "_2", fig2)

    mu_list = [2, 5, 10, 20, 30, 40, 50, 100]
    total_inner_iters = {}
    for mu in mu_list:
        _, _, hist_bar = barrier_wrapper_counted(data_kind, X, y, lam,
                                                 tolerance_inner=1e-6,
                                                 tolerance_outer=1e-5,
                                                 max_outer_iter=50,
                                                 max_inner_iter=100,
                                                 mu=mu, trace=True)
        total_inner_iters[mu] = hist_bar['inner_iters'][-1]

    fig3 = plt.figure(figsize=(8,5))
    mu_labels = [str(m) for m in mu_list]
    iters_values = [total_inner_iters[m] for m in mu_list]
    plt.bar(mu_labels, iters_values, color='skyblue')
    plt.xlabel('mu (коэффициент увеличения t)')
    plt.ylabel('Суммарное число внутренних итераций Ньютона')
    plt.title(f'{data_name}: Влияние mu на трудоёмкость барьерного метода')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    save_current_plot(prefix + "_" + data_name + "_" + str(lam) + "_3", fig3)

    return
