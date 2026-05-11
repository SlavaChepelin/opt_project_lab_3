import matplotlib.pyplot as plt
from save_graphic import save_current_plot
from optimization import *
from methods_counted_wrappers import *

def find_lambda_for_sparsity(data_kind, X, y, target_sparsity=(0.5, 0.8), max_trials=20):
    """Подбирает λ так, чтобы доля нулей в решении FISTA была в [0.5, 0.8]."""
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
    """
    Сравнение методов для L1-регуляризованной задачи.
    1. Загружает данные, подбирает λ для 50-80% нулей.
    2. Вычисляет F* с помощью FISTA (10k итераций).
    3. Запускает все 5 методов (Subgrad, ISTA, FISTA, Frank-Wolfe, Barrier).
    4. Строит графики log(F(x_k)-F*) от итераций и времени.
    """
    from libsvm_utils import load_libsvm_scaled
    X, y, info = load_libsvm_scaled(data_kind, data_name)
    m, n = info['m'], info['n']
    print(f"Датасет: {data_name}, размер: {m}x{n}")
    lam, sparsity = find_lambda_for_sparsity(data_kind, X, y)
    print(f"Подобрана λ = {lam:.5f}, доля нулей = {sparsity*100:.1f}%")
    print("Вычисление F* с помощью FISTA (10000 iter)...")
    oracle_ref = make_prox_oracle(data_kind, X, y, lam)
    x_ref, _, _ = proximal_fast_gradient_method(oracle_ref, np.zeros(n),
                                                L_0=1.0, tolerance=1e-7,
                                                max_iter=10000, trace=False)
    F_star = oracle_ref.func(x_ref)
    R_fw = np.linalg.norm(x_ref, 1)
    print(f"F* = {F_star:.6f}, R = {R_fw:.4f}")
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
    sparsity_curve = {}

    print("Запуск методов...")
    for name, method_func in methods.items():
        print(f"  {name}...")
        t_start = time()
        x_star, msg, hist = method_func()
        elapsed = time() - t_start
        histories[name] = hist
        times[name] = elapsed

    # 4. Графики log(F(x_k) - F_star) от итераций и от времени
    fig1, ax1 = plt.subplots(figsize=(10,6))
    fig2, ax2 = plt.subplots(figsize=(10,6))
    for name, hist in histories.items():
        F_vals = np.array(hist['func'])
        err = F_vals - F_star
        err[err <= 0] = 1e-16  # избегаем log(0)
        ax1.semilogy(np.arange(len(err)), err, label=name, lw=2)
        t = np.array(hist['time'])
        ax2.semilogy(t, err, label=name, lw=2)
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('F(x) - F*')
    ax1.set_title(f'{data_name}: ошибка по итерациям')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Время, с')
    ax2.set_ylabel('F(x) - F*')
    ax2.set_title(f'{data_name}: ошибка по времени')
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
    print(f"F* = {F_star:.6f}, R = {R_fw:.4f}")


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
        print(f"Запуск {name}...")
        x_star, msg, hist = method_func
        histories[name] = hist


    fig1, ax1 = plt.subplots(figsize=(10,6))
    fig2, ax2 = plt.subplots(figsize=(10,6))

    for name, hist in histories.items():
        F_vals = np.array(hist['func'])
        err = F_vals - F_star
        err[err <= 0] = 1e-16

        # По числу вызовов оракула
        calls = np.array(hist['oracle_calls'])
        ax1.semilogy(calls, err, label=name, lw=2)
        # По времени
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
    """
    Эксперимент 3: исследование свойств методов.
    3.1 Сравнение ISTA и FISTA по итерациям.
    3.2 Сравнение Frank-Wolfe с разным шагом.
    3.3 Влияние mu в методе барьеров.
    """
    from libsvm_utils import load_libsvm_scaled

    X, y, info = load_libsvm_scaled(data_kind, data_name)
    m, n = info['m'], info['n']
    print(f"Датасет: {data_name}, m={m}, n={n}")


    oracle_ref = make_prox_oracle(data_kind, X, y, lam)
    x_ref, _, _ = proximal_fast_gradient_method(oracle_ref, np.zeros(n),
                                                L_0=1.0, tolerance=1e-7,
                                                max_iter=10000, trace=False)
    F_star = oracle_ref.func(x_ref)
    R_fw = np.linalg.norm(x_ref, 1)
    print(f"F* = {F_star:.6f}, R = {R_fw:.4f}")

    # 3.1 ISTA vs FISTA
    print("3.1 ISTA vs FISTA...")
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

    # 3.2 Frank-Wolfe
    print("3.2 Frank-Wolfe: standard vs armijo...")
    _, _, hist_fw_std = frank_wolfe_wrapper_counted(data_kind, X, y, R=R_fw,
                                                    tolerance=1e-5, max_iter=5000,
                                                    step_size_strategy='standard', trace=True)
    _, _, hist_fw_armijo = frank_wolfe_wrapper_counted(data_kind, X, y, R=R_fw,
                                                       tolerance=1e-5, max_iter=5000,
                                                       step_size_strategy='armijo', trace=True)

    # Пересчитываем F(x) = f(x) + λ||x||₁ для обоих
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

    # 3.3 Barrier
    mu_list = [2, 5, 10, 20, 30, 40, 50, 100]
    print(f"3.3 Barrier: mu = {mu_list}")
    total_inner_iters = {}
    for mu in mu_list:
        print(f"  mu={mu}...")
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