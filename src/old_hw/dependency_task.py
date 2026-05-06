import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from oracles import QuadraticOracle
from save_graphic import save_current_plot
def gen_prob(n, kappa, seed):
    """
    Генерирует диагональную матрицу A с обусловленностью kappa и случайный b.
    """
    np.random.seed(seed)
    d = np.exp(np.linspace(0, np.log(kappa), n))
    np.random.shuffle(d)
    d[0], d[-1] = 1.0, kappa
    A = diags(d, format='dia')
    b = np.random.randn(n)
    return A, b

def plot_conditioning_results(exp_data, title="", lw=2.0, ax=None, method_label=None,
                              save_to_disk=False, filename=None):
    """
    Рисует зависимость числа итераций от κ для разных размерностей n.

    Параметры
    ----------
    exp_data : dict
        Словарь с ключами 'results', 'n_values', 'kappa_values'.
        results[(n, kappa)] = список числа итераций для каждого повторения.
    title : str
        Заголовок графика.
    lw : float
        Толщина линии.
    ax : matplotlib.axes.Axes или None
        Если None, создаёт новый figure и axes.
    method_label : str или None
        Если указано, добавляется в легенду перед 'n=...'.
    save_to_disk : bool
        Сохранять ли график в файл.
    filename : str or None
        Имя файла (без расширения или с расширением). Если None генерируется автоматически.

    Возвращает
    ----------
    ax : matplotlib.axes.Axes
    """
    results = exp_data['results']
    n_vals = exp_data['n_values']
    k_vals = exp_data['kappa_values']

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    colors = plt.cm.tab10(np.linspace(0, 1, len(n_vals)))
    for idx, n in enumerate(n_vals):
        c = colors[idx]
        # Точки отдельных запусков
        for k in k_vals:
            for v in results.get((n, k), []):
                ax.plot([k], [v], 'o', color=c, alpha=0.25, ms=3)
        # Среднее +- std
        mean = [np.mean(results[(n, k)]) for k in k_vals]
        std  = [np.std(results[(n, k)]) for k in k_vals]
        label = f'n={n}' if method_label is None else f'{method_label}, n={n}'
        ax.loglog(k_vals, mean, 'o-', color=c, lw=lw, label=label,
                  markersize=4, markerfacecolor='white')
        ax.fill_between(k_vals,
                        np.maximum(0.5, np.array(mean) - np.array(std)),
                        np.array(mean) + np.array(std),
                        color=c, alpha=0.15)

    ax.set_xlabel(r'$\kappa$')
    ax.set_ylabel('Итерации')
    ax.set_title(title.strip())
    ax.legend(title="Метод / n" if method_label is not None else "Размерность n")
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    if save_to_disk:
        save_current_plot(filename, fig)
    return ax

def experiment_conditioning(method_func, method_name, n_list, kappa_list, n_repeats=3,
                            tolerance=1e-6, max_iter=5000, **method_kwargs):
    """
    Проводит эксперимент: для каждой размерности n и обусловленности kappa
    запускает method_func на квадратичной задаче и собирает число итераций.

    Параметры
    ----------
    method_func : callable
        Функция, запускающая метод оптимизации. Должна иметь вид:
        method_func(oracle, x0, tolerance, max_iter, trace, **kwargs) -> (x_star, message, history)
        где history – словарь с ключом 'time' (список времён)
        Возвращаемое число итераций = len(history['time']) - 1 (если 'time' есть),
        иначе len(history['residual_norm']) - 1.
    method_name : str
        Название метода (для вывода на печать).
    n_list : list[int]
        Список размерностей.
    kappa_list : list[float]
        Список чисел обусловленности.
    n_repeats : int
        Количество повторений для усреднения.
    tolerance : float
        Критерий остановки.
    max_iter : int
        Максимальное число итераций.
    plot : bool
        Построить график результатов.
    **method_kwargs : dict
        Дополнительные параметры, передаваемые в method_func (например, line_search_options).

    Возвращает
    ----------
    results : dict
        Словарь {(n, kappa): list_of_iters}
    """
    results = {}
    for n in n_list:
        for kappa in kappa_list:
            iters_list = []
            for rep in range(n_repeats):
                seed = hash((n, kappa, rep)) % 10000
                A, b = gen_prob(n, kappa, seed)
                oracle = QuadraticOracle(A, b)
                x0 = np.zeros(n)
                try:
                    _, _, history = method_func(oracle, x0, tolerance=tolerance,
                                                max_iter=max_iter, trace=True,
                                                **method_kwargs)
                    if history is not None:
                        if 'time' in history:
                            n_iters = len(history['time']) - 1
                        elif 'residual_norm' in history:
                            n_iters = len(history['residual_norm']) - 1
                        else:
                            n_iters = max_iter  # fallback
                    else:
                        n_iters = max_iter
                except Exception as e:
                    print(f"Ошибка для n={n}, κ={kappa:.2f}, rep={rep}: {e}")
                    n_iters = max_iter
                iters_list.append(n_iters)
            results[(n, kappa)] = iters_list
            print(f"{method_name}: n={n:3d}, κ={kappa:7.2f} → {np.mean(iters_list):.1f}±{np.std(iters_list):.1f}")
    return results

