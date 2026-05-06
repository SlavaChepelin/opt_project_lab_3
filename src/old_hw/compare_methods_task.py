import numpy as np
import matplotlib.pyplot as plt
from libsvm_utils import load_libsvm_scaled, make_exp_oracle, make_logcosh_oracle
from save_graphic import save_current_plot

def plot_objective_vs_iter(histories, title="", filename=None):
    fig, ax = plt.subplots(figsize=(10,6))
    for name, hist in histories.items():
        ax.semilogy(hist['func'], label=name, lw=2)
    ax.set_xlabel('Итерация')
    ax.set_ylabel('f(x)')
    ax.set_title(title or 'Значение функции по итерациям')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if filename:
        save_current_plot(filename, fig=fig)
    else:
        plt.show()
    return fig, ax

def plot_objective_vs_time(histories, title="", filename=None):
    fig, ax = plt.subplots(figsize=(10,6))
    for name, hist in histories.items():
        ax.semilogy(hist['time'], hist['func'], label=name, lw=2)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('f(x)')
    ax.set_title(title or 'Значение функции по времени')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if filename:
        save_current_plot(filename, fig=fig)
    else:
        plt.show()
    return fig, ax

def plot_grad_sq_vs_time(histories, title="", filename=None):
    fig, ax = plt.subplots(figsize=(10,6))
    for name, hist in histories.items():
        grad0_sq = hist['grad_norm'][0]**2
        rel_grad_sq = np.array(hist['grad_norm'])**2 / max(grad0_sq, 1e-300)
        ax.semilogy(hist['time'], rel_grad_sq, label=name, lw=2)
    ax.set_xlabel('Время, с')
    ax.set_ylabel(r'$\|\nabla f\|^2 / \|\nabla f_0\|^2$')
    ax.set_title(title or 'Относительный квадрат градиента по времени')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if filename:
        save_current_plot(filename, fig=fig)
    else:
        plt.show()
    return fig, ax


def run_comparison_experiment(data_kind, data_name, method_wrappers, suffix="",
                              tolerance=1e-6, max_iter=1000):
    """
    Запускает сравнение методов на заданном датасете.

    Параметры:
    ----------
    data_kind : str
        'binary' или 'regression'
    data_name : str
        'a9a', 'gisette', 'abalone', 'bodyfat'
    method_wrappers : dict
        Ключ – имя метода (str), значение – callable с сигнатурой:
        wrapper(oracle, x0, tolerance, max_iter, trace, **kwargs)
        Возвращает (x_star, message, history)
    suffix : str
        Добавляется к именам сохраняемых файлов (например, "_a9a").
    tolerance : float
        Критерий остановки (относительная норма градиента)
    max_iter : int
        Максимальное число итераций для каждого метода.
    save_histories : bool
        Сохранять ли словарь histories в pickle-файл.
    """
    # Загрузка данных
    print(f"Загрузка {data_kind}/{data_name}...")
    X, y, info = load_libsvm_scaled(data_kind, data_name)
    regcoef = 1.0 / info['m']
    if data_kind == 'binary':
        oracle = make_exp_oracle(X, y, regcoef)
    else:
        oracle = make_logcosh_oracle(X, y, regcoef)
    x0 = np.zeros(info['n'])
    print(f"  m={info['m']}, n={info['n']}, nnz={info['nnz']}")


    histories = {}
    for name, wrapper in method_wrappers.items():
        print(f"Запуск {name}...")
        _, _, hist = wrapper(oracle, x0,
                             tolerance=tolerance,
                             max_iter=max_iter,
                             trace=True)
        histories[name] = hist
        print(f"  итераций: {len(hist['func'])}, время: {hist['time'][-1]:.2f} с")
    base_title = f"Сравнение методов на {data_name}{suffix}"
    plot_objective_vs_iter(histories,
                           title=f"{base_title} (итерации)",
                           filename=f"exp24_{data_name}{suffix}_obj_vs_iter")
    plot_objective_vs_time(histories,
                           title=f"{base_title} (время)",
                           filename=f"exp24_{data_name}{suffix}_obj_vs_time")
    plot_grad_sq_vs_time(histories,
                         title=f"{base_title} (градиент)",
                         filename=f"exp24_{data_name}{suffix}_grad_vs_time")
    return histories