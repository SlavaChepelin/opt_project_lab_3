import numpy as np
import matplotlib.pyplot as plt
import os, sys

from oracles import QuadraticOracle

sys.path.append('../src')
from libsvm_utils import load_libsvm_scaled, make_exp_oracle, make_logcosh_oracle
from optimization import lbfgs
from save_graphic import save_current_plot

def compute_rel_grad_sq(histories):
    """
    Вычисляет относительный квадрат нормы градиента для всех l.

    Параметры:
        histories: dict {l: history}, где history содержит ключ 'grad_norm'

    Возвращает:
        rel_grad_sq: dict {l: np.array} относительных квадратов норм градиента
    """
    rel_grad_sq = {}
    for l, hist in histories.items():
        grad0_sq = hist['grad_norm'][0] ** 2
        rel = np.array(hist['grad_norm']) ** 2 / grad0_sq
        rel_grad_sq[l] = rel
    return rel_grad_sq


def plot_convergence_vs_iterations(rel_grad_sq, L_VALUES, title="", filename=None):
    """
    График (а): относительный квадрат нормы градиента vs номер итерации.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for l in L_VALUES:
        ax.semilogy(rel_grad_sq[l], label=f'l={l}', lw=2)
    ax.set_xlabel('Номер итерации')
    ax.set_ylabel(r'$\|\nabla f(x_k)\|^2 / \|\nabla f(x_0)\|^2$')
    ax.set_title(title or 'Сходимость по итерациям')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if filename:
        save_current_plot(filename, fig=fig)
    else:
        plt.show()
    return fig, ax


def plot_convergence_vs_time(histories, rel_grad_sq, L_VALUES, title="", filename=None):
    """
    График (б): относительный квадрат нормы градиента vs реальное время.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for l in L_VALUES:
        ax.semilogy(histories[l]['time'], rel_grad_sq[l], label=f'l={l}', lw=2)
    ax.set_xlabel('Время, с')
    ax.set_ylabel(r'$\|\nabla f(x_k)\|^2 / \|\nabla f(x_0)\|^2$')
    ax.set_title(title or 'Сходимость по времени')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if filename:
        save_current_plot(filename, fig=fig)
    else:
        plt.show()
    return fig, ax


def plot_time_vs_l(histories, L_VALUES, title="", filename=None):
    """
    Зависимость общего времени работы от l.
    """
    total_time = {l: histories[l]['time'][-1] for l in L_VALUES}
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(list(total_time.keys()), list(total_time.values()), 'o-', lw=2, markersize=8)
    ax.set_xlabel('Размер истории l')
    ax.set_ylabel('Общее время работы, с')
    ax.set_title(title or 'Время работы vs l')
    ax.grid(True, alpha=0.3)
    if filename:
        save_current_plot(filename, fig=fig)
    else:
        plt.show()
    return fig, ax, total_time


def plot_iters_vs_l(rel_grad_sq, L_VALUES, tolerance=1e-6, title="", filename=None):
    """
    Зависимость числа итераций до сходимости от l.
    """
    iters_to_converge = {}
    for l, rel in rel_grad_sq.items():
        idx = np.argmax(rel <= tolerance) if np.any(rel <= tolerance) else len(rel)
        iters_to_converge[l] = idx
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(list(iters_to_converge.keys()), list(iters_to_converge.values()), 's-', lw=2, markersize=8)
    ax.set_xlabel('Размер истории l')
    ax.set_ylabel('Итерации до сходимости')
    ax.set_title(title or f'Итерации vs l (tolerance={tolerance})')
    ax.grid(True, alpha=0.3)
    if filename:
        save_current_plot(filename, fig=fig)
    else:
        plt.show()
    return fig, ax, iters_to_converge


def print_speedup(iters_to_converge):
    """
    Выводит ускорение относительно l=0.
    """
    iters_l0 = iters_to_converge[0]
    print("\nУскорение (по итерациям) относительно l=0:")
    for l, iters in iters_to_converge.items():
        if l > 0:
            print(f"  l={l:3d}: {iters_l0 / iters:.2f}x")


def history_check(data_kind : str, data_name: str, l_values : list[int], suffix : str = ""):
    x, y, info = load_libsvm_scaled(data_kind, data_name)
    reg_coef = 1.0/ info['m'] # по условию

    if data_kind == 'binary':
        oracle = make_exp_oracle(x, y, reg_coef)
    else:
        oracle = make_logcosh_oracle(x, y, reg_coef)

    x0 = np.zeros(info['n'])
    histories = {}
    for l in l_values:
        print(f"Запуск L-BFGS с l={l}")
        _, _, hist = lbfgs(oracle, x0,
                           tolerance=1e-6,
                           max_iter=100_000,
                           memory_size=l,
                           line_search_options={"method": "Wolfe", "c1": 1e-4, "c2": 0.9},
                           trace=True)
        histories[l] = hist

    data_name = data_name + suffix
    rel_grad_sq = compute_rel_grad_sq(histories)
    # график a -
    plot_convergence_vs_iterations(rel_grad_sq, l_values,
                                   title=f"Сходимость L-BFGS ({data_name})",
                                   filename=f"exp23_iters_{data_name}")

    # график б – время
    plot_convergence_vs_time(histories, rel_grad_sq, l_values,
                             title=f"Сходимость по времени ({data_name})",
                             filename=f"exp23_time_{data_name}")

    # дополнительный график: общее время vs l
    _, _, total_time = plot_time_vs_l(histories, l_values,
                                      title=f"Время работы от l ({data_name})",
                                      filename=f"exp23_total_time_{data_name}")

    # дополнительный график: итерации до сходимости vs l
    _, _, iters_map = plot_iters_vs_l(rel_grad_sq, l_values, tolerance=1e-6,
                                      title=f"Итерации до сходимости ({data_name})",
                                      filename=f"exp23_iters_needed_{data_name}")

    # Вывод ускорения
    print_speedup(iters_map)

def run_quadratic_experiment(A, b, name, l_values, tolerance=1e-6, max_iter=500):
    oracle = QuadraticOracle(A, b)
    x0 = np.zeros(A.shape[0])
    histories = {}
    for l in l_values:
        print(f"Запуск L-BFGS на {name}, l={l}")
        _, _, hist = lbfgs(oracle, x0,
                           tolerance=tolerance,
                           max_iter=max_iter,
                           memory_size=l,
                           line_search_options={"method": "Wolfe", "c1": 1e-4, "c2": 0.9},
                           trace=True)
        histories[l] = hist

    rel_grad_sq = compute_rel_grad_sq(histories)

    # График (а) – итерации
    plot_convergence_vs_iterations(rel_grad_sq, l_values,
                                   title=f"{name}: сходимость по итерациям",
                                   filename=f"quad_{name}_iters")

    # График (б) – время
    plot_convergence_vs_time(histories, rel_grad_sq, l_values,
                             title=f"{name}: сходимость по времени",
                             filename=f"quad_{name}_time")

    # Время vs l
    plot_time_vs_l(histories, l_values,
                   title=f"{name}: время работы от l",
                   filename=f"quad_{name}_time_vs_l")

    # Итерации до сходимости vs l
    _, _, iters_map = plot_iters_vs_l(rel_grad_sq, l_values, tolerance=tolerance,
                                      title=f"{name}: итерации до сходимости от l",
                                      filename=f"quad_{name}_iters_vs_l")


    return histories, rel_grad_sq