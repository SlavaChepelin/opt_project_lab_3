import os
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.abspath('../plots')  # или любая ваша глобальная переменная

def save_current_plot(filename, fig=None, tight=True, dpi=150):
    """
    Сохраняет текущий график в папку PLOTS_DIR.

    Параметры
    ----------
    filename : str
        Имя файла (можно с расширением .png/.jpg/.pdf/.svg, можно без).
        Если без расширения, автоматически добавляется .png.
    fig : matplotlib.figure.Figure or None
        Экземпляр фигуры. Если None, берётся текущая через plt.gcf().
    tight : bool
        Если True, сохраняет с bbox_inches='tight'.
    dpi : int
        Разрешение.

    Возвращает
    ----------
    full_path : str
        Полный путь к сохранённому файлу.
    """
    if fig is None:
        fig = plt.gcf()

    if not filename.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
        filename += '.png'

    os.makedirs(PLOTS_DIR, exist_ok=True)
    full_path = os.path.join(PLOTS_DIR, filename)

    if tight:
        fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(full_path, dpi=dpi)

    print(f"График сохранён: {full_path}")
    return full_path