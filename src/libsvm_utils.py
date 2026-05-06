import os
import urllib.request
import numpy as np
from sklearn.datasets import load_svmlight_file

# Глобальные конфигурации (можно вынести в параметры, если нужно)
DATA_DIR = os.path.abspath('../data/libsvm')
LIBSVM_BASE = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets'

def download_if_missing(url, dest):
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        import ssl
        context = ssl._create_unverified_context()
        urllib.request.urlretrieve(url, dest, context=context)

def load_libsvm_scaled(kind, name, n_features=None):
    """
    Загружает масштабированный датасет LIBSVM.

    Параметры
    ----------
    kind : str
        'regression' или 'binary'
    name : str
        Имя датасета: 'abalone', 'bodyfat', 'a9a', 'gisette'
    n_features : int, optional
        Не используется, оставлен для совместимости.

    Возвращает
    ----------
    X : scipy.sparse.csr_matrix
    y : np.ndarray
    info : dict
        С ключами 'm', 'n', 'nnz'
    """
    urls = {
        'regression': {
            'abalone': f'{LIBSVM_BASE}/regression/abalone_scale',
            'bodyfat': f'{LIBSVM_BASE}/regression/bodyfat_scale'
        },
        'binary': {
            'a9a': f'{LIBSVM_BASE}/binary/a9a',
            'gisette': f'{LIBSVM_BASE}/binary/gisette_scale.bz2'
        }
    }
    if kind not in urls or name not in urls[kind]:
        raise ValueError(f'Unknown dataset: {kind}/{name}')

    local = os.path.join(DATA_DIR, os.path.basename(urls[kind][name]))
    download_if_missing(urls[kind][name], local)

    X, y = load_svmlight_file(local)
    X = X.tocsr().astype(np.float64)
    y = np.asarray(y, dtype=float)
    info = {'m': X.shape[0], 'n': X.shape[1], 'nnz': X.nnz}
    return X, y, info


def ensure_binary_labels(y):
    """
    Приводит метки бинарной классификации к виду ±1.
    Поддерживает [0,1], [1,2] и [-1,1].
    """
    y = np.asarray(y, dtype=float).ravel()
    u = np.unique(y)
    if np.array_equal(sorted(u), [-1., 1.]):
        return y
    if np.array_equal(sorted(u), [0., 1.]):
        return 2 * y - 1
    if np.array_equal(sorted(u), [1., 2.]):
        return 2 * y - 3
    raise ValueError(f'Unsupported labels: {u}')


def matmat_ATsA_sparse(A, s):
    """
    Вычисляет A.T @ diag(s) @ A для разреженной матрицы A.
    Возвращает плотную матрицу (обычно мала, если n невелико).
    """
    s = np.asarray(s, dtype=float).reshape(-1)
    return (A.T @ A.multiply(s[:, np.newaxis])).toarray()


def sparse_mv_ops(A):
    """
    Возвращает три функции:
        matvec_Ax(x)    -> A @ x
        matvec_ATx(v)   -> A.T @ v
        matmat_ATsA(s)  -> A.T @ diag(s) @ A (плотная)
    """
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(v):
        return A.T.dot(v)

    def matmat_ATsA(s):
        return matmat_ATsA_sparse(A, s)

    return matvec_Ax, matvec_ATx, matmat_ATsA


def make_logcosh_oracle(X, b, regcoef):
    """Создаёт оракул для log-cosh регрессии с L2-регуляризацией."""
    from oracles import LogCoshL2Oracle
    Ax, ATx, ATsA = sparse_mv_ops(X)
    return LogCoshL2Oracle(Ax, ATx, ATsA, np.asarray(b, dtype=float).ravel(), regcoef)


def make_exp_oracle(X, y, regcoef):
    """Создаёт оракул для экспоненциальной (логистической) классификации с L2-регуляризацией."""
    from oracles import ExpLossL2Oracle
    return ExpLossL2Oracle(*sparse_mv_ops(X), ensure_binary_labels(y), regcoef)