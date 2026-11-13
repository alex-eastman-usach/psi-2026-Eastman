"""Convolucion lineal discreta por definicion directa y mediante FFT."""

import numpy as np


def _fft_radix2(x, inversa=False):
    """Algoritmo de Cooley-Tukey radix-2 iterativo implementado solo con NumPy.

    Requiere que el largo M de x sea una potencia de 2 (M = 2^p).
    Reduce la complejidad de O(M^2) a O(M log2 M) explotando la simetria y
    periodicidad de los factores de giro W_M^k = exp(-2j * pi * k / M).
    """
    x_arr = np.asarray(x, dtype=complex)
    m = x_arr.size
    if m <= 1:
        return x_arr.copy()

    # 1. Permutacion por inversion de bits (bit-reversal):
    # Al remodelar el arreglo en p dimensiones binarias e invertir
    # el orden de los ejes con .T, el indice binario pasa a
    # de forma vectorizada sin ciclos
    p = round(float(np.log2(m)))
    if (1 << p) != m:
        raise ValueError("El largo para _fft_radix2 debe ser potencia de 2.")

    x_work = x_arr.reshape((2,) * p).T.copy().ravel()

    # 2. Etapas de mariposa (bottom-up), doblando el tamaño de sub-DFT en cada paso:
    signo = 2j if inversa else -2j
    m_etapa = 2
    while m_etapa <= m:
        mitad = m_etapa // 2
        k = np.arange(mitad, dtype=float)
        giro = np.exp(signo * np.pi * k / m_etapa)

        bloques = x_work.reshape((-1, m_etapa))
        u = bloques[:, :mitad]
        v = bloques[:, mitad:] * giro
        x_work = np.hstack([u + v, u - v]).ravel()
        m_etapa *= 2

    if inversa:
        x_work /= m
    return x_work


def convolucion_directa(x, h):
    """Convolucion lineal completa (largo len(x) + len(h) - 1), por definicion."""
    # Por definicion: y[n] = sum_{k=0}^{N_h - 1} h[k] * x[n - k].
    # Interpretamos la suma como la superposicion de versiones desplazadas de x,
    # ponderadas por cada coeficiente h[k] del filtro.
    # De esta manera se ejecutan exactamente N_x * N_h multiplicaciones y sumas
    # (complejidad O(N_x * N_h)) aprovechando operaciones vectoriales sobre x.
    x_arr = np.asarray(x, dtype=float)
    h_arr = np.asarray(h, dtype=float)
    n_x = x_arr.size
    n_h = h_arr.size

    if n_x == 0 or n_h == 0:
        return np.array([], dtype=float)

    largo_salida = n_x + n_h - 1
    y = np.zeros(largo_salida, dtype=float)

    for k in range(n_h):
        y[k : k + n_x] += h_arr[k] * x_arr

    return y


def convolucion_fft(x, h):
    """Misma convolucion lineal, via el teorema de la convolucion."""
    # El teorema de la convolucion para la DFT establece que el producto punto a
    # punto en frecuencia IDFT(DFT(x) * DFT(h)) corresponde a la convolucion
    # CIRCULAR de periodo M.
    # Para que la convolucion circular coincida exactamente con la convolucion
    # LINEAL de largo L = N_x + N_h - 1 (evitando el traslape o aliasing temporal
    # de la cola del filtro sobre el inicio de la señal), debemos rellenar ambos
    # arreglos con ceros hasta un largo M >= L.
    # Elegimos M como la menor potencia de 2 mayor o igual a L para aplicar
    # directamente el algoritmo radix-2 en tiempo O(M log2 M).
    x_arr = np.asarray(x, dtype=float)
    h_arr = np.asarray(h, dtype=float)
    n_x = x_arr.size
    n_h = h_arr.size

    if n_x == 0 or n_h == 0:
        return np.array([], dtype=float)

    largo_lineal = n_x + n_h - 1
    # Menor potencia de 2 mayor o igual a largo_lineal
    p = int(np.ceil(np.log2(max(1, largo_lineal))))
    m_fft = 1 << p

    x_pad = np.pad(x_arr, (0, m_fft - n_x), mode="constant")
    h_pad = np.pad(h_arr, (0, m_fft - n_h), mode="constant")

    # Transformamos al dominio de la frecuencia, multiplicamos punto a punto
    # y volvemos al tiempo mediante la transformada inversa.
    x_freq = _fft_radix2(x_pad, inversa=False)
    h_freq = _fft_radix2(h_pad, inversa=False)
    y_pad = _fft_radix2(x_freq * h_freq, inversa=True)

    # Recortamos los ceros sobrantes para devolver exactamente las L muestras
    # de la convolucion lineal y descartamos la parte imaginaria residual (~1e-16).
    return np.real(y_pad[:largo_lineal])


def convolucion_circular_fft(x, h):
    """Convolucion via FFT sin relleno para convolucion lineal (solo iguala largos).

    Tarea 3.2 el efecto de aliasing
    temporal cuando M < len(x) + len(h) - 1: las ultimas N_h - 1 muestras de la
    respuesta se envuelven y se suman sobre el comienzo de la señal.
    """
    x_arr = np.asarray(x, dtype=float)
    h_arr = np.asarray(h, dtype=float)
    n_max = max(x_arr.size, h_arr.size)
    p = int(np.ceil(np.log2(max(1, n_max))))
    m_circ = 1 << p

    x_pad = np.pad(x_arr, (0, m_circ - x_arr.size), mode="constant")
    h_pad = np.pad(h_arr, (0, m_circ - h_arr.size), mode="constant")

    y_circ = _fft_radix2(
        _fft_radix2(x_pad, inversa=False) * _fft_radix2(h_pad, inversa=False),
        inversa=True,
    )
    return np.real(y_circ[:n_max])
