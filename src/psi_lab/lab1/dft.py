"""Transformada Discreta de Fourier (DFT/IDFT) por definicion y analisis espectral."""

import numpy as np


def dft(x):
    """DFT por definicion, O(N^2): X[k] = sum_n x[n] exp(-2j pi k n / N)."""
    # Convertimos la entrada a arreglo complejo para admitir tanto señales
    # reales como complejas sin perdida de precision en el producto matricial.
    x_arr = np.asarray(x, dtype=complex)
    n_total = x_arr.size
    if n_total == 0:
        return np.array([], dtype=complex)

    # En lugar de usar dos ciclos for anidados en Python construimos la matriz de
    # Fourier W de tamaño N x N mediante el producto exterior de los indices
    # k (columna) y n (fila). Esto ejecuta exactamente las N^2 exponenciales
    # y N^2 multiplicaciones de la definicion teorica O(N^2), sin usar ninguna
    # descomposicion tipo mariposa (Cooley-Tukey).
    n = np.arange(n_total, dtype=float)
    k = n.reshape((n_total, 1))
    w_mat = np.exp(-2j * np.pi * k * n / n_total)

    return w_mat @ x_arr


def idft(X):
    """DFT inversa por definicion."""
    # Por definicion: x[n] = (1/N) * sum_{k=0}^{N-1} X[k] exp(+2j pi k n / N).
    # El nucleo solo cambia el signo del exponente respecto a la DFT directa
    # y añade el factor de escala 1/N para conservar la energia/amplitud.
    x_freq = np.asarray(X, dtype=complex)
    n_total = x_freq.size
    if n_total == 0:
        return np.array([], dtype=complex)

    k = np.arange(n_total, dtype=float)
    n = k.reshape((n_total, 1))
    w_inv = np.exp(2j * np.pi * n * k / n_total)

    return (w_inv @ x_freq) / n_total


def _obtener_ventana(n, nombre):
    """Genera los coeficientes de la ventana temporal de largo n a mano."""
    nombre_limpio = nombre.lower().strip()
    if n <= 1:
        return np.ones(n, dtype=float)

    # Indice temporal normalizado sobre N - 1 para que la ventana sea simetrica
    # extremo a extremo
    idx = np.arange(n, dtype=float)
    fase = 2.0 * np.pi * idx / (n - 1)

    if nombre_limpio == "rect":
        # Ventana rectangular: truncamiento abrupto, lobulo principal mas angosto
        # pero lobulos laterales altos (-13 dB) que producen fuerte fuga espectral.
        return np.ones(n, dtype=float)
    if nombre_limpio == "hann":
        # Ventana de Hann: llega suavemente a cero en los bordes, reduciendo
        # los lobulos laterales
        return 0.5 - 0.5 * np.cos(fase)
    if nombre_limpio == "hamming":
        # Ventana de Hamming: optimiza los coeficientes para cancelar el primer lobulo lateral
        return 0.54 - 0.46 * np.cos(fase)

    raise ValueError(
        f"Ventana '{nombre}' no soportada. Use 'rect', 'hann' o 'hamming'."
    )


def espectro(x, fs, ventana="rect", nfft=None):
    """Espectro de amplitud de un lado. ventana en {"rect", "hann", "hamming"};
    nfft > len(x) rellena con ceros. Retorna (freqs, amp).
    """
    # Esta funcion es como la  `espectro(x, fs)` de la Actividad 1. Se calculaba el espectro unilateral con
    # `np.fft.rfft(x)` y ventana rectangular implicita; aqui mantenemos exactamente
    # la misma logica de normalizacion (`/ n`) y duplicacion del espectro unilateral
    # (`amp[1 : (m_fft + 1) // 2] *= 2`) pero:
    # Reemplazamos `np.fft.rfft` por nuestra propia `dft(x)` hecha a mano.
    # Agregamos multiplicacion previa por ventana ("rect", "hann", "hamming").
    # Agregamos soporte para relleno con ceros (`nfft > len(x)`).
    x_arr = np.asarray(x, dtype=float)
    n_senal = x_arr.size
    if n_senal == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # 1. Enventanado en el dominio del tiempo antes del relleno con ceros
    # la ventana debe aplicarse sobre las N muestras reales de la señal
    # si se aplicara despues del zero-padding, la discontinuidad en el borde
    # de los datos reales permaneceria intacta.
    w = _obtener_ventana(n_senal, ventana)
    x_vent = x_arr * w

    # 2. Relleno con ceros (zero-padding) si se especifica nfft > len(x):
    # agregar ceros al final interpola el espectro en una grilla mas densa de
    # frecuencias (paso fs / nfft), mejorando la visualizacion de la envolvente
    # (DTFT) aunque la resolucion fisica sigue limitada por la duracion T = N / fs.
    if nfft is None:
        m_fft = n_senal
        x_proc = x_vent
    else:
        m_fft = int(nfft)
        if m_fft > n_senal:
            x_proc = np.pad(x_vent, (0, m_fft - n_senal), mode="constant")
        else:
            x_proc = x_vent[:m_fft]
            n_senal = m_fft

    # 3. Calculo de la DFT
    x_freq = dft(x_proc)

    # 4. Dado que x es real, |X[M - k]| = |X[k]|. Nos quedamos con la primera mitad
    # (de 0 a M//2) y duplicamos todo salvo DC (indice 0) y Nyquist (si m_fft es par) para sumar
    # la energia de las frecuencias negativas sobre las positivas.
    n_un_lado = m_fft // 2 + 1
    freqs = np.arange(n_un_lado, dtype=float) * (float(fs) / m_fft)

    # Normalizamos por n_senal (cantidad de muestras efectivas de la señal)
    # para que el relleno con ceros no atenue artificialmente la amplitud de los picos.
    amp = np.abs(x_freq[:n_un_lado]) / n_senal
    amp[1 : (m_fft + 1) // 2] *= 2.0

    return freqs, amp
