"""Generacion de senales muestreadas y espectro de amplitud."""
import numpy as np


def senoidal(f, fs, duracion, amplitud=1.0, fase=0.0):
    """Senoidal de frecuencia f [Hz] muestreada a fs [Hz] durante duracion [s].
    Retorna (t, x): instantes de muestreo y valores de la senal.
    """
    n = round(fs * duracion)
    t = np.arange(n) / fs
    return t, amplitud * np.sin(2 * np.pi * f * t + fase)
def espectro(x, fs):
    """Espectro de amplitud de un lado de la senal real x muestreada a fs [Hz].
    Retorna (freqs, amp): frecuencias [Hz] y amplitud de cada componente.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    amp = np.abs(np.fft.rfft(x)) / n
    # Duplicar todo salvo DC (y Nyquist si n es par): la energia de las
    # frecuencias negativas se suma a las positivas.
    amp[1 : (n + 1) // 2] *= 2
    return freqs, amp