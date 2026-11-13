import numpy as np
import pytest

from psi_lab.lab1.dft import dft, espectro, idft


def test_dft_coincide_con_numpy_fft():
    # Tarea 2.1a: verificamos que nuestra DFT por definicion coincida con np.fft.fft
    # tanto para señales reales como complejas de distintos largos.
    rng = np.random.default_rng(42)
    for n in (16, 37, 64):
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        assert np.allclose(dft(x), np.fft.fft(x), atol=1e-10)


def test_idft_recupera_senal_original():
    # Tarea 2.1b: la composicion idft(dft(x)) debe reconstruir exactamente la señal x.
    rng = np.random.default_rng(123)
    x = rng.normal(size=50)
    x_rec = idft(dft(x))
    assert np.allclose(x_rec, x, atol=1e-10)


def test_simetria_conjugada_senal_real():
    # Tarea 2.1c: para toda señal real x[n], su DFT cumple simetria hermítica
    rng = np.random.default_rng(7)
    n = 48
    x = rng.normal(size=n)
    X = dft(x)

    k = np.arange(1, n)
    assert np.allclose(X[n - k], np.conj(X[k]), atol=1e-10)


def test_teorema_de_parseval():
    # Tarea 2.1d: conservacion de la energia entre tiempo y frecuencia:
    rng = np.random.default_rng(99)
    x = rng.normal(loc=0.5, scale=2.0, size=64)
    X = dft(x)

    energia_tiempo = np.sum(np.abs(x) ** 2)
    energia_frecuencia = np.sum(np.abs(X) ** 2) / x.size
    assert energia_tiempo == pytest.approx(energia_frecuencia, rel=1e-10)


@pytest.mark.parametrize("ventana", ["rect", "hann", "hamming"])
def test_espectro_ventanas_y_relleno_con_ceros(ventana):
    fs = 200.0
    n = 100
    t = np.arange(n) / fs
    f0 = 30.0
    x = 2.5 * np.cos(2 * np.pi * f0 * t)

    freqs, amp = espectro(x, fs=fs, ventana=ventana, nfft=400)
    assert freqs.size == amp.size == 201
    assert freqs[np.argmax(amp)] == pytest.approx(f0, abs=fs / 400)
    if ventana == "rect":
        assert amp.max() == pytest.approx(2.5, rel=1e-3)
