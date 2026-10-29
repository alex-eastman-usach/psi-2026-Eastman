import numpy as np
import pytest

from psi_lab.lab1.muestreo import frecuencia_alias, muestrear


def test_muestrear_largo_y_valores():
    fs = 100.0
    duracion = 1.5
    t, x = muestrear(lambda t: np.cos(2 * np.pi * 5.0 * t), fs=fs, duracion=duracion)
    assert t.size == x.size == 150
    assert t[0] == pytest.approx(0.0)
    assert t[1] - t[0] == pytest.approx(1.0 / fs)
    assert x[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("f", "fs", "esperado"),
    [
        (3.0, 10.0, 3.0),   # Bajo Nyquist (fs/2 = 5 Hz)
        (7.0, 10.0, 3.0),   # Plegado: 10 - 7 = 3 Hz
        (13.0, 10.0, 3.0),  # 13 - 10 = 3 Hz
        (23.0, 10.0, 3.0),  # 23 - 2*10 = 3 Hz
        (5.0, 10.0, 5.0),   # Justo en Nyquist
        (10.0, 10.0, 0.0),  # Multiplo de fs -> DC (0 Hz)
    ],
)
def test_frecuencia_alias_casos_teoricos(f, fs, esperado):
    assert frecuencia_alias(f, fs) == pytest.approx(esperado)


def test_dos_frecuencias_producen_mismas_muestras():
    fs = 10.0
    duracion = 1.0
    _, x1 = muestrear(lambda t: np.cos(2 * np.pi * 3.0 * t), fs=fs, duracion=duracion)
    _, x2 = muestrear(lambda t: np.cos(2 * np.pi * 7.0 * t), fs=fs, duracion=duracion)
    assert np.allclose(x1, x2)