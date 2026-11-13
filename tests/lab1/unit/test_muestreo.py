import numpy as np
import pytest

from psi_lab.lab1.dft import espectro
from psi_lab.lab1.muestreo import frecuencia_alias, muestrear


def test_muestrear_largo_y_valores():
    # Extiende la prueba de cantidad de muestras N = round(fs * duracion) de la
    # actividad 1, verificando ademas el instante inicial y el paso Ts = 1/fs.
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
        (3.0, 10.0, 3.0),   # Bajo Nyquist (fs/2 = 5 Hz): sin aliasing
        (7.0, 10.0, 3.0),   # Plegado sobre Nyquist: |7 - 10| = 3 Hz
        (13.0, 10.0, 3.0),  # Segundo periodo: |13 - 10| = 3 Hz
        (23.0, 10.0, 3.0),  # Tercer periodo: |23 - 2*10| = 3 Hz
        (5.0, 10.0, 5.0),   # Exactamente en la frecuencia de Nyquist
        (10.0, 10.0, 0.0),  # Multiplo entero de fs -> colapsa a DC (0 Hz)
    ],
)
def test_frecuencia_alias_casos_teoricos(f, fs, esperado):
    assert frecuencia_alias(f, fs) == pytest.approx(esperado)


def test_dos_frecuencias_producen_mismas_muestras():
    # Dos cosenos cuyas frecuencias se pliegan al mismo alias (3 Hz y 7 Hz con fs=10 Hz)
    # son indistinguibles en los instantes de muestreo t = n / fs.
    fs = 10.0
    duracion = 1.0
    _, x1 = muestrear(lambda t: np.cos(2 * np.pi * 3.0 * t), fs=fs, duracion=duracion)
    _, x2 = muestrear(lambda t: np.cos(2 * np.pi * 7.0 * t), fs=fs, duracion=duracion)
    assert np.allclose(x1, x2)


def crear_coseno(f):
    return lambda t: np.cos(2 * np.pi * f * t)


def test_pico_espectro_cae_en_frecuencia_alias_aleatoria():
    # En la actividad 1 se usa `freqs[np.argmax(amp)] == pytest.approx(f)`
    # para f < fs/2. Aqui para la tarea 1.4 se usa
    # cualquier combinacion aleatoria de f y fs usando `frecuencia_alias(f, fs)`.
    rng = np.random.default_rng(2026)
    duracion = 1.0

    for _ in range(10):
        fs = float(rng.integers(80, 300))
        f = float(rng.uniform(1.0, 3.5 * fs))

        _, x = muestrear(crear_coseno(f), fs=fs, duracion=duracion)
        freqs, amp = espectro(x, fs=fs)

        f_pico = freqs[np.argmax(amp)]
        f_predicha = frecuencia_alias(f, fs)
        # El pico discreto cae a lo sumo a medio paso de grilla (df = 1 / duracion)
        assert f_pico == pytest.approx(f_predicha, abs=1.0 / duracion)
