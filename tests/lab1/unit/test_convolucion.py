import numpy as np
import pytest

from psi_lab.lab1.convolucion import convolucion_directa, convolucion_fft


@pytest.mark.parametrize("funcion_conv", [convolucion_directa, convolucion_fft])
def test_convolucion_coincide_con_numpy(funcion_conv):
    # Tarea 3.1a: ambas implementaciones deben coincidir con np.convolve
    # y producir un arreglo de largo len(x) + len(h) - 1.
    rng = np.random.default_rng(101)
    x = rng.normal(size=75)
    h = rng.normal(size=23)

    esperado = np.convolve(x, h, mode="full")
    obtenido = funcion_conv(x, h)

    assert obtenido.size == x.size + h.size - 1
    assert np.allclose(obtenido, esperado, atol=1e-10)


@pytest.mark.parametrize("funcion_conv", [convolucion_directa, convolucion_fft])
def test_convolucion_con_impulso_desplaza_senal(funcion_conv):
    # Tarea 3.1b: convolucionar x[n] con un impulso unitario desplazado delta[n - d]
    # debe desplazar la señal d muestras hacia la derecha (precedida por d ceros).
    rng = np.random.default_rng(202)
    x = rng.normal(size=40)
    desplazamiento = 7
    largo_h = 12
    h = np.zeros(largo_h)
    h[desplazamiento] = 1.0

    y = funcion_conv(x, h)
    # Las primeras d muestras deben ser cero y luego debe aparecer x intacta
    assert np.allclose(y[:desplazamiento], 0.0, atol=1e-12)
    assert np.allclose(y[desplazamiento : desplazamiento + x.size], x, atol=1e-10)
    assert np.allclose(y[desplazamiento + x.size :], 0.0, atol=1e-12)


@pytest.mark.parametrize("funcion_conv", [convolucion_directa, convolucion_fft])
def test_convolucion_es_conmutativa(funcion_conv):
    # Tarea 3.1c: propiedad conmutativa x * h == h * x.
    rng = np.random.default_rng(303)
    x = rng.normal(size=53)
    h = rng.normal(size=19)

    assert np.allclose(funcion_conv(x, h), funcion_conv(h, x), atol=1e-10)
