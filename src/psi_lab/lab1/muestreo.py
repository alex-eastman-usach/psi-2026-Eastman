"""Muestreo uniforme en el tiempo y calculo de frecuencia aparente (aliasing)."""
#Esta parte es pasar de señal continua a muestras y alising

import numpy as np


def muestrear(fun, fs, duracion):
    """Evalua fun(t) en t = n / fs, n = 0..N-1. Retorna (t, x)."""
    # Esta parte viene de la funcion
    # senoidal(f, fs, duracion) de la Actividad 1. Donde fijabamos la
    # evaluacion a amplitud * np.sin(2 * np.pi * f * t + fase); aqui la generalizamos
    # recibiendo cualquier funcion continua `fun(t)`.

    # Se usa round(fs * duracion) en lugar de int() para evitar que la
    # representacion binaria de flotantes trunque una muestra.
    n = round(fs * duracion)

    # Construimos los instantes discretos t_n = n * Ts con Ts = 1 / fs
    # Se divide por fs en vez de usar np.linspace para asegurar que el paso
    # entre muestras consecutivas sea exactamente 1/fs y que el intervalo
    # sea semiabierto [0, T), lo cual evita duplicar el punto final en señales periodicas.
    t = np.arange(n, dtype=float) / fs

    # Evaluamos la funcion continua de forma vectorizada sobre el arreglo t
    # y forzamos dtype=float para garantizar consistencia numerica.
    x = np.asarray(fun(t), dtype=float)
    return t, x


def frecuencia_alias(f, fs):
    """Frecuencia aparente en [0, fs/2] de una senoidal de f Hz muestreada a fs."""
    # Al muestrear cos(2*pi*f*t) en t = n/fs se obtiene cos(2*pi*(f/fs)*n).
    # Como la exponencial discreta es periodica cada multiplo entero de fs y
    # el coseno es una funcion par (cos(-x) = cos(x)), cualquier frecuencia f
    # fuera de la banda base [0, fs/2] se pliega sobre dicho intervalo.
    #
    # Paso 1: llevamos |f| al intervalo principal [0, fs) mediante el modulo fs.
    r = np.mod(np.abs(f), fs)

    # Paso 2: si el residuo supera la frecuencia de Nyquist (fs / 2), la componente
    # cae en la mitad superior del periodo espectral y se refleja como fs - r
    # (correspondiente al conjugado de frecuencia negativa).
    f_aparente = np.where(r > fs / 2.0, fs - r, r)

    if np.ndim(f) == 0:
        return float(f_aparente)
    return np.asarray(f_aparente, dtype=float)
