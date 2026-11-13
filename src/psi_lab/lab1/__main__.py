"""Script principal del Laboratorio 1 (psi-lab1).
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from psi_lab.lab1.convolucion import (
    convolucion_circular_fft,
    convolucion_directa,
    convolucion_fft,
)
from psi_lab.lab1.dft import dft, espectro
from psi_lab.lab1.muestreo import frecuencia_alias, muestrear


def crear_coseno(f):
    """Devuelve la funcion cos(2*pi*f*t) para pasarla a muestrear()."""
    return lambda t: np.cos(2 * np.pi * f * t)


def experimento_tarea1_aliasing(dir_salida):
    """Tarea 1: tabla de frecuencias aparentes y grafico de aliasing."""
    fs = 10.0
    duracion = 2.0
    frecuencias = [3.0, 7.0, 13.0, 23.0]

    filas = []
    print("\n--- Tarea 1: Muestreo y Aliasing (fs = 10 Hz) ---")
    print("f real [Hz] | f predicha [Hz] | f medida [Hz]")

    for f in frecuencias:
        _, x = muestrear(crear_coseno(f), fs=fs, duracion=duracion)
        freqs, amp = espectro(x, fs=fs)

        f_medida = float(freqs[np.argmax(amp)])
        f_pred = frecuencia_alias(f, fs)
        filas.append([f, fs, f_pred, f_medida])
        print(f"{f} Hz -> predicha: {f_pred} Hz | medida: {f_medida} Hz")

    # Guardamos la tabla en CSV
    ruta_csv = dir_salida / "tabla_aliasing.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["f_real_Hz", "fs_Hz", "f_alias_predicha_Hz", "f_espectro_medida_Hz"])
        writer.writerows(filas)

    # Figura 1: comparamos f1 = 3 Hz y f2 = 7 Hz muestreadas muy fino (1000 Hz)
    # contra las muestras tomadas a fs = 10 Hz.
    t_fino, x1_fino = muestrear(lambda t: np.cos(2 * np.pi * 3.0 * t), fs=1000.0, duracion=1.0)
    t_fino, x2_fino = muestrear(lambda t: np.cos(2 * np.pi * 7.0 * t), fs=1000.0, duracion=1.0)
    t_disc, x1_disc = muestrear(lambda t: np.cos(2 * np.pi * 3.0 * t), fs=fs, duracion=1.0)
    t_disc, x2_disc = muestrear(lambda t: np.cos(2 * np.pi * 7.0 * t), fs=fs, duracion=1.0)

    # Igual que en cli.py de la Actividad 1: creamos Figure y subplots(2, 1)
    fig = Figure(figsize=(8, 6))
    ax_t, ax_f = fig.subplots(2, 1)

    ax_t.plot(t_fino, x1_fino, label="Continua f1 = 3 Hz", lw=1.2)
    ax_t.plot(t_fino, x2_fino, "--", label="Continua f2 = 7 Hz", lw=1.0)
    ax_t.plot(t_disc, x1_disc, "ro", label="Muestras a fs = 10 Hz")
    ax_t.set(xlabel="t [s]", ylabel="x(t)", title="Aliasing en el tiempo (3 Hz y 7 Hz a fs = 10 Hz)")
    ax_t.grid(True)
    ax_t.legend()

    freqs1, amp1 = espectro(x1_disc, fs=fs)
    freqs2, amp2 = espectro(x2_disc, fs=fs)
    ax_f.plot(freqs1, amp1, "o-", label="Espectro f1 = 3 Hz")
    ax_f.plot(freqs2, amp2, "x--", label="Espectro f2 = 7 Hz")
    ax_f.set(xlabel="f [Hz]", ylabel="|X(f)|", title="Espectro de amplitud (ambas caen en 3 Hz)")
    ax_f.grid(True)
    ax_f.legend()

    fig.tight_layout()
    fig.savefig(dir_salida / "fig1_aliasing_tiempo.png", dpi=120)


def experimento_tarea2_dft(dir_salida):
    """Tarea 2: tiempos de DFT vs FFT, resolucion, relleno con ceros y ventanas."""
    rng = np.random.default_rng(0)

    # 1. Tiempos de dft vs np.fft.fft para N = 2^4 .. 2^12
    tamanos_n = [2**p for p in range(4, 13)]
    tiempos_dft = []
    tiempos_fft = []

    for n in tamanos_n:
        x = rng.normal(size=n)

        t0 = time.perf_counter()
        dft(x)
        tiempos_dft.append(time.perf_counter() - t0)

        # La FFT es muy rapida, promediamos 20 corridas para medir bien el tiempo
        t0 = time.perf_counter()
        for _ in range(20):
            np.fft.fft(x)
        tiempos_fft.append((time.perf_counter() - t0) / 20)

    # Calculamos la pendiente en escala log-log con un ajuste lineal simple
    pendiente_dft, _ = np.polyfit(np.log10(tamanos_n[3:]), np.log10(tiempos_dft[3:]), 1)
    pendiente_fft, _ = np.polyfit(np.log10(tamanos_n[3:]), np.log10(tiempos_fft[3:]), 1)
    print("\n--- Tarea 2: Pendientes en escala log-log ---")
    print(f"Pendiente DFT directa: {pendiente_dft:.2f} | Pendiente FFT: {pendiente_fft:.2f}")

    fig_t = Figure(figsize=(8, 5))
    ax_t = fig_t.subplots(1, 1)
    ax_t.loglog(tamanos_n, tiempos_dft, "o-", label=f"dft propia (pendiente = {pendiente_dft:.2f})")
    ax_t.loglog(tamanos_n, tiempos_fft, "s-", label=f"np.fft.fft (pendiente = {pendiente_fft:.2f})")
    ax_t.set(xlabel="N", ylabel="Tiempo [s]", title="Tiempo de ejecucion: DFT O(N^2) vs FFT O(N log N)")
    ax_t.grid(True)
    ax_t.legend()
    fig_t.tight_layout()
    fig_t.savefig(dir_salida / "fig2_complejidad_dft.png", dpi=120)

    # 2. Resolucion: dos tonos de 50 y 52 Hz con fs = 1000 Hz y T en {0.25, 0.5, 1, 2} s
    fs = 1000.0
    duraciones = [0.25, 0.5, 1.0, 2.0]
    fig_res = Figure(figsize=(8, 6))
    axes = fig_res.subplots(2, 2).ravel()

    for ax, dur in zip(axes, duraciones):
        _, x = muestrear(
            lambda t: np.cos(2 * np.pi * 50.0 * t) + np.cos(2 * np.pi * 52.0 * t),
            fs=fs,
            duracion=dur,
        )
        freqs, amp = espectro(x, fs=fs, ventana="rect")
        ax.plot(freqs, amp, "o-", ms=3, lw=1.0)
        ax.set_xlim(40, 62)
        ax.set(xlabel="f [Hz]", ylabel="|X(f)|", title=f"T = {dur} s (df = {1 / dur:.1f} Hz)")
        ax.grid(True)

    fig_res.tight_layout()
    fig_res.savefig(dir_salida / "fig3_resolucion_espectral.png", dpi=120)

    # 3. Relleno con ceros: T = 0.25 s con nfft = N vs nfft = 8N
    _, x_corto = muestrear(
        lambda t: np.cos(2 * np.pi * 50.0 * t) + np.cos(2 * np.pi * 52.0 * t),
        fs=fs,
        duracion=0.25,
    )
    freqs_sin, amp_sin = espectro(x_corto, fs=fs, ventana="rect", nfft=None)
    freqs_pad, amp_pad = espectro(x_corto, fs=fs, ventana="rect", nfft=8 * x_corto.size)

    fig_pad = Figure(figsize=(8, 5))
    ax_p = fig_pad.subplots(1, 1)
    ax_p.plot(freqs_pad, amp_pad, "-", label="Con relleno de ceros (nfft = 8N)")
    ax_p.plot(freqs_sin, amp_sin, "ro", label="Sin relleno (nfft = N)")
    ax_p.set_xlim(35, 65)
    ax_p.set(xlabel="f [Hz]", ylabel="|X(f)|", title="Relleno con ceros en T = 0.25 s")
    ax_p.grid(True)
    ax_p.legend()
    fig_pad.tight_layout()
    fig_pad.savefig(dir_salida / "fig4_relleno_ceros.png", dpi=120)

    # 4. Fuga y ventanas: tono de 50 Hz (amplitud 1) y tono de 80 Hz (amplitud 1e-3)
    # Usamos duracion = 0.51 s para que el tono de 50 Hz tenga 25.5 ciclos y exista fuga.
    _, x_fuga = muestrear(
        lambda t: np.cos(2 * np.pi * 50.0 * t) + 1e-3 * np.cos(2 * np.pi * 80.0 * t),
        fs=fs,
        duracion=0.51,
    )
    f_rect, a_rect = espectro(x_fuga, fs=fs, ventana="rect", nfft=2048)
    f_hann, a_hann = espectro(x_fuga, fs=fs, ventana="hann", nfft=2048)

    # Pasamos las amplitudes a decibeles: 20 * log10(amp)
    db_rect = 20 * np.log10(np.maximum(a_rect, 1e-12))
    db_hann = 20 * np.log10(np.maximum(a_hann, 1e-12))

    fig_fuga = Figure(figsize=(8, 5))
    ax_f = fig_fuga.subplots(1, 1)
    ax_f.plot(f_rect, db_rect, label="Ventana rectangular", lw=1.0)
    ax_f.plot(f_hann, db_hann, label="Ventana Hann", lw=1.2)
    ax_f.set_xlim(20, 120)
    ax_f.set_ylim(-110, 5)
    ax_f.set(xlabel="f [Hz]", ylabel="Magnitud [dB]", title="Fuga espectral: Rectangular vs Hann")
    ax_f.grid(True)
    ax_f.legend()
    fig_fuga.tight_layout()
    fig_fuga.savefig(dir_salida / "fig5_fuga_y_ventanas.png", dpi=120)


def experimento_tarea3_convolucion(dir_salida):
    """Tarea 3: convolucion circular vs lineal y comparacion de tiempos."""
    # 1. Efecto de no rellenar con ceros en convolucion_fft (convolucion circular)
    x = np.zeros(64)
    x[10:55] = 1.0
    h = 0.85 ** np.arange(25)

    y_lineal = convolucion_fft(x, h)
    y_circ = convolucion_circular_fft(x, h)

    fig_c = Figure(figsize=(8, 5))
    ax_c = fig_c.subplots(1, 1)
    ax_c.plot(y_lineal, "o-", ms=3, label="Convolucion lineal (con relleno de ceros)")
    ax_c.plot(y_circ, "r--", label="Convolucion circular (sin relleno de ceros)")
    ax_c.set(xlabel="Muestra n", ylabel="y[n]", title="Convolucion lineal vs circular (sin zero-padding)")
    ax_c.grid(True)
    ax_c.legend()
    fig_c.tight_layout()
    fig_c.savefig(dir_salida / "fig6_convolucion_circular_vs_lineal.png", dpi=120)

    # 2. Comparacion de tiempos segun el largo del filtro
    rng = np.random.default_rng(0)
    x_larga = rng.normal(size=4096)
    largos_h = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    t_directa = []
    t_fft = []

    print("\n--- Tarea 3: Tiempos de convolucion (N_x = 4096) ---")
    for largo in largos_h:
        h_filtro = rng.normal(size=largo)

        t0 = time.perf_counter()
        convolucion_directa(x_larga, h_filtro)
        td = time.perf_counter() - t0
        t_directa.append(td)

        t0 = time.perf_counter()
        convolucion_fft(x_larga, h_filtro)
        tf = time.perf_counter() - t0
        t_fft.append(tf)

        print(f"Largo h = {largo} -> Directa: {td * 1000:.2f} ms | FFT: {tf * 1000:.2f} ms")

    fig_tc = Figure(figsize=(8, 5))
    ax_tc = fig_tc.subplots(1, 1)
    ax_tc.loglog(largos_h, t_directa, "o-", label="convolucion_directa")
    ax_tc.loglog(largos_h, t_fft, "s-", label="convolucion_fft")
    ax_tc.set(xlabel="Largo del filtro N_h", ylabel="Tiempo [s]", title="Tiempos: Convolucion directa vs FFT")
    ax_tc.grid(True)
    ax_tc.legend()
    fig_tc.tight_layout()
    fig_tc.savefig(dir_salida / "fig7_tiempos_convolucion.png", dpi=120)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="psi-lab1",
        description="Genera las figuras y tablas de la Parte Comun del Laboratorio 1.",
    )
    parser.add_argument("-o", "--salida", default="out/lab1", help="Carpeta de salida")
    args = parser.parse_args(argv)

    dir_salida = Path(args.salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    experimento_tarea1_aliasing(dir_salida)
    experimento_tarea2_dft(dir_salida)
    experimento_tarea3_convolucion(dir_salida)

    print("Resultados guardados en", dir_salida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
