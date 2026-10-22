"""Comando de consola psi-demo."""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure  # sin pyplot: no abre ventanas (Docker, CI)

from psi_lab.senales import espectro, senoidal


def main(argv=None):
    parser = argparse.ArgumentParser(
    prog="psi-demo",
    description="Suma de senoidales con ruido y su espectro de amplitud.",
    )
    parser.add_argument("-f", "--frecuencias", type=float, nargs="+",
    default=[50.0, 120.0], help="frecuencias [Hz]")

    parser.add_argument("--fs", type=float, default=1000.0,
    help="frecuencia de muestreo [Hz]")
    parser.add_argument("-d", "--duracion", type=float, default=1.0,

    help="duracion [s]")

    parser.add_argument("--ruido", type=float, default=0.5,

    help="desviacion estandar del ruido gaussiano")

    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument("-o", "--salida", default="out/demo.png")
    args = parser.parse_args(argv)
    t, x = senoidal(args.frecuencias[0], args.fs, args.duracion)
    for f in args.frecuencias[1:]:
        x = x + senoidal(f, args.fs, args.duracion)[1]
    rng = np.random.default_rng(args.semilla)
    x = x + rng.normal(0.0, args.ruido, size=x.size)
    freqs, amp = espectro(x, args.fs)
    k = len(args.frecuencias)
    picos = np.sort(freqs[np.argsort(amp)[-k:]])
    print("Picos detectados [Hz]:", ", ".join(f"{p:.1f}" for p in picos))
    fig = Figure(figsize=(8, 6))
    ax_t, ax_f = fig.subplots(2, 1)
    ax_t.plot(t, x, lw=0.8)
    ax_t.set(xlabel="t [s]", ylabel="x(t)", title="Senal en el tiempo")
    ax_f.plot(freqs, amp, lw=0.8)
    ax_f.set(xlabel="f [Hz]", ylabel="|X(f)|", title="Espectro de amplitud")
    fig.tight_layout()
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.salida, dpi=120)
    print("Figura guardada en", args.salida)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())