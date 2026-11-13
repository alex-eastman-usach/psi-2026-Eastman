# PSI 2-2026 · Procesamiento de Señales e Imágenes (13318)

[![CI](https://github.com/alex-eastman-usach/psi-2026-Eastman/actions/workflows/ci.yml/badge.svg)](https://github.com/alex-eastman-usach/psi-2026-Eastman/actions/workflows/ci.yml)

Repositorio de laboratorios del curso **Procesamiento de Señales e Imágenes (13318)** de la Universidad de Santiago de Chile (USACH). Incluye la infraestructura reproducible de la **Actividad 1** y el desarrollo del **Laboratorio 1 (Unidades 1–2: Del muestreo al filtrado)**.

## Integrantes

- Alexander Eastman (`alexander.eastman@usach.cl`)
- Juan Campos (`juan.campos.a@usach.cl`)
- Francisco Briones (`francisco.briones.g@usach.cl`)

---

## 1. Instalación y uso local (sin Docker)

Requisitos: Python $\ge 3.11$ y `git`.

```powershell
# Crear y activar entorno virtual (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# En Linux / macOS:
# python3 -m venv .venv && source .venv/bin/activate

# Actualizar pip e instalar el paquete en modo editable junto con herramientas de desarrollo
pip install --upgrade pip
pip install -e ".[dev]"
```

### Ejecutar pruebas y revisión de estilo

```powershell
pytest -v
ruff check .
```

### Comandos de consola disponibles

- **Reproducción completa de figuras, tablas e informe del Laboratorio 1 (`psi-lab1`)**: ejecuta los experimentos de Muestreo y Aliasing, DFT/Resolución/Fuga Espectral y Convolución, guardando todas las figuras (`.png`) y tablas (`.csv`) en `out/lab1/` y compilando `informe/lab1.pdf`.
  ```powershell
  psi-lab1 -o out/lab1
  ```

---

## 2. Ejecución con Docker y Docker Compose

El proyecto cuenta con un `Dockerfile` multi-etapa (`base`, `dev`, `runtime`) y un archivo `compose.yaml` (basados en la infraestructura reproducible de la Actividad 1) que permiten correr pruebas y regenerar todos los resultados en un contenedor aislado:

```powershell
# Construir las imagenes de los servicios
docker compose build

# Correr la suite completa de pruebas unitarias
docker compose run --rm tests

# Regenerar todas las figuras, tablas e informe del Laboratorio 1 (en ./out/lab1/ e ./informe/lab1.pdf)
docker compose run --rm lab1
```

---

## 3. Documentación de la Interfaz Pública del Paquete (`psi_lab.lab1`)

### Parte Común Laboratorio 1 (`psi_lab.lab1`)

#### 1 · Muestreo y aliasing (`psi_lab.lab1.muestreo`)

- **`muestrear(fun, fs, duracion)`**
  - **Parámetros**:
    - `fun` (`callable`): función continua `fun(t)` que recibe un arreglo 1D de tiempos en segundos $[\text{s}]$ y retorna el arreglo de valores de la señal.
    - `fs` (`float`): frecuencia de muestreo en $[\text{Hz}]$.
    - `duracion` (`float`): duración total del intervalo de observación en $[\text{s}]$.
  - **Retorno**: tupla `(t, x)` de dos `np.ndarray` 1D (`float`) de largo $N = \text{round}(f_s \cdot \text{duracion})$, evaluados en $t_n = n / f_s$ para $n = 0, \dots, N-1$.

- **`frecuencia_alias(f, fs)`**
  - **Parámetros**:
    - `f` (`float` o `np.ndarray`): frecuencia real de la senoidal continua en $[\text{Hz}]$.
    - `fs` (`float`): frecuencia de muestreo en $[\text{Hz}]$.
  - **Retorno**: frecuencia aparente plegada en la primera zona de Nyquist $[0, f_s / 2]$ en $[\text{Hz}]$ (`float` si `f` es escalar, `np.ndarray` si `f` es arreglo).

#### 2 · DFT, resolución y fuga espectral (`psi_lab.lab1.dft`)

- **`dft(x)`**
  - **Parámetros**:
    - `x` (`array_like` 1D de largo $N$): señal discreta real o compleja en el dominio del tiempo.
  - **Retorno**: `np.ndarray` 1D (`complex`) de largo $N$ con los coeficientes $X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}$ calculados por definición en $O(N^2)$.

- **`idft(X)`**
  - **Parámetros**:
    - `X` (`array_like` 1D de largo $N$): coeficientes espectrales complejos $X[k]$.
  - **Retorno**: `np.ndarray` 1D (`complex`) de largo $N$ con la secuencia temporal reconstruida $x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{+j 2\pi k n / N}$.

- **`espectro(x, fs, ventana="rect", nfft=None)`**
  - **Parámetros**:
    - `x` (`array_like` 1D de largo $N$): señal discreta real en el tiempo.
    - `fs` (`float`): frecuencia de muestreo en $[\text{Hz}]$.
    - `ventana` (`str`, opcional): tipo de ventana temporal en `{"rect", "hann", "hamming"}` (por defecto `"rect"`).
    - `nfft` (`int` o `None`, opcional): largo de la DFT; si `nfft > len(x)`, rellena con ceros al final tras aplicar la ventana.
  - **Retorno**: tupla `(freqs, amp)` de dos `np.ndarray` 1D (`float`) de largo $\lfloor M/2 \rfloor + 1$ (donde $M = \text{nfft}$ o $N$), con el eje de frecuencias en $[\text{Hz}]$ en $[0, f_s/2]$ y las amplitudes unilaterales normalizadas por $N$.

#### 3 · Convolución (`psi_lab.lab1.convolucion`)

- **`convolucion_directa(x, h)`**
  - **Parámetros**:
    - `x` (`array_like` 1D de largo $N_x$): señal de entrada.
    - `h` (`array_like` 1D de largo $N_h$): respuesta al impulso del filtro.
  - **Retorno**: `np.ndarray` 1D (`float`) de largo $L = N_x + N_h - 1$ con la convolución lineal completa calculada por definición directa en $O(N_x \cdot N_h)$.

- **`convolucion_fft(x, h)`**
  - **Parámetros**:
    - `x` (`array_like` 1D de largo $N_x$): señal de entrada.
    - `h` (`array_like` 1D de largo $N_h$): respuesta al impulso del filtro.
  - **Retorno**: `np.ndarray` 1D (`float`) de largo $L = N_x + N_h - 1$ con la convolución lineal calculada mediante el teorema de la convolución y algoritmo FFT radix-2 propio con relleno de ceros hasta $M = 2^{\lceil \log_2 L \rceil}$ en $O(M \log_2 M)$.
