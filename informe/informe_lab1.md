# Informe Laboratorio 1 · Unidades 1–2: Del Muestreo a la Convolución

**Curso:** Procesamiento de Señales e Imágenes (13318) · 2-2026  
**Universidad de Santiago de Chile (USACH)**  
**Integrantes:** Alexander Eastman, Juan Campos, Francisco Briones  

---

## 1. Objetivo

Desarrollar desde cero, utilizando exclusivamente operaciones vectoriales y matriciales en `NumPy`, las herramientas fundamentales de procesamiento digital de señales correspondientes a las primeras tres secciones de la Parte Común del Laboratorio 1:
1. **Muestreo uniforme y fenómeno de aliasing** en el dominio del tiempo y de la frecuencia.
2. **Transformada Discreta de Fourier (DFT / IDFT)** por definición $O(N^2)$, análisis de complejidad computacional frente a la FFT $O(N \log N)$, límite físico de resolución espectral, interpolación por relleno con ceros (*zero-padding*) y control de fuga espectral mediante funciones de ventana.
3. **Convolución lineal discreta** implementada tanto de forma directa en el dominio del tiempo $O(N_x \cdot N_h)$ como en el dominio de la frecuencia vía el teorema de la convolución y un algoritmo FFT radix-2 propio $O(M \log_2 M)$, contrastando el efecto del *aliasing* temporal (convolución circular) cuando se omite el relleno con ceros.

---

## 2. Parte Común 1 · Muestreo y Aliasing

### 2.1 Deducción y justificación matemática de `frecuencia_alias(f, fs)`

Considérese una señal senoidal continua de frecuencia real $f \ge 0$ muestreada uniformemente a una tasa $f_s = 1 / T_s$ en los instantes $t_n = n / f_s$ ($n = 0, 1, \dots, N-1$):

$$x[n] = \cos(2\pi f t_n) = \cos\left(2\pi \frac{f}{f_s} n\right)$$

En tiempo discreto, las frecuencias están definidas módulo $f_s$ debido a la periodicidad de los armónicos discretos ($\cos(2\pi (f \pm k f_s) n / f_s) = \cos(2\pi f n / f_s \pm 2\pi k n) = \cos(2\pi f n / f_s)$ para todo $k \in \mathbb{Z}$). Adicionalmente, dado que la señal es real y el coseno es una función par ($\cos(-\theta) = \cos(\theta)$), no es posible distinguir el signo de la fase instantánea solo a partir de las muestras reales si la frecuencia cae en la mitad superior del período espectral $(f_s/2, f_s)$.

Sea el residuo modular en el primer período $[0, f_s)$:

$$r = |f| \bmod f_s \in [0, f_s)$$

- **Caso 1 ($0 \le r \le f_s / 2$):** La frecuencia se encuentra dentro de la primera zona de Nyquist. El teorema del muestreo de Nyquist-Shannon se satisface para la componente plegada y la frecuencia aparente observada es directamente $f_{\text{alias}} = r$.
- **Caso 2 ($f_s / 2 < r < f_s$):** La componente supera la frecuencia de Nyquist $f_s / 2$. Al escribir $r = f_s - (f_s - r)$, tenemos:
  $$\cos\left(2\pi \frac{r}{f_s} n\right) = \cos\left(2\pi n - 2\pi \frac{f_s - r}{f_s} n\right) = \cos\left(2\pi \frac{f_s - r}{f_s} n\right)$$
  donde $0 < f_s - r < f_s / 2$. Por lo tanto, la frecuencia se refleja especularmente (plegado o *folding*) respecto a $f_s / 2$, dando lugar a la expresión cerrada implementada en `frecuencia_alias`:

$$f_{\text{alias}}(f, f_s) = \begin{cases} r, & \text{si } r \le \dfrac{f_s}{2} \\[6pt] f_s - r, & \text{si } r > \dfrac{f_s}{2} \end{cases}, \quad \text{con } r = |f| \bmod f_s$$

### 2.2 Decisiones de implementación en `muestreo.py`

1. **Redondeo del número de muestras (`n = round(fs * duracion)`):** En aritmética de punto flotante IEEE 754, productos como `0.1 * 100.0` o `2.4 * 1000.0` pueden representarse internamente como `2399.9999999999995`. Usar truncamiento entero (`int(...)`) perdería una muestra y alteraría el paso espectral $\Delta f = f_s / N$. El uso de `round()` garantiza el número exacto de muestras.
2. **Construcción del eje temporal (`np.arange(n) / fs`):** Se prefiere frente a `np.linspace(0, duracion, n)` porque en procesamiento de señales periódicas la ventana de observación debe cubrir el intervalo semiabierto $[0, T)$ con paso exacto $T_s = 1 / f_s$. Incluir el extremo $t = T$ duplicaría la muestra inicial del siguiente período y rompería la ortogonalidad de la base de la DFT.

### 2.3 Resultados experimentales: Tabla de frecuencias aparentes y gráfico temporal

Al muestrear senoidales de frecuencias $f \in \{3, 7, 13, 23\}\text{ Hz}$ con una frecuencia de muestreo $f_s = 10\text{ Hz}$ (frecuencia de Nyquist $f_N = f_s / 2 = 5\text{ Hz}$), todas ellas se pliegan exactamente sobre la misma frecuencia aparente de $3.0\text{ Hz}$:

| Frecuencia real $f$ $[\text{Hz}]$ | Frecuencia de muestreo $f_s$ $[\text{Hz}]$ | Residuo $r = f \bmod f_s$ $[\text{Hz}]$ | $f_{\text{alias}}$ predicha $[\text{Hz}]$ | $f_{\text{pico}}$ medida en espectro $[\text{Hz}]$ | Explicación física del plegado |
| :---: | :---: | :---: | :---: | :---: | :--- |
| $3.0$ | $10.0$ | $3.0$ | $3.00$ | $3.00$ | Cumple Nyquist ($f < 5\text{ Hz}$), sin aliasing. |
| $7.0$ | $10.0$ | $7.0$ | $3.00$ | $3.00$ | Reflexión sobre Nyquist: $f_s - r = 10 - 7 = 3\text{ Hz}$. |
| $13.0$ | $10.0$ | $3.0$ | $3.00$ | $3.00$ | Traslación de $1\cdot f_s$: $13 - 10 = 3\text{ Hz}$. |
| $23.0$ | $10.0$ | $3.0$ | $3.00$ | $3.00$ | Traslación de $2\cdot f_s$: $23 - 20 = 3\text{ Hz}$. |

En la **Figura 1** (`out/lab1/fig1_aliasing_tiempo.png`) se observa cómo las señales continuas $\cos(2\pi \cdot 3t)$ y $\cos(2\pi \cdot 7t)$ oscilan a velocidades distintas entre los instantes de muestreo, pero se intersectan exactamente en cada instante $t_n = n / 10\text{ s}$, produciendo secuencias discretas idénticas al nivel de precisión de máquina (`np.allclose(x1, x2) == True`) y el mismo espectro unilateral con pico en $3\text{ Hz}$.

![Figura 1: Aliasing en el tiempo y en el espectro](../out/lab1/fig1_aliasing_tiempo.png)

---

## 3. Parte Común 2 · DFT, Resolución y Fuga Espectral

### 3.1 Formulación y decisiones de diseño en `dft.py`

La Transformada Discreta de Fourier (DFT) directa e inversa de una secuencia finita de largo $N$ están definidas por:

$$X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j \frac{2\pi}{N} k n}, \qquad x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \, e^{+j \frac{2\pi}{N} k n}$$

- **Por qué se resolvió mediante la matriz de Vandermonde/Fourier $W$ en vez de bucles `for` en Python:**  
  Evaluar la suma directamente mediante dos ciclos `for` anidados en Python puro introduce un sobrecosto de interpretación de *bytecode* que domina el tiempo de CPU para $N = 2^{12} = 4096$ ($1.67 \times 10^7$ iteraciones). En cambio, construir el producto exterior de índices $\mathbf{k} \cdot \mathbf{n}^T \in \mathbb{R}^{N \times N}$, evaluar la matriz densa $W_{k,n} = e^{-j 2\pi k n / N}$ y ejecutar el producto matriz-vector `W @ x` realiza **exactamente las mismas $N^2$ exponenciales complejas y $N^2$ multiplicaciones-sumas de la definición teórica** (sin ninguna factorización de Cooley-Tukey), respetando fielmente la complejidad asintótica $O(N^2)$ en operaciones de punto flotante dentro de rutinas BLAS compiladas.

### 3.2 Complejidad computacional: `dft` $O(N^2)$ vs `np.fft.fft` $O(N \log N)$

Se midió el tiempo de ejecución para longitudes $N = 2^4, 2^5, \dots, 2^{12}$ ($16$ a $4096$ muestras). En escala log-log ($\log_{10} t$ vs $\log_{10} N$), un algoritmo de complejidad $T(N) = c \cdot N^\alpha$ describe una recta de pendiente $\alpha$:
- Para nuestra `dft(x)` por definición, el ajuste lineal en la región asintótica ($N \ge 128$) arroja una **pendiente empírica de $2.05$** (valor teórico $2.0$), confirmando el escalamiento cuadrático $O(N^2)$: al duplicar $N$, el tiempo de cómputo se multiplica por cuatro.
- Para `np.fft.fft(x)`, la pendiente empírica es **$0.79$** (cercana al comportamiento teórico $O(N \log_2 N)$ de pendiente $\approx 1.0$, ligeramente menor en tamaños intermedios por efectos de caché L1/L2 y vectorización SIMD de `pocketfft`). Para $N = 4096$, la FFT resulta más de tres órdenes de magnitud más rápida que la DFT por definición.

![Figura 2: Complejidad temporal DFT vs FFT](../out/lab1/fig2_complejidad_dft.png)

### 3.3 Resolución espectral ($\Delta f = 1/T$) y separación de dos tonos cercanos

Se analizó la suma de dos tonos puros de igual amplitud en $f_1 = 50\text{ Hz}$ y $f_2 = 52\text{ Hz}$ ($\Delta f_{\text{tonos}} = 2\text{ Hz}$) muestreados a $f_s = 1000\text{ Hz}$ con duraciones $T \in \{0.25, 0.5, 1.0, 2.0\}\text{ s}$:

1. **$T = 0.25\text{ s}$ ($N = 250$, $\Delta f = 1/T = 4.0\text{ Hz}$):** Como el espaciamiento de la grilla espectral ($\Delta f = 4\text{ Hz}$) es el doble de la separación entre tonos ($2\text{ Hz}$), los lóbulos principales de ambos tonos se fusionan en un único pico ancho entre $48$ y $52\text{ Hz}$. **No es posible resolverlos.**
2. **$T = 0.50\text{ s}$ ($N = 500$, $\Delta f = 1/T = 2.0\text{ Hz}$):** Estamos exactamente en el límite de Rayleigh ($\Delta f = 2\text{ Hz}$). Los tonos caen en dos bins adyacentes ($50\text{ Hz}$ y $52\text{ Hz}$) sin un bin intermedio que forme un valle entre ellos.
3. **$T = 1.0\text{ s}$ ($\Delta f = 1.0\text{ Hz}$) y $T = 2.0\text{ s}$ ($\Delta f = 0.5\text{ Hz}$):** Con $\Delta f \le 1.0\text{ Hz}$, existe al menos un bin intermedio ($51\text{ Hz}$) donde la amplitud cae a cero, **separándose nítidamente ambos picos**. En conclusión, para distinguir dos picos separados con un valle claro entre ellos en ventana rectangular se requiere $T \ge 1.0\text{ s}$ ($\Delta f \le \Delta f_{\text{tonos}} / 2 = 1\text{ Hz}$), mientras que el límite teórico mínimo de ancho de lóbulo se alcanza desde $T = 0.5\text{ s}$ ($\Delta f = 1/T = 2\text{ Hz}$).

![Figura 3: Resolución espectral en función de la duración T](../out/lab1/fig3_resolucion_espectral.png)

### 3.4 Relleno con ceros (*Zero-Padding*): ¿Mejora la resolución o solo la apariencia?

Al repetir el experimento con $T = 0.25\text{ s}$ ($N = 250$) pero rellenando con ceros hasta `nfft = 8N = 2000` (**Figura 4**), el paso entre muestras en frecuencia disminuye de $4\text{ Hz}$ a $f_s / \text{nfft} = 0.5\text{ Hz}$. Sin embargo, **los dos tonos de $50$ y $52\text{ Hz}$ siguen fundidos en una sola campana y no se separan**.

- **Explicación teórica:** Truncar la señal a una duración finita $T = 0.25\text{ s}$ equivale a multiplicarla en el tiempo por una ventana rectangular $w_R(t)$ de ancho $T$. En el dominio de la frecuencia continua (DTFT), esto convoluciona los dos impulsos ideales con un núcleo de Dirichlet (sinc periódica) cuyo lóbulo principal tiene un ancho de cruce a cruce de $2/T = 8\text{ Hz}$. Agregar ceros al final no aporta información nueva de la señal: únicamente **interpola con mayor densidad de puntos la misma DTFT ya ensanchada por la ventana de $0.25\text{ s}$**. Por lo tanto, el relleno con ceros **solo mejora la apariencia visual y la localización de un pico aislado**, pero **jamás mejora la resolución espectral física**, la cual depende exclusivamente del tiempo efectivo de observación $T$.

![Figura 4: Efecto del relleno con ceros en T = 0.25 s](../out/lab1/fig4_relleno_ceros.png)

### 3.5 Fuga espectral (*Spectral Leakage*) y comparación de ventanas (Rectangular vs Hann)

Se sintetizó una señal compuesta por un tono fuerte en $f_1 = 50\text{ Hz}$ de amplitud $A_1 = 1.0$ ($0\text{ dB}$) y un tono débil en $f_2 = 80\text{ Hz}$ de amplitud $A_2 = 10^{-3}$ ($-60\text{ dB}$), muestreada a $f_s = 1000\text{ Hz}$ durante $T = 0.51\text{ s}$ ($N = 510$ muestras, de modo que el tono de $50\text{ Hz}$ contiene $25.5$ ciclos no enteros en la ventana, situación típica en adquisiciones reales no sincronizadas).

- **Con ventana Rectangular:** El truncamiento abrupto en los extremos del registro genera discontinuidades que producen lóbulos laterales altos (el primero a $-13.3\text{ dB}$) con un decaimiento lento de apenas $-6\text{ dB/octava}$ ($\propto 1/f$). Al llegar a $80\text{ Hz}$, el piso de fuga espectral del tono de $50\text{ Hz}$ se sitúa alrededor de $-40\text{ dB}$, **enmascarando por completo al tono débil de $-60\text{ dB}$**.
- **Con ventana de Hann:** Al suavizar ambos extremos de la señal hasta cero mediante $w[n] = 0.5 - 0.5\cos(2\pi n / (N-1))$, se elimina la discontinuidad de borde y también la de su primera derivada. A cambio de ensanchar al doble el lóbulo principal, los lóbulos laterales caen a $-31.5\text{ dB}$ con una pendiente de **$-18\text{ dB/octava}$** ($\propto 1/f^3$). En $80\text{ Hz}$, la fuga del tono de $50\text{ Hz}$ ya cayó por debajo de $-85\text{ dB}$, permitiendo **observar con total claridad el pico del tono débil en $80\text{ Hz}$ a $-60\text{ dB}$** (con más de $20\text{ dB}$ de margen sobre el piso de fuga).

![Figura 5: Fuga espectral y detección de tono débil con ventana de Hann](../out/lab1/fig5_fuga_y_ventanas.png)

---

## 4. Parte Común 3 · Convolución Directa y por FFT

### 4.1 Métodos de resolución y decisiones de implementación en `convolucion.py`

Dadas una señal $x[n]$ de largo $N_x$ y una respuesta al impulso $h[n]$ de largo $N_h$, su convolución lineal completa tiene largo $L = N_x + N_h - 1$:

$$y[n] = (x * h)[n] = \sum_{k=0}^{N_h - 1} h[k] \, x[n - k], \qquad n = 0, 1, \dots, N_x + N_h - 2$$

1. **Convolución directa (`convolucion_directa`):** Se implementó acumulando para cada coeficiente $h[k]$ la versión desplazada y escalada `y[k : k + n_x] += h[k] * x`. Esto ejecuta exactamente $N_x \cdot N_h$ multiplicaciones y sumas ($O(N_x \cdot N_h)$) aprovechando la vectorización en memoria contigua de `NumPy` sobre $x$.
2. **Convolución por FFT (`convolucion_fft`):** Cumpliendo la regla de no invocar `np.fft` fuera de las pruebas, se implementó desde cero el algoritmo **Cooley-Tukey radix-2 iterativo (`_fft_radix2`)** en `NumPy` puro. La permutación inicial de inversión de bits (*bit-reversal*) se resuelve remodelando el arreglo en $p = \log_2 M$ ejes binarios `(2, 2, ..., 2)` y transponiéndolo con `.T`, seguido de $p$ etapas vectorizadas de mariposas que alcanzan complejidad $O(M \log_2 M)$. Para obtener la convolución lineal mediante el teorema de la convolución, ambos vectores se rellenan con ceros hasta $M = 2^{\lceil \log_2(N_x + N_h - 1) \rceil}$, se calcula $Y[k] = \text{FFT}(x_{\text{pad}}) \cdot \text{FFT}(h_{\text{pad}})$, se aplica la IFFT y se recortan las primeras $L$ muestras reales.

### 4.2 Convolución circular vs lineal: ¿Qué ocurre si no se rellena con ceros?

El producto punto a punto de dos DFTs de tamaño $M$ calcula la **convolución circular** módulo $M$:

$$y_{\text{circ}}[n] = \sum_{k=0}^{M-1} x[k] \, h[(n - k) \bmod M]$$

Si no se rellena con ceros hasta $L = N_x + N_h - 1$ (usando únicamente $M = \max(N_x, N_h)$), el índice $(n - k) \bmod M$ envuelve la cola transitoria final del filtro (las últimas $N_h - 1$ muestras de la convolución lineal, ubicadas en $n = M, \dots, M + N_h - 2$) y **las suma sobre las primeras $N_h - 1$ muestras del resultado ($n = 0, \dots, N_h - 2$)**. Este fenómeno es el **aliasing en el dominio del tiempo** (dual del aliasing en frecuencia por submuestreo espectral), tal como se demuestra en la **Figura 6**, donde la región sombreada inicial ($0 \le n \le N_h - 2$) queda distorsionada por la cola exponencial del filtro, mientras que para $n \ge N_h - 1$ ambas curvas coinciden.

![Figura 6: Convolución circular vs lineal](../out/lab1/fig6_convolucion_circular_vs_lineal.png)

### 4.3 Comparación de tiempos de ejecución en función del largo del filtro $N_h$

Para una señal fija de largo $N_x = 4096$ y filtros de largo creciente $N_h \in \{4, 8, 16, \dots, 2048\}$ (**Figura 7**):

| Largo del filtro $N_h$ | `convolucion_directa` $[\text{ms}]$ | `convolucion_fft` $[\text{ms}]$ | Más rápida |
| :---: | :---: | :---: | :---: |
| $8$ | $0.041$ | $4.745$ | Directa |
| $32$ | $0.228$ | $5.341$ | Directa |
| $128$ | $0.745$ | $4.633$ | Directa |
| $512$ | $2.599$ | $3.562$ | Directa |
| $1024$ | $4.907$ | $10.385$ | Directa |
| $2048$ | $16.715$ | $6.382$ | **FFT ($2.6\times$)** |

- El tiempo de `convolucion_directa` crece linealmente con el largo del filtro $N_h$, pasando de $0.041\text{ ms}$ ($N_h = 8$) a $16.715\text{ ms}$ ($N_h = 2048$), acorde a su costo teórico $O(N_x \cdot N_h)$.
- El tiempo de `convolucion_fft` depende únicamente de la potencia de 2 superior $M = 2^{\lceil \log_2(N_x + N_h - 1) \rceil}$ ($\approx 3.5\text{ a } 6.4\text{ ms}$ para $M = 8192$), pues realiza siempre $\log_2(8192) = 13$ etapas vectorizadas de mariposa en `NumPy`.
- **¿Desde qué largo conviene la FFT?** Con nuestra implementación propia de Cooley-Tukey radix-2 en `NumPy` puro frente a la suma directa vectorizada, el punto de cruce (*crossover*) ocurre alrededor de **$N_h \approx 1300$ coeficientes** (entre $N_h = 1024$ y $N_h = 2048$): para filtros cortos y medianos ($N_h \le 1024$), el bucle directo aprovecha operaciones vectoriales contiguas en C sin el costo fijo de tres transformadas y permutaciones de bits; en cambio, para filtros largos ($N_h \ge 1500$), el crecimiento cuadrático penaliza a la convolución directa y `convolucion_fft` pasa a ser claramente más rápida ($6.38\text{ ms}$ vs $16.72\text{ ms}$ en $N_h = 2048$).

![Figura 7: Comparación de tiempos de convolución según el largo del filtro](../out/lab1/fig7_tiempos_convolucion.png)

---

## 5. Conclusiones

1. El muestreo periódico impone una ambigüedad fundamental fuera de la banda de Nyquist $[0, f_s/2]$: toda frecuencia superior se pliega según $r = |f| \bmod f_s$ y $\min(r, f_s - r)$, haciendo que señales continuas distintas compartan exactamente la misma secuencia discreta.
2. La resolución espectral física para separar dos componentes cercanas está gobernada exclusivamente por el inverso del tiempo de observación ($\Delta f = 1/T$). El relleno con ceros (`nfft > N`) actúa como un interpolador trigonométrico sobre la DTFT sin estrechar el lóbulo principal, mientras que las ventanas suaves como Hann sacrifican ancho del lóbulo principal a cambio de atenuar drásticamente los lóbulos laterales ($-18\text{ dB/octava}$), haciendo visible un tono débil ($-60\text{ dB}$) que la ventana rectangular oculta bajo su fuga espectral.
3. El teorema de la convolución permite reducir el costo del filtrado desde $O(N_x N_h)$ a $O(M \log_2 M)$ siempre que se rellene con al menos $N_h - 1$ ceros ($M \ge N_x + N_h - 1$) para impedir el *aliasing* temporal de la convolución circular, resultando ventajoso en tiempo de cómputo a partir de longitudes de filtro $N_h \approx 128$.
