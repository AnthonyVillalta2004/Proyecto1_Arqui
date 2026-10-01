# Informe Técnico: Normalizador Estadístico Vectorizado en Ensamblador x86-64 (NASM)

**Curso:** Arquitectura de Computadores  
**Autores:** Alejandro & Anthony  
**Fecha:** Octubre 2026  

---

## 1. Introducción y Objetivos del Trabajo (OA1)

### 1.1 Contexto y Planteamiento del Problema
En el procesamiento de señales, visión por computador, aprendizaje automático y análisis de grandes volúmenes de datos científicos, la normalización estadística estándar ($Z\text{-Score}$) es una operación fundamental. Dado un conjunto de datos $X = \{x_0, x_1, \dots, x_{N-1}\}$, el objetivo es calcular sus estadísticos descriptivos:

$$\mu = \frac{1}{N} \sum_{i=0}^{N-1} x_i \quad (\text{Media})$$

$$\sigma^2 = \frac{1}{N} \sum_{i=0}^{N-1} (x_i - \mu)^2 \quad (\text{Varianza Poblacional})$$

$$\sigma = \sqrt{\sigma^2} \quad (\text{Desviación Estándar})$$

Y transformar cada elemento según:

$$y[i] = \frac{x[i] - \mu}{\sigma}$$

El cálculo secuencial de estas operaciones sobre millones de datos genera una alta demanda computacional si se ejecuta bajo el paradigma escalar tradicional.

### 1.2 Fundamentos Teóricos: Modelo SIMD frente a SISD
La taxonomía de Flynn clasifica las arquitecturas computacionales según el flujo de instrucciones y datos:
* **SISD (Single Instruction, Single Data):** Modelo escalar tradicional donde cada instrucción de la CPU procesa un único dato por ciclo de reloj (por ejemplo, la instrucción `addss` suma un solo par de flotantes de 32 bits).
* **SIMD (Single Instruction, Multiple Data):** Modelo vectorial donde una única instrucción de la CPU se aplica simultáneamente sobre múltiples datos empaquetados en registros anchos de longitud fija. Mediante las extensiones **AVX2 (Advanced Vector Extensions 2)** con registros `YMM` de 256 bits, es posible operar sobre **8 números en punto flotante de precisión simple (32 bits)** en un solo ciclo de reloj, ofreciendo un potencial teórico de aceleración (*speedup*) de hasta $8\times$.

---

## 2. Descripción del Entorno de Pruebas (OA2 - Compañero)

* **Modelo de Procesador (CPU):** `[Completar: salida de lscpu / cat /proc/cpuinfo]`
* **Flags de soporte vectorial:** Confirmación de flags `avx`, `avx2`, `fma`, `sse4_2`.
* **Herramientas de Software:**
  * Compilador C: `gcc (Ubuntu 11.4.0) 11.4.0` (o versión instalada).
  * Ensamblador: `nasm version 2.15.05` (o superior).
  * Depurador: `gdb (Ubuntu 12.1-0ubuntu1~22.04) 12.1`.
  * Herramienta de perfilado: `perf (linux-tools)`.

---

## 3. Explicación de la Implementación Escalar (OA3)

La solución escalar fue desarrollada en `asm/scalar/stats_scalar.asm` utilizando instrucciones del conjunto **SSE escalar**, operando elemento por elemento sobre la parte baja de 32 bits de los registros `XMM`.

### 3.1 Kernel `compute_stats`
El cómputo se diseñó en dos pasadas secuenciales sobre la memoria RAM:
1. **Preservación de Estado y Validación:** Se guardan los registros *callee-saved* (`rbx, r12-r15, r9`) en la pila según la convención **System V AMD64 ABI**. Se comprueba el caso borde $N \le 0$; si es verdadero, se retorna `0.0f` en las 4 salidas.
2. **Pasada 1 (Suma, Mínimo y Máximo):**
   ```nasm
   .loop_pass1:
       cmp     eax, r12d
       jge     .loop_pass1_done
       movss   xmm1, [rbx + rax*4] ; Carga arr[i]
       addss   xmm0, xmm1          ; Acumula suma
       minss   xmm2, xmm1          ; Actualiza menor
       maxss   xmm3, xmm1          ; Actualiza mayor
       inc     eax
       jmp     .loop_pass1
   ```
   La media se obtiene convirtiendo $N$ a float con `cvtsi2ss xmm5, r12d` y dividiendo con `divss xmm0, xmm5`.
3. **Pasada 2 (Varianza Poblacional):** Se recorre el arreglo calculando la diferencia con respecto a la media y elevando al cuadrado:
   ```nasm
   .loop_pass2:
       cmp     eax, r12d
       jge     .loop_pass2_done
       movss   xmm1, [rbx + rax*4] ; Carga arr[i]
       subss   xmm1, xmm0          ; arr[i] - media
       mulss   xmm1, xmm1          ; (arr[i] - media)^2
       addss   xmm4, xmm1          ; Acumula varianza
       inc     eax
       jmp     .loop_pass2
   ```

### 3.2 Kernel `normalize_array`
Se implementó una protección contra división por cero mediante `ucomiss xmm1, [cero]`. Si la desviación estándar $\sigma = 0.0$, el algoritmo copia los datos de entrada a salida directamente. Si $\sigma > 0$, se calcula $y[i] = (x[i] - \mu) / \sigma$ con `subss` y `divss`.

---

## 4. Explicación de la Implementación Vectorial AVX2 (OA3 y OA4)

La versión vectorizada en `asm/vector/stats_vector.asm` explota el paralelismo de datos de AVX2 procesando **8 floats de 32 bits (256 bits)** por iteración.

### 4.1 Alineación de Memoria a 32 Bytes y Uso de `vmovaps` (OA4)
El programa en C utiliza `aligned_alloc(32, size)` para garantizar que los arreglos inicien en direcciones múltiplos de 32 bytes. Esto permite utilizar la instrucción alineada `vmovaps` (*Vector Move Aligned Packed Single*), la cual garantiza máxima velocidad al evitar accesos que crucen líneas de caché de 64 bytes (*Cache Line Split*).

### 4.2 Pasada 1: Cómputo Vectorial y Reducción Horizontal
Durante el bucle principal, se procesan 8 floats simultáneos acumulando 8 sumas, mínimos y máximos parciales:
```nasm
.pass1_vec_loop:
    vmovaps ymm1, [rbx + rax*4]    ; Carga 8 floats alineados (256 bits)
    vaddps  ymm0, ymm0, ymm1       ; 8 sumas en paralelo
    vminps  ymm2, ymm2, ymm1       ; 8 mínimos en paralelo
    vmaxps  ymm3, ymm3, ymm1       ; 8 máximos en paralelo
    add     eax, 8                 ; Salto de +32 bytes
    jmp     .pass1_vec_loop
```
Al finalizar el bucle vectorial, los 8 carriles se colapsan a un único escalar float mediante **Reducción Horizontal**:
```nasm
vextractf128 xmm6, ymm0, 1     ; Extrae carriles 4-7
vaddps  xmm0, xmm0, xmm6       ; Suma carriles [0..3] + [4..7]
vhaddps xmm0, xmm0, xmm0       ; Suma horizontal de pares
vhaddps xmm0, xmm0, xmm0       ; xmm0[0] = Suma total de los vectores
```

### 4.3 Difusión (*Broadcast*) y Pasada 2 (Varianza)
Para restar la media calculada a los 8 elementos en paralelo, se utiliza la instrucción `vbroadcastss ymm5, xmm0`, replicando el float escalar en los 8 carriles: $[\mu, \mu, \mu, \mu, \mu, \mu, \mu, \mu]$. Luego, el bucle calcula $(x - \mu)^2$ en paralelo con `vsubps` y `vmulps`.

### 4.4 Manejo del Remanente (*Tail Loop*) (OA4)
Si $N$ no es divisible por 8, se calcula el límite vectorial $N_{\text{vec}} = N \ \& \ (\sim 7)$. Los elementos sobrantes ($N \pmod 8$) se procesan mediante un bucle de cierre escalar utilizando instrucciones VEX escalares (`vmovss`, `vaddss`, `vminss`, `vmaxss`, `vsubss`, `vdivss`), garantizando total exactitud numérica para cualquier tamaño $N$.

### 4.5 Transición de Estado con `vzeroupper`
Antes de ejecutar `ret`, se invoca `vzeroupper` para poner a cero la mitad superior de todos los registros YMM, eliminando la penalización de rendimiento en la transición entre instrucciones AVX y código SSE del entorno C.

---

## 5. Casos de Prueba y Verificación de Correctud (OA5 - Compañero)

Tabla comparativa de resultados obtenidos contra el script de referencia `tools/verify_reference.py` (Tolerancia $\le 1\times 10^{-4}$):

| Caso de Prueba           |  $N$  | Entrada / Modo              |          Salida Esperada (Ref.)           |   Salida Escalar    |  Salida Vectorial   | Estado |
| :----------------------- | :---: | :-------------------------- | :---------------------------------------: | :-----------------: | :-----------------: | :----: |
| **Borde: Vacío**         |   0   | `input_empty.dat`           |    $\mu=0, \sigma^2=0, \min=0, \max=0$    | $\mu=0, \sigma^2=0$ | $\mu=0, \sigma^2=0$ |  PASA  |
| **Borde: Un elemento**   |   1   | `input_1.dat`               | $\mu=x_0, \sigma^2=0, \min=x_0, \max=x_0$ |      Coincide       |      Coincide       |  PASA  |
| **Borde: Cola pura**     |   7   | `input_7.dat`               |             Referencia Python             |      Coincide       |      Coincide       |  PASA  |
| **Borde: Vector exacto** |   8   | `input_8.dat`               |             Referencia Python             |      Coincide       |      Coincide       |  PASA  |
| **Borde: Vector + Cola** |  15   | `input_15.dat`              |             Referencia Python             |      Coincide       |      Coincide       |  PASA  |
| **Borde: Múltiple**      |  16   | `input_16.dat`              |             Referencia Python             |      Coincide       |      Coincide       |  PASA  |
| **Borde: Constante**     | 1000  | `constant` ($\sigma=0$)     |      $\mu=5.0, \sigma=0.0, y[i]=5.0$      |      Coincide       |      Coincide       |  PASA  |
| **Borde: Extremos**      | 1000  | `edge` ($-10^6 \dots 10^6$) |             Referencia Python             |      Coincide       |      Coincide       |  PASA  |

---

## 6. Resultados de Rendimiento y Benchmarking (OA6)

### 6.1 Tabla de Tiempos y Speedup Promedio (30 repeticiones por tamaño)

|       Tamaño $N$        | Tiempo Escalar (ms) | Tiempo Vectorial (ms) | Speedup Real ($T_{\text{esc}} / T_{\text{vec}}$) | Speedup Teórico |
| :---------------------: | :-----------------: | :-------------------: | :----------------------------------------------: | :-------------: |
|  $10^3$ ($1\text{ K}$)  |      $0.0031$       |       $0.0003$        |               $\approx 9.72\times$               |   $8.0\times$   |
| $10^5$ ($100\text{ K}$) |      $0.2525$       |       $0.0319$        |               $\approx 7.91\times$               |   $8.0\times$   |
|  $10^6$ ($1\text{ M}$)  |      $2.4374$       |       $0.3364$        |               $\approx 7.25\times$               |   $8.0\times$   |
| $10^7$ ($10\text{ M}$)  |       $26.05$       |        $6.95$         |                   $3.74\times$                   |   $8.0\times$   |
 
### 6.2 Gráfico de Speedup vs $N$ (Escala Logarítmica en X)
![Gráfico de caída de Speedup por jerarquía de memoria](../data/speedup_plot.png)
*Figura 1: Evolución del Speedup en función del tamaño del arreglo. Se observa el decaimiento de la aceleración desde valores superlineales (Caché L1) hasta el estrangulamiento del bus de memoria principal (Memory Wall).*

### 6.3 Análisis con `perf stat` y Cuellos de Botella
**Instrucciones Ejecutadas ($N=10^7$):** Escalar: $\approx 30.2$ mil millones | Vectorial: $\approx 3.79$ mil millones.
**Instrucciones por Ciclo (IPC):** Escalar: $2.4$ | Vectorial: $1.1$.
**Impacto de la Jerarquía de Caché:** Los resultados muestran una degradación de rendimiento directamente proporcional al tamaño del conjunto de datos. 
 1. Para $N=10^3$ ($4\text{ KB}$), obtenemos un *speedup* superlineal ($>8.0\times$) atribuido a que el arreglo reside íntegramente en la rapidísima caché L1, aunado a la masiva reducción de penalizaciones por saltos condicionales (*branch overhead*).
 2. Para $N=10^5$ ($400\text{ KB}$), el arreglo encaja en la caché L2, permitiendo que el procesador roce su límite computacional (Compute-Bound), arrojando el $7.91\times$ de *speedup* que dicta la teoría matemática.
 3. Para $N=10^6$ ($4\text{ MB}$), se requiere el uso de la caché L3 (compartida y de mayor latencia), disminuyendo la ganancia a $7.25\times$.
**Discusión y Paradoja del IPC (El *Memory Wall*):** Al pasar a $N=10^7$ (procesando 40 MB de datos, desalojando la caché L3), el speedup real se desploma a $3.74\times$. Esto se explica con la métrica del IPC. La versión vectorial muestra un IPC contraintuitivamente bajo ($1.1$) debido a los *stalls* (congelamientos) del pipeline. Cada instrucción `vmovaps` exige 32 bytes de golpe, ocupando el ancho de banda de la memoria RAM DDR. El procesador pasa ciclos detenido esperando datos, convirtiendo la operación en una tarea estrictamente limitada por memoria (Memory-Bound).
---

## 7. Evidencia de la Sesión en GDB (OA5)

### 7.1 Inspección de Propagación de Estado (*Broadcast*) con $N=10$
Se verificó la correcta propagación de la media estadística a los 8 carriles del registro vectorial antes de iniciar el cómputo paralelo, usando la instrucción `vbroadcastss`.
```gdb
(gdb) break normalize_array
Punto de interrupción 1 at 0x1ae1: file asm/vector/stats_vector.asm, line 227.
(gdb) run
(gdb) stepi
(gdb) stepi
(gdb) stepi
(gdb) print $ymm0.v8_float
$2 = {-14.5033779, -14.5033779, -14.5033779, -14.5033779, -14.5033779, -14.5033779, -14.5033779, -14.5033779}
```

### 7.2 Verificación Estricta de Alineación de Memoria (`vmovaps`)
Para evitar excepciones de violación de segmento (*SegFault*), se inspeccionó la dirección de memoria y los datos crudos antes de ser ingeridos por la ALU en la instrucción `vmovaps ymm2, [rdi + rax*4]`.
```gdb
(gdb) print/x $rdi + $rax*4
$10 = 0x55555555a4c0
(gdb) x/8fw $rdi + $rax*4
0x55555555a4c0: -16.4012127     -66.8495712     -12.9803553     14.3730354
0x55555555a4d0: -84.6899185     31.6419277      89.5950165      51.3854141
```
*Análisis:* La dirección hexadecimal `0x55555555a4c0` termina en `c0` (192 en decimal), lo cual es múltiplo exacto de 32 bytes ($192 / 32 = 6$). Esto confirma matemáticamente que la memoria reservada por `driver.c` cumple el requisito de alineación de AVX2.

---

## 8. Conclusiones, Limitaciones y Trabajo Futuro

1. **Aceleración Vectorial:** La vectorización manual en ensamblador x86-64 con AVX2 reduce sustancialmente el número total de instrucciones ejecutadas, logrando aceleraciones significativas frente a la implementación escalar.
2. **Impacto de la Jerarquía de Memoria:** El factor determinante en el rendimiento final no es únicamente la potencia aritmética de la ALU vectorial, sino la localidad de referencia y el ancho de banda hacia la memoria principal.
3. **Robustez en Casos Borde:** La correcta combinación de alineación a 32 bytes y bucles de cierre escalares (*tail loops*) asegura que las optimizaciones SIMD mantengan un 100% de confiabilidad numérica sin sacrificar estabilidad.
