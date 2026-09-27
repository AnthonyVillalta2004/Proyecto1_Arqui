# Tabla de Asignación de Registros (Rúbrica 3.a.3)

Mapeo técnico de los registros de la CPU x86-64 utilizados en los núcleos de cómputo (`compute_stats` y `normalize_array`) respetando la convención System V AMD64 ABI.

---

## 1. Kernel `compute_stats`

| Registro | Ancho / Tipo | Estado ABI | Rol en Versión Escalar (`stats_scalar.asm`) | Rol en Versión Vectorial (`stats_vector.asm`) |
| :--- | :---: | :---: | :--- | :--- |
| **rdi** | 64 bits | Caller-saved | Dirección base del arreglo (`arr`) | Dirección base del arreglo alineada a 32 bytes (`arr`) |
| **esi / r12d** | 32 bits | Caller-saved | Cantidad de elementos N del arreglo | Cantidad de elementos N del arreglo |
| **rdx / r13** | 64 bits | Caller-saved | Puntero a variable de salida `*mean` | Puntero a variable de salida `*mean` |
| **rcx / r14** | 64 bits | Caller-saved | Puntero a variable de salida `*var` | Puntero a variable de salida `*var` |
| **r8 / r15** | 64 bits | Caller-saved | Puntero a variable de salida `*min` | Puntero a variable de salida `*min` |
| **r9 / [rsp]** | 64 bits | Caller-saved | Puntero a variable de salida `*max` (guardado en pila) | Puntero a variable de salida `*max` (guardado en pila) |
| **rbx** | 64 bits | **Callee-saved** | Preserva la dirección del arreglo `arr` | Preserva la dirección del arreglo `arr` |
| **rax** | 64 bits | Caller-saved | Contador de índice `i` (avanza +1) | Contador de índice `i` (+8 en bucle vector, +1 en tail) |
| **ecx** | 32 bits | Caller-saved | No se requiere | Límite del bucle vectorial: N_vec = N & ~7 (múltiplo de 8) |
| **ymm0 / xmm0** | 256b / 32b | Volátil | Acumulador suma / Resultado de media | 8 sumas parciales -> reducida a media escalar |
| **ymm1 / xmm1** | 256b / 32b | Volátil | Buffer temporal del float actual arr[i] | Buffer de 8 floats leídos con vmovaps (256 bits) |
| **ymm2 / xmm2** | 256b / 32b | Volátil | Mínimo local acumulado escalar | 8 mínimos parciales con vminps -> reducido a escalar |
| **ymm3 / xmm3** | 256b / 32b | Volátil | Máximo local acumulado escalar | 8 máximos parciales con vmaxps -> reducido a escalar |
| **ymm4 / xmm4** | 256b / 32b | Volátil | Acumulador de varianza sum((arr[i] - media)^2) | Acumulador vectorial de varianza -> reducido a escalar |
| **ymm5** | 256 bits | Volátil | No utilizado | Broadcast de la media: [media, media, ..., media] |
| **xmm6 / xmm7** | 128b / 32b | Volátil | Conversión de N a float (cvtsi2ss) | Registros temporales para reducción horizontal (vextractf128) |

---

## 2. Kernel `normalize_array`

| Registro | Ancho / Tipo | Rol en Versión Escalar (`stats_scalar.asm`) | Rol en Versión Vectorial (`stats_vector.asm`) |
| :--- | :---: | :--- | :--- |
| **rdi** | 64 bits | Puntero arreglo entrada `in` | Puntero arreglo entrada `in` (alineado a 32 bytes) |
| **rsi** | 64 bits | Puntero arreglo salida `out` | Puntero arreglo salida `out` (alineado a 32 bytes) |
| **edx** | 32 bits | Longitud N del arreglo | Longitud N del arreglo |
| **rax** | 64 bits | Contador de índice `i` (avanza +1) | Contador de índice `i` (+8 en bucle vector, +1 en tail) |
| **ecx** | 32 bits | No se requiere | Límite vectorial: N_vec = N & ~7 |
| **ymm0 / xmm0** | 256b / 32b | Media escalar (`mean`) | Media duplicada en 8 carriles con vbroadcastss |
| **ymm1 / xmm1** | 256b / 32b | Desviación estándar escalar (`stddev`) | Desviación estándar duplicada en 8 carriles |
| **ymm2 / xmm2** | 256b / 32b | Float actual: in[i] | Carga de 8 floats de entrada con vmovaps |
| **ymm3 / xmm3** | 256b / 32b | Resultado calculado: (in[i] - media) / stddev | 8 floats normalizados calculados en paralelo (vdivps) |
