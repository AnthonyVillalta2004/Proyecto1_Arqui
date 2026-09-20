; =============================================================
; stats_vector.asm
; Version VECTORIZADA (AVX2, 8 floats por iteracion) de los
; kernels de computo. Misma ABI que la version escalar.
;
; Antes de compilar/ejecutar en su maquina, confirme soporte AVX2:
;   lscpu | grep avx2
;   cat /proc/cpuinfo | grep avx2
; =============================================================

    global sum_array
    global compute_stats
    global normalize_array

    section .text

; ---------------------------------------------------------------
; float sum_array(const float *arr, int n)
;   rdi = arr, esi = n -> retorna la suma en xmm0
;
; IMPLEMENTADA COMO EJEMPLO. Fijense especialmente en:
;   (1) como se calcula cuantos elementos entran en bucles de 8
;       ("and ecx, ~7" redondea n hacia abajo al multiplo de 8),
;   (2) la REDUCCION HORIZONTAL para pasar de 8 sumas parciales
;       (un YMM) a un unico escalar,
;   (3) el BUCLE ESCALAR DE CIERRE para el remanente (n % 8 != 0).
; Reutilicen este mismo patron en compute_stats y normalize_array.
; ---------------------------------------------------------------
sum_array:
    xor     eax, eax               ; eax = i = 0
    vxorps  ymm0, ymm0, ymm0       ; ymm0 = acumulador vectorial (8 carriles) = 0

    mov     ecx, esi
    and     ecx, ~7                ; ecx = n redondeado hacia abajo, multiplo de 8
    test    ecx, ecx
    jle     .sum_reduce

.sum_vec_loop:
    cmp     eax, ecx
    jge     .sum_reduce
    vmovups ymm1, [rdi + rax*4]    ; carga 8 floats (unaligned: siempre valido)
    vaddps  ymm0, ymm0, ymm1       ; acumula por carril
    add     eax, 8
    jmp     .sum_vec_loop

.sum_reduce:
    ; --- reduccion horizontal: 8 carriles de ymm0 -> un escalar ---
    vextractf128 xmm2, ymm0, 1     ; xmm2 = mitad alta (carriles 4-7)
    vaddps  xmm0, xmm0, xmm2       ; xmm0 = 4 sumas parciales (carriles 0-3 + 4-7)
    vhaddps xmm0, xmm0, xmm0       ; suma horizontal dentro de 128 bits
    vhaddps xmm0, xmm0, xmm0       ; xmm0[0] = suma total de los 8 carriles originales

.sum_scalar_tail:
    ; --- elementos sobrantes (n % 8), uno a la vez ---
    cmp     eax, esi
    jge     .sum_done
    vmovss  xmm1, [rdi + rax*4]
    vaddss  xmm0, xmm0, xmm1
    inc     eax
    jmp     .sum_scalar_tail

.sum_done:
    vzeroupper                     ; evita penalizacion de transicion AVX/SSE
    ret

; ---------------------------------------------------------------
; void compute_stats(const float *arr, int n,
;                     float *mean, float *var, float *min, float *max)
;   rdi = arr, esi = n, rdx = mean*, rcx = var*, r8 = min*, r9 = max*
;
;   var = varianza POBLACIONAL = sum((x - mean)^2) / n
;   Caso borde: si n <= 0, escriba 0.0 en mean/var/min/max.
; ---------------------------------------------------------------
compute_stats:
    ; 1. Preservar registros callee-saved segun la ABI System V
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    push    r9                     ; Guardar puntero max* en pila

    ; 2. Caso borde: si n <= 0, escribir 0.0 en las salidas
    test    esi, esi
    jle     .stats_vec_zero

    ; 3. Guardar argumentos en registros preservados
    mov     rbx, rdi               ; rbx = arr
    mov     r12d, esi              ; r12d = n
    mov     r13, rdx               ; r13 = mean*
    mov     r14, rcx               ; r14 = var*
    mov     r15, r8                ; r15 = min*

    ; 4. Calcular limite vectorial N_vec = n & ~7 (multiplo de 8)
    mov     ecx, esi
    and     ecx, ~7                ; ecx = limite del bucle AVX2

    ; 5. Inicializar acumuladores vectoriales
    xor     eax, eax               ; eax = i = 0
    vxorps  ymm0, ymm0, ymm0       ; ymm0 = acumulador suma = [0, ..., 0]
    vbroadcastss ymm2, [rbx]       ; ymm2 = acumulador min = [arr[0], ..., arr[0]]
    vbroadcastss ymm3, [rbx]       ; ymm3 = acumulador max = [arr[0], ..., arr[0]]

    test    ecx, ecx
    jle     .pass1_vec_reduce

.pass1_vec_loop:
    cmp     eax, ecx
    jge     .pass1_vec_reduce
    vmovaps ymm1, [rbx + rax*4]    ; Carga 8 floats alineados a 32 bytes
    vaddps  ymm0, ymm0, ymm1       ; Suma vectorial
    vminps  ymm2, ymm2, ymm1       ; Minimo vectorial
    vmaxps  ymm3, ymm3, ymm1       ; Maximo vectorial
    add     eax, 8
    jmp     .pass1_vec_loop

.pass1_vec_reduce:
    ; --- Reduccion horizontal de suma (ymm0 -> xmm0) ---
    vextractf128 xmm6, ymm0, 1
    vaddps  xmm0, xmm0, xmm6
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

    ; --- Reduccion horizontal de minimo (ymm2 -> xmm2) ---
    vextractf128 xmm6, ymm2, 1
    vminps  xmm2, xmm2, xmm6
    vshufps xmm6, xmm2, xmm2, 0x4E
    vminps  xmm2, xmm2, xmm6
    vshufps xmm6, xmm2, xmm2, 0xB1
    vminps  xmm2, xmm2, xmm6

    ; --- Reduccion horizontal de maximo (ymm3 -> xmm3) ---
    vextractf128 xmm6, ymm3, 1
    vmaxps  xmm3, xmm3, xmm6
    vshufps xmm6, xmm3, xmm3, 0x4E
    vmaxps  xmm3, xmm3, xmm6
    vshufps xmm6, xmm3, xmm3, 0xB1
    vmaxps  xmm3, xmm3, xmm6

.pass1_tail_loop:
    ; --- Remanente escalar (n % 8 elementos sobrantes) ---
    cmp     eax, r12d
    jge     .pass1_done
    vmovss  xmm1, [rbx + rax*4]
    vaddss  xmm0, xmm0, xmm1
    vminss  xmm2, xmm2, xmm1
    vmaxss  xmm3, xmm3, xmm1
    inc     eax
    jmp     .pass1_tail_loop

.pass1_done:
    vcvtsi2ss xmm7, xmm7, r12d     ; xmm7 = (float)n
    vdivss  xmm0, xmm0, xmm7       ; xmm0 = mean = suma / n
    vmovss  [r13], xmm0            ; Guardar *mean = mean

    ; 6. Pasada 2 Vectorial: Varianza sum((x - mean)^2)
    vbroadcastss ymm5, xmm0        ; ymm5 = [mean, mean, ..., mean]
    vxorps  ymm4, ymm4, ymm4       ; ymm4 = acumulador varianza = [0, ..., 0]
    xor     eax, eax

    test    ecx, ecx
    jle     .pass2_vec_reduce

.pass2_vec_loop:
    cmp     eax, ecx
    jge     .pass2_vec_reduce
    vmovaps ymm1, [rbx + rax*4]    ; Carga 8 floats
    vsubps  ymm1, ymm1, ymm5       ; ymm1 = (arr[i..i+7] - mean)
    vmulps  ymm1, ymm1, ymm1       ; ymm1 = (arr[i..i+7] - mean)^2
    vaddps  ymm4, ymm4, ymm1       ; Acumula varianza
    add     eax, 8
    jmp     .pass2_vec_loop

.pass2_vec_reduce:
    ; --- Reduccion horizontal de varianza (ymm4 -> xmm4) ---
    vextractf128 xmm6, ymm4, 1
    vaddps  xmm4, xmm4, xmm6
    vhaddps xmm4, xmm4, xmm4
    vhaddps xmm4, xmm4, xmm4

.pass2_tail_loop:
    ; --- Remanente escalar de varianza ---
    cmp     eax, r12d
    jge     .pass2_done
    vmovss  xmm1, [rbx + rax*4]
    vsubss  xmm1, xmm1, xmm0
    vmulss  xmm1, xmm1, xmm1
    vaddss  xmm4, xmm4, xmm1
    inc     eax
    jmp     .pass2_tail_loop

.pass2_done:
    vdivss  xmm4, xmm4, xmm7       ; xmm4 = var = sum_sq / n
    vmovss  [r14], xmm4            ; Guardar *var = var
    vmovss  [r15], xmm2            ; Guardar *min = min
    pop     r9                     ; Recuperar puntero max*
    vmovss  [r9], xmm3             ; Guardar *max = max
    jmp     .stats_vec_end

.stats_vec_zero:
    pop     r9                     ; Limpiar pila si n <= 0
    vxorps  xmm0, xmm0, xmm0
    vmovss  [rdx], xmm0            ; *mean = 0.0
    vmovss  [rcx], xmm0            ; *var  = 0.0
    vmovss  [r8], xmm0             ; *min  = 0.0
    vmovss  [r9], xmm0             ; *max  = 0.0

.stats_vec_end:
    ; 7. Restaurar registros y limpiar estado AVX
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    vzeroupper                     ; Limpia mitad superior de registros YMM
    ret

; ---------------------------------------------------------------
; void normalize_array(const float *in, float *out, int n,
;                       float mean, float stddev)
;   rdi = in, rsi = out, edx = n, xmm0 = mean, xmm1 = stddev
;
;   out[i] = (in[i] - mean) / stddev
;   Caso borde: si stddev == 0.0, copie in[i] en out[i] tal cual.
; ---------------------------------------------------------------
normalize_array:
    test    edx, edx
    jle     .norm_vec_done         ; Si n <= 0, salir inmediatamente

    vxorps  xmm2, xmm2, xmm2
    vucomiss xmm1, xmm2
    je      .norm_vec_copy         ; Caso borde: si stddev == 0.0, solo copiar

    ; Broadcast de mean y stddev a registros YMM de 256 bits
    vbroadcastss ymm0, xmm0        ; ymm0 = [mean, mean, ..., mean]
    vbroadcastss ymm1, xmm1        ; ymm1 = [stddev, stddev, ..., stddev]

    mov     ecx, edx
    and     ecx, ~7                ; Limite vectorial multiplo de 8
    xor     eax, eax

.norm_vec_loop:
    cmp     eax, ecx
    jge     .norm_vec_tail
    vmovaps ymm2, [rdi + rax*4]    ; Carga 8 floats alineados a 32 bytes
    vsubps  ymm2, ymm2, ymm0       ; in[i..i+7] - mean
    vdivps  ymm2, ymm2, ymm1       ; (in[i..i+7] - mean) / stddev
    vmovaps [rsi + rax*4], ymm2    ; Guarda 8 floats alineados
    add     eax, 8
    jmp     .norm_vec_loop

.norm_vec_tail:
    ; Remanente escalar para los ultimos n % 8 elementos
    cmp     eax, edx
    jge     .norm_vec_finish
    vmovss  xmm2, [rdi + rax*4]
    vsubss  xmm2, xmm2, xmm0
    vdivss  xmm2, xmm2, xmm1
    vmovss  [rsi + rax*4], xmm2
    inc     eax
    jmp     .norm_vec_tail

.norm_vec_copy:
    ; Copia en caso de desviacion estandar == 0.0
    mov     ecx, edx
    and     ecx, ~7
    xor     eax, eax

.copy_vec_loop:
    cmp     eax, ecx
    jge     .copy_vec_tail
    vmovaps ymm2, [rdi + rax*4]
    vmovaps [rsi + rax*4], ymm2
    add     eax, 8
    jmp     .copy_vec_loop

.copy_vec_tail:
    cmp     eax, edx
    jge     .norm_vec_finish
    vmovss  xmm2, [rdi + rax*4]
    vmovss  [rsi + rax*4], xmm2
    inc     eax
    jmp     .copy_vec_tail

.norm_vec_finish:
    vzeroupper                     ; Limpia estado YMM
.norm_vec_done:
    ret
