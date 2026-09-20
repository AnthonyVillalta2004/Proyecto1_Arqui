; =============================================================
; stats_scalar.asm
; Version ESCALAR (referencia) de los kernels de computo.
;
; Convencion de llamada: System V AMD64 ABI
;   enteros/punteros: rdi, rsi, rdx, rcx, r8, r9
;   flotantes:        xmm0, xmm1, xmm2, ...
;   retorno float:    xmm0
;   callee-saved:     rbx, rbp, r12-r15 (si los usa, debe preservarlos)
; =============================================================

    global sum_array
    global compute_stats
    global normalize_array

    section .text

; ---------------------------------------------------------------
; float sum_array(const float *arr, int n)
;   rdi = arr, esi = n
;   retorna la suma en xmm0
;
; IMPLEMENTADA COMO EJEMPLO: estudien este patron (recorrido,
; acumulador, condicion de salida) antes de escribir compute_stats
; y normalize_array.
; ---------------------------------------------------------------
sum_array:
    xor     eax, eax           ; eax = i = 0
    xorps   xmm0, xmm0         ; xmm0 = acumulador = 0.0

.sum_loop:
    cmp     eax, esi
    jge     .sum_done
    movss   xmm1, [rdi + rax*4]
    addss   xmm0, xmm1
    inc     eax
    jmp     .sum_loop

.sum_done:
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
    push    r9                 ; Guardar puntero max* en la pila

    ; 2. Caso borde: si n <= 0, poner 0.0 en todas las salidas
    test    esi, esi
    jle     .stats_scalar_zero

    ; 3. Guardar argumentos en registros callee-saved
    mov     rbx, rdi           ; rbx = arr (puntero base)
    mov     r12d, esi          ; r12d = n (longitud del arreglo)
    mov     r13, rdx           ; r13 = mean*
    mov     r14, rcx           ; r14 = var*
    mov     r15, r8            ; r15 = min*

    ; 4. Pasada 1: Suma, Minimo y Maximo
    xor     eax, eax           ; eax = i = 0
    xorps   xmm0, xmm0         ; xmm0 = acumulador suma = 0.0
    movss   xmm2, [rbx]        ; xmm2 = min = arr[0]
    movss   xmm3, [rbx]        ; xmm3 = max = arr[0]

.loop_pass1:
    cmp     eax, r12d
    jge     .loop_pass1_done
    movss   xmm1, [rbx + rax*4] ; xmm1 = arr[i]
    addss   xmm0, xmm1          ; suma += arr[i]
    minss   xmm2, xmm1          ; min = min(min, arr[i])
    maxss   xmm3, xmm1          ; max = max(max, arr[i])
    inc     eax
    jmp     .loop_pass1

.loop_pass1_done:
    cvtsi2ss xmm5, r12d        ; xmm5 = (float)n
    divss   xmm0, xmm5         ; xmm0 = mean = suma / n
    movss   [r13], xmm0        ; Guardar *mean = mean

    ; 5. Pasada 2: Varianza sum((x - mean)^2) / n
    xor     eax, eax           ; eax = i = 0
    xorps   xmm4, xmm4         ; xmm4 = acumulador varianza = 0.0

.loop_pass2:
    cmp     eax, r12d
    jge     .loop_pass2_done
    movss   xmm1, [rbx + rax*4] ; xmm1 = arr[i]
    subss   xmm1, xmm0          ; xmm1 = arr[i] - mean
    mulss   xmm1, xmm1          ; xmm1 = (arr[i] - mean)^2
    addss   xmm4, xmm1          ; acumula diferencia al cuadrado
    inc     eax
    jmp     .loop_pass2

.loop_pass2_done:
    divss   xmm4, xmm5         ; xmm4 = var = acumulador / n
    movss   [r14], xmm4        ; Guardar *var = var
    movss   [r15], xmm2        ; Guardar *min = min
    pop     r9                 ; Recuperar puntero max*
    movss   [r9], xmm3         ; Guardar *max = max
    jmp     .stats_scalar_end

.stats_scalar_zero:
    pop     r9                 ; Limpiar puntero de la pila
    xorps   xmm0, xmm0
    movss   [rdx], xmm0        ; *mean = 0.0
    movss   [rcx], xmm0        ; *var  = 0.0
    movss   [r8], xmm0         ; *min  = 0.0
    movss   [r9], xmm0         ; *max  = 0.0

.stats_scalar_end:
    ; 6. Restaurar registros y retornar
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
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
    jle     .norm_scalar_done   ; Si n <= 0, salir inmediatamente

    xor     eax, eax           ; eax = i = 0
    xorps   xmm2, xmm2         ; xmm2 = 0.0
    ucomiss xmm1, xmm2         ; Comparar stddev con 0.0
    je      .norm_copy_loop    ; Caso borde: si stddev == 0.0, solo copiar

.norm_calc_loop:
    cmp     eax, edx
    jge     .norm_scalar_done
    movss   xmm2, [rdi + rax*4] ; xmm2 = in[i]
    subss   xmm2, xmm0          ; xmm2 = in[i] - mean
    divss   xmm2, xmm1          ; xmm2 = (in[i] - mean) / stddev
    movss   [rsi + rax*4], xmm2 ; out[i] = xmm2
    inc     eax
    jmp     .norm_calc_loop

.norm_copy_loop:
    cmp     eax, edx
    jge     .norm_scalar_done
    movss   xmm2, [rdi + rax*4] ; xmm2 = in[i]
    movss   [rsi + rax*4], xmm2 ; out[i] = in[i]
    inc     eax
    jmp     .norm_copy_loop

.norm_scalar_done:
    ret
