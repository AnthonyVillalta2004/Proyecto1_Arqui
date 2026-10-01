#!/usr/bin/env python3
"""
Genera el gráfico de Speedup vs N para el informe técnico del OA6.
El eje X se configura en escala logarítmica para visualizar correctamente
la magnitud de los arreglos.
"""
import matplotlib.pyplot as plt

def main():
    # Datos extraídos del benchmark (Sección 6.1)
    N_values = [1000, 100000, 1000000, 10000000]
    speedups = [9.72, 7.91, 7.25, 3.74]
    
    # Límite teórico de AVX2 (8 floats por instrucción)
    theoretical_limit = 8.0

    # Configuración de la figura
    plt.figure(figsize=(10, 6))
    
    # Trazar datos experimentales
    plt.plot(N_values, speedups, marker='o', linestyle='-', color='b', linewidth=2, markersize=8, label='Speedup Medido')
    
    # Trazar límite teórico
    plt.axhline(y=theoretical_limit, color='r', linestyle='--', linewidth=2, label='Límite Teórico (8.0x)')
    
    # Configuración de ejes
    plt.xscale('log')
    plt.xlabel('Tamaño del Arreglo (N)', fontsize=12)
    plt.ylabel('Speedup (T_escalar / T_vectorial)', fontsize=12)
    plt.title('Impacto de la Jerarquía de Memoria en el Speedup Vectorial (AVX2)', fontsize=14)
    
    # Limites del eje Y para visualizar bien la caída
    plt.ylim(0, 12)
    
    # Rejilla y leyenda
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    
    # Anotaciones para los puntos clave (Cachés vs RAM)
    plt.annotate('L1 Cache (~4 KB)', xy=(1000, 9.72), xytext=(1200, 10.2),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.annotate('L2 Cache (~400 KB)', xy=(100000, 7.91), xytext=(120000, 8.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.annotate('Memory Wall (RAM, 40 MB)', xy=(10000000, 3.74), xytext=(1500000, 2.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Guardar gráfico
    output_path = '/docs/informe/data/speedup_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico generado exitosamente en: '{output_path}'")

if __name__ == "__main__":
    main()