# Visualización de Atención con PyTorch y Matplotlib

Este proyecto explora los conceptos de **hard attention**, **soft attention** y **self-attention** usando vectores simples y operaciones matriciales en PyTorch, con visualización en 2D mediante Matplotlib.

## Contenido

- `attention_visualization.py`: Código principal que:
  - Define un conjunto de vectores `X`.
  - Implementa hard attention (selección uno a uno de vectores).
  - Implementa soft attention (distribución de atención sobre todos los vectores).
  - Implementa self-attention (atención basada en similitud entre vectores).
  - Genera gráficos para visualizar cómo se aplican las atenciones.

## Conceptos aprendidos

1. **Hard Attention**
   - Solo un vector es atendido, los demás se ignoran.
   - Se representa multiplicando un vector one-hot `a` por la matriz de vectores `X`.

2. **Soft Attention**
   - La atención se distribuye entre todos los vectores.
   - Cada vector de salida es la suma ponderada de los vectores de entrada.
   - Se representa usando una matriz `A` con valores que suman 1 en cada fila.

3. **Self-Attention**
   - Calcula la atención basada en la similitud entre vectores de entrada.
   - Uso de `softmax(X @ X.T)` para obtener la matriz de atención.
   - Es la base de mecanismos de atención en modelos Transformers.

4. **Visualización**
   - Flechas (`ax.arrow`) representan vectores originales y vectores de salida.
   - Colores distintos muestran qué vectores están siendo atendidos.

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
