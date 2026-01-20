# Word2Vec con Skip-Gram – Práctica de PLN

Este repositorio contiene la implementación completa de un modelo **Word2Vec basado en Skip-Gram**, desarrollada desde cero con fines académicos para la asignatura de **Procesamiento del Lenguaje Natural (PLN)**.

El objetivo del trabajo es comprender el funcionamiento interno del modelo Word2Vec, evitando el uso de librerías de alto nivel como *gensim*, y abordando explícitamente todas las etapas del proceso: tratamiento del corpus, entrenamiento y análisis de los embeddings obtenidos.

---

## Objetivos del trabajo

- Implementar el modelo Skip-Gram de Word2Vec desde cero.
- Comprender cómo se construyen representaciones distribucionales del lenguaje.
- Analizar el efecto de distintos hiperparámetros en la calidad de los embeddings.
- Evaluar el modelo mediante análisis cuantitativo y cualitativo.
- Reflexionar sobre las limitaciones del enfoque en corpus de tamaño moderado.

---

## Descripción general

El modelo se entrena utilizando **softmax completo sobre todo el vocabulario**, lo que permite seguir de forma directa la formulación original del modelo y analizar con claridad el cálculo de probabilidades y gradientes. Aunque este enfoque no es escalable a grandes vocabularios, resulta adecuado en un contexto docente donde se prioriza la comprensión conceptual.

Con el objetivo de mejorar la calidad de los embeddings obtenidos, se ha llevado a cabo una **búsqueda  de hiperparámetros**, evaluando distintas combinaciones de dimensión del espacio vectorial, tamaño de la ventana de contexto, tasa de aprendizaje y número de épocas de entrenamiento. Cada configuración fue analizada mediante métricas cuantitativas, como la pérdida final del modelo y la similitud media entre palabras y sus vecinos más cercanos, así como mediante inspección cualitativa de vecinos semánticos y analogías. Este proceso permitió seleccionar un conjunto de hiperparámetros que ofrece un equilibrio adecuado entre estabilidad del entrenamiento y coherencia semántica de las representaciones aprendidas.


El corpus utilizado ha sido construido de forma controlada para incluir patrones lingüísticos claros, como:
- relaciones de género gramatical,
- asociaciones entre agentes y acciones,
- atributos de objetos,
- relaciones geográficas simples.



