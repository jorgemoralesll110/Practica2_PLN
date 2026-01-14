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

El corpus utilizado ha sido construido de forma controlada para incluir patrones lingüísticos claros, como:
- relaciones de género gramatical,
- asociaciones entre agentes y acciones,
- atributos de objetos,
- relaciones geográficas simples.

La evaluación del modelo combina:
- un **análisis cuantitativo**, basado en la pérdida final del entrenamiento y en la similitud media entre palabras y sus vecinos más cercanos,
- y un **análisis cualitativo**, mediante la inspección de vecinos semánticos y analogías no triviales diseñadas de acuerdo con el contenido del corpus.

