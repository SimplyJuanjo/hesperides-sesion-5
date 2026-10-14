# AGENTS.md

## Qué es esto

Un RAG pequeño en Python y NumPy sobre tres informes públicos del Banco de España, el BCE y la CNMV, para entender qué pasa entre la pregunta y la respuesta con citas. Es para aprender: quien lo usa tiene que entender cada línea.

## Reglas

- Usa solo NumPy y la biblioteca estándar de Python.
- Deja como están `tests/`, `corpus/`, `indices/`, `preguntas.json`, `modelos.py` y las funciones de `rag.py` que ya vienen escritas.
- En `buscar` y `fusionar`, operaciones con vectores y matrices de NumPy, sin bucles sobre los fragmentos.
- Las claves viven en `.env`. No las escribas en el código ni las muestres por pantalla.

## Cómo trabajar

- Un peldaño cada vez: implementa solo la función que te pidan y ejecuta su test, por ejemplo `pytest tests/test_1_trocear.py`.
- Antes de escribir código, di en dos frases qué vas a hacer y, si hay matrices, la forma (filas, columnas) de cada una, y espera.
- Después, explica cada línea en una frase, para alguien que no ha usado NumPy.
