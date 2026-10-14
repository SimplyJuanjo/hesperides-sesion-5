# Sesión 5 · From Foundational Papers to Practical Patterns

Repositorio de partida del laboratorio del 14 de octubre de 2026: un RAG pequeño, con citas, sobre tres informes públicos del Banco de España, el BCE y la CNMV. Máster en Finanzas Cuantitativas y Métodos Computacionales, Universidad de las Hespérides.

## Antes de clase: preparar

Unos 20 minutos, la mayoría esperando descargas. Hazlo antes del miércoles, no en clase.

```text
git clone https://github.com/SimplyJuanjo/hesperides-sesion-5.git
cd hesperides-sesion-5
pip install -r requirements.txt
copy .env.ejemplo .env
```

En Mac o Linux, `cp .env.ejemplo .env`. Abre `.env`, pon tus claves (abajo, cómo conseguirlas) y ejecuta:

```text
python preparar.py
```

La primera vez descarga EmbeddingGemma, el modelo de embeddings que corre en tu ordenador (1,26 GB). Después prueba cada clave y, con las de Gemini y Cohere, calcula tu índice en la nube. Puedes repetirlo cuantas veces quieras. Para llegar a clase hace falta «bien» en la instalación, en EmbeddingGemma y en al menos un LLM.

### Las claves

Todas son gratis y ninguna pide tarjeta. Usa una cuenta personal: con la de la universidad, Gemini puede dar error 403.

- **Groq**, el LLM que redacta la respuesta (la única imprescindible): entra en [console.groq.com](https://console.groq.com) con GitHub o con tu correo, ve a «API Keys» y crea una. Va en `GROQ_API_KEY`. El modelo es `openai/gpt-oss-120b`.
- **Gemini** (opcional): en [Google AI Studio](https://aistudio.google.com), «Get API key». Va en `GEMINI_API_KEY`. Sirve para los embeddings en la nube y, si Groq falla, como LLM de reserva (`gemini-3.5-flash-lite`).
- **Cohere** (opcional, embeddings en la nube): al registrarte en [dashboard.cohere.com](https://dashboard.cohere.com) se crea una clave de prueba. Va en `CO_API_KEY`.

## En clase

```text
pytest
```

Todo en rojo: es el punto de partida. Abre tu agente dentro de esta carpeta. Cinco peldaños, uno cada vez, cada uno con su test:

1. `trocear`: `pytest tests/test_1_trocear.py`
2. `buscar`: `pytest tests/test_2_buscar.py`
3. `fusionar`: `pytest tests/test_3_fusionar.py`, y después `python evaluar.py`
4. `prompt_con_citas`: `pytest tests/test_4_prompt.py`
5. `verificar`: `pytest tests/test_5_verificar.py`, y después `python preguntar.py S1`

## Qué hay

- `rag.py`: las cinco funciones de la clase, sin hacer, y las que ya vienen escritas, entre ellas BM25, el buscador por palabras.
- `modelos.py`: las llamadas a los modelos, ya escritas. Embeddings en local (EmbeddingGemma) o en la nube (Gemini, Cohere) y el LLM que redacta.
- `corpus/`: el texto de los tres informes, una línea por página, y sus fuentes.
- `indices/`: los 350 fragmentos y sus vectores de EmbeddingGemma, ya calculados. Los de Gemini y Cohere los calcula `preparar.py` con tus claves.
- `preguntas.json`: 12 preguntas con su respuesta de referencia y la cita que la prueba. Una no tiene respuesta en los informes.
- `banco.py`: el «banco» de la sesión 4 con un modelo de embeddings de verdad.
- `evaluar.py`: qué buscador encuentra la evidencia de cada pregunta.
- `preguntar.py`: el RAG completo. `--directo`, `--descomponer` y `--python` para comparar formas de responder.
- `preparar.py`: la comprobación de antes de clase.
- `AGENTS.md`: las reglas para tu agente.

## Los documentos

Tres informes públicos, reproducidos con fines docentes y citando la fuente, como permiten sus avisos legales:

- Banco de España, *Proyecciones macroeconómicas e informe trimestral de la economía española. Tercer trimestre de 2026*, 9 de octubre de 2026: https://www.bde.es/f/webbe/SES/Secciones/Publicaciones/InformesBoletinesRevistas/BoletinEconomico/26/T3/be2603-it.pdf
- Banco Central Europeo, *Declaración de política monetaria*, 10 de septiembre de 2026, traducción del Banco de España: https://www.bde.es/f/webbe/GAP/Secciones/SalaPrensa/ComunicadosBCE/DecisionesPoliticaMonetaria/26/mps260910.pdf
- CNMV, *Nota de estabilidad financiera. Primer semestre de 2026*, 30 de julio de 2026: https://www.cnmv.es/DocPortal/Publicaciones/NotaEstabilidad/Nota_de_estabilidad_jul_2026_ES.pdf

El texto salió de los PDF con pdfplumber, con los espacios normalizados y sin las páginas vacías. El BCE pide que se diga cuándo un dato es un cálculo propio: lo son las respuestas que restan o multiplican cifras de los informes.

## Reglas

- Las claves van en `.env`, que no se sube a git. Nada de claves, contraseñas ni datos de tu trabajo en el código.
- Los informes son públicos. A los servicios gratuitos no les mandes datos personales ni de tu trabajo.
