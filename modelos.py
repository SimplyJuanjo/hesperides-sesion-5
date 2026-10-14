"""Los modelos: embeddings en local o en la nube, y el LLM que redacta la respuesta.

Ya está escrito. Las claves salen de las variables de entorno o del archivo `.env` de esta carpeta: copia
`.env.ejemplo` como `.env` y rellénalo. `.env` no se sube a git.
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

AQUI = Path(__file__).resolve().parent
INDICES = AQUI / "indices"

LOCAL = "embeddinggemma-300m"
GEMINI = "gemini-embedding-2"
COHERE = "embed-v5.0-pro"
CLAVE_DE = {LOCAL: None, GEMINI: "GEMINI_API_KEY", COHERE: "CO_API_KEY"}

# Pregunta y documento no se codifican igual. EmbeddingGemma y Gemini lo piden como prefijo del texto.
PREFIJO = {"pregunta": "task: search result | query: ", "documento": "title: none | text: "}

# Proveedores gratuitos con la misma API que OpenAI: (dirección, variable de la clave, modelo, ajustes).
# gpt-oss razona siempre; con "low" gasta menos. En Gemini 3, Google pide no bajar la temperatura de 1.
LLMS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b",
             {"temperature": 0, "reasoning_effort": "low", "include_reasoning": False}),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-3.5-flash-lite", {}),
}
# Incluye lo que el modelo razona antes de contestar. Groq descuenta entrada + MAX_TOKENS de sus 8.000 tokens por minuto.
MAX_TOKENS = 1000


def leer_env() -> None:
    ruta = AQUI / ".env"
    if not ruta.exists():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            nombre, valor = linea.split("=", 1)
            if valor.strip():
                os.environ.setdefault(nombre.strip(), valor.strip().strip('"'))


leer_env()


def clave(variable: str) -> str | None:
    return os.environ.get(variable) or None


# ---------- Embeddings ----------

_gemma = None


def embeber(textos: list[str], modelo: str, tipo: str) -> np.ndarray:
    """Una fila por texto. `tipo` es "pregunta" o "documento"."""
    if modelo == LOCAL:
        global _gemma
        if _gemma is None:
            _gemma = TextEmbedding("google/embeddinggemma-300m", cache_dir=str(AQUI / "modelos"))
        return np.array(list(_gemma.embed([PREFIJO[tipo] + t for t in textos])))
    if modelo == GEMINI:
        filas = []
        for i in range(0, len(textos), 100):
            peticiones = [{"model": f"models/{GEMINI}",
                           "content": {"parts": [{"text": PREFIJO[tipo] + t}]},
                           "embedContentConfig": {"outputDimensionality": 768}} for t in textos[i:i + 100]]
            r = enviar(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI}:batchEmbedContents",
                       {"requests": peticiones}, {"x-goog-api-key": clave("GEMINI_API_KEY")})
            filas += [e["values"] for e in r["embeddings"]]
        return np.array(filas)
    if modelo == COHERE:
        filas = []
        for i in range(0, len(textos), 96):
            cuerpo = {"model": COHERE, "texts": textos[i:i + 96], "output_dimension": 1024, "embedding_types": ["float"],
                      "input_type": "search_query" if tipo == "pregunta" else "search_document"}
            r = enviar("https://api.cohere.com/v2/embed", cuerpo, {"Authorization": f"Bearer {clave('CO_API_KEY')}"})
            filas += r["embeddings"]["float"]
        return np.array(filas)
    raise ValueError(f"No conozco el modelo {modelo}. Los que hay: {', '.join(CLAVE_DE)}")


def hay_indice(modelo: str) -> bool:
    return (INDICES / f"{modelo}.npy").exists()


def cargar_indice(modelo: str) -> np.ndarray:
    """La matriz (N, d) con un vector por fragmento, calculada antes con `preparar.py`."""
    return np.load(INDICES / f"{modelo}.npy")


def cargar_preguntas_embebidas(modelo: str) -> np.ndarray:
    """La matriz (12, d) con un vector por pregunta de `preguntas.json`, en el mismo orden."""
    return np.load(INDICES / f"{modelo}-preguntas.npy")


def modelos_listos() -> list[str]:
    """Los modelos de embeddings que se pueden usar ahora: con índice y, si son de la nube, con clave."""
    return [m for m in CLAVE_DE if hay_indice(m) and (CLAVE_DE[m] is None or clave(CLAVE_DE[m]))]


# ---------- El LLM que redacta ----------

def llm_elegido() -> tuple[str, str]:
    """(proveedor, modelo). Manda LLM_PROVEEDOR si está; si no, el primero con clave."""
    elegido = os.environ.get("LLM_PROVEEDOR")
    for nombre, (_, variable, modelo, _) in LLMS.items():
        if (elegido is None or elegido == nombre) and clave(variable):
            return nombre, os.environ.get("LLM_MODELO") or modelo
    raise RuntimeError("No hay clave para ningún LLM. Mira .env.ejemplo y ejecuta python preparar.py")


def llamar_llm(proveedor: str, modelo: str, prompt: str) -> str:
    url, variable, _, ajustes = LLMS[proveedor]
    cuerpo = {"model": modelo, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS, **ajustes}
    eleccion = enviar(f"{url}/chat/completions", cuerpo, {"Authorization": f"Bearer {clave(variable)}"})["choices"][0]
    texto = (eleccion["message"].get("content") or "").strip()
    if not texto and eleccion.get("finish_reason") == "length":
        raise RuntimeError(f"{modelo} gastó los {MAX_TOKENS} tokens razonando y no llegó a contestar. "
                           "Sube MAX_TOKENS en modelos.py.")
    return texto


def redactar(prompt: str) -> str:
    """Manda el prompt al LLM elegido y devuelve su respuesta."""
    return llamar_llm(*llm_elegido(), prompt)


# ---------- HTTP con la biblioteca estándar ----------

def enviar(url: str, cuerpo: dict, cabeceras: dict, reintentos: int = 3) -> dict:
    """POST con JSON. Si el servicio responde 429 (demasiadas peticiones por minuto), espera y repite."""
    servidor = url.split("/")[2]
    peticion = urllib.request.Request(url, data=json.dumps(cuerpo).encode("utf-8"), method="POST", headers={
        "Content-Type": "application/json", "User-Agent": "hesperides-sesion-5", **cabeceras})
    for intento in range(reintentos + 1):
        try:
            with urllib.request.urlopen(peticion, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detalle = e.read().decode("utf-8", errors="replace")[:400]
            try:
                espera = float(e.headers.get("retry-after") or 10)
            except ValueError:
                espera = 10
            # Una espera de más de un minuto es el cupo del día, no el del minuto: no merece la pena esperar.
            if e.code == 429 and intento < reintentos and espera <= 60:
                print(f"  {servidor} pide esperar {espera:.0f} s: límite de peticiones por minuto.")
                time.sleep(espera)
                continue
            raise RuntimeError(f"{servidor} respondió {e.code}: {detalle}") from None
