"""Un RAG pequeño sobre tres informes públicos: Banco de España, BCE y CNMV.

Las cinco primeras funciones están sin hacer: son los peldaños de la clase. Los tests de `tests/` dicen
qué tiene que ser verdad para cada una. Las de abajo ya están escritas.

Convención: un fragmento es un diccionario {"doc", "pagina", "texto"}. N fragmentos convertidos en
vectores de d números son una matriz de N filas: la fila i es el fragmento i.
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
DOCUMENTOS = json.loads((AQUI / "corpus" / "fuentes.json").read_text(encoding="utf-8"))
NO_ESTA = "No está en los documentos."


# ---------- Peldaño 1: trocear ----------

def trocear(paginas: list[dict], tam: int = 800, solape: int = 100) -> list[dict]:
    """Corta el texto de cada página en fragmentos de `tam` caracteres como mucho.

    Cada fragmento empieza `tam - solape` caracteres después del anterior, así que dos seguidos comparten
    `solape` caracteres. El último de cada página termina donde termina la página. Un fragmento nunca
    mezcla dos páginas y lleva el "doc" y la "pagina" de donde sale.
    """
    raise NotImplementedError


# ---------- Peldaño 2: buscar por parecido ----------

def buscar(q: np.ndarray, E: np.ndarray, k: int = 5) -> np.ndarray:
    """Posiciones de las k filas de E más parecidas a q por coseno, de más a menos parecida.

    q tiene forma (d,) y E forma (N, d). Las filas de E y q no tienen por qué medir 1.
    """
    raise NotImplementedError


# ---------- Peldaño 3: juntar dos buscadores ----------

def fusionar(ordenes: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Junta varias ordenaciones de los mismos N fragmentos (la mejor posición primero) en una sola.

    Reciprocal Rank Fusion: en cada ordenación, el fragmento del puesto r suma 1 / (k + r), con r = 1 para
    el primero. Devuelve las N posiciones ordenadas por la suma, de más a menos.
    """
    raise NotImplementedError


# ---------- Peldaño 4: el prompt con citas ----------

def prompt_con_citas(pregunta: str, fragmentos: list[dict]) -> str:
    """El texto que se manda al modelo: instrucciones, fragmentos numerados y la pregunta, en ese orden.

    Cada fragmento va con su número entre corchetes, desde [1], y con `fuente(fragmento)`. Las instrucciones
    piden responder solo con los fragmentos, poner [n] detrás de cada dato y, si la respuesta no está,
    contestar exactamente NO_ESTA.
    """
    raise NotImplementedError


# ---------- Peldaño 5: verificar con Python ----------

def verificar(respuesta: str, fragmentos: list[dict]) -> list[str]:
    """Lista de problemas de la respuesta, cada uno una frase que nombra la cita o el número que falla.
    Vacía si todo cuadra.

    - Cada cita [n] tiene que existir: n entre 1 y len(fragmentos).
    - Cada número de la respuesta tiene que aparecer en alguno de los fragmentos que cita. 2,6 y 2.6 son el
      mismo número. Los números de las citas [n] no cuentan como números de la respuesta.
    - Si la respuesta tiene números y ninguna cita, también es un problema.
    """
    raise NotImplementedError


# ---------- Ya escritas ----------

def numeros_de(texto: str) -> list[str]:
    """Los números del texto, con coma decimal: «2.6» y «2,6» salen los dos como «2,6»."""
    return [x.replace(".", ",") for x in re.findall(r"\d+(?:[.,]\d+)?", texto)]


def cargar_paginas() -> list[dict]:
    """Las páginas con texto de los tres informes, en orden: {"doc", "pagina", "texto"}."""
    lineas = (AQUI / "corpus" / "paginas.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(linea) for linea in lineas if linea]


def fuente(fragmento: dict) -> str:
    """De dónde sale un fragmento, para citarlo: «BdE, informe trimestral (9 oct 2026), p. 9»."""
    return f"{DOCUMENTOS[fragmento['doc']]['corto']}, p. {fragmento['pagina']}"


PALABRAS_VACIAS = set("""a al algo ante con cual cuales cuando cuanto cuanta cuantas cuantos de del desde donde
el en entre es esta este esto ha han hasta la las lo los mas muy no o para pero por que se segun si sin sobre
su sus un una uno y ya hay como""".split())


def palabras(texto: str) -> list[str]:
    """Minúsculas, sin tildes y sin palabras vacías: «¿Cuál será el desempleo?» → ["sera", "desempleo"]."""
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")
    return [p for p in re.findall(r"\w+", sin_tildes) if p not in PALABRAS_VACIAS]


def orden_bm25(pregunta: str, fragmentos: list[dict], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Las N posiciones ordenadas por BM25, el buscador por palabras de toda la vida: cuenta cuántas veces sale
    cada palabra de la pregunta en cada fragmento, premia las palabras raras y corrige por el largo."""
    cuentas = [Counter(palabras(f["texto"])) for f in fragmentos]
    largo = np.array([sum(c.values()) for c in cuentas], dtype=float)
    puntos = np.zeros(len(fragmentos))
    for p in set(palabras(pregunta)):
        tf = np.array([c[p] for c in cuentas], dtype=float)
        n = np.count_nonzero(tf)
        idf = np.log((len(fragmentos) - n + 0.5) / (n + 0.5) + 1)
        puntos += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * largo / largo.mean()))
    return np.argsort(-puntos, kind="stable")
