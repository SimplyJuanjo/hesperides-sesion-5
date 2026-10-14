"""¿Qué buscador encuentra la evidencia? Las 11 preguntas de preguntas.json que tienen respuesta, con cada uno.

    python evaluar.py

Cada celda es el puesto del primer fragmento que contiene la evidencia (1 = el primero). Abajo, en cuántas
preguntas sale entre los 5 primeros, que son los que verá el LLM. Hace falta tener en verde los peldaños
1 a 3 (trocear, buscar y fusionar). No llama a ninguna API: las preguntas ya están convertidas en vectores.
"""

import json
import sys

import numpy as np

import modelos
from rag import AQUI, buscar, cargar_paginas, fusionar, orden_bm25, trocear

TOP = 5
CORTO = {modelos.LOCAL: "gemma", modelos.GEMINI: "gemini", modelos.COHERE: "cohere"}


def es_evidencia(fragmento: dict, ev: dict) -> bool:
    cita = ev["cita"]
    return (fragmento["doc"] == ev["doc"] and fragmento["pagina"] == ev["pagina"]
            and (cita in fragmento["texto"] or cita[:40] in fragmento["texto"] or cita[-40:] in fragmento["texto"]))


def puesto(orden: np.ndarray, fragmentos: list[dict], evidencias: list[dict]) -> int:
    for r, i in enumerate(orden, 1):
        if any(es_evidencia(fragmentos[i], ev) for ev in evidencias):
            return r
    return len(orden) + 1


def tabla(fragmentos: list[dict]) -> dict[str, dict[str, int]]:
    preguntas = json.loads((AQUI / "preguntas.json").read_text(encoding="utf-8"))
    filas = {}
    for i, p in enumerate(preguntas):
        if not p["evidencias"]:
            continue
        bm25 = orden_bm25(p["pregunta"], fragmentos)
        filas[p["id"]] = {"BM25": puesto(bm25, fragmentos, p["evidencias"])}
        for m in modelos.modelos_listos():
            E = modelos.cargar_indice(m)
            denso = buscar(modelos.cargar_preguntas_embebidas(m)[i], E, k=len(E))
            filas[p["id"]][CORTO[m]] = puesto(denso, fragmentos, p["evidencias"])
            filas[p["id"]][f"{CORTO[m]}+BM25"] = puesto(fusionar([bm25, denso]), fragmentos, p["evidencias"])
    return filas


def main() -> None:
    try:
        fragmentos = trocear(cargar_paginas())
        if len(fragmentos) != len(modelos.cargar_indice(modelos.LOCAL)):
            sys.exit(f"Tu trocear da {len(fragmentos)} fragmentos y el índice tiene 350. Pasa el test 1.")
        filas = tabla(fragmentos)
    except NotImplementedError:
        sys.exit("Te falta trocear, buscar o fusionar en tu rag.py. Ponlos en verde con pytest y vuelve.")
    columnas = list(next(iter(filas.values())))
    print("      " + "".join(f"{c:>13}" for c in columnas))
    for qid, celdas in filas.items():
        print(f"  {qid:<4}" + "".join(f"{celdas[c]:>13}" for c in columnas))
    aciertos = [sum(celdas[c] <= TOP for celdas in filas.values()) for c in columnas]
    print(f"  top{TOP}" + "".join(f"{f'{a} de {len(filas)}':>13}" for a in aciertos))


if __name__ == "__main__":
    main()
