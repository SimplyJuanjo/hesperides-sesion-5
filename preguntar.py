"""Pregunta a los tres informes. Busca, arma el prompt con citas, se lo da al LLM y verifica la respuesta.

    python preguntar.py "¿Cuál será el desempleo en España el año que viene?"
    python preguntar.py S1                    una pregunta de preguntas.json, por su id
    python preguntar.py S1 --directo          sin documentos: lo que el modelo sabe o se inventa
    python preguntar.py M2 --descomponer      parte la pregunta en otras más simples y busca cada una
    python preguntar.py N1 --python           las cuentas las hace Python, no el modelo
    python preguntar.py P1 --modelo gemini-embedding-2 --k 8

Hacen falta los cinco peldaños en verde y una clave de LLM en .env.
"""

import argparse
import ast
import json
import operator
import re
import sys

import modelos
from rag import AQUI, buscar, cargar_paginas, fuente, fusionar, orden_bm25, prompt_con_citas, trocear, verificar

PARTIR = """Parte esta pregunta en las preguntas más simples que hay que buscar por separado en unos informes \
para poder responderla. Una por línea, sin numerar, como mucho cuatro. Si no hace falta partirla, \
devuelve la misma pregunta.

Pregunta: {}"""

CUENTA = """

Si la respuesta necesita una cuenta, no la hagas tú. Da los datos con su cita y termina con una línea \
que empiece por CUENTA: y lleve la operación solo con números, + - * / y paréntesis. Por ejemplo:
CUENTA: (3,9 - 3,0) * 2"""

OPERACIONES = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
               ast.USub: operator.neg, ast.UAdd: operator.pos}


def calcular(expresion: str) -> float:
    """Evalúa una cuenta con + - * / y paréntesis, y nada más: lo que escribe un LLM no se ejecuta tal cual."""
    def valor(nodo):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)):
            return nodo.value
        if isinstance(nodo, ast.BinOp) and type(nodo.op) in OPERACIONES:
            return OPERACIONES[type(nodo.op)](valor(nodo.left), valor(nodo.right))
        if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in OPERACIONES:
            return OPERACIONES[type(nodo.op)](valor(nodo.operand))
        raise ValueError(f"En una cuenta solo valen números y + - * /: {expresion}")
    return valor(ast.parse(re.sub(r"(\d),(\d)", r"\1.\2", expresion), mode="eval").body)


def recuperar(pregunta: str, fragmentos: list[dict], modelo: str, k: int) -> list[int]:
    E = modelos.cargar_indice(modelo)
    q = modelos.embeber([pregunta], modelo, "pregunta")[0]
    return list(fusionar([orden_bm25(pregunta, fragmentos), buscar(q, E, k=len(E))])[:k])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pregunta", help="la pregunta, o el id de una de preguntas.json (P1, M2, S1...)")
    p.add_argument("--modelo", default=modelos.LOCAL, choices=list(modelos.CLAVE_DE))
    p.add_argument("--k", type=int, default=6, help="cuántos fragmentos ve el LLM")
    p.add_argument("--directo", action="store_true")
    p.add_argument("--descomponer", action="store_true")
    p.add_argument("--python", action="store_true")
    args = p.parse_args()

    conocidas = {q["id"]: q for q in json.loads((AQUI / "preguntas.json").read_text(encoding="utf-8"))}
    conocida = conocidas.get(args.pregunta.upper())
    pregunta = conocida["pregunta"] if conocida else args.pregunta
    proveedor, llm = modelos.llm_elegido()
    print(f"Pregunta: {pregunta}\nLLM: {llm} ({proveedor})")

    if args.directo:
        print(f"\n{modelos.redactar(f'Responde en español, en dos o tres frases: {pregunta}')}")
        print("\nSin fragmentos no hay nada que verificar.")
    else:
        try:
            fragmentos = trocear(cargar_paginas())
            subpreguntas = [pregunta]
            if args.descomponer:
                subpreguntas = [s.strip("-• ").strip() for s in modelos.redactar(PARTIR.format(pregunta)).splitlines()]
                subpreguntas = [s for s in subpreguntas if s][:4]
                print("Subpreguntas:\n" + "\n".join(f"  {s}" for s in subpreguntas))
            elegidos = []
            for s in subpreguntas:
                for i in recuperar(s, fragmentos, args.modelo, args.k if len(subpreguntas) == 1 else 4):
                    if i not in elegidos:
                        elegidos.append(i)
            elegidos = [fragmentos[i] for i in elegidos]
            prompt = prompt_con_citas(pregunta, elegidos) + (CUENTA if args.python else "")
            respuesta = modelos.redactar(prompt)
            problemas = verificar(respuesta, elegidos)
        except NotImplementedError:
            sys.exit("Te falta algún peldaño en tu rag.py. Ponlos en verde con pytest y vuelve.")
        print(f"Embeddings: {args.modelo}, {len(elegidos)} fragmentos\n\n{respuesta}\n")
        for n, f in enumerate(elegidos, 1):
            print(f"  [{n}] {fuente(f)}")
        if args.python:
            for cuenta in re.findall(r"CUENTA:\s*(.+)", respuesta):
                print(f"\nPython: {cuenta.strip()} = {calcular(cuenta):.4g}".replace(".", ","))
        print("\nVerificación: todos los números y citas cuadran." if not problemas else "\nVerificación:")
        for problema in problemas:
            print(f"  {problema}")

    if conocida:
        print(f"\nRespuesta de referencia: {conocida['respuesta']}")


if __name__ == "__main__":
    main()
