"""Deja todo listo antes de clase. Ejecútalo las veces que haga falta: lo que ya está hecho no se repite.

    python preparar.py

1. Comprueba Python y los paquetes de requirements.txt.
2. Descarga EmbeddingGemma (1,26 GB, solo la primera vez) en la carpeta modelos/.
3. Con cada clave que encuentre en .env, prueba el servicio. Con las de Gemini y Cohere calcula además tu
   índice en la nube: 350 fragmentos y 12 preguntas, unos segundos.
"""

import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent


def linea(bien: bool, texto: str) -> None:
    print(f"  {'bien ' if bien else 'falta'}  {texto}")


def paquetes() -> bool:
    print("Instalación")
    linea(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (hace falta 3.10 o más)")
    faltan = [p for p in ("numpy", "pytest", "fastembed") if importlib.util.find_spec(p) is None]
    linea(not faltan, "paquetes: " + (f"faltan {', '.join(faltan)}. pip install -r requirements.txt" if faltan
                                      else "numpy, pytest y fastembed"))
    return sys.version_info >= (3, 10) and not faltan


def modelo_local(modelos) -> None:
    print("\nEmbeddings en tu ordenador")
    primera = not (AQUI / "modelos").exists()
    if primera:
        print("  Descargando EmbeddingGemma, 1,26 GB. Puede tardar varios minutos.")
    inicio = time.time()
    v = modelos.embeber(["¿Qué decidió el BCE sobre los tipos de interés?"], modelos.LOCAL, "pregunta")
    linea(v.shape == (1, 768), f"{modelos.LOCAL}: vector de {v.shape[1]} números en {time.time() - inicio:.0f} s")


def indice_nube(modelos, modelo: str) -> None:
    variable = modelos.CLAVE_DE[modelo]
    if not modelos.clave(variable):
        linea(False, f"{modelo}: no hay {variable} en .env (opcional)")
        return
    if modelos.hay_indice(modelo) and (modelos.INDICES / f"{modelo}-preguntas.npy").exists():
        linea(True, f"{modelo}: tu índice ya está calculado")
        return
    lineas = (modelos.INDICES / "fragmentos.jsonl").read_text(encoding="utf-8").splitlines()
    textos = [json.loads(x)["texto"] for x in lineas]
    preguntas = [p["pregunta"] for p in json.loads((AQUI / "preguntas.json").read_text(encoding="utf-8"))]
    try:
        inicio = time.time()
        E = modelos.embeber(textos, modelo, "documento").astype(np.float32)
        Q = modelos.embeber(preguntas, modelo, "pregunta").astype(np.float32)
    except Exception as e:
        linea(False, f"{modelo}: {e}")
        if "403" in str(e) and modelo == modelos.GEMINI:
            print("         Si usas la cuenta de la universidad, prueba con una cuenta personal de Gmail.")
        return
    np.save(modelos.INDICES / f"{modelo}.npy", E)
    np.save(modelos.INDICES / f"{modelo}-preguntas.npy", Q)
    linea(True, f"{modelo}: índice de {E.shape[0]} fragmentos y {E.shape[1]} números en {time.time() - inicio:.0f} s")


def llms(modelos) -> None:
    print("\nEl LLM que redacta")
    alguno = False
    for nombre, (_, variable, modelo, _) in modelos.LLMS.items():
        if not modelos.clave(variable):
            linea(False, f"{nombre}: no hay {variable} en .env")
            continue
        try:
            inicio = time.time()
            texto = modelos.llamar_llm(nombre, modelo, "Contesta solo con la palabra: listo")
            linea(True, f"{nombre} ({modelo}) contesta «{texto[:30]}» en {time.time() - inicio:.1f} s")
            alguno = True
        except Exception as e:
            linea(False, f"{nombre} ({modelo}): {e}")
    if not alguno:
        print("  Hace falta al menos una clave de LLM. Mira .env.ejemplo.")


def main() -> None:
    if not paquetes():
        sys.exit("\nArregla la instalación y vuelve a ejecutar python preparar.py")
    # modelos importa fastembed: solo se puede importar después de comprobar que está instalado.
    modelos = importlib.import_module("modelos")
    if not (AQUI / ".env").exists():
        print("\nNo hay .env: copia .env.ejemplo como .env y pon tus claves.")
    modelo_local(modelos)
    print("\nEmbeddings en la nube")
    for modelo in (modelos.GEMINI, modelos.COHERE):
        indice_nube(modelos, modelo)
    llms(modelos)
    print("\nModelos de embeddings listos para evaluar.py y preguntar.py:", ", ".join(modelos.modelos_listos()))


if __name__ == "__main__":
    main()
