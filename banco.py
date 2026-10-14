"""El «banco» de la sesión 4, con un modelo de embeddings de verdad.

    python banco.py
    python banco.py "me senté en el banco" "pedí un crédito al banco" "un banco de peces"
    python banco.py --modelo gemini-embedding-2

Cada frase entra entera en el modelo y sale un vector. La tabla es el coseno entre cada par de frases.
"""

import argparse

import numpy as np

import modelos

FRASES = [
    "me senté en el banco del parque",
    "descansé en un asiento del jardín",
    "pedí un crédito al banco",
    "la entidad me concedió un préstamo",
    "el BCE subió los tipos de interés",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("frases", nargs="*", default=FRASES)
    p.add_argument("--modelo", default=modelos.LOCAL, choices=list(modelos.CLAVE_DE))
    args = p.parse_args()
    V = modelos.embeber(args.frases, args.modelo, "documento")
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    C = V @ V.T
    print(f"{args.modelo}: {len(args.frases)} frases, cada una un vector de {V.shape[1]} números\n")
    for i, frase in enumerate(args.frases, 1):
        print(f"  {i}  {frase}")
    print("\n     " + "".join(f"{j:>7}" for j in range(1, len(args.frases) + 1)))
    for i, fila in enumerate(C, 1):
        print(f"  {i:>2} " + "".join(f"{x:7.2f}" for x in fila))


if __name__ == "__main__":
    main()
