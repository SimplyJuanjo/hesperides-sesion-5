import json
from pathlib import Path

from rag import cargar_paginas, trocear

INDICES = Path(__file__).resolve().parent.parent / "indices"


def pagina(texto, doc="bde_it_2026t3", n=1):
    return {"doc": doc, "pagina": n, "texto": texto}


def textos(fragmentos):
    return [f["texto"] for f in fragmentos]


def test_trocear_corta_con_solape():
    assert textos(trocear([pagina("abcdefghij")], tam=4, solape=1)) == ["abcd", "defg", "ghij"]


def test_ningun_fragmento_pasa_de_tam():
    assert all(len(t) <= 800 for t in textos(trocear([pagina("x" * 2500)])))


def test_dos_fragmentos_seguidos_comparten_el_solape():
    t = textos(trocear([pagina("".join(chr(65 + i % 26) for i in range(2000)))]))
    for a, b in zip(t, t[1:]):
        assert a[-100:] == b[:100]


def test_una_pagina_corta_es_un_solo_fragmento():
    assert textos(trocear([pagina("hola")])) == ["hola"]


def test_el_ultimo_termina_donde_termina_la_pagina():
    texto = "".join(str(i % 10) for i in range(1234))
    fragmentos = trocear([pagina(texto)])
    assert texto.startswith(fragmentos[0]["texto"]) and texto.endswith(fragmentos[-1]["texto"])


def test_un_fragmento_nunca_mezcla_paginas():
    fragmentos = trocear([pagina("a" * 900, n=1), pagina("b" * 300, n=2)])
    assert [(f["pagina"], set(f["texto"])) for f in fragmentos] == [(1, {"a"}), (1, {"a"}), (2, {"b"})]


def test_cada_fragmento_lleva_doc_y_pagina():
    fragmentos = trocear([pagina("texto", doc="cnmv_nota_estabilidad_jul2026", n=17)])
    assert fragmentos == [{"doc": "cnmv_nota_estabilidad_jul2026", "pagina": 17, "texto": "texto"}]


def test_con_los_tres_informes_salen_los_350_fragmentos_del_indice():
    lineas = (INDICES / "fragmentos.jsonl").read_text(encoding="utf-8").splitlines()
    assert trocear(cargar_paginas()) == [json.loads(linea) for linea in lineas]
