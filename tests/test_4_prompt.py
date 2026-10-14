from rag import NO_ESTA, fuente, prompt_con_citas

FRAGMENTOS = [
    {"doc": "bde_it_2026t3", "pagina": 9, "texto": "anticipa un crecimiento del PIB del 2,6 % en 2026"},
    {"doc": "bce_declaracion_20260910", "pagina": 3, "texto": "La inflación aumentó hasta el 3,3 % en agosto"},
]
PREGUNTA = "¿Cuánto crecerá el PIB en 2026?"


def test_cada_fragmento_va_detras_de_su_numero():
    p = prompt_con_citas(PREGUNTA, FRAGMENTOS)
    a, b = p.index(FRAGMENTOS[0]["texto"]), p.index(FRAGMENTOS[1]["texto"])
    assert "[1]" in p[:a] and "[2]" in p[a:b]


def test_cada_fragmento_dice_de_donde_sale():
    p = prompt_con_citas(PREGUNTA, FRAGMENTOS)
    assert fuente(FRAGMENTOS[0]) in p and fuente(FRAGMENTOS[1]) in p


def test_la_pregunta_va_despues_de_los_fragmentos():
    p = prompt_con_citas(PREGUNTA, FRAGMENTOS)
    assert p.index(PREGUNTA) > p.index(FRAGMENTOS[1]["texto"])


def test_dice_que_contestar_si_no_esta():
    assert NO_ESTA in prompt_con_citas(PREGUNTA, FRAGMENTOS)
