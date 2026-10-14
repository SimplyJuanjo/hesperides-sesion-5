from rag import NO_ESTA, verificar

FRAGMENTOS = [
    {"doc": "bde_it_2026t3", "pagina": 9, "texto": "anticipa un crecimiento del PIB del 2,6 % en 2026 y del 2,2 % en 2027"},
    {"doc": "bce_declaracion_20260910", "pagina": 3, "texto": "La inflación aumentó hasta el 3,3 % en agosto"},
]


def test_una_respuesta_bien_citada_no_tiene_problemas():
    assert verificar("El PIB crecerá un 2,6 % en 2026 [1].", FRAGMENTOS) == []


def test_dos_datos_de_dos_fragmentos():
    assert verificar("El PIB crecerá un 2,6 % en 2026 [1] y la inflación fue del 3,3 % [2].", FRAGMENTOS) == []


def test_una_cita_que_no_existe():
    assert any("[3]" in p for p in verificar("El PIB crecerá un 2,6 % [3].", FRAGMENTOS))


def test_un_numero_inventado():
    problemas = verificar("El PIB crecerá un 2,8 % en 2026 [1].", FRAGMENTOS)
    assert len(problemas) == 1 and "2,8" in problemas[0]


def test_un_numero_de_otro_fragmento_no_vale():
    problemas = verificar("La inflación fue del 3,3 % [1].", FRAGMENTOS)
    assert len(problemas) == 1 and "3,3" in problemas[0]


def test_coma_y_punto_decimal_son_el_mismo_numero():
    assert verificar("El PIB crecerá un 2.6 % [1].", FRAGMENTOS) == []


def test_numeros_sin_ninguna_cita():
    assert verificar("El PIB crecerá un 2,6 %.", FRAGMENTOS) != []


def test_decir_que_no_esta_no_es_un_problema():
    assert verificar(NO_ESTA, FRAGMENTOS) == []


def test_los_numeros_de_las_citas_no_cuentan():
    assert verificar("Crecerá [1] y la inflación subirá [2].", FRAGMENTOS) == []
