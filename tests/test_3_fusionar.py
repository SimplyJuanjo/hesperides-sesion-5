import numpy as np

from rag import fusionar


def test_con_una_sola_ordenacion_no_cambia_nada():
    assert list(fusionar([np.array([3, 0, 2, 1])])) == [3, 0, 2, 1]


def test_devuelve_todas_las_posiciones_una_vez():
    assert sorted(fusionar([np.array([4, 2, 0, 1, 3]), np.array([0, 1, 2, 3, 4])])) == [0, 1, 2, 3, 4]


def test_el_que_sale_primero_en_las_dos_gana():
    assert fusionar([np.array([2, 0, 1]), np.array([2, 1, 0])])[0] == 2


def test_segundo_en_las_dos_gana_a_primero_en_una_y_ultimo_en_otra():
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.array([9, 1, 2, 3, 4, 5, 6, 7, 8, 0])
    assert fusionar([a, b])[0] == 1


def test_el_primer_puesto_es_r_igual_a_1():
    a = np.array([0, 1, 2, 3, 4, 5])
    b = np.array([2, 1, 3, 4, 5, 0])
    assert list(fusionar([a, b], k=1)) == [2, 1, 0, 3, 4, 5]
