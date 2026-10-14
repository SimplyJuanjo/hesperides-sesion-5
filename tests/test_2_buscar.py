import numpy as np

from rag import buscar


def test_devuelve_las_k_mas_parecidas_de_mas_a_menos():
    E = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert list(buscar(np.array([1.0, 0.2]), E, k=3)) == [0, 2, 1]


def test_devuelve_k_posiciones():
    E = np.random.default_rng(0).normal(size=(50, 8))
    assert len(buscar(E[7], E, k=5)) == 5


def test_cada_fila_se_parece_sobre_todo_a_si_misma():
    E = np.random.default_rng(1).normal(size=(50, 8))
    assert buscar(E[7], E, k=1)[0] == 7


def test_es_coseno_y_no_producto_escalar():
    E = np.array([[10.0, 10.0], [1.0, 0.0]])
    assert buscar(np.array([1.0, 0.0]), E, k=1)[0] == 1


def test_no_cambia_si_la_pregunta_mide_mas():
    E = np.random.default_rng(2).normal(size=(30, 4))
    q = np.random.default_rng(3).normal(size=4)
    assert list(buscar(q, E, k=5)) == list(buscar(100 * q, E, k=5))
