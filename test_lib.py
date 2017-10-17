import pytest
from lib import *

def test_simple():
    x = Variable('x', lo=0)
    y = Variable('y', lo=0)

    polytope = (x + 3*y <= 10) & \
               (3*x + y <= 10)

    solution = polytope.maximize(x+y)
    assert solution[x] == pytest.approx(10./4)
    assert solution[y] == pytest.approx(10./4)

    solution = polytope.maximize(5*x + y)
    assert solution[x] == pytest.approx(10./3)
    assert solution[y] == pytest.approx(0)


def test_simple_integer():
    x = Variable('x', lo=0, type=int)
    y = Variable('y', lo=0, type=int)

    polytope = (x + 3*y <= 10) & \
               (3*x + y <= 10)

    solution = polytope.maximize(x+y)
    assert solution[x] == pytest.approx(10//4)
    assert solution[y] == pytest.approx(10//4)

    solution = polytope.maximize(5*x + y)
    assert solution[x] == pytest.approx(3)
    assert solution[y] == pytest.approx(1)


def test_or():
    x = Variable('x', lo=0)
    y = Variable('y', lo=0)

    polytope = ((x <= 10) & (y <= 1)) | \
               ((x <= 3) & (y <= 9))

    solution = polytope.maximize(x+y)
    assert solution[x] == pytest.approx(3)
    assert solution[y] == pytest.approx(9)

    solution = polytope.maximize(2*x+y)
    assert solution[x] == pytest.approx(10)
    assert solution[y] == pytest.approx(1)


def test_abs():
    x = Variable('x', lo=0)
    y = Variable('y', lo=0)

    polytope = (x + 3*y <= 10) & \
               (3*x + y <= 10)

    solution = polytope.minimize(abs(x-7))
    assert solution[x] == pytest.approx(10./3)
    assert solution[y] == pytest.approx(0.0)

    solution = polytope.minimize(abs(y-7))
    assert solution[x] == pytest.approx(0.0)
    assert solution[y] == pytest.approx(10./3)

    solution = polytope.minimize(abs(x-7) + abs(y-7))
    assert solution[x] == pytest.approx(10./4)
    assert solution[y] == pytest.approx(10./4)
