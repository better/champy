import pytest
from lib import *


def test_1d_constraint():
    x = Scalar('x')
    polytope = (x >= 3) & (x <= 5)
    solution = polytope.maximize(x)
    assert solution[x] == pytest.approx(5)
    solution = polytope.minimize(x)
    assert solution[x] == pytest.approx(3)


def test_1d_equality():
    x = Scalar('x')
    polytope = (x == 7)
    solution = polytope.maximize(x)
    assert solution[x] == pytest.approx(7)


def test_simple():
    x = Scalar('x', lo=0)
    y = Scalar('y', lo=0)

    polytope = (x + 3*y <= 10) & \
               (3*x + y <= 10)

    solution = polytope.maximize(x+y)
    assert solution[x] == pytest.approx(10./4)
    assert solution[y] == pytest.approx(10./4)

    solution = polytope.maximize(5*x + y)
    assert solution[x] == pytest.approx(10./3)
    assert solution[y] == pytest.approx(0)


def test_same_name():
    x1 = Scalar('x', lo=0)
    x2 = Scalar('x', lo=0)
    assert str(x1) == str(x2)
    assert x1 is not x2

    polytope = (x1 + 3*x2 <= 10) & \
               (3*x1 + x2 <= 10)

    solution = polytope.maximize(5*x1 + x2)
    assert solution[x1] == pytest.approx(10./3)
    assert solution[x2] == pytest.approx(0)


def test_simple_integer():
    x = Scalar('x', lo=0, type=int)
    y = Scalar('y', lo=0, type=int)

    polytope = (x + 3*y <= 10) & \
               (3*x + y <= 10)

    solution = polytope.maximize(x+y)
    assert solution[x] == pytest.approx(10//4)
    assert solution[y] == pytest.approx(10//4)

    solution = polytope.maximize(5*x + y)
    assert solution[x] == pytest.approx(3)
    assert solution[y] == pytest.approx(1)


def test_quotient():
    x = Scalar('x', lo=0)
    y = Scalar('y', lo=0)

    polytope = (1 / y / 2 >= 1) & \
               (y / (1 - x) * 10 <= 10)

    solution = polytope.maximize(x + 2*y)
    assert solution[x] == pytest.approx(0.5)
    assert solution[y] == pytest.approx(0.5)


def test_or():
    x = Scalar('x', lo=0, hi=100)
    y = Scalar('y', lo=0, hi=100)

    polytope = ((x <= 10) & (y <= 1)) | \
               ((x <= 3) & (y <= 9))

    solution = polytope.maximize(x+y)
    assert solution[x] == pytest.approx(3)
    assert solution[y] == pytest.approx(9)

    solution = polytope.maximize(2*x+y)
    assert solution[x] == pytest.approx(10)
    assert solution[y] == pytest.approx(1)


def test_or_equality():
    x = Scalar('x', lo=0, hi=100)
    polytope = (x == 3) | (x == 5)

    solution = polytope.maximize(x)
    assert solution[x] == pytest.approx(5)

    solution = polytope.minimize(x)
    assert solution[x] == pytest.approx(3)


def test_abs():
    x = Scalar('x', lo=0)
    y = Scalar('y', lo=0)

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


def test_abs_2():
    z = Scalar('z', lo=0, hi=100)
    polytope = z >= 5
    solution = polytope.minimize(abs(z-1))
    assert solution[z] == 5


def test_categorical():
    x = Categorical('x', ['A', 'B', 'C'])
    y = Categorical('y', ['A', 'B', 'C'])

    polytope = (x == 'C') & (x == y)
    solution = polytope.minimize(1)

    assert solution[x] == 'C'
    assert solution[y] == 'C'


def test_categorical_inequality_checkerboard():
    n = 4
    board = [[Categorical('x', [1, 0]) for col in range(n)]
             for row in range(n)]
    constraints = [board[row][col] != board[row+1][col] for col in range(n) for row in range(n-1)] + \
                  [board[row][col] != board[row][col+1] for col in range(n-1) for row in range(n)]
    solution = Polytope.all(constraints).minimize(1)
    solution = [[solution[board[row][col]] for col in range(n)]
                for row in range(n)]
    assert solution == [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]] or \
        solution == [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]


def test_switch():
    x = Scalar(lo=0, hi=10)
    y = Scalar(lo=0, hi=10)

    polytope = Polytope.switch({
        y <= 5: x <= 10,
        y <= 10: x <= 5
    })

    solution = polytope.minimize(abs(x-9) + abs(y-10))
    assert solution[x] == pytest.approx(5)
    assert solution[y] == pytest.approx(10)

    solution = polytope.minimize(abs(x-10) + abs(y-9))
    assert solution[x] == pytest.approx(10)
    assert solution[y] == pytest.approx(5)
