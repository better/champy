from ortools.linear_solver import pywraplp

class Expression:
    def __init__(self, expr, constant=0):
        self._expr = expr
        self._constant = constant

    def __mul__(self, rhs):
        assert type(rhs) in [float, int]
        return Expression(tuple((k*rhs, v) for k, v in self._expr))

    def __rmul__(self, lhs):
        return self * lhs

    def __truediv__(self, rhs):
        return self * (1./rhs)

    def __add__(self, rhs):
        if type(rhs) in [float, int]:
            return Expression(self._expr, self._constant + rhs)
        else:
            assert isinstance(rhs, Expression)
            coeffs = {}
            vs = {}
            for c, v in self._expr:
                coeffs[v._id] = c
                vs[v._id] = v
            for c, v in rhs._expr:
                coeffs[v._id] = coeffs.get(v._id, 0) + c
                vs[v._id] = v
            return Expression(tuple((coeffs[v._id], v) for v in vs.values()), self._constant + rhs._constant)

    def __radd__(self, lhs):
        return self * lhs

    def __neg__(self):
        return self * -1

    def __sub__(self, rhs):
        return self + (-rhs)

    def __rsub__(self, lhs):
        return lhs + (-self)

    def __abs__(self):
        return Problem.abs(self)
    
    #def __lt__(self, rhs):
    #    return Polytope(((-self - rhs, '>'),))

    def __le__(self, rhs):
        return Polytope(((-self - rhs, '>='),))

    #def __gt__(self, rhs):
    #    return Polytope(((self - rhs, '>'),))

    def __ge__(self, rhs):
        return Polytope(((self - rhs, '>='),))

    def __eq__(self, rhs):
        return Polytope(((self - rhs, '=='),))

    def __str__(self):
        return 'Expression(%s+%.2f)' % (''.join('%+.2f%s' % (k, v) for k, v in self._expr), self._constant)


class Variable(Expression):
    _ids = {}
    def __init__(self, name=None, lo=None, hi=None, cat=None):
        self._name = name
        self._expr = ((1, self),)
        self._constant = 0
        self._lo = lo
        self._hi = hi
        self._cat = cat
        if name is None:
            id = 'untitled'
        else:
            id = name
        if name in Variable._ids:
            id += '_%d' % Variable._ids[name]
        Variable._ids[name] = Variable._ids.get(name, 0) + 1
        self._id = id

    def __str__(self):
        return self._id

    def __hash__(self):
        return hash(self._id)


class CategoricalVariable:
    # This actually isn't a variable, just a convenience wrapper
    def __init__(self, name='untitled', options=[]):
        self._name = name
        self._options = {o: Variable(name='%s==%s' % (name, o)) for o in options}

    def __eq__(self, rhs):
        return self == self._options[rhs]


class Polytope:
    def __init__(self, constraints):
        self._constraints = tuple(constraints)

    @staticmethod
    def all(polytopes):
        return Polytope(sum((c._constraints for c in polytopes), tuple()))

    @staticmethod
    def any(polytopes, big_M=999999):
        # This one is the most interesting
        polytopes = tuple(polytopes)
        magic = [Variable('magic', 0, 1, 'binary') for p in polytopes]
        new_constraints = []
        for polytope, m in zip(polytopes, magic):
            for expr, op in polytope._constraints:
                new_constraints.append((expr + m * big_M, op))
        return Polytope(new_constraints) & \
            (sum(magic) <= len(polytopes)-1)

    def __and__(self, rhs):
        return Polytope.all((self, rhs))

    def __or__(self, rhs):
        return Polytope.any((self, rhs))
    
    def __str__(self):
        return 'Polytope(%s)' % ', '.join('%s %s 0' % (str(expr), op) for expr, op in self._constraints)


class Problem:
    # TODO: some types of problems, like abs(x-y), are more like a *constrained* expression,
    # so it would make sense to make this class a subclass of Expression, or merge the two.
    # Open question: how do you perform expression algebra in the presence of constraints
    # Eg. should you merge the constraints when you do abs(x+y) < abs(z+w)? I think so.
    def __init__(self, polytope, objective):
        self._polytope = polytope
        self._objective = objective

    @staticmethod
    def abs(m):
        z_neg = Variable('z_neg', lo=0, hi=None)
        z_pos = Variable('z_pos', lo=0, hi=None)
        return Problem(m + z_neg + z_pos == 0, z_neg + z_pos)

    def solve(self):
        solver = pywraplp.Solver('test', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        vs = {}
        objective = solver.Objective()
        def get_var(v):
            if v._id not in vs:
                vs[v._id] = (v, solver.NumVar(v._lo if v._lo is not None else -solver.infinity(),
                                              v._hi if v._hi is not None else solver.infinity(),
                                              '%s_%s' % (v._name, v._id)))
            return vs[v._id][1]
            
        for k, v in self._objective._expr:
            objective.SetCoefficient(get_var(v), k)
        objective.SetMaximization()
        for c, op in self._polytope._constraints:
            if op == '==':
                constraint = solver.Constraint(c._constant, c._constant)
            elif op == '>=':
                constraint = solver.Constraint(c._constant, solver.infinity())
            for k, v in c._expr:
                constraint.SetCoefficient(get_var(v), k)
        res = solver.Solve()
        solution = {}
        for k, (v, nv) in vs.items():
            solution[v] = nv.solution_value()
        return solution

    def __str__(self):
        return 'Problem(min %s s.t. %s)' % (self._objective, self._polytope)
