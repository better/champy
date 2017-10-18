from ortools.linear_solver import pywraplp

class Expression:
    def __init__(self, expr, constant=0, polytope=None):
        self._expr = expr
        self._constant = constant
        if polytope is None:
            polytope = Polytope()
        self._polytope = polytope # only used for certain ones tied to the definition (see abs)

    def __mul__(self, rhs):
        assert type(rhs) in [float, int]
        return Expression(
            tuple((k*rhs, v) for k, v in self._expr),
            self._constant*rhs,
            self._polytope
        )

    def __rmul__(self, lhs):
        return self * lhs

    def __truediv__(self, rhs):
        if type(rhs) in [float, int]:
            return self * (1./rhs)
        else:
            assert isinstance(rhs, Expression)
            return QuotientExpression(self, rhs)

    def __rtruediv__(self, lhs):
        if type(lhs) in [float, int]:
            lhs = Expression(tuple(), lhs)
        assert isinstance(lhs, Expression)
        return QuotientExpression(lhs, self)

    def __add__(self, rhs):
        if type(rhs) in [float, int]:
            return Expression(
                self._expr,
                self._constant + rhs,
                self._polytope
            )
        else:
            assert isinstance(rhs, Expression)
            vs = {}
            for c, v in self._expr:
                vs[v] = vs.get(v, 0) + c
            for c, v in rhs._expr:
                vs[v] = vs.get(v, 0) + c
            return Expression(
                tuple((c, v) for v, c in vs.items()),
                self._constant + rhs._constant,
                self._polytope & rhs._polytope
            )

    def __radd__(self, lhs):
        return self + lhs

    def __neg__(self):
        return self * -1

    def __sub__(self, rhs):
        return self + (-rhs)

    def __rsub__(self, lhs):
        return lhs + (-self)
    
    #def __lt__(self, rhs):
    #    return Polytope(((-self - rhs, '>'),))

    def __le__(self, rhs):
        return Polytope(((rhs - self, '>='),))

    #def __gt__(self, rhs):
    #    return Polytope(((self - rhs, '>'),))

    def __ge__(self, rhs):
        return Polytope(((self - rhs, '>='),))

    def __eq__(self, rhs):
        return Polytope(((self - rhs, '=='),))

    def __abs__(self):
        z_neg = Variable('z_neg', lo=0, hi=None)
        z_pos = Variable('z_pos', lo=0, hi=None)
        return Expression(
            ((1, z_neg), (1, z_pos)),
            0,
            self + z_neg + z_pos == 0)

    def __str__(self):
        return 'Expression(%s+%.2f)' % (''.join('%+.2f%s' % (k, v) for k, v in self._expr), self._constant)

    def __repr__(self):
        return str(self)


class QuotientExpression:
    # Not an Expression subclass. Can only be used to compare with constants.
    # Useful for saying eg (x + 5) / (y + 2) <= 3
    def __init__(self, nom, den):
        self._nom = nom
        self._den = den

    def __le__(self, rhs):
        return Polytope(((self._den * rhs - self._nom, '>='),))

    def __ge__(self, rhs):
        return Polytope(((self._nom - self._den * rhs, '>='),))

    def __eq__(self, rhs):
        return Polytope(((self._den * rhs - self._nom, '=='),))


class Variable(Expression):
    _ids = {}
    def __init__(self, name=None, lo=None, hi=None, type=float):
        self._name = name
        self._lo = lo
        self._hi = hi
        self._type=type
        if name is None:
            id = 'untitled'
        else:
            id = name
        if name in Variable._ids:
            id += '_%d' % Variable._ids[name]
        Variable._ids[name] = Variable._ids.get(name, 0) + 1
        self._id = id
        super(Variable, self).__init__(((1, self),))

    def __str__(self):
        return self._id

    def __repr__(self):
        return 'Variable#%s' % self._id

    def __hash__(self):
        return hash(self._id)


class CategoricalVariable:
    # This actually isn't a variable, just a convenience wrapper
    def __init__(self, name='untitled', options=[]):
        self._name = name
        self._options = options
        self._vars = {o: Variable(name='[%s==%s]' % (name, o), type=bool) for o in options}
        polytope = sum(self._vars.values()) == 1
        for v in self._vars.values():
            v._polytope = polytope

    def __eq__(self, rhs):
        if isinstance(rhs, CategoricalVariable):
            assert self._options == rhs._options
            return Polytope.all(self._vars[o] == rhs._vars[o] for o in self._options)
        else:
            assert rhs in self._vars
            return self._vars[rhs] == 1

    def __ne__(self, rhs):
        # TODO: support rhs being another CategoricalVariable
        return self != self._options[rhs]


class Polytope:
    def __init__(self, constraints=tuple()):
        self._constraints = tuple(constraints)

    @staticmethod
    def all(polytopes):
        return Polytope(sum((c._constraints for c in polytopes), tuple()))

    @staticmethod
    def any(polytopes, big_M=999999):
        # This one is the most interesting
        polytopes = tuple(polytopes)
        magic = [Variable('magic', type=bool) for p in polytopes]
        new_constraints = []
        for polytope, m in zip(polytopes, magic):
            for expr, op in polytope._constraints:
                # TODO: need to handle equality constraints here
                new_constraints.append((expr + m * big_M, op))
        return Polytope(new_constraints) & (sum(magic) <= len(polytopes)-1)

    def __and__(self, rhs):
        return Polytope.all((self, rhs))

    def __or__(self, rhs):
        return Polytope.any((self, rhs))
    
    def optimize(self, objective, sign):
        if not isinstance(objective, Expression):
            # Not a function of the variables, so let's just replace it with a constant 0
            objective = Expression(tuple())
        ot_solver = pywraplp.Solver('test', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        ot_variables = {}
        ot_objective = ot_solver.Objective()
        cls = {int: lambda lo, hi, name: ot_solver.IntVar(lo, hi, name),
               float: lambda lo, hi, name: ot_solver.NumVar(lo, hi, name),
               bool: lambda lo, hi, name: ot_solver.BoolVar(name)}
        def get_var(v):
            if v not in ot_variables:
                ot_variables[v] = cls[v._type](v._lo if v._lo is not None else -ot_solver.infinity(),
                                               v._hi if v._hi is not None else ot_solver.infinity(),
                                               v._id)
            return ot_variables[v]

        for k, v in objective._expr:
            ot_objective.SetCoefficient(get_var(v), k * sign)
        ot_objective.SetMaximization()
        added_polytopes = set()
        def add_polytope(p):
            if p in added_polytopes:
                return
            added_polytopes.add(p)
            for c, op in p._constraints:
                if op == '==':
                    ot_constraint = ot_solver.Constraint(-c._constant, -c._constant)
                elif op == '>=':
                    ot_constraint = ot_solver.Constraint(-c._constant, ot_solver.infinity())
                for k, v in c._expr:
                    ot_constraint.SetCoefficient(get_var(v), k)
                if c._polytope:
                    add_polytope(c._polytope)
        add_polytope(self)
        if objective._polytope:
            add_polytope(objective._polytope)
        res = ot_solver.Solve()
        solution = {}
        for v, nv in ot_variables.items():
            solution[v] = nv.solution_value()
        return solution

    def maximize(self, objective):
        return self.optimize(objective, 1)

    def minimize(self, objective):
        return self.optimize(objective, -1)

    def __str__(self):
        return 'Polytope(%s)' % ', '.join('%s %s 0' % (str(expr), op) for expr, op in self._constraints)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))
