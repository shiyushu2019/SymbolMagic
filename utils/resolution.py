import ast
from itertools import product
import math

import torch

class MathExpr:
    """
    Wrap a mathematical expression string as a callable object.

    =================== Format requirements for the formula ===================
    1. Syntax: a valid Python math expression.
       - The multiplication sign is required: write 2*x, not 2x; functions
         must be called with parentheses: sin(x).

    2. Unknowns: any single English letter (case-sensitive), e.g. x, y, z, a, B.
       Exception: e and i are treated as constants, so they are not unknowns.

    3. Math constants:
       e      base of the natural logarithm
       i      imaginary unit (the result becomes complex)
       pi or π
       tau or τ
       inf or ∞
       nan    not a number

    4. Operators:
       binary arithmetic: +  -  *  /  //  %  **
       unary arithmetic:  +x  -x
       comparison:        ==  !=  <  <=  >  >=
       boolean:           and  or  not
       grouping:          ()

    5. Math functions (collected into funs):
       every function in the math module (sin/cos/tan/exp/log/sqrt/ceil/
       factorial/gcd/hypot/...) plus abs pow min max round.

    6. Not supported: attribute access ., subscripts [], lambda, if-else
       conditional expressions, list/dict/set literals and comprehensions,
       assignment, :=, f-strings, keyword arguments, etc.

    ========================= Attributes and calling ==========================
    expr      : the original string
    vars      : list of unknown names (in order of first appearance)
    funs      : set of math function names used

    e(*args, **kwargs) : evaluate
       - positional arguments are passed in the order of self.vars;
       - or pass them as keyword arguments by variable name.
    """

    # Whitelist: only allow the syntax needed by math expressions
    _NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.Call, ast.Name, ast.Load, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.UAdd, ast.USub, ast.Not,
        ast.And, ast.Or,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )
    _FUNCS = {k: v for k, v in vars(math).items()
              if not k.startswith('_') and callable(v)}
    _FUNCS.update(abs=abs, pow=pow, min=min, max=max, round=round)
    _CONSTS = {
        'e': math.e, 'i': 1j,
        'pi': math.pi, 'tau': math.tau, 'inf': math.inf, 'nan': math.nan,
    }
    _SYMBOLS = str.maketrans({'π': ' pi ', 'τ': ' tau ', '∞': ' inf '})

    def __init__(self, expr: str):
        self.expr = expr
        s = expr.translate(self._SYMBOLS).strip()
        if not s:
            raise ValueError("expression is empty")
        try:
            tree = ast.parse(s, mode='eval')
        except SyntaxError as exc:
            raise ValueError(f"expression syntax error: {exc}") from exc

        found, funs = [], set()
        for node in ast.walk(tree):
            if not isinstance(node, self._NODES):
                raise ValueError(f"unsupported syntax construct: {type(node).__name__}")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else None
                if name not in self._FUNCS:
                    raise ValueError(f"unknown function: {name}")
                funs.add(name)
            elif isinstance(node, ast.Name):
                if node.id in self._FUNCS or node.id in self._CONSTS:
                    continue
                if len(node.id) == 1 and node.id.isalpha():
                    found.append((node.lineno, node.col_offset, node.id))
                else:
                    raise ValueError(f"invalid identifier: {node.id}")
            elif isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float, complex)):
                    raise ValueError(f"only numeric constants are supported, got {node.value!r}")

        # Unknowns ordered by first appearance
        seen = set()
        self.vars = [name for _, _, name in sorted(found)
                     if not (name in seen or seen.add(name))]
        self.funs = funs

        self._code = compile(tree, '<expr>', 'eval')
        self._env = {**self._FUNCS, **self._CONSTS}

    def __call__(self, *args, **kwargs):
        if args and kwargs:
            raise TypeError("positional and keyword arguments cannot be mixed")
        if args:
            if len(args) != len(self.vars):
                raise ValueError(f"expected {len(self.vars)} unknowns {self.vars}, "
                                 f"got {len(args)}")
            env = dict(zip(self.vars, args))
        else:
            if set(kwargs) != set(self.vars):
                raise ValueError(f"expected variables {self.vars}, got {sorted(kwargs)}")
            env = kwargs
        return eval(self._code, {'__builtins__': {}}, {**self._env, **env})

    def __repr__(self):
        return f"MathExpr({self.expr!r})"

class LazyDataset:
    def __init__(self,
                 formula: str,
                 lower_bound: float,
                 upper_bound: float,
                 step: float):
        self.expr=MathExpr(formula)
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.step=step

        # [TODO] 这个计数部分较为低效，但是前期先这样保留
        self.one_dim_count = 0
        self.total_count = 1
        count = 0
        x = self.lower_bound
        while x < self.upper_bound -1e-10:
            count += 1
            x = self.lower_bound + count * self.step
        self.one_dim_count = count
        k = len(self.expr.vars)
        self.total_count = count ** k
    def __len__(self):
        return self.total_count
    def __getitem__(self, index):
        count = self.one_dim_count    
        total = self.total_count
        k = len(self.expr.vars)

        if isinstance(index, slice):
            start, stop, step = index.indices(total)
            return [self._at(i, count, k) for i in range(start, stop, step)]

        if index < 0:
            index += total

        if not (0 <= index < total):
            raise IndexError(f"index out of range for ExprDataset of size {total}")

        return self._at(index, count, k)
    def _at(self, index, count, k):
        coords = [0] * k
        for j in range(k - 1, -1, -1):
            coords[j] = index % count
            index //= count
        args = tuple(self.lower_bound + c * self.step for c in coords) 
        x = torch.tensor(args, dtype=torch.float32)          # (k,)，k=1 时 (1,)
        y = torch.tensor([self.expr(*args)], dtype=torch.float32)  # (1,)
        return x, y

class ActDataset:
    def __init__(self,
                 formula: str,
                 lower_bound: float,
                 upper_bound: float,
                 step: float):
        self.expr=MathExpr(formula)
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.step=step

        self.one_dim_count = 0
        self.total_count = 1
        count = 0
        x = self.lower_bound
        while x < self.upper_bound -1e-10:
            count += 1
            x = self.lower_bound + count * self.step
        self.one_dim_count = count
        k = len(self.expr.vars)
        self.total_count = count ** k

        axis = [self.lower_bound + c * self.step for c in range(count)]
        self._values = tuple(
            (torch.tensor(args, dtype=torch.float32),
            torch.tensor([self.expr(*args)], dtype=torch.float32))
            for args in product(axis, repeat=k)
        )
    def __len__(self):
        return self.total_count
    def __getitem__(self, index):
        return self._values[index]
        