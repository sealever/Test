import math


def func(num1):
    if type(num1) == Fraction:
        return num1
    if type(num1) == int:
        res = Fraction(num1, 1)
        return res
    if type(num1) == float:
        p = 10 ** len(str(num1).split('.')[1])
        res = Fraction(num1*p, p)
        return res
    raise TypeError('数值类型错误')


class Fraction:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        gcd = math.gcd(self.a, self.b)
        a = self.a // gcd
        b = self.b // gcd
        if b < 0:
            a = -a
            b = -b
        if b == 1:
            return f'{a}'
        return f'{a}/{b}'

    def __neg__(self):
        return Fraction(-self.a, self.b)

    def __add__(self, other):  # self:左 other： 右
        other = func(other)
        return Fraction(self.a * other.b + self.b * other.a, self.b * other.b)

    def __radd__(self, other):  # self:右 other： 左
        return self + other

    def __sub__(self, other):   # self:左 other： 右
        other = func(other)
        return Fraction(self.a * other.b - self.b * other.a, self.b * other.b)

    def __rsub__(self, other):  # self:右 other： 左
        return -(self - other)

    def __mul__(self, other):
        other = func(other)
        return Fraction(self.a * other.a, self.b * other.b)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = func(other)
        return Fraction(self.a * other.b, self.b * other.a)

    def __rtruediv__(self, other):
        other = func(other)
        return func(1) / (self / other)


f1 = Fraction(1, 2)
f2 = Fraction(1, -3)
print(f1 / f2)
