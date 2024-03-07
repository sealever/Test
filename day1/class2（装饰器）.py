import time

''' 函数装饰器 '''


def timer(name):  # 先传装饰器导入的参数‘张三’
    def namer(f):  # 导入被装饰的函数 machine1
        def inner(s):  # 导入被装饰的函数machine的参数
            start = time.time()
            f(s)
            end = time.time()
            print(f'{name},你好,程序所花费的时间为：{end - start}')

        return inner

    return namer


@timer('张三')  # 函数装饰器
def machine1(s):
    time.sleep(s)
    print('ending...')


machine1(2.5)

''' 类装饰器'''


class Timer:

    def __init__(self, name):
        self.name = name

    def __call__(self, f):
        def inner(s):  # 导入被装饰的函数machine的参数
            start = time.time()
            f(s)
            end = time.time()
            print(f'{self.name},你好,程序所花费的时间为：{end - start}')

        return inner


@Timer('张三')  # 类装饰器
def machine1(s):
    time.sleep(s)
    print('ending...')


machine1(2.5)

''' 多个装饰器 '''


def dec(func):  # func =wp2
    def wp1(*args):   # 执行wp1（3，4，5）
        res = func(*args)  # res = wp2(3,4,5) = 12
        return res
    return wp1


def timer(func):  # func = add
    def wp2(*args):
        start = time.time()
        res = func(*args)  # res =add(3,4,5) =12
        end = time.time()
        print(f'函数耗时{end - start}')
        return res
    return wp2


@dec
@timer
def add(*args):
    time.sleep(2)
    return sum(args)


print(add(3, 4, 5))  # add =wp1(3,4,5) = 12
