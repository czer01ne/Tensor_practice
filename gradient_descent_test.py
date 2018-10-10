## 학습 알고리듬 설정
x = 6 # The algorithm starts at x=6
delta = 0.01 # step size
n = 50 # number of learning iterations

## 손실 함수
def f(x):
    return x ** 4 - 3 * x ** 3 + 2

## 수작업 미분 함수
def f_derivative(x):
    return 4 * x**3 - 9 * x**2

## 경사하강 학습
print("%3d: f(%f) = %f" % (0, x, f(x)))

for count in range(n):
    x -= delta * f_derivative(x)
    print("%3d: f(%f) = %f" % ((count+1), x, f(x)))
