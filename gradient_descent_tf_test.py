import tensorflow as tf

precision = 0.00001

## 최적화 함수 정의

x = tf.get_variable("x", initializer=6.0)
y = x**4 - 3 * x**3 + 2

## 경사 하강 알고리듬 설정

a = tf.get_variable("a", initializer=0.01) # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(y)

# 변수 초기화
init = tf.global_variables_initializer()

## 훈련
sess = tf.Session()
sess.run(init)

count = 0
x_old = 0
x_new = sess.run(x)
print("%3d: f(%f) = %f, precision: %f" % (count, x_new, sess.run(y), x_new - x_old))

while abs(x_new - x_old) > precision:
    count += 1
    x_old = x_new
    sess.run(train)
    x_new = sess.run(x)
    print("%3d: f(%f) = %f, precision: %f" % (count, x_new, sess.run(y), x_new - x_old))

print("Local minimum occurs at", x_new)
