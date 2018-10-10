import tensorflow as tf

## 데이터 수집

x_data = [1, 2, 3]
y_data = [1, 2, 3]

## 예측 모델 정의

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
y = W * x_data + b

## 비용 함수, 최적화 함수 정의

cost = tf.reduce_mean(tf.square(y - y_data))
a = tf.Variable(0.1) # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# 변수 초기화
init = tf.global_variables_initializer()

## 훈련
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
