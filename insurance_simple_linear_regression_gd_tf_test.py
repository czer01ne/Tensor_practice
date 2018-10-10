import tensorflow as tf
import math
import numpy as np
import pandas as pd

## 데이터 수집

insurance = pd.read_csv("insurance.csv")
print(insurance[0:5])

age = insurance["age"].values
expenses = insurance["expenses"].values

## 예측 모델 정의

X = tf.placeholder("float")
Y = tf.placeholder("float")

tf_coef = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
tf_intercept = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
tf_expenses_pred = tf_coef * X + tf_intercept

## 비용 함수, 최적화 함수 정의

tf_cost = tf.reduce_mean(tf.square(tf_expenses_pred - Y))
a = tf.Variable(0.1) # 학습률 alpha. 가중치가 infinity, nan으로 발산
#a = tf.Variable(0.0001) # 학습률 alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(tf_cost)
# 변수 초기화
init = tf.global_variables_initializer()

## 훈련
sess = tf.Session()
sess.run(init)
for step in range(300000):
    sess.run(train, feed_dict={X: age, Y: expenses})
    if step % 10000 == 0:
        cost, coef, intercept = sess.run([tf_cost, tf_coef, tf_intercept],
                                         feed_dict={X: age, Y: expenses})
        print(step, cost, math.sqrt(cost), coef, intercept)

## 예측
coef = sess.run(tf_coef)
intercept = sess.run(tf_intercept)
print(coef * np.array([20, 40, 60]) + intercept)

print(sess.run(tf_expenses_pred, feed_dict={X: [20.0, 40.0, 60.0]}))
