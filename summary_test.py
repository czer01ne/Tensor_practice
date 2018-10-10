import tensorflow as tf

tf.reset_default_graph()

## 데이터 수집

x_data = [1, 2, 3]
y_data = [1, 2, 3]

## 예측 모델 정의

W = tf.get_variable("W", initializer=tf.random_uniform([1], -1.0, 1.0))
b = tf.get_variable("b", initializer=tf.random_uniform([1], -1.0, 1.0))
y = W * x_data + b

## 비용 함수, 최적화 함수 정의

cost = tf.reduce_mean(tf.square(y - y_data))
alpha = tf.get_variable("alpha", initializer=0.01) # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

## 초기화, summary and graph log

init = tf.global_variables_initializer()

tf.summary.scalar("cost", cost)
tf.summary.scalar("W", W[0])
tf.summary.scalar("b", b[0])

summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

sess = tf.Session()
summary_writer = tf.summary.FileWriter("summary_logs/", sess.graph)
sess.run(init)

## 훈련

for step in range(500):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, step)
    saver.save(sess, "summary_logs/model-checkpoint", global_step=step)
