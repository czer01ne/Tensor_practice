import tensorflow as tf

# 텐서플로 기본 그래프 생성
a = tf.constant(1)
b = tf.constant(2)
c = a + b
addOp = tf.get_default_graph().get_operation_by_name("add")

# 세션 생성 및 닫기
sess = tf.Session()
v = sess.run(c)
print("%s: %r\n" % (type(v), v))
v = sess.run(addOp)
print("%s: %r\n" % (type(v), v))
sess.close()

# 세션 생성 및 자동 닫기
with tf.Session() as sess:
    v = sess.run(c)
    print("%s: %r\n" % (type(v), v))

# 기본 세션 (as_default())
sess = tf.Session()
with sess.as_default():
    v = addOp.run()
    print("%s: %r\n" % (type(v), v))
    v = c.eval()
    print("%s: %r\n" % (type(v), v))

sess.close()

# 기본 세션 (tf.InteractiveSession)
sess = tf.InteractiveSession()
v = addOp.run()
print("%s: %r\n" % (type(v), v))
v = c.eval()
print("%s: %r\n" % (type(v), v))
sess.close()
