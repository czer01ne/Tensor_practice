import tensorflow as tf

a = tf.constant(1.0)
b = tf.constant(2.0)
c = a + b

print(type(a), ",", a)
print(type(b), ",", b)
print(type(c), ",", c)

print("get_operations():")
for op in tf.get_default_graph().get_operations():
    print("op.name: ", op.name, ", op.type: ", op.type, ", op.inputs: ", [x for x in op.inputs],
              ", op.outputs: ", [x for x in op.outputs], sep="")
g = tf.get_default_graph()
print(a.graph is g)

print("Const:", g.get_operation_by_name("Const"))
print("Const.output[0]:", g.get_operation_by_name("Const").outputs[0])
print("Const:0", g.get_tensor_by_name("Const:0"))
print("a:", a, type(a))
print("a.name: ", a.name, ", a.op.name: ", a.op.name, ", a.value_index: ", a.value_index,
          ", a.shape: ", a.shape, ", a.dtype: ", a.dtype, sep="")

print("Const_1:", g.get_operation_by_name("Const_1"))
print("Const_1.outputs[0]:", g.get_operation_by_name("Const_1").outputs[0])
print("Const_1:0", g.get_tensor_by_name("Const_1:0"))
print("b:", b, type(b))
print("b.name: ", b.name, ", b.op.name: ", b.op.name, ", b.value_index: ", b.value_index,
          ", b.shape: ", b.shape, ", b.dtype: ", b.dtype, sep="")

print("add:", g.get_operation_by_name("add"))
print("add.outputs[0]:", g.get_operation_by_name("add").outputs[0])
print("add.inputs[0]:", g.get_operation_by_name("add").inputs[0])
print("add.inputs[1]:", g.get_operation_by_name("add").inputs[1])
print("c:", c, type(c))
print("c.name: ", c.name, ", c.op.name: ", c.op.name, ", c.value_index: ", c.value_index,
          ", c.shape: ", c.shape, ", c.dtype: ", c.dtype, sep="")

sess = tf.Session()
v = sess.run(g.get_tensor_by_name("Const:0"))
print("%s: %r\n" % (type(v), v))
sess.close()
