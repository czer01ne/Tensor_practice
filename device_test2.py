with g.device('/gpu:0'):
  # All operations constructed in this context will be placed on GPU 0.
  with g.device(None):
    # All operations constructed in this context will have no assigned device.

# Defines a function from `Operation` to device string.
def matmul_on_gpu(n):
  if n.type == "MatMul":
    return "/gpu:0"
  else:
    return "/cpu:0"

with g.device(matmul_on_gpu):
  # All operations of type "MatMul" constructed in this context
  # will be placed on GPU 0; all other operations will be placed
  # on CPU 0.
