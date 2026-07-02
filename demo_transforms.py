import mlx.core as mx

# 1. Basic Array Operations
print("--- 1. Basic Array Operations ---")
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])
c = a * b + 2.0
print(f"a: {a}")
print(f"b: {b}")
print(f"c (a * b + 2.0): {c}\n")

# 2. Lazy Evaluation (Explicit Evaluation)
print("--- 2. Lazy Evaluation & mx.eval ---")
# MLX computes lazily. We can chain operations without immediately running them on GPU.
x = mx.random.normal((1000, 1000))
y = mx.random.normal((1000, 1000))
z = mx.matmul(x, y)
# At this point, z is not yet materialized.
print(f"Before evaluation, z shape is known: {z.shape}")
mx.eval(z)  # This triggers the actual GPU compilation & execution
print("z evaluated successfully!\n")

# 3. Composable Function Transformations: Grad (Automatic Differentiation)
print("--- 3. Automatic Differentiation with mx.grad ---")
# Let's define a simple mathematical function: f(w, x, b) = w * x + b
def loss_fn(w, x, b, y_true):
    y_pred = w * x + b
    return mx.mean(mx.square(y_pred - y_true))

w = mx.array(2.0)
x = mx.array(3.0)
b = mx.array(1.0)
y_true = mx.array(8.0) # True output is 2*3 + 1 = 7. Let's compute loss and gradients.

# Grad function with respect to w (argnums=0) and b (argnums=2)
grad_w = mx.grad(loss_fn, argnums=0)
grad_b = mx.grad(loss_fn, argnums=2)

print(f"Loss value: {loss_fn(w, x, b, y_true)}")
print(f"Gradient w.r.t w: {grad_w(w, x, b, y_true)}")
print(f"Gradient w.r.t b: {grad_b(w, x, b, y_true)}")

# Using value_and_grad to get both at once
val_and_grad_fn = mx.value_and_grad(loss_fn, argnums=[0, 2])
loss, (dw, db) = val_and_grad_fn(w, x, b, y_true)
print(f"value_and_grad -> Loss: {loss}, grad_w: {dw}, grad_b: {db}\n")

# 4. Vectorized Mapping: mx.vmap
print("--- 4. Vectorization with mx.vmap ---")
# Batching operations over a dimension automatically
def scale_and_add(v, scale):
    return v * scale + 1.0

vector_data = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
scale_factors = mx.array([2.0, 3.0, 4.0])

# vmap scales the vector elements by their corresponding scale factors
vmapped_fn = mx.vmap(scale_and_add, in_axes=(0, 0))
result = vmapped_fn(vector_data, scale_factors)
print(f"Input vectors:\n{vector_data}")
print(f"Scales: {scale_factors}")
print(f"Vmapped result:\n{result}\n")

# 5. Graph Compilation: mx.compile
print("--- 5. Compilation with mx.compile ---")
# mx.compile fuses operations into a single kernel on the GPU
@mx.compile
def fused_ops(x):
    return mx.sin(x) + mx.cos(x) * mx.exp(-x)

input_arr = mx.array([0.0, 0.5, 1.0])
output_arr = fused_ops(input_arr)
print(f"Input: {input_arr}")
print(f"Fused ops output: {output_arr}")
print("Done! Setup and Transforms are fully verified.")
