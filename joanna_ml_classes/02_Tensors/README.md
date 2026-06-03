# Moving from Scalars to Tensors

## The Fundamental Question

In the previous lecture, we learned Gradient Descent using a single parameter:

$$
y = wx + b
$$

where:

- $x$ = Scalar Input
- $w$ = Scalar Weight
- $b$ = Scalar Bias

This helps us understand the learning process, but real-world AI systems cannot work with only scalars.

---

## We Have No Choice

Real-world data is naturally multi-dimensional.

### Image

An RGB image is represented as:

$$
X \in \mathbb{R}^{3 \times H \times W}
$$

Example:

```text
Channels × Height × Width

3 × 224 × 224
```

Each pixel contains information.

Reducing an image to a single scalar would destroy almost all useful information.

---

### Text

A sentence:

```text
I love Deep Learning
```

is not a scalar.

It is represented as a sequence of tokens:

```text
[I] [love] [Deep] [Learning]
```

which later become vectors and tensors.

---

### Video

A video contains:

```text
Frames × Channels × Height × Width
```

which is naturally a higher-dimensional tensor.

---

## Better Representation Power

Suppose we try to classify a cat using only one scalar:

$$
x = 5
$$

The model learns only one relationship:

$$
y = wx + b
$$

Very little information can be captured.

---

With tensors:

$$
Y = WX + b
$$

the model can learn:

- Edges
- Shapes
- Colors
- Textures
- Object Parts
- Semantic Features

Instead of learning from one value, it learns from thousands or millions of values simultaneously.

This gives neural networks their expressive power.

---

## Memory and Computational Efficiency

Modern hardware such as GPUs is optimized for tensor operations.

Instead of:

```python
for value in data:
    process(value)
```

we perform:

```python
process(entire_tensor)
```

Benefits:

- Parallel computation
- Faster training
- Better GPU utilization
- Reduced execution overhead
- Large-scale learning

Tensor operations are highly optimized inside libraries such as:

- NumPy
- PyTorch
- TensorFlow

---

## Types of Data Structures

### Scalar (0D Tensor)

A single number.

$$
x = 5
$$

```text
5
```

---

### Vector (1D Tensor)

A collection of numbers.

$$
[1,2,3,4]
$$

```text
[1 2 3 4]
```

Shape:

$$
(4)
$$

---

### Matrix (2D Tensor)

Rows and columns.

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

```text
[[1, 2],
 [3, 4]]
```

Shape:

$$
(2,2)
$$

---

### Tensor (3D+)

Multiple matrices stacked together.

Example RGB Image:

```text
3 × 224 × 224
```

Shape:

$$
(C,H,W)
$$

where:

- $C$ = Channels
- $H$ = Height
- $W$ = Width

---

## Key Takeaways

- Real-world data is not scalar.
- Images, text, audio, and videos naturally exist as tensors.
- Tensors provide significantly more representation power than scalars.
- Neural networks learn richer relationships using tensor-valued weights and biases.
- GPUs are designed to process tensors efficiently in parallel.
- Modern AI systems such as CNNs, Transformers, U-Nets, and GPTs are built entirely on tensor operations.

> Scalars help us understand learning. Tensors make modern AI possible.