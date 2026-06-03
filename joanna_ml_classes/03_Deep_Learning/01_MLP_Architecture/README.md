## MLP Architecture for Image Data

Images are naturally represented as **multi-dimensional tensors** rather than flat vectors. This representation preserves the spatial organization of pixels and serves as the standard format for image data in deep learning.

### 1. Representation of a Single Image

A color image contains three channels:

* Red (R)
* Green (G)
* Blue (B)

Each channel is a matrix of pixel values with dimensions **Height × Width**.

Therefore, a single RGB image is naturally represented as a **3D tensor**:

$$
(3, H, W)
$$

where:

* $3$ = Number of channels (RGB)
* $H$ = Height of the image
* $W$ = Width of the image

For example, a color image of size $224 \times 224$ is represented as:

$$
(3, 224, 224)
$$

---

### 2. Representation of Multiple Images

During training, neural networks process multiple images simultaneously in the form of batches.

If a batch contains $N$ images, the images are stacked together to form a **4D tensor**:

$$
(N, 3, H, W)
$$

where:

* $N$ = Number of images in the batch
* $3$ = RGB channels
* $H$ = Height
* $W$ = Width

For example:

$$
(128, 3, 224, 224)
$$

represents a batch of 128 RGB images, each having dimensions $224 \times 224$.

This is the natural representation of image datasets in deep learning frameworks such as PyTorch and TensorFlow.

---

### 3. Grayscale Images

Unlike RGB images, grayscale images contain only a single channel representing pixel intensity.

A single grayscale image is represented as:

$$
(1, H, W)
$$

and a batch of grayscale images is represented as:

$$
(N, 1, H, W)
$$

where:

* $1$ = Single grayscale channel
* $N$ = Number of images in the batch

For example:

$$
(64, 1, 28, 28)
$$

represents a batch of 64 grayscale images of size $28 \times 28$.

---

### 4. Input Requirement of an MLP

A Multi-Layer Perceptron (MLP) operates on feature vectors and expects its input to be a **2D tensor** of shape:

$$
(N, D)
$$

where:

* $N$ = Number of samples
* $D$ = Number of features per sample

The forward pass through an MLP is given by:

$$
y = Wx + b
$$

Since image data is naturally stored as a 4D tensor, it cannot be directly passed into an MLP.

---

### 5. Flattening the Image

To make image data compatible with an MLP, every image must be flattened into a one-dimensional vector.

For RGB images:

$$
(N, 3, H, W)
$$

is reshaped into:

$$
(N, 3 \times H \times W)
$$

For example:

$$
(128, 3, 224, 224)
$$

becomes:

$$
(128, 150528)
$$

because:

$$
3 \times 224 \times 224 = 150528
$$

Each image is now represented as a single feature vector containing all pixel values.

---

### 6. Complete Data Flow

The complete pipeline for using images with an MLP is:

$$
(N, 3, H, W)
\rightarrow
(N, 3 \times H \times W)
\rightarrow
\text{MLP Layers}
\rightarrow
\text{Prediction}
$$

or for grayscale images:

$$
(N, 1, H, W)
\rightarrow
(N, H \times W)
\rightarrow
\text{MLP Layers}
\rightarrow
\text{Prediction}
$$

---

### Summary

* A single RGB image is naturally represented as:

$$
(3, H, W)
$$

* A batch of RGB images is represented as:

$$
(N, 3, H, W)
$$

* A batch of grayscale images is represented as:

$$
(N, 1, H, W)
$$

* MLPs expect a 2D tensor of shape:

$$
(N, D)
$$

* Therefore, image tensors must be reshaped before being passed into an MLP:

$$
(N, 3, H, W)
\rightarrow
(N, 3 \times H \times W)
$$

This flattening step converts the naturally occurring 4D image representation into the 2D format required by a Multi-Layer Perceptron.
