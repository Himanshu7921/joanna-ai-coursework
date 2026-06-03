# Joanna AI Coursework

AI/ML coursework repository for Joanna, conducted by **Himanshu Singh** under **ModernAgeCoders**.

This repository contains lecture notes, mathematical derivations, coding exercises, implementations, assignments, and learning resources covered throughout the course.

---

## Course Philosophy

This course focuses on building AI knowledge from **first principles**.

Instead of treating AI as a collection of libraries and frameworks, we gradually build the mathematical and engineering intuition behind modern systems such as:

- Neural Networks
- CNNs
- GPT Models
- Generative AI Systems
- Reinforcement Learning Agents

The goal is to understand **how modern AI systems learn internally**.

---

## Recommended Reading Order

The materials are intentionally organized in a progressive manner.

Follow them in the exact order below.

### 1. Gradient Descent

Start here.

This module introduces:

- AI as Function Approximation
- Model Parameters
- Forward Pass
- Loss Functions
- Mean Squared Error (MSE)
- Numerical Derivatives
- Gradient Descent
- Parameter Optimization

**Read Order**

```text
01_Gradient_Descent/
├── README.md
├── mathematical_derivation.md
└── gradient_descent.ipynb
```

---

### 2. Tensors

After understanding how a model learns, we move from simple scalar values to tensors.

Topics covered:

- Scalars
- Vectors
- Matrices
- Tensors
- Tensor Shapes
- Reshaping
- Batch Dimensions
- Why Modern AI Uses Tensors

**Read Order**

```text
02_Tensors/
├── README.md
└── tensors.ipynb
```

---

### 3. Deep Learning

Once tensors are understood, we can build neural networks.

---

#### 3.1 MLP Architecture

Topics covered:

- Neurons
- Fully Connected Layers
- Matrix Multiplication
- Multi-Layer Perceptrons
- Universal Function Approximation
- Deep Neural Networks

**Read Order**

```text
03_Deep_Learning/
└── 01_MLP_Architecture/
    ├── README.md
    └── mlp.ipynb
```

---

#### 3.2 CNN Architecture

Topics covered:

- Why MLPs struggle with images
- Spatial Information
- Convolutions
- Kernels
- Feature Maps
- CNN Intuition

**Read Order**

```text
03_Deep_Learning/
└── 02_CNN_Architecture/
```

---

## Repository Structure

```text
joanna_ml_classes/
│
├── 01_Gradient_Descent/
│   ├── README.md
│   ├── mathematical_derivation.md
│   └── gradient_descent.ipynb
│
├── 02_Tensors/
│   ├── README.md
│   └── tensors.ipynb
│
├── 03_Deep_Learning/
│   │
│   ├── 01_MLP_Architecture/
│   │   ├── README.md
│   │   └── mlp.ipynb
│   │
│   └── 02_CNN_Architecture/
│
├── assets/
│
└── README.md
```

---

## Learning Path

```text
Gradient Descent
        ↓
Tensors
        ↓
Matrix Multiplication
        ↓
Neurons
        ↓
Fully Connected Layers
        ↓
MLPs
        ↓
CNNs
        ↓
Modern Deep Learning
        ↓
Transformers & GPTs
```

Each module builds directly on concepts introduced in previous lectures.

Skipping topics is strongly discouraged.

---

## Author

**Himanshu Singh**  
AI/ML Instructor  
ModernAgeCoders