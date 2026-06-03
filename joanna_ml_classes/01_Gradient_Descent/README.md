# Lecture 01: Gradient Descent

### *Understanding How Modern AI Systems Learn*

**By Himanshu Singh**

---

## The Big Idea

Artificial Intelligence is nothing more than a function approximator.

Given an input $X$, the model tries to learn a function:

$$
\hat{y} = f(X;\theta)
$$

where:

* $X$ = Input data
* $\hat{y}$ = Prediction
* $\theta$ = Parameters of the model (weights and biases)

The entire goal of training is to find the best possible values of $\theta$.

---

## A Single Neuron

Everything in Deep Learning starts from a simple equation of a line:

genui{"math_block_widget_always_prefetch_v2":{"content":"y = mx + c"}}

In neural networks, we write it as:

genui{"math_block_widget_always_prefetch_v2":{"content":"y = wx + b"}}

where:

* $w$ → Weight
* $b$ → Bias
* $x$ → Input

A neuron is simply learning the best values of $w$ and $b$.

---

## AI as a Classification System

Imagine we have a dataset:

| Image     | Label |
| --------- | ----- |
| Dog Image | 1     |
| Cat Image | 0     |
| Dog Image | 1     |
| Cat Image | 0     |

The model sees many examples and learns patterns.

After training:

$$
f(X;\theta) \rightarrow 1
$$

means **Dog**

and

$$
f(X;\theta) \rightarrow 0
$$

means **Cat**

---

## Forward Pass

The Forward Pass is simply the model making a prediction.

For every training example:

$$
\hat{y}^{(1)} = f(X_1;\theta)
$$

$$
\hat{y}^{(2)} = f(X_2;\theta)
$$

$$
\hat{y}^{(3)} = f(X_3;\theta)
$$

The model is answering:

> "Based on my current knowledge, what do I think this image is?"

At this stage, the model is only predicting.

No learning happens yet.

---

## The Problem

The model predictions are often wrong.

We have:

$$
\hat{y}
$$

Predicted Value

and

$$
y
$$

Ground Truth Value

Our objective is:

$$
\hat{y} \approx y
$$

The closer the prediction is to the true value, the better the model.

---

## The Learning Story

Think of the model as a student taking a test.

1. Look at the question (Input)
2. Write an answer (Prediction)
3. Compare with correct answer (Ground Truth)
4. Calculate mistake (Loss)
5. Improve for next attempt

This cycle repeats thousands or millions of times.

---

## Gradient Descent

The question becomes:

> How do we find the best values of the parameters $\theta$?

The answer is **Gradient Descent**.

Gradient Descent continuously updates the parameters to reduce prediction error.

Parameter update rule:

\theta = \theta - \eta \nabla J(\theta)

where:

* $\theta$ = Parameters
* $\eta$ = Learning Rate
* $\nabla J(\theta)$ = Gradient of Loss

The gradient tells us:

> Which direction increases the error?

Gradient Descent moves in the opposite direction.

---

## Backward Pass

After the Forward Pass:

1. Calculate prediction error
2. Compute gradients
3. Update weights and biases

This process is called:

**Backward Pass**

or

**Backpropagation**

This is where actual learning happens.

---

## Complete Training Pipeline

```text
Input Image
      ↓
 Forward Pass
      ↓
 Prediction
      ↓
 Calculate Error
      ↓
 Backward Pass
      ↓
 Update Parameters
      ↓
 Better Prediction
```

This loop is repeated until the model becomes good at the task.

---

## Why This Matters

Whether it is:

* CNNs
* RNNs
* LSTMs
* GRUs
* Transformers
* GPT Models
* U-Net
* Diffusion Models
* GANs

Every modern AI system follows the same fundamental learning cycle:

```text
Forward Pass
      ↓
 Prediction
      ↓
 Error Calculation
      ↓
 Backward Pass
      ↓
 Gradient Descent
      ↓
 Better Parameters
```

The architecture changes.

The learning mechanism remains the same.

---

## Key Takeaways

* AI models learn a function $$f(X;\theta)$$
* Parameters $$\theta$$ represent weights and biases
* Forward Pass produces predictions
* Predictions are compared with ground truth labels
* Gradient Descent updates parameters to reduce error
* Backpropagation computes gradients
* Every modern Deep Learning model is built on this learning process

> Deep Learning is ultimately the search for the best set of parameters such that the model's prediction becomes as close as possible to reality.
