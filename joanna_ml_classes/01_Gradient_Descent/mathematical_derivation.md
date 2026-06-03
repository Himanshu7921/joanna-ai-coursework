# Gradient Descent: A Mathematical Derivation

## Step 1: Training Data

Assume we have a dataset:

$$
X = [X_1, X_2, X_3, \dots, X_N]
$$

with corresponding ground-truth labels:

$$
Y_t = [Y_{t_1}, Y_{t_2}, Y_{t_3}, \dots, Y_{t_N}]
$$

where:

- $X_i$ denotes the $i^{th}$ training sample.
- $Y_{t_i}$ denotes the true label of the $i^{th}$ sample.

---

## Step 2: Forward Pass

Given model parameters:

$$
\theta \in \Theta
$$

the model produces predictions:

$$
Y_p = f(X;\theta)
$$

or equivalently:

$$
Y_p =
[
Y_{p_1},
Y_{p_2},
\dots,
Y_{p_N}
]
$$

This process is known as the **Forward Pass**.

---

## Step 3: Measuring Prediction Error

A model is useful only if its predictions are close to the ground truth:

$$
Y_p \approx Y_t
$$

To quantify the discrepancy between prediction and reality, we define a loss function.

For regression problems, one of the most common choices is the Mean Squared Error (MSE):

$$
L(Y_t,Y_p)
=
\frac{1}{N}
\sum_{i=1}^{N}
(Y_{t_i}-Y_{p_i})^2
$$

---

### Example: Mini-Batch Loss

For a mini-batch of size:

$$
N = 8
$$

the loss becomes:

$$
L
=
\frac{1}{8}
\Big[
(Y_{t_1}-Y_{p_1})^2
+
(Y_{t_2}-Y_{p_2})^2
+
\dots
+
(Y_{t_8}-Y_{p_8})^2
\Big]
$$

Each sample contributes to the overall training signal.

---

## Step 4: Learning Objective

The lower the loss, the better the model.

Therefore, the entire training process can be formulated as:

$$
\theta^*
=
\arg\min_{\theta}
L(Y_t,Y_p)
$$

where:

$$
Y_p = f(X;\theta)
$$

and

$$
\theta^*
$$

represents the optimal set of model parameters.

---

## Step 5: Which Parameters Are Better?

Consider two possible parameter configurations:

### Model 1

$$
Y_{p_1}
=
f(X;\theta_1)
=
3X - 1
$$

### Model 2

$$
Y_{p_2}
=
f(X;\theta_2)
=
5X - 4
$$

Suppose:

$$
Y_t = 1
$$

and

$$
Y_{p_1}=0
$$

while

$$
Y_{p_2}=0.9
$$

Clearly:

$$
Y_{p_2}
\approx
Y_t
$$

Therefore:

$$
L(\theta_2)
<
L(\theta_1)
$$

which implies that $\theta_2$ is a better parameter configuration.

---

## Step 6: The Core Question

Finding a good parameter set is not enough.

We need to answer:

> How does a change in a parameter influence the loss?

For example:

$$
\theta_2
\rightarrow
\theta_2 + h
$$

where

$$
h \ll 1
$$

is a very small perturbation.

If increasing $\theta_2$ causes the loss to decrease, then we should continue moving in that direction.

If increasing $\theta_2$ causes the loss to increase, then we should move in the opposite direction.

---

## Step 7: Quantifying Influence

Define the loss as a function of a parameter:

$$
L(\theta)
$$

We want to measure:

> How rapidly does the loss change when $\theta$ changes?

Using finite differences:

$$
d_\theta
=
\frac
{
L(\theta+h)-L(\theta)
}
{h}
$$

where:

$$
h \rightarrow 0
$$

This quantity approximates the derivative:

$$
\frac{\partial L}{\partial \theta}
$$

---

## Interpretation

### Positive Gradient

If

$$
d_\theta > 0
$$

then increasing $\theta$ increases the loss.

Therefore we should move left.

---

### Negative Gradient

If

$$
d_\theta < 0
$$

then increasing $\theta$ decreases the loss.

Therefore we should move right.

---

### Zero Gradient

If

$$
d_\theta = 0
$$

then we may be near a minimum.

---

## Step 8: Gradient Descent

The derivative tells us **which direction increases the loss**.

To reduce the loss, we move in the opposite direction:

$$
\theta
=
\theta
-
\eta d_\theta
$$

where:

- $\eta$ = Learning Rate
- $d_\theta$ = Gradient

Substituting the finite-difference estimate:

$$
\theta
=
\theta
-
\eta
\Bigg(
\frac
{
L(\theta+h)-L(\theta)
}
{h}
\Bigg)
$$

This is the fundamental idea behind Gradient Descent.

---

## Step 9: Searching for the Best Parameters

Repeated application of

$$
\theta
=
\theta
-
\eta d_\theta
$$

gradually pushes the parameters toward:

$$
\theta^*
=
\arg\min_{\theta}
L(\theta)
$$

Ideally:

$$
L(\theta^*)
\approx 0
$$

and therefore:

$$
Y_p
\approx
Y_t
$$

---

## Local vs Global Minima

The loss landscape is rarely simple.

Multiple minima may exist:

- Local Minima
- Global Minimum

Mathematically:

$$
L(\theta_A)
<
L(\theta_{neighbors})
$$

does not necessarily imply

$$
L(\theta_A)
=
\min_\theta L(\theta)
$$

A local minimum is only optimal within a small neighborhood.

A global minimum is optimal over the entire parameter space.

---

## Before Training

Predictions are generated using random parameters:

$$
Y_p = f(X;\theta)
$$

which often leads to:

$$
Y_p \neq Y_t
$$

and therefore:

$$
L(Y_t,Y_p)
\gg 0
$$

---

## After Training

Gradient Descent discovers:

$$
\theta^*
$$

such that:

$$
Y_p^*
=
f(X;\theta^*)
$$

and

$$
Y_p^*
\approx
Y_t
$$

Consequently:

$$
L(Y_t,Y_p^*)
\rightarrow 0
$$

---

## Final Conclusion

Training a neural network is fundamentally an optimization problem:

$$
\theta^*
=
\arg\min_{\theta}
L(Y_t,f(X;\theta))
$$

Gradient Descent provides a systematic mechanism for achieving this objective by repeatedly:

1. Computing predictions.
2. Measuring error.
3. Estimating parameter influence through gradients.
4. Updating parameters in the direction of lower loss.

This principle forms the foundation of modern machine learning, deep learning, CNNs, Transformers, GPTs, U-Nets, Diffusion Models, and virtually every trainable AI system.