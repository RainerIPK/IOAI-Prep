# Statistics, Linear Algebra, Calculus, And Optimization Mastery

This is a dense foundations course for AI, machine learning, and IOAI-style problem solving.

It covers:

- Statistics: mean, variance, standard deviation, percentiles, random variables, distributions, sample variance, covariance, correlation, and causation.
- Linear algebra: vectors, distances, dot products, matrices, inverses, systems of equations, projections, embeddings, PCA intuition, and NumPy implementation.
- Calculus and optimization: derivatives, partial derivatives, gradients, critical points, minima/maxima, loss functions, gradient descent, learning rates, and least squares.
- Practice: hand calculations, coding drills, NumPy implementation, applied mini-projects, and final exams.

Verified resource map:

- OpenIntro Statistics: https://www.openintro.org/book/os/
- Think Stats: https://allendowney.github.io/ThinkStats2/
- Harvard CS109: https://harvard-iacs.github.io/CS109/
- Mathematics for Machine Learning: https://mml-book.github.io/
- Mathematics for Machine Learning PDF: https://mml-book.github.io/book/mml-book.pdf
- 3Blue1Brown Linear Algebra: https://www.3blue1brown.com/topics/linear-algebra
- 3Blue1Brown Calculus: https://www.3blue1brown.com/topics/calculus
- MIT OCW 18.06 Linear Algebra: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- MIT OCW 18.06 Assignments: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/assignments/
- MIT OCW 18.06 Exams: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/exams/
- Khan Academy Calculus: https://www.khanacademy.org/math/calculus-1
- Google ML Crash Course: https://developers.google.com/machine-learning/crash-course

## 0. How To Study This File

Do not try to memorize every formula first.

Use this cycle:

1. Read the concept.
2. Work one hand example.
3. Implement the same idea in Python or NumPy.
4. Explain it in plain words.
5. Solve problems without looking.
6. Keep a mistake log.

Mastery means you can move between four representations:

```text
plain language <-> formula <-> hand computation <-> code
```

Example:

```text
Plain language:
Variance measures average squared distance from the mean.

Formula:
var = (1/n) * sum((x_i - mean)^2)

Hand:
Data [2, 4, 6], mean 4, squared deviations 4, 0, 4, variance 8/3.

Code:
np.mean((x - x.mean()) ** 2)
```

If you can only recite formulas but cannot compute or code them, you do not own the topic yet.

## 1. Why These Topics Matter For AI

Statistics answers:

- What does the data look like?
- How spread out is it?
- How unusual is this value?
- How uncertain is this estimate?
- Are two variables related?
- Does this relationship imply causation?

Linear algebra answers:

- How do we represent data as vectors and matrices?
- How do we compare examples?
- How do embeddings work?
- How do linear models compute predictions?
- What is PCA doing geometrically?
- How do neural networks batch many computations efficiently?

Calculus and optimization answer:

- How does a model know which direction improves?
- How do loss functions get minimized?
- Why does gradient descent work?
- What can go wrong with learning rates?
- Why are derivatives everywhere in ML?

In machine learning, a dataset is usually a matrix, a model is often a function, training means minimizing a loss, and evaluation is statistical reasoning.

## 2. Notation Cheat Sheet

Scalars:

```text
x, y, a, b
```

Vectors:

```text
x = [x1, x2, ..., xd]
```

Matrices:

```text
X = n by d data matrix
n = number of examples
d = number of features
```

Targets:

```text
y = target vector
y_hat = predicted target vector
```

Mean:

```text
mean(x) = (1/n) * sum(x_i)
```

Dot product:

```text
a dot b = sum(a_i * b_i)
```

Matrix-vector prediction:

```text
y_hat = Xw + b
```

Loss:

```text
L(w) = measure of model error
```

Gradient:

```text
grad L = direction of steepest increase
```

Gradient descent:

```text
w_new = w_old - learning_rate * grad L
```

## Part I: Statistics

## 3. Data Types And First Questions

Before computing anything, ask:

1. What is one row?
2. What is one column?
3. Is the column numerical or categorical?
4. Is the data a sample or the whole population?
5. What question are we trying to answer?

### Numerical Data

Examples:

- Height.
- Weight.
- Age.
- Price.
- Temperature.
- Score.

Numerical data can be:

- Discrete: counts, like number of clicks.
- Continuous: measurements, like height or time.

### Categorical Data

Examples:

- Species.
- City.
- Blood type.
- Class label.
- Browser.

Categorical data can be:

- Nominal: no order, like red/blue/green.
- Ordinal: ordered categories, like low/medium/high.

### Population vs Sample

Population:

```text
Everyone or everything you care about.
```

Sample:

```text
The subset you actually observed.
```

Example:

- Population: all Indonesian students preparing for IOAI.
- Sample: 300 students who took one online selection.

Statistics often uses sample data to estimate population properties.

## 4. Mean, Median, Mode, And Robustness

### Mean

The mean is the arithmetic average:

```text
mean = (x1 + x2 + ... + xn) / n
```

Example:

```text
Data: 2, 4, 6, 8
Mean: (2 + 4 + 6 + 8) / 4 = 5
```

The mean is sensitive to outliers.

Example:

```text
Data: 2, 4, 6, 8, 100
Mean: 120 / 5 = 24
```

Most values are near 2-8, but the mean jumps to 24.

### Median

The median is the middle value after sorting.

Odd number of values:

```text
Data: 2, 4, 6, 8, 100
Median: 6
```

Even number of values:

```text
Data: 2, 4, 6, 8
Median: (4 + 6) / 2 = 5
```

The median is robust to outliers.

### Mode

The mode is the most common value.

```text
Data: 1, 2, 2, 3, 3, 3, 4
Mode: 3
```

Categorical example:

```text
Data: cat, dog, dog, fish
Mode: dog
```

### When To Use Which

| Situation | Use |
|---|---|
| Symmetric numerical data | Mean |
| Skewed data with outliers | Median |
| Categorical data | Mode |
| Need total-preserving average | Mean |
| Need typical person/item | Median |

## 5. Variance And Standard Deviation

Spread tells you how far values are from the center.

Two datasets can have the same mean but very different spread.

```text
A: 9, 10, 11
B: 0, 10, 20
```

Both have mean 10. Dataset B is much more spread out.

### Population Variance

For population data:

```text
variance = (1/n) * sum((x_i - mean)^2)
```

Example:

```text
Data: 2, 4, 6
Mean: 4
Deviations: -2, 0, 2
Squared deviations: 4, 0, 4
Population variance: (4 + 0 + 4) / 3 = 8/3
Population standard deviation: sqrt(8/3)
```

### Why Square Deviations?

If you average raw deviations, they cancel:

```text
Data: 2, 4, 6
Mean: 4
Deviations: -2, 0, 2
Average deviation: 0
```

Squaring:

- Makes all deviations positive.
- Penalizes large deviations more.
- Leads to useful mathematical properties.

### Standard Deviation

Standard deviation is the square root of variance:

```text
std = sqrt(variance)
```

Variance is in squared units. Standard deviation is in original units.

Example:

- If data is in kilograms, variance is in kilograms squared.
- Standard deviation is in kilograms.

### Sample Variance

For sample data:

```text
sample variance = (1/(n - 1)) * sum((x_i - sample_mean)^2)
```

Why divide by `n - 1`?

Because the sample mean is estimated from the same data. Once you know `n - 1` deviations, the last deviation is forced by the fact that deviations sum to zero. This is called degrees of freedom.

### Sample Variance Intuition

If you sample only a few points, the sample mean tends to be closer to those sampled points than the true population mean would be. Dividing by `n` would underestimate population variance. Dividing by `n - 1` corrects that bias on average.

Example:

```text
Data: 2, 4, 6
Mean: 4
Squared deviations: 4, 0, 4
Population variance: 8/3
Sample variance: 8/2 = 4
```

### Python

```python
import numpy as np

x = np.array([2, 4, 6])

population_var = np.var(x, ddof=0)
sample_var = np.var(x, ddof=1)

population_std = np.std(x, ddof=0)
sample_std = np.std(x, ddof=1)
```

`ddof` means delta degrees of freedom.

```text
ddof=0 -> divide by n
ddof=1 -> divide by n - 1
```

## 6. Percentiles, Quantiles, IQR, And Z-Scores

### Percentiles

The `p`th percentile is a value such that about `p` percent of the data is at or below it.

Examples:

- 50th percentile = median.
- 25th percentile = first quartile, Q1.
- 75th percentile = third quartile, Q3.

If your test score is at the 90th percentile, you scored at or above about 90 percent of people.

### Interquartile Range

```text
IQR = Q3 - Q1
```

IQR measures spread of the middle 50 percent of the data.

It is robust to outliers.

### Outlier Rule

A common rough rule:

```text
Lower fence = Q1 - 1.5 * IQR
Upper fence = Q3 + 1.5 * IQR
```

Values outside this range are potential outliers.

Potential does not mean wrong. It means investigate.

### Z-Score

A z-score says how many standard deviations a value is from the mean:

```text
z = (x - mean) / std
```

Example:

```text
mean = 70
std = 10
x = 90
z = (90 - 70) / 10 = 2
```

The value is 2 standard deviations above the mean.

### Python

```python
import numpy as np

x = np.array([10, 20, 30, 40, 100])

q1 = np.percentile(x, 25)
median = np.percentile(x, 50)
q3 = np.percentile(x, 75)
iqr = q3 - q1

z = (x - x.mean()) / x.std(ddof=0)
```

## 7. Random Variables

A random variable assigns a number to an uncertain outcome.

Example:

Flip a coin:

```text
X = 1 if heads
X = 0 if tails
```

Roll a die:

```text
X = result from 1 to 6
```

Random variables are not mysterious. They are variables whose value depends on chance.

### Discrete Random Variables

Discrete random variables take countable values.

Examples:

- Number of heads in 10 flips.
- Number of correct answers.
- Number of customers arriving in an hour.

### Continuous Random Variables

Continuous random variables take values on an interval.

Examples:

- Height.
- Weight.
- Time.
- Temperature.

### Probability Mass Function

For discrete variables:

```text
P(X = x)
```

Example fair die:

```text
P(X = 1) = 1/6
P(X = 2) = 1/6
...
P(X = 6) = 1/6
```

### Probability Density Function

For continuous variables, probability at exactly one point is usually zero.

Instead, we talk about area under a density curve:

```text
P(a <= X <= b) = area under density from a to b
```

## 8. Expected Value And Variance Of A Random Variable

### Expected Value

Expected value is the long-run average.

For discrete random variables:

```text
E[X] = sum(x * P(X = x))
```

Example fair die:

```text
E[X] = 1*(1/6) + 2*(1/6) + ... + 6*(1/6)
     = 21/6
     = 3.5
```

You cannot roll a 3.5. Expected value is an average over many trials.

### Variance Of A Random Variable

```text
Var(X) = E[(X - E[X])^2]
```

Equivalent formula:

```text
Var(X) = E[X^2] - (E[X])^2
```

This shortcut is often easier.

Example fair coin with `X=1` for heads and `X=0` for tails:

```text
E[X] = 0.5
E[X^2] = 1^2*0.5 + 0^2*0.5 = 0.5
Var(X) = 0.5 - 0.5^2 = 0.25
```

## 9. Important Distributions

Distributions describe how random values behave.

### Bernoulli Distribution

One yes/no trial.

```text
X = 1 with probability p
X = 0 with probability 1 - p
```

Examples:

- Coin flip.
- Pass/fail.
- Click/no click.

Mean:

```text
E[X] = p
```

Variance:

```text
Var(X) = p(1 - p)
```

### Binomial Distribution

Number of successes in `n` independent Bernoulli trials.

```text
X ~ Binomial(n, p)
```

Example:

```text
Number of heads in 10 coin flips.
```

Probability:

```text
P(X = k) = C(n, k) * p^k * (1 - p)^(n - k)
```

Mean:

```text
np
```

Variance:

```text
np(1 - p)
```

### Uniform Distribution

All values in a range are equally likely.

Discrete example:

```text
Fair die: 1, 2, 3, 4, 5, 6
```

Continuous example:

```text
Random number between 0 and 1
```

### Normal Distribution

The normal distribution is bell-shaped.

It is described by:

```text
mean mu
standard deviation sigma
```

Notation:

```text
X ~ Normal(mu, sigma^2)
```

Why it matters:

- Measurement errors are often roughly normal.
- Sample means often become approximately normal.
- Many statistical tools use normal approximations.

### Standard Normal

```text
Z ~ Normal(0, 1)
```

Convert normal value to standard normal:

```text
z = (x - mu) / sigma
```

### Distribution Cheat Sheet

| Distribution | Use when |
|---|---|
| Bernoulli | One yes/no trial |
| Binomial | Count successes in fixed trials |
| Uniform | Equal likelihood across values |
| Normal | Bell-shaped measurement/noise |

## 10. Duplicating Data And Combining Groups

### What Happens When You Duplicate A Dataset?

Suppose:

```text
x = [1, 2, 3]
duplicated = [1, 2, 3, 1, 2, 3]
```

Mean stays the same.

Population variance stays the same, because the distribution is identical.

But sample variance can change slightly because the denominator changes from `n - 1` to `2n - 1`.

Example:

```text
x = [1, 2, 3]
mean = 2
sum squared deviations = 1 + 0 + 1 = 2
sample variance = 2 / (3 - 1) = 1
```

Duplicated:

```text
sum squared deviations = 4
sample variance = 4 / (6 - 1) = 0.8
```

Important:

- The population-style spread did not change.
- The sample variance estimate changed because of the `n - 1` correction.

This distinction matters in contest questions.

### Combined Mean

If group A has:

```text
n1 values, mean m1
```

and group B has:

```text
n2 values, mean m2
```

combined mean:

```text
(n1*m1 + n2*m2) / (n1 + n2)
```

Example:

```text
Class A: 10 students, mean 80
Class B: 20 students, mean 70
Combined mean = (10*80 + 20*70) / 30
              = 2200 / 30
              = 73.333...
```

Do not average means equally unless group sizes are equal.

Wrong:

```text
(80 + 70) / 2 = 75
```

### Combining Variance

For population variance, if groups have means `m1`, `m2`, variances `v1`, `v2`, sizes `n1`, `n2`:

```text
combined_mean = (n1*m1 + n2*m2) / (n1 + n2)

combined_variance =
  [n1*(v1 + (m1 - combined_mean)^2)
   + n2*(v2 + (m2 - combined_mean)^2)]
  / (n1 + n2)
```

Intuition:

Total spread = spread within groups + spread between group means.

## 11. Covariance And Correlation

### Covariance

Covariance measures whether two variables move together.

Population covariance:

```text
cov(X, Y) = (1/n) * sum((x_i - mean_x)(y_i - mean_y))
```

Positive covariance:

```text
When x is above its mean, y tends to be above its mean.
```

Negative covariance:

```text
When x is above its mean, y tends to be below its mean.
```

Near-zero covariance:

```text
No strong linear movement together.
```

Covariance depends on units, so it can be hard to interpret.

### Correlation

Correlation standardizes covariance:

```text
corr(X, Y) = cov(X, Y) / (std_X * std_Y)
```

It ranges from `-1` to `1`.

```text
1: perfect positive linear relationship
0: no linear relationship
-1: perfect negative linear relationship
```

### Correlation Is Linear

Correlation can be zero even when variables have a strong nonlinear relationship.

Example:

```text
y = x^2
```

If `x` is symmetric around 0, correlation between `x` and `x^2` may be near zero.

### Correlation vs Causation

Correlation does not prove causation.

Possible explanations when A and B are correlated:

1. A causes B.
2. B causes A.
3. A and B share a hidden cause C.
4. The relationship is coincidence.
5. The data collection process creates the pattern.

Example:

```text
Ice cream sales and drowning incidents rise together.
```

Ice cream does not cause drowning. Hot weather increases both swimming and ice cream purchases.

### To Argue Causation

You need more than correlation:

- Randomized experiment.
- Natural experiment.
- Causal design.
- Strong domain reasoning.
- Temporal order.
- Control for confounders.

In AI competitions, you usually need prediction, not causation. But if you interpret model features as causes, be careful.

## Part II: Linear Algebra And Vectors

## 12. Scalars, Vectors, And Matrices

### Scalar

A scalar is a single number:

```text
3
-1.5
0.02
```

### Vector

A vector is an ordered list of numbers:

```text
v = [2, -1, 5]
```

In ML, one row of features is often a vector.

Example student vector:

```text
[hours_studied, sleep_hours, prior_score]
```

### Matrix

A matrix is a rectangular table of numbers:

```text
X = [
  [1, 2, 3],
  [4, 5, 6]
]
```

Shape:

```text
2 rows, 3 columns
```

In ML:

```text
X = data matrix
rows = examples
columns = features
```

## 13. Vector Arithmetic

Given:

```text
a = [1, 2]
b = [3, 5]
```

Addition:

```text
a + b = [4, 7]
```

Subtraction:

```text
b - a = [2, 3]
```

Scalar multiplication:

```text
3a = [3, 6]
```

Vector arithmetic is componentwise.

### Geometric Meaning

Vectors can represent:

- Points.
- Directions.
- Feature sets.
- Embeddings.

Adding a vector means moving by that direction.

Example:

```text
position + velocity = next_position
```

## 14. Norms And Euclidean Distance

### Vector Norm

A norm measures vector length.

Euclidean norm:

```text
||v|| = sqrt(v1^2 + v2^2 + ... + vd^2)
```

Example:

```text
v = [3, 4]
||v|| = sqrt(3^2 + 4^2) = 5
```

### Euclidean Distance

Distance between `a` and `b`:

```text
distance(a, b) = ||a - b||
```

Example:

```text
a = [1, 2]
b = [4, 6]
b - a = [3, 4]
distance = 5
```

### Manhattan Distance

```text
sum(abs(a_i - b_i))
```

Example:

```text
a = [1, 2]
b = [4, 6]
Manhattan distance = |3| + |4| = 7
```

### Why Distance Matters In AI

Distance is used in:

- K-nearest neighbors.
- Clustering.
- Similarity search.
- Embedding retrieval.
- Anomaly detection.

But distance depends on scale.

Example:

```text
height in meters: 1.4 to 2.0
income in dollars: 0 to 100000
```

Income dominates Euclidean distance unless features are scaled.

## 15. Dot Product

The dot product multiplies corresponding entries and adds:

```text
a dot b = a1*b1 + a2*b2 + ... + ad*bd
```

Example:

```text
a = [1, 2, 3]
b = [4, 5, 6]
a dot b = 1*4 + 2*5 + 3*6 = 32
```

### Dot Product As Weighted Sum

Linear model:

```text
y_hat = w dot x + b
```

Example:

```text
x = [hours, sleep]
w = [5, 2]
b = 10

y_hat = 5*hours + 2*sleep + 10
```

Dot product is how linear models turn features into predictions.

### Dot Product And Angle

```text
a dot b = ||a|| ||b|| cos(theta)
```

So:

- Positive dot product: angle less than 90 degrees.
- Zero dot product: perpendicular.
- Negative dot product: angle greater than 90 degrees.

### Cosine Similarity

```text
cosine_similarity(a, b) = (a dot b) / (||a|| ||b||)
```

Used for:

- Text embeddings.
- Search.
- Recommendation.
- Semantic similarity.

Cosine similarity cares about direction, not magnitude.

## 16. Matrix Operations

### Matrix Addition

Matrices must have same shape.

```text
[[1, 2],      [[5, 6],      [[6, 8],
 [3, 4]]  +    [7, 8]]  =    [10, 12]]
```

### Matrix-Vector Multiplication

If `A` is `m by n` and `x` is length `n`, then `Ax` is length `m`.

Example:

```text
A = [[1, 2],
     [3, 4]]
x = [10, 20]

Ax = [
  1*10 + 2*20,
  3*10 + 4*20
] = [50, 110]
```

### Matrix-Matrix Multiplication

If:

```text
A is m by n
B is n by p
```

then:

```text
AB is m by p
```

Entry `(i, j)` of `AB` is:

```text
row i of A dot column j of B
```

### Shape Discipline

Always check shapes.

```text
(m by n) @ (n by p) = (m by p)
```

If inner dimensions do not match, multiplication is invalid.

### NumPy

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
x = np.array([10, 20])

print(A @ x)
```

Use `@` for matrix multiplication.

Use `*` for elementwise multiplication.

```python
A * A   # elementwise
A @ A   # matrix multiplication
```

## 17. Identity, Inverse, And Systems Of Equations

### Identity Matrix

The identity matrix acts like 1:

```text
AI = A
IA = A
```

Example:

```text
I = [[1, 0],
     [0, 1]]
```

### Inverse Matrix

The inverse of `A` is `A^-1` such that:

```text
A A^-1 = I
```

If:

```text
Ax = b
```

and `A` has an inverse:

```text
x = A^-1 b
```

### Important Warning

In numerical computing, do not usually compute inverse directly to solve systems.

Prefer:

```python
np.linalg.solve(A, b)
```

over:

```python
np.linalg.inv(A) @ b
```

`solve` is more stable and efficient.

### Systems Of Linear Equations

Example:

```text
2x + y = 5
x - y = 1
```

Matrix form:

```text
[[2, 1],
 [1, -1]] [x, y] = [5, 1]
```

Solve:

```python
A = np.array([[2, 1], [1, -1]])
b = np.array([5, 1])
x = np.linalg.solve(A, b)
```

## 18. Linear Independence, Span, Basis, Rank

### Linear Combination

A linear combination of vectors is:

```text
c1*v1 + c2*v2 + ... + ck*vk
```

### Span

The span of vectors is all possible linear combinations of them.

Example:

```text
[1, 0] and [0, 1] span all of 2D space.
```

### Linear Independence

Vectors are linearly independent if no vector can be made from the others.

Example independent:

```text
[1, 0], [0, 1]
```

Example dependent:

```text
[1, 2], [2, 4]
```

The second is just `2 * first`.

### Basis

A basis is a set of independent vectors that span a space.

Standard 2D basis:

```text
e1 = [1, 0]
e2 = [0, 1]
```

### Rank

Rank is the number of independent directions in a matrix.

In data:

- Full rank means features contain independent information.
- Low rank means redundancy.

Example:

```text
feature_2 = 2 * feature_1
```

These two columns add only one independent direction.

## 19. Projections And Least Squares Geometry

Projection means dropping a vector onto a line or subspace.

Projection of `b` onto vector `a`:

```text
proj_a(b) = ((a dot b) / (a dot a)) * a
```

### Why Projection Matters

Least squares regression finds the prediction vector in the column space of `X` closest to `y`.

In words:

```text
Linear regression projects y onto the space of possible predictions Xw.
```

This is why linear algebra and regression are deeply connected.

### Residual

```text
residual = y - y_hat
```

At the least-squares solution, residual is perpendicular to the column space of `X`.

This gives the normal equation:

```text
X^T X w = X^T y
```

If `X^T X` is invertible:

```text
w = (X^T X)^-1 X^T y
```

Again, in code prefer stable solvers:

```python
w, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
```

## 20. Embeddings As Vectors

An embedding represents an object as a vector.

Objects can be:

- Words.
- Images.
- Users.
- Products.
- DNA sequences.
- Audio clips.

The goal is that meaningful relationships become geometric relationships.

Example:

```text
similar words -> nearby vectors
opposite sentiment -> different directions
same category -> clusters
```

Operations:

```text
distance(a, b)
cosine_similarity(a, b)
a + b - c
nearest neighbors
```

### Important Warning

Embedding dimensions are not usually individually interpretable.

You usually interpret:

- Distances.
- Directions.
- Neighborhoods.
- Clusters.

not dimension 17 by itself.

## 21. PCA Intuition

PCA stands for Principal Component Analysis.

It finds directions of maximum variance.

Imagine points shaped like a long tilted ellipse.

PCA finds:

- PC1: direction along the longest spread.
- PC2: next biggest spread, perpendicular to PC1.

### Why PCA Works

Data may have many columns but fewer meaningful directions.

Example:

```text
height in cm
height in inches
```

These are redundant. PCA can compress them into one main direction.

### PCA Workflow

1. Center data by subtracting feature means.
2. Often scale features.
3. Find directions of high variance.
4. Project data onto top directions.

### PCA In ML

Use PCA for:

- Visualization.
- Compression.
- Noise reduction.
- Understanding structure.

Do not assume PCA improves prediction. It preserves variance, not target relevance.

## Part III: Calculus And Optimization

## 22. Functions And Rates Of Change

A function maps input to output.

```text
f(x) = x^2
```

Derivative means instantaneous rate of change.

```text
f'(x) = derivative of f at x
```

For:

```text
f(x) = x^2
```

derivative:

```text
f'(x) = 2x
```

Interpretation:

- At `x = 3`, slope is `6`.
- Around `x = 3`, increasing x slightly increases f by about `6 * change_in_x`.

## 23. Derivative Rules

### Power Rule

```text
d/dx x^n = n*x^(n-1)
```

Examples:

```text
d/dx x^2 = 2x
d/dx x^3 = 3x^2
d/dx sqrt(x) = d/dx x^(1/2) = (1/2)x^(-1/2)
```

### Constant Rule

```text
d/dx c = 0
```

### Constant Multiple Rule

```text
d/dx [c f(x)] = c f'(x)
```

Example:

```text
d/dx 7x^3 = 21x^2
```

### Sum Rule

```text
d/dx [f(x) + g(x)] = f'(x) + g'(x)
```

Example:

```text
d/dx (x^2 + 3x + 1) = 2x + 3
```

### Product Rule

```text
d/dx [f(x)g(x)] = f'(x)g(x) + f(x)g'(x)
```

### Chain Rule

If:

```text
h(x) = f(g(x))
```

then:

```text
h'(x) = f'(g(x)) * g'(x)
```

Example:

```text
h(x) = (3x + 1)^2
h'(x) = 2(3x + 1) * 3 = 6(3x + 1)
```

The chain rule is the heart of backpropagation.

## 24. Critical Points, Minima, And Maxima

Critical points occur when:

```text
f'(x) = 0
```

or derivative is undefined.

For smooth functions, minima and maxima often occur at critical points.

Example:

```text
f(x) = (x - 3)^2
f'(x) = 2(x - 3)
Set derivative to 0:
2(x - 3) = 0
x = 3
```

At `x=3`, the function has minimum value 0.

### Second Derivative

Second derivative tells curvature.

```text
f''(x) > 0 -> curves upward -> local minimum
f''(x) < 0 -> curves downward -> local maximum
```

Example:

```text
f(x) = x^2
f''(x) = 2 > 0
minimum at x = 0
```

```text
f(x) = -x^2
f''(x) = -2 < 0
maximum at x = 0
```

## 25. Partial Derivatives And Gradients

ML losses usually depend on many parameters.

Example:

```text
L(w1, w2) = (w1 - 2)^2 + (w2 + 3)^2
```

Partial derivative with respect to `w1`:

```text
dL/dw1 = 2(w1 - 2)
```

Partial derivative with respect to `w2`:

```text
dL/dw2 = 2(w2 + 3)
```

Gradient:

```text
grad L = [dL/dw1, dL/dw2]
```

The gradient points in the direction of steepest increase.

So:

```text
-grad L
```

points in the direction of steepest decrease.

This is why gradient descent subtracts the gradient.

## 26. Loss Functions

A loss function measures how wrong a model is.

### Mean Squared Error

For regression:

```text
MSE = (1/n) * sum((y_i - y_hat_i)^2)
```

Why useful:

- Smooth.
- Penalizes large errors.
- Works well with linear regression.

### Mean Absolute Error

```text
MAE = (1/n) * sum(abs(y_i - y_hat_i))
```

Why useful:

- More robust to outliers.
- Easier to interpret.

But absolute value is not differentiable at 0, though optimization tools can still handle it in many ways.

### Cross-Entropy

For classification:

```text
loss = -[y log(p) + (1-y) log(1-p)]
```

Cross-entropy punishes confident wrong predictions heavily.

Example:

If true label is 1:

```text
p = 0.9 -> small loss
p = 0.1 -> large loss
```

## 27. Gradient Descent

Gradient descent is an iterative optimization method.

Update rule:

```text
parameter_new = parameter_old - learning_rate * gradient
```

### Example

Minimize:

```text
f(w) = (w - 5)^2
```

Derivative:

```text
f'(w) = 2(w - 5)
```

Start:

```text
w = 0
learning_rate = 0.1
```

Step 1:

```text
gradient = 2(0 - 5) = -10
w_new = 0 - 0.1*(-10) = 1
```

Step 2:

```text
gradient = 2(1 - 5) = -8
w_new = 1 - 0.1*(-8) = 1.8
```

The parameter moves toward 5.

### Learning Rate

Learning rate controls step size.

Too small:

- Slow.
- Many iterations.

Too large:

- Overshoots.
- May diverge.

Good:

- Decreases loss steadily.

### Batch, Stochastic, Mini-Batch

Batch gradient descent:

```text
Use all data for each update.
```

Stochastic gradient descent:

```text
Use one example per update.
```

Mini-batch gradient descent:

```text
Use a small batch, like 32 or 128 examples.
```

Most deep learning uses mini-batches.

## 28. Least Squares Minimization

Linear model:

```text
y_hat = Xw
```

MSE loss:

```text
L(w) = (1/n) ||Xw - y||^2
```

Gradient:

```text
grad L = (2/n) X^T (Xw - y)
```

Gradient descent update:

```text
w = w - alpha * (2/n) X^T (Xw - y)
```

Closed-form least squares solution:

```text
X^T X w = X^T y
```

If invertible:

```text
w = (X^T X)^-1 X^T y
```

Use gradient descent when:

- Dataset is huge.
- Closed-form inverse is expensive.
- Model is nonlinear.
- You are training neural networks.

Use closed-form or `np.linalg.lstsq` when:

- Dataset is small/medium.
- You want exact least-squares solution.

## 29. Convexity Intuition

A convex function is bowl-shaped.

For convex minimization:

- Any local minimum is a global minimum.
- Gradient descent is easier to reason about.

Example convex:

```text
f(x) = x^2
```

Example nonconvex:

```text
f(x) = x^4 - 3x^2 + x
```

Neural networks are usually nonconvex. Linear regression with MSE is convex.

## 30. NumPy Toolkit

```python
import numpy as np
```

Arrays:

```python
x = np.array([1, 2, 3])
A = np.array([[1, 2], [3, 4]])
```

Mean/variance:

```python
x.mean()
x.var(ddof=0)
x.var(ddof=1)
x.std(ddof=1)
```

Percentiles:

```python
np.percentile(x, [25, 50, 75])
```

Vector operations:

```python
a + b
a - b
3 * a
np.dot(a, b)
np.linalg.norm(a)
```

Distance:

```python
np.linalg.norm(a - b)
```

Matrix multiplication:

```python
A @ x
A @ B
```

Solve systems:

```python
np.linalg.solve(A, b)
```

Least squares:

```python
np.linalg.lstsq(X, y, rcond=None)
```

Random sampling:

```python
rng = np.random.default_rng(42)
sample = rng.normal(loc=0, scale=1, size=1000)
```

## 31. Problemsets Overview

The problemsets are staged.

Level A:

```text
Hand computation.
```

Level B:

```text
Python/NumPy implementation.
```

Level C:

```text
AI/ML application.
```

Level D:

```text
Explanation, diagnosis, and proof-style reasoning.
```

Do not skip Level A. If you cannot compute a tiny example by hand, code will feel like magic.

## Problemset 1: Mean, Median, Variance, Standard Deviation

### A. Hand Computation

1. For data `[2, 4, 6, 8]`, compute mean, median, population variance, population standard deviation.
2. For data `[2, 4, 6, 8, 100]`, compute mean and median. Which is more representative?
3. For data `[1, 1, 2, 3, 8]`, compute sample variance and population variance.
4. Create two datasets with the same mean but different variance.
5. Explain why deviations from the mean sum to zero.

### B. Python

Write functions without NumPy:

```python
mean(xs)
median(xs)
population_variance(xs)
sample_variance(xs)
standard_deviation(xs, sample=True)
```

Then compare with NumPy.

### C. AI Application

You have model errors:

```text
Model A errors: [1, 1, 1, 1, 10]
Model B errors: [3, 3, 3, 3, 3]
```

Compute MAE and RMSE for both.

Explain which model you prefer if large errors are dangerous.

### D. Mastery Explanation

Explain the difference between variance and standard deviation to a 12-year-old.

## Problemset 2: Percentiles, IQR, Z-Scores

### A. Hand Computation

Given sorted data:

```text
[10, 12, 14, 15, 18, 21, 22, 25, 30]
```

1. Find median.
2. Find Q1 and Q3 using the median-of-halves method.
3. Compute IQR.
4. Compute outlier fences.
5. Is 45 an outlier by the 1.5 IQR rule?

### B. Python

Generate 1000 values from a normal distribution.

Compute:

- Mean.
- Standard deviation.
- 5th, 25th, 50th, 75th, 95th percentiles.
- Number of values with absolute z-score above 2.

### C. AI Application

Suppose a feature has extreme outliers.

Compare:

1. Standard scaling using mean/std.
2. Robust scaling using median/IQR.

Explain which is safer for outlier-heavy data.

## Problemset 3: Random Variables And Distributions

### A. Hand Computation

Let `X` be a die roll.

1. Write the probability mass function.
2. Compute `E[X]`.
3. Compute `E[X^2]`.
4. Compute `Var(X)`.

Let `Y ~ Bernoulli(0.3)`.

5. Compute `E[Y]`.
6. Compute `Var(Y)`.

### B. Binomial

Let `X ~ Binomial(5, 0.2)`.

1. Compute `P(X = 0)`.
2. Compute `P(X = 1)`.
3. Compute `P(X >= 1)`.
4. Compute mean and variance.

### C. Python Simulation

Simulate 10000 coin flips with `p=0.3`.

Compare:

- Simulated mean.
- Theoretical mean.
- Simulated variance.
- Theoretical variance.

### D. AI Application

Suppose a classifier has 90 percent accuracy on independent samples.

If it classifies 10 samples, model the number correct as binomial.

Compute:

- Expected number correct.
- Probability all 10 are correct.
- Probability at least 8 are correct.

## Problemset 4: Combining Groups And Duplicating Data

### A. Combined Mean

Group A:

```text
n = 12, mean = 80
```

Group B:

```text
n = 18, mean = 70
```

Compute combined mean.

### B. Reverse Engineering

Two groups combine to mean 75.

Group A:

```text
n = 10, mean = 90
```

Group B:

```text
n = 30, mean = ?
```

Find Group B mean.

### C. Duplicating Dataset

For data:

```text
[1, 2, 3, 4]
```

1. Compute population variance.
2. Compute sample variance.
3. Duplicate the dataset.
4. Recompute both.
5. Explain what changed and why.

### D. Contest-Style Problem

A batch of `N` samples has mean `m`. A second identical batch of `N` samples is added.

1. What is the new mean?
2. What is the population variance compared with before?
3. What happens to sample variance?

## Problemset 5: Covariance, Correlation, Causation

### A. Hand Computation

Given:

```text
x = [1, 2, 3]
y = [2, 4, 6]
```

1. Compute mean of x and y.
2. Compute covariance.
3. Compute correlation.
4. Interpret.

Given:

```text
x = [1, 2, 3]
y = [3, 2, 1]
```

5. Compute correlation.

### B. Python

Create:

```python
x = np.linspace(-5, 5, 200)
y = x ** 2
```

Compute correlation between `x` and `y`.

Plot `x` vs `y`.

Explain why correlation can miss nonlinear relationships.

### C. Causation Reasoning

For each pair, list at least three causal explanations:

1. Hours studied and test score.
2. Ice cream sales and drowning.
3. Number of firefighters and fire damage.
4. Shoe size and reading ability in children.
5. Social media usage and anxiety.

## Problemset 6: Vector Arithmetic And Distance

### A. Hand Computation

Given:

```text
a = [1, 2, 3]
b = [4, 0, -1]
```

Compute:

1. `a + b`
2. `a - b`
3. `2a`
4. `||a||`
5. Euclidean distance between `a` and `b`
6. Manhattan distance between `a` and `b`

### B. Nearest Neighbor

Points:

```text
A = [0, 0]
B = [2, 0]
C = [0, 3]
D = [10, 10]
```

Query:

```text
q = [1, 1]
```

Compute distances and find nearest neighbor.

### C. Python

Write:

```python
euclidean(a, b)
manhattan(a, b)
nearest_neighbor(points, q)
```

Then implement the same using NumPy.

### D. AI Application

Explain why feature scaling changes nearest-neighbor predictions.

Create a small example where one large-scale feature dominates the distance.

## Problemset 7: Dot Product, Cosine Similarity, Embeddings

### A. Hand Computation

Given:

```text
a = [1, 2]
b = [3, 4]
c = [-2, 1]
```

Compute:

1. `a dot b`
2. `a dot c`
3. `||a||`
4. cosine similarity of `a` and `b`
5. cosine similarity of `a` and `c`

### B. Weighted Sum

A model uses:

```text
w = [0.5, -2, 3]
b = 4
x = [10, 1, 2]
```

Compute:

```text
y_hat = w dot x + b
```

Interpret each weight.

### C. Embedding Matching

Known word vectors:

```text
cat = [1, 2]
dog = [1, 3]
car = [10, 0]
bus = [11, 1]
```

1. Which word is closest to cat by Euclidean distance?
2. Which word is closest to car?
3. Compute cosine similarity between cat/dog and cat/car.

### D. Python

Write a function:

```python
top_k_similar(embeddings, query, k)
```

Use cosine similarity.

## Problemset 8: Matrix Operations

### A. Hand Computation

Given:

```text
A = [[1, 2],
     [3, 4]]

B = [[0, 1],
     [2, 3]]

x = [5, 6]
```

Compute:

1. `A + B`
2. `A - B`
3. `A @ x`
4. `A @ B`
5. `B @ A`

Is `A @ B = B @ A`?

### B. Shapes

For each multiplication, say valid or invalid and give output shape:

1. `(10 by 3) @ (3 by 1)`
2. `(10 by 3) @ (10 by 3)`
3. `(5 by 7) @ (7 by 2)`
4. `(7 by 2) @ (2 by 7)`
5. `(100 by 20) @ (20 by 4)`

### C. NumPy

Create random:

```text
X shape: 100 by 3
w shape: 3
```

Compute:

```text
y_hat = X @ w
```

Then add scalar bias.

## Problemset 9: Systems, Inverses, Rank

### A. Solve By Hand

Solve:

```text
2x + y = 5
x - y = 1
```

### B. Matrix Form

Write the system as:

```text
Ax = b
```

Then solve with NumPy.

### C. Rank

For:

```text
A = [[1, 2],
     [2, 4]]
```

1. Are columns independent?
2. What is rank?
3. Does inverse exist?
4. Explain geometrically.

### D. AI Application

Create a dataset where one feature is exactly twice another.

Fit linear regression.

Explain why redundant features can make coefficients unstable.

## Problemset 10: Projections And Least Squares

### A. Projection

Let:

```text
a = [1, 0]
b = [3, 4]
```

Compute projection of `b` onto `a`.

Now:

```text
a = [1, 1]
b = [2, 0]
```

Compute projection of `b` onto `a`.

### B. Least Squares By Matrix

Fit a line:

```text
y = w0 + w1*x
```

Data:

```text
x = [0, 1, 2]
y = [1, 3, 5]
```

Build:

```text
X = [[1, 0],
     [1, 1],
     [1, 2]]
```

Solve with:

```text
w = (X^T X)^-1 X^T y
```

### C. NumPy

Use `np.linalg.lstsq` on the same data.

Confirm the line.

### D. Explanation

Explain linear regression as projecting `y` onto the column space of `X`.

## Problemset 11: PCA

### A. Concept

Answer:

1. What does PC1 mean?
2. Why do we center data before PCA?
3. Why might we scale data before PCA?
4. Does PCA use labels?
5. Why can PCA preserve variance but lose predictive information?

### B. Hand Intuition

Points:

```text
(1, 1), (2, 2), (3, 3), (4, 4)
```

1. What is the main direction of variance?
2. What is the second direction?
3. Can these points be mostly represented in 1D?

### C. Python

Use `sklearn.datasets.load_iris`.

1. Standardize features.
2. Apply PCA to 2 components.
3. Plot points colored by species.
4. Print explained variance ratio.

### D. AI Application

Use PCA before KNN on a dataset.

Compare performance with:

1. No PCA.
2. PCA to 2 components.
3. PCA preserving 95 percent variance.

## Problemset 12: Derivatives

### A. Compute Derivatives

Find derivatives:

1. `f(x) = x^2`
2. `f(x) = 3x^4 - 2x + 7`
3. `f(x) = (2x + 1)^3`
4. `f(x) = x^2 + 1/x`
5. `f(x) = (x^2 + 3x)(x - 1)`

### B. Evaluate Slopes

For:

```text
f(x) = x^2 - 4x + 1
```

1. Find `f'(x)`.
2. Evaluate slope at `x=0`.
3. Evaluate slope at `x=2`.
4. Evaluate slope at `x=5`.
5. Interpret signs.

### C. Numerical Derivative

Implement:

```python
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2*h)
```

Compare numerical and symbolic derivatives for `x^2`.

## Problemset 13: Critical Points And Optimization

### A. Hand Computation

For each function:

1. Find derivative.
2. Set derivative to zero.
3. Classify min/max using second derivative.

Functions:

```text
f(x) = (x - 3)^2 + 2
g(x) = -x^2 + 4x
h(x) = x^3 - 3x
q(x) = x^4 - 4x^2
```

### B. Applied

Profit:

```text
P(x) = -2x^2 + 40x - 100
```

Find production amount `x` that maximizes profit.

### C. Python

Plot each function and mark critical points.

## Problemset 14: Gradients

### A. Hand Computation

Given:

```text
L(w1, w2) = (w1 - 2)^2 + (w2 + 3)^2
```

1. Compute gradient.
2. Evaluate gradient at `(0, 0)`.
3. What direction should gradient descent move?
4. What is the minimum?

### B. More Gradients

Compute gradients:

```text
L(x, y) = x^2 + y^2
L(x, y) = 3x^2 + 2xy + y^2
L(x, y) = (x + y - 1)^2
```

### C. Numerical Gradient

Implement:

```python
def numerical_gradient(f, w, h=1e-5):
    ...
```

Use finite differences.

Compare to analytic gradient.

## Problemset 15: Loss Functions

### A. MSE And MAE

True:

```text
y = [2, 4, 6]
```

Predictions:

```text
y_hat = [1, 5, 10]
```

Compute:

1. Errors.
2. MAE.
3. MSE.
4. RMSE.

### B. Cross-Entropy

For binary classification, true label `y=1`.

Compute loss:

```text
p = 0.9
p = 0.5
p = 0.1
```

Use:

```text
-log(p)
```

Explain why confident wrong predictions are punished.

### C. AI Application

Create two regression models:

```text
Model A: small errors most of the time, one huge error
Model B: moderate errors always
```

Compare MAE and RMSE.

## Problemset 16: Gradient Descent

### A. Hand Iterations

Minimize:

```text
f(w) = (w - 4)^2
```

Start:

```text
w = 0
learning_rate = 0.25
```

Do 5 gradient descent steps.

### B. Learning Rate

Repeat with:

```text
learning_rate = 0.01
learning_rate = 0.5
learning_rate = 1.1
```

Describe behavior.

### C. Python

Implement gradient descent for:

```text
f(w) = (w - 4)^2
```

Plot `w` over iterations and loss over iterations.

### D. Explanation

Explain why the gradient points uphill and why subtracting it goes downhill.

## Problemset 17: Least Squares And Gradient Descent

### A. Matrix Gradient

For:

```text
L(w) = (1/n) ||Xw - y||^2
```

Show:

```text
grad L = (2/n) X^T (Xw - y)
```

You do not need a rigorous proof, but explain shapes.

### B. Coding

Generate synthetic data:

```python
y = 3*x + 5 + noise
```

Fit line using:

1. Closed-form `np.linalg.lstsq`.
2. Gradient descent.
3. scikit-learn `LinearRegression`.

Compare results.

### C. Learning Rate Experiment

Try learning rates:

```text
0.0001, 0.001, 0.01, 0.1, 1.0
```

Plot loss curves.

Explain which diverge, which are slow, and which work.

## Problemset 18: Integrated Statistics And Linear Algebra

You are given student data:

```text
hours_studied, sleep_hours, practice_tests, score
```

Create or simulate 200 rows.

Tasks:

1. Compute summary statistics for each feature.
2. Compute percentiles for scores.
3. Compute correlation matrix.
4. Explain which variables correlate with score.
5. Standardize features.
6. Fit linear regression using NumPy least squares.
7. Interpret coefficients.
8. Compute MAE and RMSE.
9. Identify outliers by z-score.
10. Explain correlation vs causation for hours studied and score.

## Problemset 19: Integrated Embeddings And PCA

Create fake 2D or 5D embeddings for 20 words or objects.

Tasks:

1. Store embeddings in a matrix.
2. Compute nearest neighbors using Euclidean distance.
3. Compute nearest neighbors using cosine similarity.
4. Compare differences.
5. Center the embedding matrix.
6. Apply PCA to 2D if original dimension is greater than 2.
7. Plot embeddings.
8. Explain clusters.
9. Try vector arithmetic: `king - man + woman` style with your own toy vectors.
10. Write limitations of interpreting toy embeddings.

## Problemset 20: Final Exam

### Part A: Statistics

1. Compute mean, median, population variance, sample variance, and standard deviation for `[3, 5, 5, 7, 20]`.
2. Explain why sample variance divides by `n - 1`.
3. Compute Q1, Q3, IQR, and outlier fences for a small dataset.
4. Compute expectation and variance of a Bernoulli random variable.
5. Compute a binomial probability.
6. Explain normal distribution and z-score.
7. Compute covariance and correlation for a small pair of vectors.
8. Give three reasons correlation may not imply causation.

### Part B: Linear Algebra

1. Compute vector norm and distance.
2. Compute dot product and cosine similarity.
3. Multiply a matrix by a vector.
4. Check matrix multiplication shapes.
5. Solve a 2 by 2 linear system.
6. Determine whether two vectors are independent.
7. Explain rank.
8. Explain projection.
9. Solve a least-squares problem.
10. Explain PCA in plain language.

### Part C: Calculus And Optimization

1. Differentiate five functions.
2. Find minima/maxima using derivatives.
3. Compute a gradient.
4. Run gradient descent by hand for three steps.
5. Explain learning rate failure modes.
6. Derive or explain MSE gradient.
7. Fit a line using gradient descent.
8. Compare MAE and MSE.
9. Explain why `-gradient` lowers loss.
10. Explain the connection between optimization and model training.

### Part D: Coding Project

Build one clean notebook or script:

1. Simulate a dataset with 3 features and 1 numeric target.
2. Summarize statistics.
3. Plot histograms and scatter plots.
4. Compute correlations.
5. Standardize the feature matrix.
6. Fit linear regression using `np.linalg.lstsq`.
7. Fit the same model using gradient descent.
8. Compare parameters.
9. Compute MAE, MSE, RMSE.
10. Write a report explaining every formula you used.

Passing standard:

- You can do small examples by hand.
- You can implement core operations in NumPy.
- You can explain every formula in plain language.
- You can connect the math to ML models.

## 32. External Problem Sources

Use these after finishing the custom problemsets above.

### Statistics

OpenIntro Statistics:

- Chapter 2: summarizing data.
- Chapter 3: probability.
- Chapter 4: distributions.
- Use the exercises at the end of each chapter.

Link: https://www.openintro.org/book/os/

Think Stats:

- Use for computational statistics in Python.
- Recreate examples in notebooks.

Link: https://allendowney.github.io/ThinkStats2/

Harvard CS109:

- Use for statistics and data science style practice.
- Focus on probability, inference, regression, and Python analysis.

Link: https://harvard-iacs.github.io/CS109/

### Linear Algebra

MIT OCW 18.06:

- Use assignments 1-10.
- Use exams after you finish the assignments.
- Focus especially on systems, vector spaces, orthogonality, determinants, eigenvalues, and positive definite matrices.

Assignments: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/assignments/

Exams: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/exams/

3Blue1Brown Linear Algebra:

- Use for geometric intuition.
- Watch before or alongside MIT problem sets.

Link: https://www.3blue1brown.com/topics/linear-algebra

Mathematics for Machine Learning:

- Use chapters on linear algebra, analytic geometry, matrix decompositions, probability, vector calculus, and continuous optimization.
- Use its additional exercises and notebooks if you want harder ML-oriented practice.

Link: https://mml-book.github.io/

### Calculus And Optimization

3Blue1Brown Calculus:

- Use for intuition about derivatives, chain rule, and integrals.

Link: https://www.3blue1brown.com/topics/calculus

Khan Academy Calculus:

- Use for many routine derivative practice problems.

Link: https://www.khanacademy.org/math/calculus-1

Google ML Crash Course:

- Use linear regression, generalization, overfitting, and numerical data sections.
- Connect gradients and loss functions to ML training.

Link: https://developers.google.com/machine-learning/crash-course

## 33. Six-Week Mastery Plan

### Week 1: Descriptive Statistics

Study:

- Mean, median, mode.
- Variance and standard deviation.
- Percentiles and z-scores.

Do:

- Problemsets 1 and 2.
- OpenIntro Chapter 2 exercises.

Deliverable:

- A Python script implementing summary statistics from scratch.

### Week 2: Random Variables, Distributions, Correlation

Study:

- Random variables.
- Expected value.
- Bernoulli, binomial, uniform, normal.
- Covariance and correlation.
- Correlation vs causation.

Do:

- Problemsets 3, 4, and 5.
- OpenIntro probability/distribution exercises.

Deliverable:

- Simulation notebook comparing theoretical and empirical probabilities.

### Week 3: Vectors And Matrices

Study:

- Vector arithmetic.
- Norms and distances.
- Dot product.
- Matrix multiplication.
- Systems of equations.

Do:

- Problemsets 6, 7, 8, and 9.
- MIT 18.06 early assignments.

Deliverable:

- NumPy vector/matrix toolkit.

### Week 4: Projections, Least Squares, PCA

Study:

- Projection.
- Least squares geometry.
- Rank.
- PCA intuition.
- Embeddings.

Do:

- Problemsets 10 and 11.
- MML linear algebra/PCA sections.

Deliverable:

- PCA visualization of Iris or digits dataset.

### Week 5: Derivatives And Optimization

Study:

- Derivative rules.
- Chain rule.
- Critical points.
- Partial derivatives.
- Gradients.

Do:

- Problemsets 12, 13, and 14.
- Khan Academy derivative practice.

Deliverable:

- Numerical derivative and numerical gradient implementation.

### Week 6: Losses, Gradient Descent, Integrated Projects

Study:

- MSE and MAE.
- Cross-entropy intuition.
- Gradient descent.
- Learning rate.
- Least squares optimization.

Do:

- Problemsets 15, 16, 17, 18, 19, and 20.
- Google ML Crash Course linear regression/generalization sections.

Deliverable:

- Final script fitting linear regression by closed form and gradient descent.

## 34. Selected Answer Key

### Problemset 1, A1

Data:

```text
[2, 4, 6, 8]
```

Mean:

```text
20 / 4 = 5
```

Median:

```text
(4 + 6) / 2 = 5
```

Population variance:

```text
deviations = -3, -1, 1, 3
squared = 9, 1, 1, 9
variance = 20 / 4 = 5
```

Population standard deviation:

```text
sqrt(5)
```

### Problemset 3, Die Variance

For fair die:

```text
E[X] = 3.5
E[X^2] = (1 + 4 + 9 + 16 + 25 + 36) / 6 = 91/6
Var(X) = 91/6 - 3.5^2 = 35/12
```

### Problemset 5, Positive Correlation

For:

```text
x = [1, 2, 3]
y = [2, 4, 6]
```

`y = 2x`, so correlation is `1`.

### Problemset 6, Vector Distance

Given:

```text
a = [1, 2, 3]
b = [4, 0, -1]
```

```text
a + b = [5, 2, 2]
a - b = [-3, 2, 4]
2a = [2, 4, 6]
||a|| = sqrt(1 + 4 + 9) = sqrt(14)
distance = sqrt((-3)^2 + 2^2 + 4^2) = sqrt(29)
Manhattan = 3 + 2 + 4 = 9
```

### Problemset 7, Dot Product

```text
a = [1, 2]
b = [3, 4]
a dot b = 1*3 + 2*4 = 11
```

### Problemset 8, Matrix-Vector

```text
A = [[1, 2],
     [3, 4]]
x = [5, 6]

A @ x = [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
```

### Problemset 12, Derivative Example

```text
f(x) = 3x^4 - 2x + 7
f'(x) = 12x^3 - 2
```

### Problemset 14, Gradient Example

```text
L(w1, w2) = (w1 - 2)^2 + (w2 + 3)^2
grad L = [2(w1 - 2), 2(w2 + 3)]
```

At `(0, 0)`:

```text
grad L = [-4, 6]
-grad L = [4, -6]
```

Gradient descent moves toward increasing `w1` and decreasing `w2`, heading toward `(2, -3)`.

### Problemset 16, First Step

```text
f(w) = (w - 4)^2
f'(w) = 2(w - 4)
w0 = 0
learning_rate = 0.25
gradient = -8
w1 = 0 - 0.25*(-8) = 2
```

## 35. Mistake Log Template

Use this after every study session:

```text
Date:
Topic:
Problem:
My answer:
Correct answer:
What concept I missed:
How to recognize this pattern next time:
One similar problem to solve tomorrow:
```

The fastest learners are not the ones who avoid mistakes. They are the ones who convert mistakes into patterns.

## 36. Final Mastery Checklist

Statistics:

- I can compute mean, median, variance, standard deviation by hand.
- I understand sample vs population variance.
- I can compute percentiles, IQR, and z-scores.
- I can define random variables and expected value.
- I know Bernoulli, binomial, uniform, and normal distributions.
- I can simulate random processes in Python.
- I can combine group means correctly.
- I understand what changes when data is duplicated.
- I can compute covariance and correlation.
- I can explain correlation vs causation.

Linear algebra:

- I can add, subtract, and scale vectors.
- I can compute norms and distances.
- I can compute dot product and cosine similarity.
- I understand embeddings as vectors.
- I can multiply matrices and track shapes.
- I can solve linear systems.
- I understand inverse, rank, span, basis, and independence.
- I can explain projection.
- I can connect least squares to projection.
- I can explain PCA intuitively.
- I can implement matrix operations in NumPy.

Calculus and optimization:

- I can compute derivatives using basic rules.
- I understand chain rule.
- I can find critical points.
- I can classify minima and maxima.
- I can compute partial derivatives.
- I understand gradients.
- I can explain loss functions.
- I can run gradient descent by hand.
- I can implement gradient descent in Python.
- I understand learning-rate tradeoffs.
- I can fit a line by minimizing MSE.

If you can do every item above, you have the math foundation needed for classical ML, deep learning intros, and many IOAI-style reasoning problems.
