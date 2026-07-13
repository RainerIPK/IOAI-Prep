# Selection Reasoning Mastery

This guide is for the OSN AI / IOAI-style selection problem family summarized by:

| Questions | Topic | What to learn |
|---|---|---|
| 1 | Optimal-play interval game | Game states, rational intervals, coverage, strategy, invariants. |
| 2 | Grid path counting | DFS/dynamic counting, 8-neighbor movement, repeated-cell paths. |
| 3-4 | Statistics and algebra | Variance under duplicated samples, combined means, symbolic manipulation. |
| 5-7 | Probability | Law of total probability, conditional probability, Bayes-style reasoning. |
| 8-9 | Linear model | Mean squared error, least squares, optimizing alpha and beta. |
| 10-21 | Vector similarity | Euclidean distance, vector arithmetic, embeddings, matching by geometry. |
| 22-25 | Convolution | Kernel, stride, padding, output shape formulas. |
| 26-40 | Simple sequence classification | Feature counting, language-model/classification intuition, nearest pattern matching. |

The goal is not to memorize answers. The goal is to master the reasoning patterns well enough that a new selection problem feels like a variation, not a surprise.

## 0. How To Use This Guide

Use this loop:

```text
read concept -> solve tiny hand example -> brute force with code -> find pattern -> prove/explain -> solve timed drills
```

For every topic, you should be able to do four things:

1. **Model** the problem in clean mathematical language.
2. **Compute** small examples by hand.
3. **Verify** your reasoning with Python.
4. **Explain** the final method in a few sentences.

Contest reasoning is often won by turning a story into one of these shapes:

```text
state game
grid walk
weighted average
conditional probability table
least-squares optimization
nearest-neighbor geometry
shape formula
feature-count classifier
```

If you can recognize the shape, the problem becomes manageable.

## 1. General Contest Method

When you see a long problem statement:

1. Identify the objects.
2. Identify the operation.
3. Identify what is being counted, optimized, predicted, or compared.
4. Rewrite in symbols.
5. Solve a tiny version.
6. Look for invariant or recurrence.
7. Check boundary cases.
8. Only then compute the actual answer.

### The Tiny-Version Rule

Never start with the full numbers if the problem is new.

Examples:

- If the board is `7 x 7`, test `2 x 2` and `3 x 3`.
- If sequence length is 12, test length 2 or 3.
- If interval is `[0, n]`, test `n = 1, 2, 3, 4`.
- If a model has parameters alpha and beta, test one data point, then two.

Small cases reveal:

- parity
- symmetry
- impossible moves
- off-by-one errors
- whether repetition is allowed
- whether endpoints matter
- whether strict inequality matters

### The Strict-Inequality Rule

Many contest traps hide in symbols:

```text
> 1 is not >= 1
inclusive interval [0, n] includes endpoints
path may revisit cells unless forbidden
probability in percent may require flooring
padding = same differs from padding = 0
```

Underline these constraints before solving.

### The Python Verification Rule

For any combinatorics/game/counting problem, write a brute force for tiny cases.

Your brute force does not need to be efficient. It needs to be correct for small cases.

Then compare:

```text
hand reasoning vs brute force
```

If they disagree, the brute force is your lantern.

## Part I: Optimal-Play Interval Games

## 2. What Is An Optimal-Play Game?

An optimal-play game has:

- players
- legal moves
- turns
- terminal states
- winning/losing condition

In the interval game:

```text
Players choose rational numbers in [0, n].
A new number must be more than 1 away from every previously chosen number.
A player unable to move loses.
```

This is an impartial game:

- both players have the same legal moves
- no hidden information
- no randomness

The only difference between players is whose turn it is.

## 3. State Representation

A state is everything needed to determine legal future moves.

For interval games, the state can be represented as:

```text
chosen points sorted increasingly
```

or as:

```text
remaining legal intervals
```

If a point `x` is chosen, future points cannot lie in:

```text
[x - 1, x + 1]
```

intersected with `[0, n]`.

Because the next point must be distance `> 1` from all chosen points, endpoints exactly distance 1 are also illegal.

### Important Subtlety

If a chosen point is `x`, a future point `y` is legal only if:

```text
|x - y| > 1
```

So:

```text
y = x + 1
```

is not legal.

This strict inequality changes maximum packing counts.

## 4. Packing Points More Than 1 Apart

Question:

```text
How many points can fit in [0, n] if all pairwise distances are > 1?
```

If there are `k` points sorted:

```text
x1 < x2 < ... < xk
```

then:

```text
x2 - x1 > 1
x3 - x2 > 1
...
xk - x(k-1) > 1
```

Adding:

```text
xk - x1 > k - 1
```

Since:

```text
x1 >= 0
xk <= n
```

we need:

```text
n > k - 1
```

So:

```text
k < n + 1
```

If `n` is an integer, the maximum possible `k` is at most `n`.

You can achieve `n` points by choosing:

```text
0, 1 + eps, 2 + 2eps, ..., n - 1 + (n - 1)eps
```

for sufficiently small positive `eps` such that the last point stays at most `n`.

So for integer `n`, the maximum packing size is:

```text
n
```

But a game is not just maximum packing. Players can make moves that reduce or preserve future move count. Optimal strategy matters.

## 5. Winning And Losing States

A state is:

```text
losing if every legal move gives the opponent a winning state
winning if at least one legal move gives the opponent a losing state
```

This recursive definition is the backbone of impartial game analysis.

For finite games:

```python
def is_winning(state):
    moves = legal_moves(state)
    if not moves:
        return False
    return any(not is_winning(next_state) for next_state in moves)
```

For continuous rational games, you cannot enumerate all moves directly. You need structure:

- symmetry
- intervals
- pairing strategies
- maximum/minimum remaining moves
- invariants

## 6. Pairing Strategy

A pairing strategy divides the board into pairs or mirrored regions.

If player 1 chooses a point, player 2 responds with the paired point.

For interval `[0, n]`, symmetry around the center:

```text
x -> n - x
```

is often useful.

If a move at `x` is legal, its mirror may or may not be legal depending on chosen points and distance constraints.

Pairing works when:

- every move has a valid response
- responses do not interfere
- the responder always makes the last move

This is common in interval, board, and geometric games.

## 7. Splitting Into Subgames

Choosing a point can split the remaining legal region into left and right intervals.

Example:

Choose `x` in `[0, n]`.

Forbidden:

```text
[x - 1, x + 1]
```

Remaining:

```text
[0, x - 1) and (x + 1, n]
```

Each remaining interval can be treated as an independent region, except for strict endpoints.

This resembles games like:

- Kayles
- interval deletion games
- impartial splitting games

For contest use, you usually do not need full Sprague-Grundy theory, but you should know the idea:

```text
one move can split a game into independent subgames
```

## 8. Interval Game Mastery Checklist

You should be able to:

- Represent chosen numbers as sorted points.
- Translate distance constraints into forbidden intervals.
- Compute maximum number of points possible.
- Analyze small `n` cases.
- Identify symmetry.
- Build a brute-force approximation by discretizing points.
- Explain why discretization is only evidence, not proof.
- Look for pairing or invariant strategies.

## Part II: Grid Path Counting

## 9. Grid Paths With Repetition

A grid path problem usually defines:

1. starting cell
2. allowed moves
3. path length
4. whether revisiting cells is allowed
5. target word/pattern

In the IOAI-style path problem:

```text
7 x 7 grid
letters I, O, A
word = IOAI
path length = 4 cells
move to any of 8 neighbors each step
cells may be revisited
```

Because revisiting is allowed, the state does not need a visited set.

State can be:

```text
(row, col, index_in_word)
```

## 10. Eight-Direction Movement

From cell `(r, c)`, neighbors are:

```text
(r-1, c-1), (r-1, c), (r-1, c+1)
(r,   c-1),           (r,   c+1)
(r+1, c-1), (r+1, c), (r+1, c+1)
```

Keep only those inside grid.

Python directions:

```python
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
```

## 11. DFS Counting

For word `W`, count paths:

```python
def dfs(r, c, i):
    if grid[r][c] != W[i]:
        return 0
    if i == len(W) - 1:
        return 1
    total = 0
    for dr, dc in DIRS8:
        nr, nc = r + dr, c + dc
        if inside(nr, nc):
            total += dfs(nr, nc, i + 1)
    return total
```

Total:

```python
sum(dfs(r, c, 0) for all cells)
```

Because repeated cells are allowed, this is simple.

If repeated cells were forbidden, state would include:

```text
visited cells
```

That is a much harder problem.

## 12. Dynamic Programming Counting

Instead of starting DFS from each cell, build counts by step.

Let:

```text
dp[i][r][c] = number of paths that end at cell (r, c) after matching W[0..i]
```

Initialization:

```text
dp[0][r][c] = 1 if grid[r][c] == W[0], else 0
```

Transition:

```text
dp[i][r][c] =
    0 if grid[r][c] != W[i]
    otherwise sum dp[i-1][neighbor_r][neighbor_c]
```

Final answer:

```text
sum dp[len(W)-1][r][c]
```

This is efficient:

```text
O(word_length * rows * cols * 8)
```

## 13. Matrix View

You can flatten grid cells into nodes of a graph.

Let:

```text
A = adjacency matrix
```

where:

```text
A[u][v] = 1 if v is a neighbor of u
```

A path of length 4 cells has 3 moves.

For word constraints, you apply letter masks at each step.

This is overkill for small grids but useful conceptually:

```text
grid path counting = graph walk counting with label constraints
```

## 14. Path Counting Traps

Common mistakes:

- Count moves instead of cells.
- Forget diagonal moves.
- Forbid revisiting when allowed.
- Allow staying in place when not allowed.
- Count only paths starting from a highlighted example.
- Mix row/column coordinates.
- Double-count reversed paths if direction matters.

In this problem, direction matters:

```text
I -> O -> A -> I
```

Reversed path spells:

```text
I -> A -> O -> I
```

which is different.

## Part III: Statistics And Algebra

## 15. Mean

Mean:

```text
mean = (sum of values) / n
```

If a dataset has `N` values with mean `m`, its sum is:

```text
N*m
```

This is the key to combined-mean problems.

## 16. Combined Mean

Group 1:

```text
n1 values, mean m1
```

Group 2:

```text
n2 values, mean m2
```

Combined mean:

```text
(n1*m1 + n2*m2) / (n1 + n2)
```

Do not average group means unless group sizes are equal.

Example:

```text
10 students mean 80
30 students mean 60
combined = (10*80 + 30*60) / 40 = 65
```

Wrong:

```text
(80 + 60) / 2 = 70
```

## 17. Variance

Population variance:

```text
var_pop = (1/n) * sum((x_i - mean)^2)
```

Sample variance:

```text
var_sample = (1/(n-1)) * sum((x_i - mean)^2)
```

Always check which definition the problem uses.

Contest problems sometimes say "sample variance" specifically.

## 18. Duplicating A Dataset

Suppose data:

```text
x1, x2, ..., xN
```

is duplicated exactly:

```text
x1, x2, ..., xN, x1, x2, ..., xN
```

Mean:

```text
unchanged
```

Population variance:

```text
unchanged
```

because every deviation is repeated and denominator doubles.

Sample variance:

Original:

```text
S / (N - 1)
```

Duplicated:

```text
2S / (2N - 1)
```

These are not generally equal.

If a contest answer expects sample variance, duplication can change the numerical sample variance even though the distribution looks the same.

## 19. Symbolic Manipulation

Many selection questions are algebra in disguise.

Example:

Batch A and B together have `2N` people with mean `m`.

Batch C has `M` people with mean `m/2`.

After combining all, new mean is `3m/4`.

Equation:

```text
(2N*m + M*(m/2)) / (2N + M) = 3m/4
```

Assuming `m != 0`, divide by `m`:

```text
(2N + M/2) / (2N + M) = 3/4
```

Solve:

```text
4(2N + M/2) = 3(2N + M)
8N + 2M = 6N + 3M
2N = M
```

So:

```text
M = 2N
```

The method is more important than this one result:

```text
turn words into weighted-average equation
cancel common factors
solve symbolically
```

## Part IV: Probability

## 20. Probability Basics

Probability is a number from 0 to 1.

```text
P(A) = probability event A happens
```

Complement:

```text
P(not A) = 1 - P(A)
```

Intersection:

```text
P(A and B)
```

Union:

```text
P(A or B)
```

Formula:

```text
P(A or B) = P(A) + P(B) - P(A and B)
```

## 21. Conditional Probability

Conditional probability:

```text
P(A | B) = P(A and B) / P(B)
```

Read:

```text
probability of A given B
```

Important:

```text
P(A | B) is not the same as P(B | A)
```

## 22. Law Of Total Probability

If events `B1, B2, ..., Bk` partition all possibilities:

```text
P(A) = sum P(A | Bi) P(Bi)
```

Example:

```text
P(buy popcorn)
= P(popcorn | action)*P(action)
+ P(popcorn | family)*P(family)
+ P(popcorn | horror)*P(horror)
```

## 23. Bayes Rule

Bayes rule:

```text
P(A | B) = P(B | A) P(A) / P(B)
```

Often:

```text
P(cause | evidence) = P(evidence | cause) P(cause) / P(evidence)
```

For selection problems:

1. Compute numerator.
2. Compute denominator using total probability.
3. Divide.

## 24. Probability Tables

When given categories and conditional percentages, build a joint table.

Example:

```text
P(Film = Action) = 0.40
P(Popcorn | Action) = 0.60
```

Then:

```text
P(Action and Popcorn) = 0.40 * 0.60 = 0.24
```

For conditional questions:

```text
P(Horror | SoftDrinkOnly)
= P(Horror and SoftDrinkOnly) / P(SoftDrinkOnly)
```

## 25. Rounding And Percent Trap

If the problem says:

```text
answer as integer percent, round down
```

then:

```text
64.71 percent -> 64
```

Do not round to nearest unless instructed.

## Part V: Linear Models

## 26. Linear Prediction

A linear model with one feature:

```text
f(M; alpha, beta) = alpha + beta*M
```

`alpha`:

```text
intercept
```

`beta`:

```text
slope
```

If `beta = 6`, increasing `M` by 1 increases prediction by 6.

## 27. Mean Squared Error

For data `(M_i, y_i)`:

```text
prediction_i = alpha + beta*M_i
error_i = y_i - prediction_i
```

MSE:

```text
L(alpha, beta) = (1/n) * sum(error_i^2)
```

MSE punishes large errors strongly.

## 28. Computing A Given Loss

Steps:

1. Compute predictions.
2. Compute errors.
3. Square errors.
4. Average.
5. Apply requested rounding.

Example:

```text
M = [1, 2]
y = [5, 7]
alpha = 1
beta = 2
pred = [3, 5]
errors = [2, 2]
squared = [4, 4]
MSE = 4
```

## 29. Least Squares Optimum

For simple linear regression:

```text
beta = sum((M_i - mean_M)(y_i - mean_y)) / sum((M_i - mean_M)^2)
alpha = mean_y - beta*mean_M
```

This minimizes MSE.

For multiple features:

```text
w = (X^T X)^-1 X^T y
```

if invertible.

In code:

```python
import numpy as np
w, *_ = np.linalg.lstsq(X, y, rcond=None)
```

## 30. Derivative View

MSE is a quadratic bowl in `alpha` and `beta`.

At optimum:

```text
dL/dalpha = 0
dL/dbeta = 0
```

For contest problems with 2 parameters and 3 points, either:

- use the regression formulas
- solve normal equations
- use a small Python verification

## Part VI: Vector Similarity

## 31. Vectors

A vector is a list of numbers.

Example:

```text
senang = (21, 10)
```

Vector operations:

```text
a + b
a - b
k*a
```

Example:

```text
(21, 10) + (23, 16) - (26, 8)
= (18, 18)
```

## 32. Euclidean Distance

For two 2D points:

```text
d((x1,y1), (x2,y2)) = sqrt((x1-x2)^2 + (y1-y2)^2)
```

Often the square root is unnecessary for comparison.

Squared distance:

```text
(x1-x2)^2 + (y1-y2)^2
```

Preserves ordering because square root is increasing.

## 33. Dot Product And Cosine Similarity

Dot product:

```text
a dot b = sum a_i b_i
```

Cosine similarity:

```text
cos(a,b) = (a dot b) / (||a|| ||b||)
```

Euclidean distance compares absolute positions.

Cosine similarity compares direction.

For many embedding tasks:

- cosine is common for text embeddings
- Euclidean can work when vector magnitudes matter

## 34. Embedding Analogies

Embedding analogies use vector arithmetic.

Classic shape:

```text
king - man + woman ~= queen
```

This works when a semantic relationship is represented as a consistent vector offset.

In contest problems, vectors are often designed so these relationships are exact or nearly exact.

Method:

1. Compute target vector.
2. Compare to candidate vectors.
3. Choose nearest candidate.

## 35. Matching Two Vector Spaces

Suppose you have two sets of vectors:

```text
Indonesian words
Alien words
```

and they are translations under a structure-preserving transformation.

If structure is preserved, then pairwise distances are preserved.

Strategy:

1. Compute all pairwise distance patterns in set A.
2. Compute all pairwise distance patterns in set B.
3. Match points with same distance fingerprints.

Distance fingerprint for a point:

```text
sorted list of squared distances to all other points
```

If fingerprints are unique, matching is easy.

If not unique:

- use triangle patterns
- use known analogies
- use brute force search with pruning

## 36. Brute Force Matching

If there are 10 words, naive permutations:

```text
10! = 3,628,800
```

This is feasible in Python for well-pruned checks.

Pruning idea:

```text
when assigning a new pair, check distances to already assigned pairs
```

Pseudo-code:

```python
def backtrack(mapping):
    if complete:
        return mapping
    choose unmapped alien point
    for candidate Indonesian point:
        if candidate unused and distances match assigned pairs:
            assign
            recurse
            unassign
```

This is powerful for contest vector-matching problems.

## Part VII: Convolution Shape Problems

## 37. What Is Convolution?

In image processing, convolution applies a kernel over an image.

For each position:

```text
multiply kernel entries by image patch entries
sum products
```

Kernel:

```text
small matrix, e.g. 3 x 3
```

Stride:

```text
how far kernel moves each step
```

Padding:

```text
extra border pixels
```

## 38. Output Size Formula

For one dimension:

```text
out = floor((in + 2*padding - kernel) / stride + 1)
```

More general with dilation:

```text
out = floor((in + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
```

Most selection problems use dilation 1.

## 39. Same Padding

If:

```text
padding = "same"
stride = 1
```

then output size equals input size.

If stride is not 1, "same" can mean framework-specific rounding.

For contest problems, read the statement carefully. If it says:

```text
padding = same, stride = 1
```

use:

```text
out = in
```

## 40. Tuple Kernels, Padding, Stride

If input is:

```text
height x width
```

and kernel/padding/stride are tuples:

```text
kernel = (kh, kw)
padding = (ph, pw)
stride = (sh, sw)
```

compute each dimension separately:

```text
out_h = floor((h + 2*ph - kh) / sh + 1)
out_w = floor((w + 2*pw - kw) / sw + 1)
```

Do not mix height and width.

## 41. Sequential Layers

Apply layers one by one.

Example:

```text
Input 32 x 32
Conv 3x3, same, stride 1 -> 32 x 32
Conv 5x5, padding 0, stride 2
out = floor((32 - 5)/2 + 1) = 14
Final: 14 x 14
```

Most errors come from:

- forgetting floor
- forgetting padding on both sides
- applying stride before padding
- mixing width/height
- misunderstanding same padding

## Part VIII: Simple Sequence Classification

## 42. Sequence Classification Without Fancy AI

A sequence classifier maps:

```text
sequence -> class
```

Examples:

- DNA sequence -> species
- text -> sentiment
- click sequence -> user type
- sound event sequence -> label

For small contest problems, simple feature counting often works.

## 43. Feature Counting

For DNA-like sequences:

```text
A, C, G, T
```

Features:

- count of A
- count of C
- count of G
- count of T
- proportions
- bigram counts
- longest run
- GC content
- position-specific patterns

GC content:

```text
(count(G) + count(C)) / length
```

If one class has high GC content and another has high AT content, classification can be done by counting.

## 44. Nearest Prototype Classification

For each class, compute prototype feature vector:

```text
mean feature vector of training examples in class
```

For a new sequence:

1. compute feature vector
2. compute distance to each class prototype
3. choose nearest

This is simple and often strong in designed contest problems.

## 45. Naive Bayes Intuition For Sequences

For class `c`, estimate:

```text
P(symbol | class)
```

Then for sequence:

```text
score(c) = P(c) * product P(symbol_i | c)
```

Use log probabilities to avoid underflow:

```text
log_score(c) = log P(c) + sum log P(symbol_i | c)
```

This is a simple language model.

## 46. Nearest Neighbor Classification

Instead of prototypes, compare to all training sequences.

Distance choices:

- Hamming distance for equal-length sequences
- edit distance for variable-length strings
- Euclidean distance over feature counts
- cosine similarity over count vectors

For contest DNA strings of equal length, Hamming distance is easy:

```text
number of positions where symbols differ
```

## 47. Sequence Classification Traps

Common mistakes:

- using raw counts when lengths differ
- ignoring class priors
- missing obvious position-specific signal
- overfitting to one example
- not checking bigrams/trigrams
- assuming AI is needed when simple counting works

In selection problems, if the dataset is tiny and handcrafted, look for visible patterns first.

## 48. Problemsets Overview

Levels:

```text
A: hand computation
B: Python brute force
C: strategy explanation
D: timed mixed drills
```

Do not skip hand computation. These problems are designed so a human can solve them with careful arithmetic and modeling.

## Problemset 1: Interval Game Foundations

### A. Packing

For each interval, find the maximum number of points with pairwise distance `> 1`.

1. `[0, 1]`
2. `[0, 2]`
3. `[0, 3]`
4. `[0, 4]`
5. `[0, 5]`

Explain the strict inequality.

### B. Legal Moves

Already chosen:

```text
[2.5]
```

Interval:

```text
[0, 6]
```

Describe remaining legal regions.

Now chosen:

```text
[1, 4]
```

Describe remaining legal regions.

### C. Small Game Analysis

For discretized board:

```text
positions = {0, 1.1, 2.2, 3.3}
distance must be > 1
```

Analyze who wins under optimal play.

### D. Strategy

Try to find a first-player or second-player pairing strategy for small integer `n`.

Write your reasoning, not just the answer.

## Problemset 2: Interval Game Brute Force

### A. Discretization

Write code to approximate `[0, n]` by positions:

```text
0, step, 2*step, ..., n
```

Legal if distance `> 1`.

### B. Recursive Solver

Implement:

```python
is_winning(chosen_positions, remaining_positions)
```

for small discretized boards.

### C. Pattern Search

Run for:

```text
n = 2, 3, 4, 5, 6
step = 0.5, 0.25
```

Look for patterns.

### D. Warning

Explain why discretized evidence is not a proof for rational intervals.

## Problemset 3: Grid Path Counting

### A. Hand Count

Grid:

```text
I O
A I
```

Allowed moves:

```text
8 directions
```

Cells may be revisited.

Count paths spelling:

```text
IOI
```

### B. DFS

Implement DFS to count a word in a grid.

Test on:

```text
2 x 2
3 x 3
```

### C. DP

Implement dynamic programming version.

Compare with DFS.

### D. Trap Test

Modify code for:

1. no diagonal moves
2. no revisits
3. staying in place allowed

Compare counts and explain differences.

## Problemset 4: Word IOAI On Grids

### A. Custom Grid

Create a `4 x 4` grid using letters I/O/A.

Count paths spelling:

```text
IOAI
```

### B. Contribution By Start

For each starting I cell, count how many paths start there.

### C. Contribution By Direction Pattern

Classify paths by first move direction.

### D. Report

Explain how repeated cells affect the count.

## Problemset 5: Mean And Variance

### A. Hand

Data:

```text
[2, 4, 6]
```

Compute:

1. mean
2. population variance
3. sample variance

Duplicate the dataset and recompute all three.

### B. General

If original sample has `N` values and sum of squared deviations `S`, compare:

```text
S/(N-1)
```

and:

```text
2S/(2N-1)
```

### C. Explanation

Explain why population variance is unchanged by exact duplication.

## Problemset 6: Combined Means

### A. Weighted Mean

Group A:

```text
N = 20, mean = 70
```

Group B:

```text
N = 30, mean = 80
```

Find combined mean.

### B. Solve For Unknown Group Size

Existing group:

```text
2N people, mean m
```

New group:

```text
M people, mean m/2
```

Combined mean:

```text
3m/4
```

Solve for `M` in terms of `N`.

### C. Create Your Own

Create 5 combined-mean problems where the unknown is:

1. group mean
2. group size
3. final mean
4. original mean
5. added total sum

## Problemset 7: Probability Tables

### A. Total Probability

Movie probabilities:

```text
Action 40%
Family 35%
Horror 25%
```

Popcorn probabilities:

```text
P(popcorn | Action) = 60%
P(popcorn | Family) = 40%
P(popcorn | Horror) = 40%
```

Compute `P(popcorn)`.

### B. Conditional Probability

Suppose:

```text
P(soft only | Action) = 10%
P(soft only | Family) = 15%
P(soft only | Horror) = 10%
```

Compute:

```text
P(Horror | soft only)
```

### C. Floor Rule

Convert final probability to integer percent rounded down.

### D. Python

Represent the table as dictionaries and compute all joint probabilities.

## Problemset 8: Bayes Rule

### A. Medical Test

Disease prevalence:

```text
1%
```

Test sensitivity:

```text
95%
```

False positive rate:

```text
5%
```

Compute:

```text
P(disease | positive)
```

### B. Classifier Analogy

Explain how this relates to precision and recall.

### C. Contest Reflection

Why do base rates matter?

## Problemset 9: Linear Model Loss

### A. Hand

Data:

```text
M = [95, 86, 68]
y = [785, 790, 600]
```

For:

```text
alpha = 227
beta = 6
```

Compute predictions, errors, squared errors, and MSE.

Round to 1 decimal.

### B. Python

Write a function:

```python
mse(alpha, beta, M, y)
```

### C. Grid Search

Try alpha from 0 to 400 and beta from 0 to 10 in steps of 0.1.

Find approximate best pair.

## Problemset 10: Least Squares

### A. Formula

Using:

```text
beta = sum((x-mean_x)(y-mean_y)) / sum((x-mean_x)^2)
alpha = mean_y - beta*mean_x
```

fit a line for:

```text
x = [1, 2, 3]
y = [2, 4, 5]
```

### B. Matrix

Build:

```text
X = [[1, x1], [1, x2], [1, x3]]
```

Solve with `np.linalg.lstsq`.

### C. Explanation

Explain why optimal rounded parameters may not be optimal after rounding.

## Problemset 11: Vector Arithmetic

### A. Hand

Vectors:

```text
senang = (21, 10)
kalah = (23, 16)
menang = (26, 8)
```

Compute:

```text
senang + kalah - menang
```

### B. Distances

Compute Euclidean distance:

```text
raja = (8, 36)
kalah = (23, 16)
```

### C. Nearest

Given 10 vectors, compute nearest to a target vector by squared distance.

## Problemset 12: Vector Matching

### A. Fingerprints

Create two sets of 5 points where one is translated/rotated version of the other.

Compute sorted squared-distance fingerprints.

### B. Match

Use fingerprints to match points.

### C. Ambiguity

Create a square:

```text
(0,0), (1,0), (0,1), (1,1)
```

Show why fingerprints are not unique.

### D. Backtracking

Implement distance-preserving backtracking for 6-10 points.

## Problemset 13: Embedding Analogies

### A. Toy

Define:

```text
man = (0, 0)
woman = (10, 0)
king = (0, 5)
queen = (10, 5)
```

Compute:

```text
king - man + woman
```

### B. Noisy

Add small noise to vectors.

Use nearest neighbor to recover analogy.

### C. Explain

When does analogy arithmetic fail?

## Problemset 14: Convolution Shapes

### A. Single Layer

Compute output sizes:

1. `32 x 32`, kernel `3 x 3`, same padding, stride 1
2. `32 x 32`, kernel `5 x 5`, padding 0, stride 2
3. `64 x 64`, kernel `7 x 7`, padding 3, stride 2
4. `128 x 64`, kernel `7 x 3`, padding `(3,1)`, stride `(2,1)`

### B. Sequential

Start:

```text
64 x 64
```

Layers:

```text
Conv 7x7 p=3 s=2
Conv 3x3 same s=2
Conv 1x1 p=0 s=1
```

Compute final size.

### C. Python

Write:

```python
conv_out(size, kernel, padding, stride)
```

Support integers and tuples.

## Problemset 15: Sequence Feature Counting

### A. Counts

For:

```text
ACGACGCGATCG
```

Compute:

1. count A
2. count C
3. count G
4. count T
5. GC content

### B. Class Prototypes

Given training sequences for 3 species, compute mean feature counts.

Classify new sequences by nearest prototype.

### C. Bigrams

Compute bigram counts:

```text
AA, AC, AG, AT, CA, ...
```

Compare whether bigrams improve classification.

## Problemset 16: Naive Bayes Sequence Classifier

### A. Hand

Class A sequences:

```text
AAAA
AAAT
```

Class B sequences:

```text
GGGG
GGGA
```

With Laplace smoothing, classify:

```text
AAAG
```

### B. Code

Implement Naive Bayes over symbols.

### C. Logs

Use log probabilities.

Explain why multiplying many probabilities can underflow.

## Problemset 17: Mixed Timed Drill 1

Set a timer for 60 minutes.

Solve:

1. One interval game small case.
2. One grid path count on `3 x 3`.
3. One combined mean problem.
4. One Bayes probability problem.
5. One convolution shape chain.

Afterwards, write:

```text
Which topic cost the most time?
Which mistakes were arithmetic?
Which mistakes were modeling?
```

## Problemset 18: Mixed Timed Drill 2

Set a timer for 90 minutes.

Solve:

1. A vector matching problem with 6 points.
2. A least-squares regression problem.
3. A sequence classification problem.
4. A probability table problem.
5. A grid path DP problem.

Then verify all answers with Python.

## Problemset 19: Full Synthetic Selection

Create your own 20-question mini selection:

- 2 game questions
- 2 grid counting questions
- 2 stats/algebra questions
- 3 probability questions
- 2 linear model questions
- 4 vector similarity questions
- 2 convolution questions
- 3 sequence classification questions

Write answer key and explanations.

This is one of the best ways to master the format.

## Problemset 20: Final Exam

### Part A: Concepts

Explain:

1. How to represent an interval game state.
2. Why strict `> 1` matters.
3. DFS vs DP for grid path counting.
4. Sample variance vs population variance.
5. Combined mean formula.
6. Law of total probability.
7. Bayes rule.
8. MSE and least squares.
9. Euclidean distance and squared distance.
10. Distance-preserving vector matching.
11. Convolution output formula.
12. Feature-count sequence classification.

### Part B: Hand Computation

Compute:

1. Maximum packing count in `[0, 8]`.
2. Number of paths spelling a 3-letter word in a `2 x 2` grid.
3. Sample variance before and after duplication.
4. A conditional probability from a table.
5. MSE for a line on 3 points.
6. Nearest vector by squared distance.
7. Final convolution size after 4 layers.
8. GC content and nearest species prototype.

### Part C: Coding

Implement:

1. grid word counter
2. probability table calculator
3. least-squares solver
4. vector matching by distance fingerprints
5. convolution shape calculator
6. sequence feature classifier

Passing standard:

- You can solve small cases by hand.
- You can verify with Python.
- You can explain every formula.
- You can avoid strict-inequality and rounding traps.
- You can complete a mixed drill under time pressure.

## 49. Eight-Week Mastery Plan

### Week 1: Contest Modeling And Interval Games

Study:

- state representation
- legal moves
- strict inequalities
- winning/losing recursion
- symmetry and pairing

Do:

- Problemsets 1-2

Deliverable:

- small interval-game brute force solver

### Week 2: Grid Path Counting

Study:

- DFS
- DP
- 8-neighbor movement
- repeated-cell paths

Do:

- Problemsets 3-4

Deliverable:

- reusable grid word counter

### Week 3: Statistics, Algebra, Probability

Study:

- mean
- variance
- duplication
- combined means
- total probability
- Bayes

Do:

- Problemsets 5-8

Deliverable:

- probability table and weighted-mean calculator

### Week 4: Linear Models

Study:

- MSE
- least squares
- rounding
- grid search verification

Do:

- Problemsets 9-10

Deliverable:

- line-fitting script with plots/loss values

### Week 5: Vector Similarity

Study:

- vector arithmetic
- Euclidean distance
- embeddings
- distance fingerprints
- matching by geometry

Do:

- Problemsets 11-13

Deliverable:

- vector matching solver

### Week 6: Convolution And Sequence Classification

Study:

- kernel/stride/padding
- sequential output shapes
- feature counts
- nearest prototype
- Naive Bayes sequence model

Do:

- Problemsets 14-16

Deliverable:

- convolution calculator and DNA classifier

### Week 7: Mixed Timed Drills

Do:

- Problemsets 17-18

Deliverable:

- mistake log with topic-by-topic weaknesses

### Week 8: Create And Solve A Mini Selection

Do:

- Problemsets 19-20

Deliverable:

- full synthetic selection with answer key

## 50. Selected Answer Key

### Packing In `[0, n]`

For integer `n`, maximum number of points pairwise distance `> 1` is:

```text
n
```

Reason:

If there are `k` sorted points, then:

```text
xk - x1 > k - 1
```

and since `xk - x1 <= n`:

```text
k < n + 1
```

For integer `n`, `k <= n`.

### Combined Mean Unknown

Equation:

```text
(2N*m + M*(m/2)) / (2N + M) = 3m/4
```

Cancel `m`:

```text
(2N + M/2) / (2N + M) = 3/4
```

Solve:

```text
8N + 2M = 6N + 3M
M = 2N
```

### IoU-Style Distance Reminder

For vector nearest-neighbor comparisons, squared distance is enough:

```text
d2 = (x1-x2)^2 + (y1-y2)^2
```

The smallest squared distance also has the smallest Euclidean distance.

### Convolution Formula

```text
out = floor((in + 2*padding - kernel) / stride + 1)
```

For tuples:

```text
out_h = floor((h + 2*ph - kh) / sh + 1)
out_w = floor((w + 2*pw - kw) / sw + 1)
```

### Grid DP Recurrence

```text
dp[i][r][c] = 0 if grid[r][c] != word[i]
otherwise sum dp[i-1][nr][nc] over 8-neighbors
```

Final:

```text
sum dp[last_index][r][c]
```

## 51. Mistake Log

Use this after every drill:

```text
Date:
Topic:
Problem:
My first model:
Correct model:
Arithmetic mistake:
Conceptual mistake:
Trap missed:
How I verified:
What pattern I should recognize next time:
One similar problem to retry:
```

Most selection mistakes are not from lacking advanced AI knowledge. They are from:

- misreading constraints
- skipping small cases
- arithmetic drift
- using the wrong variance/probability formula
- forgetting floors
- ignoring strict inequalities
- failing to verify with simple code

Master those, and these problems become much calmer.

## 52. Final Mastery Checklist

Interval games:

- I can model legal moves and forbidden intervals.
- I can analyze small cases.
- I can explain winning/losing state recursion.
- I can look for symmetry and invariants.

Grid counting:

- I can count paths with DFS.
- I can count paths with DP.
- I can handle 8-neighbor movement.
- I can correctly allow or forbid revisits.

Statistics/algebra:

- I can compute mean and variance.
- I know population vs sample variance.
- I understand exact dataset duplication.
- I can solve combined-mean equations.

Probability:

- I can build joint probability tables.
- I can apply total probability.
- I can apply Bayes rule.
- I can handle percent rounding instructions.

Linear models:

- I can compute MSE by hand.
- I can fit least-squares lines.
- I can optimize alpha and beta.
- I can verify with Python.

Vector similarity:

- I can compute vector arithmetic.
- I can compute Euclidean distances.
- I can solve analogy-style vector questions.
- I can match vector spaces by distance structure.

Convolution:

- I can compute output sizes layer by layer.
- I can handle tuple stride/padding/kernel.
- I remember floor and both-side padding.

Sequence classification:

- I can count symbols and bigrams.
- I can build class prototypes.
- I can use nearest-prototype classification.
- I can explain simple language-model classification.

If you can complete the problemsets and final exam without looking at the notes, you are ready for this selection-problem family.
