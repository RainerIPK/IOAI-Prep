# Classical Machine Learning Mastery

This is a dense, self-contained learning path for classical machine learning:

- Supervised learning: linear regression, logistic regression, regularization, KNN, trees, random forests, bagging, gradient boosting, SVMs.
- Unsupervised learning: K-means, PCA, t-SNE, UMAP, DBSCAN, hierarchical clustering, spectral clustering.
- Evaluation: accuracy, precision, recall, F1, confusion matrix, ROC curves, underfitting, overfitting, hyperparameter tuning, cross-validation.
- Practice: hand problems, coding problems, dataset tasks, and capstone projects.

Verified source resources:

- IOAI Module 2: https://ioai.toki.id/assets/modul/Modul_2_IOAI__Pengenalan_Machine_Learning__ML_.pdf
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course
- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- scikit-learn MOOC: https://inria.github.io/scikit-learn-mooc/
- ISLR: https://www.statlearning.com/
- ISLR Python labs: https://www.statlearning.com/resources-python
- StatQuest: https://statquest.org/
- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- Kaggle Datasets: https://www.kaggle.com/datasets

## 0. What Classical ML Is Really About

Classical machine learning is the art of learning patterns from examples without explicitly programming every rule.

You are usually given a dataset:

```text
X = inputs / features
y = target / label
```

Example:

```text
House size, bedrooms, distance to city -> house price
Age, blood pressure, cholesterol -> disease yes/no
Petal length, petal width -> iris species
Customer history -> likely to churn yes/no
```

A model is a function:

```text
f(X) -> prediction
```

Training means choosing model parameters so predictions match known answers as well as possible.

Evaluation means testing whether the model learned a real pattern or merely memorized noise.

The central questions in all classical ML:

1. What is the input representation?
2. What kind of prediction do we need?
3. What model family fits the structure of the problem?
4. What loss or metric defines success?
5. How do we prevent overfitting?
6. How do we verify that performance generalizes to new data?
7. Can we explain the result enough to trust it?

## 1. The Full ML Workflow

Every serious ML project follows this skeleton:

1. Define the task.
2. Collect or load data.
3. Inspect and clean the data.
4. Split into train, validation, and test sets.
5. Build a baseline.
6. Train models.
7. Tune hyperparameters.
8. Evaluate on validation data.
9. Select one final model.
10. Test once on held-out test data.
11. Interpret errors.
12. Improve features, data, or model.

### Train, Validation, Test

The most common split:

```text
Training set:   used to fit model parameters.
Validation set: used to choose hyperparameters and model type.
Test set:       used once at the end to estimate final performance.
```

Never tune your model on the test set. If you repeatedly check test performance, the test set stops being a true test and becomes another validation set.

### Baseline

A baseline is a simple first model. If your fancy model cannot beat the baseline, something is wrong.

Regression baselines:

- Always predict the training mean.
- Linear regression.

Classification baselines:

- Always predict the majority class.
- Logistic regression.
- K-nearest neighbors with a small feature set.

Clustering baselines:

- Try K-means with different values of `k`.
- Compare against obvious feature-based grouping.

### Leakage

Data leakage happens when information from the future, target, validation set, or test set sneaks into training.

Examples:

- Scaling the entire dataset before splitting.
- Filling missing values using the mean of the full dataset before splitting.
- Including an ID column that encodes the answer.
- Training on duplicated samples that also appear in the test set.
- Using a feature collected after the event you are trying to predict.

Correct pattern:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression())
])

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

The pipeline learns preprocessing only from training data.

## 2. Data And Features

Models do not understand real life. They understand numbers.

Feature engineering means turning raw data into useful numeric representation.

### Numeric Features

Examples:

- Age.
- Height.
- Price.
- Temperature.
- Number of previous purchases.

Common preprocessing:

- Standardization: subtract mean and divide by standard deviation.
- Min-max scaling: map values to `[0, 1]`.
- Log transform: reduce extreme skew.
- Clipping: cap extreme outliers.

Standardization:

```text
z = (x - mean) / standard_deviation
```

Important for:

- KNN.
- SVM.
- Logistic regression with regularization.
- PCA.
- K-means.

Less important for:

- Decision trees.
- Random forests.
- Gradient boosted trees.

### Categorical Features

Examples:

- City.
- Product category.
- Browser type.
- Species name.

Common encodings:

- One-hot encoding: each category becomes a 0/1 column.
- Ordinal encoding: category becomes an integer. Only use if order makes sense or the model can handle it.

Bad idea:

```text
red = 1, green = 2, blue = 3
```

This implies blue > green > red, which may be meaningless.

### Missing Values

Common approaches:

- Drop rows if few are missing.
- Fill numeric values with mean or median.
- Fill categorical values with most frequent category.
- Add a missing-value indicator column.

Practical rule:

- If missingness itself may carry information, add an indicator.
- Example: missing income may mean a user declined to answer, which can be predictive.

### Outliers

Outliers can be:

- Errors.
- Rare but real cases.
- Important anomalies.

Do not blindly delete outliers. Ask:

1. Is this physically possible?
2. Is it a data entry error?
3. Does the model need to handle cases like this?
4. Does the metric become dominated by these values?

### Feature Scaling Cheat Sheet

| Model | Scaling needed? | Why |
|---|---:|---|
| Linear regression | Sometimes | Helps optimization and regularization. |
| Logistic regression | Usually | Regularization is scale-sensitive. |
| KNN | Yes | Distances depend on feature scale. |
| SVM | Yes | Margins and kernels depend on scale. |
| K-means | Yes | Euclidean distance depends on scale. |
| PCA | Yes | Variance depends on scale. |
| Decision tree | No | Trees split by thresholds. |
| Random forest | No | Tree-based. |
| Gradient boosted trees | Usually no | Tree-based. |

## 3. Supervised Learning

Supervised learning uses labeled examples.

```text
Training data: (X_1, y_1), (X_2, y_2), ..., (X_n, y_n)
Goal: learn f(X) that predicts y for new X.
```

Two main types:

```text
Regression: predict a number.
Classification: predict a class.
```

## 4. Linear Regression

Linear regression predicts a continuous numeric value.

### Model

For one feature:

```text
y_hat = b0 + b1*x
```

For many features:

```text
y_hat = b0 + b1*x1 + b2*x2 + ... + bp*xp
```

Each coefficient answers:

> If this feature increases by 1 unit while other features stay fixed, how much does the prediction change?

### Loss Function

The usual loss is mean squared error:

```text
MSE = (1/n) * sum((y_i - y_hat_i)^2)
```

Why square errors?

- Penalizes large errors more strongly.
- Makes the math smooth.
- Leads to a closed-form solution for ordinary least squares.

### Interpretation Example

Suppose:

```text
exam_score = 40 + 5*hours_studied
```

Interpretation:

- `40` is the predicted score at 0 hours.
- `5` means each extra hour studied adds 5 predicted points.

This does not prove causation. It only describes the fitted relationship in the data.

### Assumptions To Know

Linear regression works best when:

- Relationship is roughly linear.
- Errors are not extremely skewed.
- Features are not redundant copies of each other.
- There are no overwhelming outliers.

But in competitions, you do not need perfect assumptions. You need good generalization.

### Common Failure Modes

1. Nonlinear relationship.
2. Outliers dominate MSE.
3. Important feature missing.
4. Target leakage.
5. Multicollinearity: features highly correlated with each other.

### Metrics

Regression metrics:

```text
MAE  = mean absolute error
MSE  = mean squared error
RMSE = sqrt(MSE)
R2   = fraction of variance explained
```

MAE is easier to interpret. RMSE punishes large errors more.

### Minimal scikit-learn Pattern

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(mean_absolute_error(y_test, pred))
print(mean_squared_error(y_test, pred))
print(r2_score(y_test, pred))
```

## 5. Logistic Regression

Despite the name, logistic regression is usually used for classification.

### Binary Classification

Binary classification predicts one of two classes:

```text
spam / not spam
disease / no disease
survived / died
fraud / not fraud
```

Linear regression outputs any number. For classification, we want a probability between 0 and 1.

Logistic regression computes:

```text
z = b0 + b1*x1 + ... + bp*xp
p = sigmoid(z)
```

Where:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

The sigmoid maps any real number to `[0, 1]`.

Decision rule:

```text
if p >= 0.5: predict class 1
else: predict class 0
```

The threshold does not have to be 0.5. In medical screening, you may lower the threshold to catch more true positives.

### Log Odds

Logistic regression models log odds:

```text
log(p / (1 - p)) = b0 + b1*x1 + ... + bp*xp
```

A positive coefficient increases the probability of class 1. A negative coefficient decreases it.

### Multiclass Classification

For more than two classes, logistic regression can use:

- One-vs-rest.
- Multinomial softmax.

Softmax converts scores into probabilities across classes:

```text
P(class k) = exp(score_k) / sum(exp(score_j))
```

### Minimal scikit-learn Pattern

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

## 6. L1 And L2 Regularization

Regularization discourages overly complex models.

It modifies the training objective:

```text
objective = loss + penalty
```

### L2 Regularization

L2 adds squared coefficient size:

```text
penalty = lambda * sum(w_j^2)
```

Effect:

- Shrinks coefficients toward zero.
- Usually keeps all features.
- Helps when features are correlated.
- Makes model smoother.

In scikit-learn:

```python
from sklearn.linear_model import Ridge
```

or logistic regression:

```python
LogisticRegression(penalty="l2")
```

### L1 Regularization

L1 adds absolute coefficient size:

```text
penalty = lambda * sum(abs(w_j))
```

Effect:

- Can push some coefficients exactly to zero.
- Performs feature selection.
- Useful when many features are irrelevant.

In scikit-learn:

```python
from sklearn.linear_model import Lasso
```

or logistic regression:

```python
LogisticRegression(penalty="l1", solver="liblinear")
```

### Elastic Net

Elastic net combines L1 and L2.

Useful when:

- Many features.
- Some irrelevant.
- Some correlated groups.

```python
from sklearn.linear_model import ElasticNet
```

### Important scikit-learn Detail

For `LogisticRegression`, the parameter `C` is inverse regularization strength:

```text
large C  -> weak regularization
small C  -> strong regularization
```

This is easy to forget.

## 7. K-Nearest Neighbors

KNN predicts by looking at nearby training examples.

### Classification

Given a new point:

1. Find the `k` closest training points.
2. Let them vote.
3. Predict the majority class.

### Regression

Given a new point:

1. Find the `k` closest training points.
2. Average their target values.

### Distance

Most common:

```text
Euclidean distance = sqrt(sum((x_j - z_j)^2))
```

### Choosing k

Small `k`:

- Flexible.
- Low bias.
- High variance.
- Can overfit noise.

Large `k`:

- Smoother.
- Higher bias.
- Lower variance.
- May underfit.

### Why Scaling Matters

Suppose one feature is age from 0 to 100, and another is income from 0 to 1,000,000. Euclidean distance will be dominated by income.

Always scale for KNN unless all features are naturally comparable.

### Minimal Pattern

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])

pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

### Strengths

- Simple.
- No training cost beyond storing data.
- Works well on small, clean datasets.
- Flexible decision boundaries.

### Weaknesses

- Slow prediction on large datasets.
- Sensitive to irrelevant features.
- Sensitive to feature scaling.
- Poor in high-dimensional spaces.

The high-dimensional problem is called the curse of dimensionality: distances become less meaningful as dimension grows.

## 8. Decision Trees

A decision tree asks a sequence of questions.

Example:

```text
if petal_width <= 0.8:
    class = setosa
else:
    if petal_length <= 4.8:
        class = versicolor
    else:
        class = virginica
```

### Splits

A tree chooses splits that make child nodes purer.

For classification, common impurity measures:

- Gini impurity.
- Entropy.

Gini impurity:

```text
Gini = 1 - sum(p_k^2)
```

If a node contains only one class, Gini is 0.

Entropy:

```text
Entropy = -sum(p_k * log2(p_k))
```

### Regression Trees

Regression trees split to reduce variance or squared error.

Leaves predict the average target value of training examples in that leaf.

### Tree Hyperparameters

Important knobs:

- `max_depth`: maximum depth.
- `min_samples_split`: minimum samples needed to split a node.
- `min_samples_leaf`: minimum samples in each leaf.
- `max_leaf_nodes`: maximum leaves.

Small tree:

- More interpretable.
- Less variance.
- More bias.

Huge tree:

- Can memorize training data.
- Low training error.
- Often high test error.

### Minimal Pattern

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

### Strengths

- Easy to interpret.
- Handles nonlinear relationships.
- Does not need scaling.
- Handles mixed feature scales.

### Weaknesses

- High variance.
- Small data changes can change the tree.
- Deep trees overfit easily.

## 9. Bagging And Random Forests

Random forests fix the high-variance weakness of decision trees.

### Bagging

Bagging means bootstrap aggregating.

Process:

1. Create many bootstrap samples from training data.
2. Train one tree on each sample.
3. Average predictions for regression or vote for classification.

Bootstrap sample:

- Same size as original training set.
- Sampled with replacement.
- Some rows appear multiple times.
- Some rows do not appear.

Effect:

- Reduces variance.
- Makes predictions more stable.

### Random Forest

Random forest adds another trick:

At each split, each tree considers only a random subset of features.

This decorrelates trees, making the ensemble stronger.

### Minimal Pattern

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

### Important Hyperparameters

- `n_estimators`: number of trees.
- `max_depth`: depth limit.
- `min_samples_leaf`: minimum examples per leaf.
- `max_features`: number of features considered per split.

### Strengths

- Strong default model.
- Handles nonlinearities.
- Robust to feature scaling.
- Less overfitting than one tree.
- Can estimate feature importance.

### Weaknesses

- Less interpretable than one tree.
- Larger memory and compute cost.
- Feature importance can be misleading when features are correlated.

## 10. Gradient Boosting

Gradient boosting builds an ensemble sequentially.

Instead of training many independent trees, it trains each new tree to correct mistakes from the previous ensemble.

Core idea:

```text
prediction_0 = simple prediction
for each round:
    compute errors
    train small tree to predict errors
    add small tree to ensemble
```

### Learning Rate

Gradient boosting usually adds each tree with a small weight:

```text
new_prediction = old_prediction + learning_rate * tree_prediction
```

Small learning rate:

- Slower.
- Often better generalization.
- Needs more trees.

Large learning rate:

- Faster.
- More risk of overfitting or instability.

### Common Models

In scikit-learn:

- `GradientBoostingClassifier`
- `GradientBoostingRegressor`
- `HistGradientBoostingClassifier`
- `HistGradientBoostingRegressor`

Outside scikit-learn:

- XGBoost.
- LightGBM.
- CatBoost.

For IOAI/classical ML mastery, scikit-learn is enough.

### Minimal Pattern

```python
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    random_state=42
)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

### Strengths

- Often excellent on tabular data.
- Handles nonlinearities and interactions.
- Strong competition baseline.

### Weaknesses

- More hyperparameter-sensitive.
- Can overfit if too many rounds or too high learning rate.
- Less interpretable than simple models.

## 11. Support Vector Machines

SVMs try to find a decision boundary with the largest margin.

### Margin

The margin is the distance between the decision boundary and the nearest training points.

SVM wants:

```text
Correct classification + largest possible margin
```

Nearest boundary points are support vectors.

### Linear SVM

Useful when classes are roughly linearly separable.

```python
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LinearSVC(C=1.0, random_state=42))
])
```

### Kernel SVM

Kernels let SVM create nonlinear boundaries.

Common kernel:

```text
RBF kernel
```

RBF kernel has important hyperparameters:

- `C`: regularization strength.
- `gamma`: how local each training point's influence is.

Large `gamma`:

- Very local influence.
- More complex boundary.
- Can overfit.

Small `gamma`:

- Smoother boundary.
- Can underfit.

### Minimal RBF Pattern

```python
from sklearn.svm import SVC

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", SVC(kernel="rbf", C=1.0, gamma="scale"))
])
```

### Strengths

- Good on small/medium datasets.
- Effective in high-dimensional spaces.
- Kernel trick can model nonlinear boundaries.

### Weaknesses

- Scaling is essential.
- Can be slow on large datasets.
- Hyperparameters matter a lot.
- Less interpretable.

## 12. Unsupervised Learning

Unsupervised learning uses data without labels.

Goals:

- Find clusters.
- Compress data.
- Visualize high-dimensional data.
- Detect anomalies.
- Discover structure.

There is no single correct answer in many unsupervised tasks. Evaluation is harder because no labels are given.

## 13. K-Means Clustering

K-means partitions data into `k` clusters.

Algorithm:

1. Choose `k` initial centroids.
2. Assign each point to nearest centroid.
3. Move each centroid to mean of assigned points.
4. Repeat until assignments stop changing or improvement is tiny.

### Objective

K-means minimizes within-cluster squared distance:

```text
sum over clusters sum over points in cluster distance(point, centroid)^2
```

### Choosing k

Methods:

- Domain knowledge.
- Elbow method.
- Silhouette score.

Elbow method:

- Plot inertia for different `k`.
- Look for the point where improvement slows.

Silhouette score:

- Measures how close each point is to its own cluster vs other clusters.
- Range: `-1` to `1`.
- Higher is better.

### Minimal Pattern

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KMeans(n_clusters=3, n_init="auto", random_state=42))
])

labels = pipe.fit_predict(X)
```

### Strengths

- Simple.
- Fast.
- Works well for round, similarly-sized clusters.

### Weaknesses

- Must choose `k`.
- Sensitive to scale.
- Sensitive to initialization.
- Poor for irregular cluster shapes.
- Poor when clusters have very different densities.

## 14. PCA

Principal Component Analysis reduces dimensionality while preserving as much variance as possible.

Imagine data points in 2D lying mostly along a diagonal line. PCA finds that main direction.

### Core Idea

PCA creates new axes:

```text
PC1: direction of maximum variance
PC2: next direction, perpendicular to PC1
PC3: next direction, perpendicular to earlier PCs
```

The components are linear combinations of original features.

### Why Use PCA?

- Visualization.
- Noise reduction.
- Compression.
- Speeding up models.
- Handling correlated features.

### Explained Variance

If first 2 components explain 95 percent of variance, you can often reduce many dimensions to 2 with limited information loss.

### Minimal Pattern

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=2))
])

X_2d = pipe.fit_transform(X)
```

### Warning

PCA is unsupervised. It preserves variance, not necessarily class separability.

A low-variance direction may still be highly predictive.

## 15. t-SNE And UMAP

t-SNE and UMAP are mainly for visualization, not preprocessing before ordinary models.

They map high-dimensional data to 2D or 3D while trying to preserve local neighborhood structure.

### t-SNE

Good for:

- Visualizing clusters.
- Exploring embeddings.

Weaknesses:

- Slow on large datasets.
- Global distances can be misleading.
- Different random seeds can change plots.
- Cluster sizes in plots are not always meaningful.

### UMAP

Good for:

- Faster visualization.
- Preserving some local/global structure better than t-SNE in many cases.
- Embedding visualization.

Weaknesses:

- Extra dependency: `umap-learn`.
- Hyperparameters matter.
- Still mainly exploratory.

### Rule

Use PCA first if dimensions are huge, then t-SNE/UMAP for visualization.

```text
Raw high-dimensional data -> scale -> PCA to 30-50 dims -> t-SNE/UMAP to 2 dims
```

## 16. DBSCAN

DBSCAN is a density-based clustering algorithm.

It finds dense regions and labels sparse points as noise.

Important parameters:

- `eps`: radius around a point.
- `min_samples`: minimum points required to form dense region.

Point types:

- Core point: enough neighbors within `eps`.
- Border point: near a core point but does not itself have enough neighbors.
- Noise point: not part of any cluster.

### Strengths

- Does not require choosing number of clusters.
- Can find irregular shapes.
- Can detect outliers.

### Weaknesses

- Sensitive to `eps`.
- Struggles with varying-density clusters.
- Scaling is important.

### Minimal Pattern

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)
```

`label = -1` means noise.

## 17. Hierarchical Clustering

Hierarchical clustering builds a tree of clusters.

Agglomerative approach:

1. Start with each point as its own cluster.
2. Repeatedly merge closest clusters.
3. Stop when desired number of clusters is reached.

### Linkage Types

- Single linkage: distance between closest points.
- Complete linkage: distance between farthest points.
- Average linkage: average pairwise distance.
- Ward linkage: minimizes within-cluster variance.

### Dendrogram

A dendrogram shows the merge history.

You can "cut" the dendrogram at different heights to get different cluster counts.

### Strengths

- Does not require choosing cluster count upfront if using dendrogram.
- Useful for nested structure.

### Weaknesses

- Can be slow for large datasets.
- Sensitive to distance metric and linkage.

## 18. Spectral Clustering

Spectral clustering treats data as a graph.

Idea:

1. Build a similarity graph among points.
2. Use eigenvectors of a graph Laplacian.
3. Cluster in the transformed space.

Good for:

- Non-convex clusters.
- Shapes K-means cannot capture.

Weaknesses:

- More complex.
- Sensitive to similarity graph settings.
- Can be expensive.

Use it after understanding K-means and DBSCAN.

## 19. Evaluation For Classification

Classification predictions can be summarized by a confusion matrix.

For binary classification:

```text
                 Predicted positive   Predicted negative
Actual positive  True positive        False negative
Actual negative  False positive       True negative
```

### Accuracy

```text
accuracy = correct predictions / all predictions
```

Good when classes are balanced and mistake costs are similar.

Bad when classes are imbalanced.

Example:

- 99 percent of emails are not spam.
- Model always predicts not spam.
- Accuracy = 99 percent.
- Spam detection = useless.

### Precision

```text
precision = TP / (TP + FP)
```

Of predicted positives, how many were correct?

High precision matters when false positives are costly.

Example:

- Accusing someone of fraud.
- Sending a patient for risky treatment.

### Recall

```text
recall = TP / (TP + FN)
```

Of actual positives, how many did we catch?

High recall matters when false negatives are costly.

Example:

- Cancer screening.
- Security threat detection.

### F1 Score

```text
F1 = 2 * precision * recall / (precision + recall)
```

F1 balances precision and recall.

Use F1 when:

- Classes are imbalanced.
- You care about both false positives and false negatives.

### ROC Curve And AUC

Many classifiers output probabilities or scores.

Changing the threshold changes:

- True positive rate.
- False positive rate.

ROC curve plots:

```text
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
```

AUC summarizes the ROC curve:

- 0.5 means random ranking.
- 1.0 means perfect ranking.

Warning:

For highly imbalanced data, precision-recall curves can be more informative than ROC curves.

## 20. Evaluation For Regression

### MAE

```text
MAE = mean(abs(y - y_hat))
```

Interpretation: average absolute error in target units.

### MSE

```text
MSE = mean((y - y_hat)^2)
```

Penalizes large errors more.

### RMSE

```text
RMSE = sqrt(MSE)
```

Same units as target.

### R2

```text
R2 = 1 - model_error / baseline_error
```

Baseline error usually means predicting the mean.

Interpretation:

- `R2 = 1`: perfect.
- `R2 = 0`: no better than predicting mean.
- `R2 < 0`: worse than mean baseline.

## 21. Underfitting And Overfitting

### Underfitting

Model too simple.

Symptoms:

- High training error.
- High validation error.
- Model cannot capture pattern.

Fixes:

- Use more expressive model.
- Add useful features.
- Reduce regularization.
- Train longer for iterative models.

### Overfitting

Model memorizes training noise.

Symptoms:

- Low training error.
- High validation error.
- Large train-validation gap.

Fixes:

- Use simpler model.
- Add regularization.
- Get more data.
- Remove leakage.
- Reduce tree depth.
- Increase `min_samples_leaf`.
- Cross-validate.

### Bias-Variance Intuition

High bias:

- Model is too rigid.
- Underfits.

High variance:

- Model is too sensitive to training data.
- Overfits.

Goal:

```text
Low enough bias + low enough variance.
```

## 22. Hyperparameter Tuning

Parameters are learned from data.

Examples:

- Linear regression coefficients.
- Tree split thresholds.

Hyperparameters are chosen before training.

Examples:

- `k` in KNN.
- Tree `max_depth`.
- Random forest `n_estimators`.
- SVM `C` and `gamma`.
- Regularization strength.

### Grid Search

Try all combinations from a grid.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__gamma": ["scale", 0.1, 1]
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1")
search.fit(X_train, y_train)
print(search.best_params_)
print(search.best_score_)
```

### Random Search

Try random combinations.

Good when:

- Many hyperparameters.
- Some matter more than others.
- You have limited compute.

### Practical Tuning Order

1. Start with simple baseline.
2. Fix data leakage and preprocessing.
3. Try 2-4 model families.
4. Tune most promising model.
5. Inspect errors.
6. Add/change features.
7. Final test once.

## 23. Cross-Validation

Cross-validation estimates generalization more reliably than one train/validation split.

### K-Fold Cross-Validation

Process:

1. Split training data into `k` folds.
2. Train on `k-1` folds.
3. Validate on remaining fold.
4. Repeat so each fold validates once.
5. Average scores.

Common:

```text
k = 5 or 10
```

### Stratified K-Fold

For classification, preserve class proportions in each fold.

Use this especially for imbalanced classes.

### Time Series Warning

Do not randomly shuffle time series when predicting the future.

Use time-aware splits:

```text
train on past -> validate on later future
```

## 24. Model Choice Cheat Sheet

| Situation | Try first |
|---|---|
| Need interpretability, numeric target | Linear regression / Ridge / Lasso |
| Need interpretable binary classifier | Logistic regression |
| Small dataset, nonlinear boundary | KNN or SVM |
| Tabular data, strong baseline | Random forest |
| Tabular data, best performance | Gradient boosting |
| Need simple decision rules | Decision tree |
| Need clustering with known k | K-means |
| Need irregular clusters/outliers | DBSCAN |
| Need dimensionality reduction | PCA |
| Need visualization | PCA, t-SNE, UMAP |

## 25. scikit-learn Project Template

Use this pattern repeatedly until it becomes automatic.

```python
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")

target = "target_column"
X = df.drop(columns=[target])
y = df[target]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

## 26. Problemsets

The problemsets are staged. Do them in order.

Mastery rule:

- Level A: you can compute by hand.
- Level B: you can implement from scratch or with NumPy.
- Level C: you can use scikit-learn correctly.
- Level D: you can diagnose, tune, explain, and improve.

Do not skip hand problems. They make the APIs meaningful.

## Problemset 1: Data, Splits, And Leakage

### A. Concept Questions

1. Explain the difference between a parameter and a hyperparameter. Give two examples of each.
2. Why is it wrong to scale the entire dataset before train/test split?
3. You train a model to predict whether a customer will cancel tomorrow. One feature is `cancellation_email_sent`. Why might this be leakage?
4. A dataset has 10,000 rows, and 9,700 belong to class 0. A classifier gets 97 percent accuracy by always predicting 0. Why is this weak?
5. Why do we use validation data?
6. What is a baseline model for classification? For regression?
7. What does `random_state=42` do?
8. Why might you use `stratify=y` in `train_test_split`?

### B. Coding Drills

1. Create a synthetic dataset with 100 rows and 3 numeric columns. Split it into train/test sets.
2. Fit `StandardScaler` only on training data, then transform both train and test.
3. Intentionally scale before splitting. Explain why the result is less trustworthy.
4. Create a classification dataset with 95 percent class 0 and 5 percent class 1. Compute majority-class baseline accuracy.

### C. Mastery Task

Use the UCI Iris dataset or scikit-learn built-in Iris dataset:

1. Split with and without stratification.
2. Print class proportions in train/test.
3. Explain the difference.

## Problemset 2: Linear Regression

### A. Hand Problems

1. Given `y_hat = 3 + 2x`, compute predictions for `x = 0, 1, 2, 5`.
2. True values are `[3, 5, 9]`; predictions are `[2, 5, 10]`. Compute MAE, MSE, and RMSE.
3. Fit the best line intuitively to points `(0, 1)`, `(1, 3)`, `(2, 5)`. What are intercept and slope?
4. If a model has coefficient `-4` for `distance_to_city`, interpret it.
5. Why can MSE be more sensitive to outliers than MAE?

### B. Coding Drills

1. Generate `y = 5 + 3x + noise`. Fit `LinearRegression`.
2. Print fitted intercept and coefficient.
3. Plot true points and fitted line.
4. Compare MAE and RMSE when you add one huge outlier.
5. Fit Ridge and Lasso on the same data. Compare coefficients.

### C. From-Scratch Challenge

Implement simple linear regression for one feature using formulas:

```text
slope = sum((x - mean_x)(y - mean_y)) / sum((x - mean_x)^2)
intercept = mean_y - slope * mean_x
```

Then compare your result with scikit-learn.

### D. Dataset Task

Use UCI Wine Quality or a Kaggle house-price style dataset:

1. Predict a numeric target.
2. Build mean baseline.
3. Build linear regression.
4. Build Ridge.
5. Compare MAE, RMSE, R2.
6. Explain which model you would submit and why.

## Problemset 3: Logistic Regression

### A. Hand Problems

1. Compute `sigmoid(0)`.
2. Is `sigmoid(3)` closer to 0 or 1?
3. If a coefficient is positive, what happens as the feature increases?
4. A classifier outputs probability 0.72 for class 1. What prediction is made at threshold 0.5? At threshold 0.8?
5. Explain why logistic regression can still produce a linear decision boundary.

### B. Coding Drills

1. Load breast cancer dataset from scikit-learn.
2. Train logistic regression with scaling.
3. Print confusion matrix and classification report.
4. Change threshold from 0.5 to 0.3 and compare precision/recall.
5. Tune `C` with cross-validation.

### C. Interpretation Task

Train logistic regression on a dataset with named features.

Report:

- Top 5 positive coefficients.
- Top 5 negative coefficients.
- Which features increase predicted class 1 probability?
- Which features decrease it?

### D. Failure Analysis

Find 10 false positives and 10 false negatives. Inspect their feature values. Write what patterns confuse the model.

## Problemset 4: Regularization

### A. Concept Questions

1. What is the purpose of regularization?
2. Difference between L1 and L2?
3. Why can L1 perform feature selection?
4. In logistic regression, what happens when `C` becomes smaller?
5. Why should features be scaled before regularized linear models?

### B. Coding Drills

1. Create a dataset with 5 useful features and 20 noisy features.
2. Fit unregularized linear/logistic model.
3. Fit Ridge/L2.
4. Fit Lasso/L1.
5. Count how many coefficients are near zero.

### C. Mastery Task

Make a plot:

```text
regularization strength vs train score and validation score
```

Explain:

- Which region underfits?
- Which region overfits?
- Which value would you choose?

## Problemset 5: KNN

### A. Hand Problems

Given points:

```text
A: (0, 0), class red
B: (1, 0), class red
C: (0, 1), class blue
D: (5, 5), class blue
```

1. For query `(0.8, 0.2)`, compute distances to all points.
2. Predict with `k=1`.
3. Predict with `k=3`.
4. Explain how prediction changes if feature 1 is multiplied by 100.

### B. Coding Drills

1. Load Iris.
2. Scale features.
3. Train KNN for `k = 1, 3, 5, 11, 21`.
4. Plot `k` vs validation accuracy.
5. Repeat without scaling. Compare.

### C. Mastery Task

Use a dataset with at least 10 features.

1. Train KNN.
2. Add 50 random noise features.
3. Train KNN again.
4. Explain why performance changes.

## Problemset 6: Decision Trees

### A. Concept Questions

1. What is a leaf?
2. What is tree depth?
3. What does Gini impurity measure?
4. Why do deep trees overfit?
5. Why do trees not require feature scaling?

### B. Coding Drills

1. Train a decision tree on Iris.
2. Try `max_depth = 1, 2, 3, None`.
3. Compare train and test accuracy.
4. Plot or print the tree rules.
5. Tune `min_samples_leaf`.

### C. Hand Impurity Problem

A node has 10 samples:

```text
6 class A, 4 class B
```

Compute Gini:

```text
1 - (6/10)^2 - (4/10)^2
```

Now split into:

```text
Left: 5 A, 1 B
Right: 1 A, 3 B
```

Compute weighted child Gini. Did the split improve purity?

### D. Mastery Task

Train a tree on a tabular dataset. Write a human-readable explanation of the first three splits.

## Problemset 7: Random Forests And Bagging

### A. Concept Questions

1. What is bootstrapping?
2. Why does averaging many trees reduce variance?
3. What extra randomness does a random forest add beyond bagging?
4. What does `n_estimators` control?
5. Why can feature importance be misleading?

### B. Coding Drills

1. Train one decision tree and one random forest on the same data.
2. Compare train/test performance.
3. Tune `n_estimators`, `max_depth`, and `min_samples_leaf`.
4. Plot feature importances.
5. Compare out-of-bag score if using `oob_score=True`.

### C. Mastery Task

Use the UCI Adult dataset or another tabular classification dataset.

1. Build preprocessing with numeric and categorical columns.
2. Train logistic regression baseline.
3. Train random forest.
4. Compare metrics.
5. Explain which errors matter more in this problem.

## Problemset 8: Gradient Boosting

### A. Concept Questions

1. How is boosting different from bagging?
2. What does learning rate do?
3. Why can too many boosting rounds overfit?
4. Why is gradient boosting strong on tabular data?
5. What is the tradeoff between `learning_rate` and number of estimators/iterations?

### B. Coding Drills

1. Train `HistGradientBoostingClassifier`.
2. Compare with random forest.
3. Try learning rates `0.01`, `0.05`, `0.1`, `0.3`.
4. Try different `max_leaf_nodes`.
5. Plot validation score by hyperparameter.

### C. Mastery Task

Choose a tabular Kaggle or UCI dataset.

1. Build logistic regression, random forest, and gradient boosting.
2. Use cross-validation.
3. Tune only the best two models.
4. Write a model selection report.

## Problemset 9: Support Vector Machines

### A. Concept Questions

1. What is a margin?
2. What are support vectors?
3. Why does SVM need scaling?
4. What does `C` control?
5. What does `gamma` control in RBF SVM?

### B. Coding Drills

1. Train linear SVM on breast cancer dataset.
2. Train RBF SVM.
3. Tune `C` and `gamma`.
4. Compare with logistic regression.
5. Compare training time as dataset size grows.

### C. Mastery Task

Create a 2D toy dataset with nonlinear class boundary using `make_moons`.

1. Train logistic regression.
2. Train linear SVM.
3. Train RBF SVM.
4. Plot decision boundaries.
5. Explain why RBF wins.

## Problemset 10: K-Means

### A. Hand Problems

Given points on a line:

```text
0, 1, 2, 10, 11, 12
```

Initial centroids:

```text
0 and 12
```

1. Assign each point to nearest centroid.
2. Recompute centroids.
3. Repeat one more iteration.
4. What clusters do you get?

### B. Coding Drills

1. Generate blobs with `make_blobs`.
2. Fit K-means with `k=2,3,4,5`.
3. Plot inertia.
4. Plot silhouette score.
5. Choose best `k`.

### C. Mastery Task

Use customer-like synthetic data:

```text
annual_income
spending_score
visits_per_month
average_order_value
```

Cluster customers and name each cluster in business language.

Example:

- High-income low-spend.
- Frequent small buyers.
- Premium loyal customers.

## Problemset 11: PCA

### A. Concept Questions

1. What does PC1 represent?
2. Why should data be scaled before PCA?
3. What is explained variance?
4. Why can PCA hurt classification performance?
5. Is PCA supervised or unsupervised?

### B. Coding Drills

1. Load digits dataset from scikit-learn.
2. Scale it.
3. Use PCA to reduce to 2D.
4. Plot points colored by digit.
5. Find how many components explain 95 percent variance.

### C. Mastery Task

Train logistic regression on:

1. Original features.
2. PCA features preserving 95 percent variance.
3. PCA features with only 2 components.

Compare speed and accuracy.

## Problemset 12: t-SNE And UMAP

### A. Concept Questions

1. Why are t-SNE and UMAP mostly visualization tools?
2. Why should you not over-interpret distances between far-away clusters in t-SNE?
3. Why might you run PCA before t-SNE?
4. What can random seed change?
5. What does local neighborhood preservation mean?

### B. Coding Drills

1. Use digits dataset.
2. Run PCA to 30 dimensions.
3. Run t-SNE to 2D.
4. Plot.
5. Change perplexity and random seed. Compare.

### C. Optional UMAP Task

Install `umap-learn` if available.

1. Run UMAP on digits.
2. Compare plot with t-SNE.
3. Explain which visualization seems more useful and why.

## Problemset 13: DBSCAN

### A. Concept Questions

1. What is a core point?
2. What does `eps` mean?
3. What does `min_samples` mean?
4. Why can DBSCAN detect outliers?
5. Why does DBSCAN struggle with varying density?

### B. Coding Drills

1. Generate `make_moons`.
2. Fit K-means.
3. Fit DBSCAN.
4. Compare cluster shapes.
5. Tune `eps`.

### C. Mastery Task

Create a dataset with:

- Two dense clusters.
- One sparse cluster.
- Random noise points.

Try DBSCAN. Explain what it succeeds at and what it fails at.

## Problemset 14: Hierarchical And Spectral Clustering

### A. Concept Questions

1. What is agglomerative clustering?
2. What is a dendrogram?
3. Difference between single, complete, average, and Ward linkage?
4. Why can spectral clustering solve non-convex clusters?
5. Why is spectral clustering more expensive?

### B. Coding Drills

1. Generate blobs.
2. Run agglomerative clustering with different linkage methods.
3. Compare cluster assignments.
4. Generate moons.
5. Compare K-means, DBSCAN, and spectral clustering.

### C. Mastery Task

Write a one-page comparison:

```text
K-means vs DBSCAN vs hierarchical vs spectral
```

Include:

- Assumptions.
- Hyperparameters.
- Strengths.
- Weaknesses.
- Best use cases.

## Problemset 15: Evaluation Metrics

### A. Hand Confusion Matrix

Given:

```text
TP = 40
FP = 10
FN = 20
TN = 130
```

Compute:

1. Accuracy.
2. Precision.
3. Recall.
4. F1.
5. False positive rate.

Then answer:

1. Is this model better at avoiding false positives or false negatives?
2. If this is cancer screening, what would you improve?
3. If this is fraud accusation, what would you improve?

### B. Threshold Task

Train logistic regression and get predicted probabilities.

Evaluate thresholds:

```text
0.1, 0.2, 0.3, ..., 0.9
```

For each threshold, compute:

- Precision.
- Recall.
- F1.

Choose a threshold for:

1. High recall.
2. High precision.
3. Balanced F1.

### C. ROC Task

1. Plot ROC curve.
2. Compute ROC AUC.
3. Plot precision-recall curve.
4. Explain which curve is more useful if positives are rare.

## Problemset 16: Cross-Validation And Tuning

### A. Concept Questions

1. Why is one train/test split sometimes unreliable?
2. What is 5-fold cross-validation?
3. Why use stratified folds for classification?
4. What is nested cross-validation?
5. Why should preprocessing be inside the pipeline during CV?

### B. Coding Drills

1. Create a pipeline with scaler and SVM.
2. Use `cross_val_score`.
3. Use `GridSearchCV`.
4. Tune `C` and `gamma`.
5. Report mean and standard deviation of CV scores.

### C. Mastery Task

Build a comparison table:

```text
model | best params | CV mean | CV std | test score | notes
```

Models:

- Logistic regression.
- KNN.
- Random forest.
- Gradient boosting.
- SVM.

## Problemset 17: Full Mini-Competition

Choose one dataset:

- Iris: beginner multiclass classification.
- Breast Cancer Wisconsin: binary classification.
- Wine: multiclass classification.
- Wine Quality: regression or classification.
- Adult: tabular classification with categorical features.
- Bank Marketing: imbalanced classification.
- Heart Disease: medical classification.

Rules:

1. Create a clean notebook or script.
2. Define the prediction problem.
3. Perform EDA.
4. Split data correctly.
5. Build at least 5 models.
6. Use pipelines.
7. Use cross-validation.
8. Tune at least 2 models.
9. Pick final model.
10. Evaluate once on test set.
11. Write error analysis.
12. Write final model card.

Model card template:

```text
Task:
Dataset:
Target:
Features:
Train/validation/test strategy:
Models tried:
Final model:
Metric:
Performance:
Most important errors:
Risks:
How to improve:
```

## 27. Online Practice Map

Use these as external reps.

### Google ML Crash Course

Best sections:

- Linear regression.
- Logistic regression.
- Classification.
- Regularization.
- Training and test sets.
- Generalization.
- Validation.
- Numerical data.
- Categorical data.

Link: https://developers.google.com/machine-learning/crash-course

### scikit-learn User Guide

Best sections:

- Supervised learning.
- Model selection and evaluation.
- Dataset transformations.
- Clustering.
- Decomposition.
- Pipelines and composite estimators.

Link: https://scikit-learn.org/stable/user_guide.html

### scikit-learn MOOC

Use this for structured coding practice:

- Predictive modeling pipeline.
- Model evaluation.
- Linear models.
- Decision tree models.
- Ensemble models.

Link: https://inria.github.io/scikit-learn-mooc/

### ISLR

Use this for theory and deeper statistical understanding:

- Chapter 2: statistical learning.
- Chapter 3: linear regression.
- Chapter 4: classification.
- Chapter 5: resampling methods.
- Chapter 6: linear model selection and regularization.
- Chapter 8: tree-based methods.
- Chapter 10: unsupervised learning.

Link: https://www.statlearning.com/

### StatQuest

Use this when a formula feels mysterious. Search the site/videos for:

- Logistic regression.
- ROC and AUC.
- Cross-validation.
- Regularization.
- Random forest.
- Gradient boosting.
- PCA.
- K-means.
- SVM.

Link: https://statquest.org/

## 28. Mastery Schedule

### Week 1: Workflow, Data, Linear Regression

Study:

- ML workflow.
- Train/test/validation.
- Leakage.
- Scaling.
- Linear regression.
- MAE, MSE, RMSE, R2.

Do:

- Problemsets 1 and 2.
- One regression dataset.

Mastery check:

- You can explain leakage.
- You can build a regression pipeline.
- You can interpret coefficients and regression metrics.

### Week 2: Classification, Logistic Regression, Metrics

Study:

- Logistic regression.
- Confusion matrix.
- Accuracy, precision, recall, F1.
- Thresholds.
- ROC and PR curves.

Do:

- Problemsets 3 and 15.
- Breast cancer classification.

Mastery check:

- You can choose a metric for imbalanced classification.
- You can tune a decision threshold.

### Week 3: Regularization, KNN, SVM

Study:

- L1/L2.
- KNN and scaling.
- SVM margin, C, gamma.

Do:

- Problemsets 4, 5, and 9.

Mastery check:

- You can explain why scaling matters.
- You can tune regularization and SVM parameters.

### Week 4: Trees And Ensembles

Study:

- Decision trees.
- Bagging.
- Random forests.
- Gradient boosting.

Do:

- Problemsets 6, 7, and 8.

Mastery check:

- You can diagnose overfit trees.
- You can compare forest vs boosting.
- You can tune tree depth and leaf size.

### Week 5: Unsupervised Learning

Study:

- K-means.
- PCA.
- t-SNE/UMAP.
- DBSCAN.
- Hierarchical and spectral clustering.

Do:

- Problemsets 10, 11, 12, 13, and 14.

Mastery check:

- You can choose a clustering method based on shape and density.
- You can use PCA for compression/visualization.
- You do not over-interpret t-SNE/UMAP plots.

### Week 6: Full Project And Review

Study:

- Cross-validation.
- Hyperparameter tuning.
- Model comparison.
- Error analysis.

Do:

- Problemsets 16 and 17.
- One full mini-competition.

Mastery check:

- You can produce a clean model report.
- You can defend metric choice.
- You can explain final model strengths and weaknesses.

## 29. What Mastery Looks Like

You are comfortable with classical ML when you can do all of this without copying blindly:

1. Load a dataset and identify target/features.
2. Split data correctly.
3. Build preprocessing pipelines.
4. Choose metrics based on task.
5. Train linear/logistic baselines.
6. Train KNN, trees, forests, boosting, and SVM.
7. Tune hyperparameters with cross-validation.
8. Detect underfitting and overfitting from train/validation scores.
9. Explain confusion matrix tradeoffs.
10. Interpret coefficients or feature importances cautiously.
11. Use PCA and clustering appropriately.
12. Write a short model report.

## 30. Final Exam

Do this only after finishing the problemsets.

### Part 1: Theory

Answer in writing:

1. Compare linear regression and logistic regression.
2. Explain L1 vs L2 regularization.
3. Explain why KNN suffers in high dimensions.
4. Explain why a single decision tree overfits.
5. Explain how random forests reduce variance.
6. Explain how boosting differs from bagging.
7. Explain SVM margin and kernel trick.
8. Explain K-means objective and limitations.
9. Explain PCA and explained variance.
10. Explain DBSCAN and what `eps` means.
11. Explain precision vs recall using a medical example.
12. Explain why cross-validation must include preprocessing inside the pipeline.

### Part 2: Hand Calculations

1. Compute MAE, MSE, RMSE for a small regression example.
2. Compute precision, recall, F1 from a confusion matrix.
3. Compute one KNN prediction by hand.
4. Compute Gini impurity before and after a split.
5. Compute one K-means iteration by hand.
6. Compute simple PCA intuition for points mostly along a line.

### Part 3: Coding

Use one unseen dataset.

Required:

1. EDA.
2. Proper split.
3. Baseline.
4. At least 5 models.
5. Cross-validation.
6. Hyperparameter tuning.
7. Final test evaluation.
8. Error analysis.
9. Final report.

Passing standard:

- You can explain every line of your notebook/script.
- You know why each model was tried.
- You can justify the final metric.
- You can explain what failed and what improved.
- You do not leak test data.

## 31. Short Answer Key For Selected Hand Problems

Use this after attempting the problems.

### Problemset 2, A2

True: `[3, 5, 9]`

Pred: `[2, 5, 10]`

Errors:

```text
1, 0, -1
```

Absolute errors:

```text
1, 0, 1
```

MAE:

```text
2/3
```

Squared errors:

```text
1, 0, 1
```

MSE:

```text
2/3
```

RMSE:

```text
sqrt(2/3)
```

### Problemset 5, A

Query `(0.8, 0.2)`.

Distances:

```text
to A(0,0): sqrt(0.8^2 + 0.2^2) = sqrt(0.68)
to B(1,0): sqrt(0.2^2 + 0.2^2) = sqrt(0.08)
to C(0,1): sqrt(0.8^2 + (-0.8)^2) = sqrt(1.28)
to D(5,5): sqrt((-4.2)^2 + (-4.8)^2) = sqrt(40.68)
```

Nearest is B, class red.

`k=1`: red.

For `k=3`, nearest are B red, A red, C blue, so red.

### Problemset 6, C

Parent:

```text
Gini = 1 - 0.6^2 - 0.4^2 = 0.48
```

Left:

```text
5 A, 1 B
Gini = 1 - (5/6)^2 - (1/6)^2 = 10/36 = 0.2778
```

Right:

```text
1 A, 3 B
Gini = 1 - (1/4)^2 - (3/4)^2 = 6/16 = 0.375
```

Weighted:

```text
(6/10)*0.2778 + (4/10)*0.375 = 0.3167
```

The split improves purity because `0.3167 < 0.48`.

### Problemset 15, A

Given:

```text
TP = 40
FP = 10
FN = 20
TN = 130
```

Total:

```text
200
```

Accuracy:

```text
(40 + 130) / 200 = 0.85
```

Precision:

```text
40 / (40 + 10) = 0.80
```

Recall:

```text
40 / (40 + 20) = 0.6667
```

F1:

```text
2 * 0.80 * 0.6667 / (0.80 + 0.6667) = 0.7273
```

False positive rate:

```text
10 / (10 + 130) = 0.0714
```

The model is better at avoiding false positives than catching all positives.

## 32. Keep A Mistake Log

Create a mistake log while studying:

```text
Date:
Problem:
Topic:
My wrong idea:
Correct idea:
How to recognize next time:
One similar problem to retry:
```

Most people do not master ML because they only read. Mastery comes from:

- Predicting before running code.
- Computing small examples by hand.
- Comparing models.
- Explaining errors.
- Repeating the full workflow until it feels boring.

That boredom is a good sign: it means the workflow is becoming automatic.
