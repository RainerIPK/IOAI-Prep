# Neural Networks And Deep Learning Mastery

This is a dense, self-contained course for learning neural networks and deep learning for AI contests, IOAI preparation, and practical PyTorch work.

It covers:

- Theory: perceptrons, activation functions, loss functions, gradient descent, backpropagation, MLPs, embeddings, pooling, attention, transformers.
- Practice: PyTorch tensors, autograd, modules, datasets, dataloaders, training loops, SGD, mini-batches, Adam, AdamW, learning rates, validation, early stopping, dropout, weight decay, initialization, batch normalization, autoencoders, fine-tuning, and parameter-efficient fine-tuning.
- Mastery: hand calculations, from-scratch NumPy drills, PyTorch drills, debugging drills, and final projects.

Verified resource map:

- IOAI Module 3: https://ioai.toki.id/assets/modul/Modul_3_IOAI__Pengenalan_Deep_Learning.pdf
- Dive into Deep Learning: https://d2l.ai/
- fast.ai Practical Deep Learning: https://course.fast.ai/
- Deep Learning with Python chapters: https://deeplearningwithpython.io/chapters/
- PyTorch Tutorials: https://docs.pytorch.org/tutorials/
- PyTorch `torch.nn`: https://pytorch.org/docs/stable/nn.html
- PyTorch optimizers: https://pytorch.org/docs/stable/optim.html

## 0. How To Use This File

Deep learning has three layers of understanding:

1. Math: tensors, matrix multiplication, derivatives, gradients, losses.
2. Model design: layers, activations, architectures, regularization.
3. Training discipline: data splits, batching, optimization, validation, debugging.

You need all three.

Use this cycle:

```text
read concept -> compute tiny example -> implement from scratch -> implement in PyTorch -> debug failure -> explain result
```

Do not only watch videos. Do not only copy notebooks. A deep learning practitioner must be able to answer:

- What shape is this tensor?
- What does this layer compute?
- What loss is being minimized?
- Which parameters receive gradients?
- Is the model underfitting or overfitting?
- Is the learning rate too high or too low?
- Does validation loss agree with training loss?
- What baseline does this beat?

## 1. What Deep Learning Is

Classical machine learning often depends on manually chosen features.

Example:

```text
Image -> edges, corners, color histogram -> classifier
```

Deep learning learns representations automatically:

```text
Image pixels -> low-level edges -> textures -> parts -> object class
Text tokens -> embeddings -> syntax/semantics -> task output
Audio waveform -> local patterns -> phonemes/sounds -> transcription/class
```

A neural network is a function with many parameters:

```text
y_hat = f(x; theta)
```

Training means finding parameters `theta` that minimize a loss:

```text
theta_new = theta_old - learning_rate * gradient
```

The basic loop:

1. Feed input through model.
2. Compute prediction.
3. Compute loss.
4. Backpropagate gradients.
5. Update parameters.
6. Repeat.

## 2. Tensors

Deep learning frameworks operate on tensors.

A tensor is a multidimensional array.

```text
Scalar:       shape []
Vector:       shape [d]
Matrix:       shape [rows, cols]
Image batch:  shape [batch, channels, height, width]
Text batch:   shape [batch, sequence_length]
Embedding:    shape [batch, sequence_length, embedding_dim]
```

PyTorch:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
```

Always know tensor shape. Shape confusion is the most common beginner bug.

PyTorch shape tools:

```python
x.shape
x.ndim
x.dtype
x.device
x.reshape(...)
x.view(...)
x.unsqueeze(dim)
x.squeeze(dim)
x.permute(...)
```

### Tensor Dtypes

Common dtypes:

```text
torch.float32: neural network weights and real-valued inputs
torch.float16 / bfloat16: faster mixed precision on supported hardware
torch.long: class labels and token IDs
torch.bool: masks
```

Important:

`nn.CrossEntropyLoss` expects class labels as integer class IDs, usually `torch.long`, not one-hot vectors.

### Devices

Tensors and models must be on the same device.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
xb = xb.to(device)
yb = yb.to(device)
```

If data is on CPU and model is on GPU, PyTorch will complain.

## 3. Perceptron

A perceptron is the simplest neural unit.

It computes:

```text
z = w dot x + b
output = step(z)
```

Where:

- `x` is input vector.
- `w` is weight vector.
- `b` is bias.
- `step(z)` returns 1 if `z >= 0`, else 0.

Example:

```text
x = [2, 3]
w = [4, -1]
b = -2

z = 4*2 + (-1)*3 - 2 = 3
output = 1
```

### Perceptron Learning Rule

For binary labels `0` or `1`:

```text
prediction = step(w dot x + b)
error = y - prediction
w = w + learning_rate * error * x
b = b + learning_rate * error
```

If prediction is correct, error is 0 and nothing changes.

If prediction is too low, weights move toward the input.

If prediction is too high, weights move away from the input.

### Limitation

A single perceptron can only solve linearly separable problems.

It can solve AND and OR.

It cannot solve XOR.

XOR table:

```text
x1 x2 | y
0  0  | 0
0  1  | 1
1  0  | 1
1  1  | 0
```

There is no single straight line that separates positives from negatives.

This motivates multi-layer neural networks.

## 4. Activations

If every layer is linear, stacking layers still gives one linear function.

Example:

```text
Linear(linear(x)) = still linear
```

Activation functions add nonlinearity.

Without nonlinear activations, deep networks cannot model complex patterns.

## 5. Sigmoid

Formula:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

Range:

```text
0 to 1
```

Use:

- Binary probability output.
- Older hidden layers, though less common now.

Derivative:

```text
sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
```

Problem:

- Saturates near 0 and 1.
- Gradients become tiny for large positive/negative inputs.
- Can cause vanishing gradients.

## 6. Tanh

Formula:

```text
tanh(x)
```

Range:

```text
-1 to 1
```

Compared with sigmoid:

- Zero-centered.
- Still saturates.

Use:

- Older recurrent networks.
- Some bounded hidden states.

## 7. ReLU

Formula:

```text
ReLU(x) = max(0, x)
```

Range:

```text
0 to infinity
```

Why it is popular:

- Simple.
- Cheap.
- Does not saturate for positive values.
- Helps deep networks train.

Derivative:

```text
0 if x < 0
1 if x > 0
```

Problem:

- Dead ReLUs: if a unit always receives negative input, gradient is zero and it may stop learning.

Variants:

- LeakyReLU.
- GELU.
- ELU.

Modern transformer models often use GELU or variants, but ReLU remains essential to understand.

## 8. Loss Functions

A loss function measures how wrong the model is.

Training minimizes loss.

### MSE

Mean squared error:

```text
MSE = (1/n) * sum((y - y_hat)^2)
```

Use:

- Regression.
- Autoencoder reconstruction.

Properties:

- Smooth.
- Punishes large errors heavily.

PyTorch:

```python
loss_fn = torch.nn.MSELoss()
```

### MAE

Mean absolute error:

```text
MAE = (1/n) * sum(abs(y - y_hat))
```

Use:

- Regression when outliers should not dominate as much.

PyTorch:

```python
loss_fn = torch.nn.L1Loss()
```

### Binary Cross Entropy

For binary classification:

```text
loss = -[y log(p) + (1-y) log(1-p)]
```

Use:

- Binary output probability.
- Multi-label classification where each label is independent.

Numerically stable PyTorch pattern:

```python
loss_fn = torch.nn.BCEWithLogitsLoss()
```

This expects raw logits, not sigmoid probabilities.

Bad pattern:

```python
p = torch.sigmoid(logits)
loss = BCELoss(p, y)
```

Better:

```python
loss = BCEWithLogitsLoss(logits, y)
```

### Multiclass Cross Entropy

For one correct class among `C` classes.

PyTorch:

```python
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

Important:

- `logits` shape: `[batch, num_classes]`
- `labels` shape: `[batch]`
- `labels` dtype: `torch.long`
- Do not apply softmax before `CrossEntropyLoss`

`CrossEntropyLoss` internally combines log-softmax and negative log-likelihood.

## 9. Gradient Descent

Gradient descent updates parameters opposite the gradient.

```text
theta = theta - learning_rate * gradient
```

Gradient points uphill.

Negative gradient points downhill.

### Batch Gradient Descent

Uses all training examples for each update.

Pros:

- Stable gradient.

Cons:

- Slow for large datasets.

### Stochastic Gradient Descent

Uses one training example per update.

Pros:

- Frequent updates.
- Can escape some shallow traps.

Cons:

- Noisy.

### Mini-Batch Gradient Descent

Uses a small batch, usually:

```text
32, 64, 128, 256
```

This is the standard deep learning approach.

Pros:

- Efficient on GPUs.
- Less noisy than pure SGD.
- More scalable than full-batch.

## 10. Backpropagation

Backpropagation computes gradients efficiently using the chain rule.

For a network:

```text
x -> layer1 -> activation -> layer2 -> loss
```

Backpropagation works backward:

```text
dLoss/dLayer2
dLoss/dActivation
dLoss/dLayer1
dLoss/dWeights
```

You do not manually write backprop for normal PyTorch models. Autograd does it.

But you must understand:

1. The forward pass builds a computation graph.
2. The loss is a scalar.
3. `loss.backward()` computes gradients for parameters that require gradients.
4. The optimizer uses those gradients to update parameters.
5. Gradients accumulate unless you clear them.

PyTorch pattern:

```python
optimizer.zero_grad()
pred = model(xb)
loss = loss_fn(pred, yb)
loss.backward()
optimizer.step()
```

If you forget `zero_grad()`, gradients accumulate across batches.

## 11. Multi-Layer Perceptrons

An MLP is a stack of linear layers and nonlinear activations.

Example:

```text
input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> output
```

PyTorch:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 3)
)
```

For classification with 3 classes, output has 3 logits.

Do not put softmax at the end if using `CrossEntropyLoss`.

### Universal Approximation

An MLP with enough hidden units can approximate many functions.

But:

- It may need many parameters.
- It may be hard to train.
- Architecture matters.
- Data matters.
- Optimization matters.

## 12. Data Embeddings

An embedding maps an object to a vector.

Examples:

```text
word -> vector
image patch -> vector
audio segment -> vector
user -> vector
product -> vector
```

Embeddings let neural networks represent meaning geometrically.

Similar objects should have nearby vectors.

### Embedding Layer

For token IDs:

```python
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=128)
token_vectors = embedding(token_ids)
```

If:

```text
token_ids shape = [batch, sequence_length]
```

then:

```text
token_vectors shape = [batch, sequence_length, embedding_dim]
```

An embedding layer is just a learnable lookup table.

### Embeddings For Images And Audio

Images:

- CNN feature vectors.
- Vision transformer patch embeddings.

Audio:

- Spectrogram patches.
- Learned audio encoder outputs.

Text:

- Token embeddings.
- Contextual transformer embeddings.

## 13. Pooling

Pooling reduces spatial or sequence dimensions.

### Max Pooling

Takes maximum over a region.

Example:

```text
[1, 3, 2, 0] -> max = 3
```

In CNNs:

- Reduces height and width.
- Keeps strongest local response.
- Adds some translation robustness.

PyTorch:

```python
nn.MaxPool2d(kernel_size=2)
```

### Average Pooling

Takes average over a region.

PyTorch:

```python
nn.AvgPool2d(kernel_size=2)
```

### Global Average Pooling

Averages over all spatial positions.

Often used before final classifier in CNNs.

Shape:

```text
[batch, channels, height, width] -> [batch, channels]
```

## 14. Convolutional Neural Networks

CNNs are designed for grid-like data such as images.

Key ideas:

- Local receptive fields.
- Shared weights.
- Translation equivariance.
- Pooling or strided convolution for downsampling.

Convolution layer:

```python
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
```

Input image batch:

```text
[batch, channels, height, width]
```

Output:

```text
[batch, out_channels, new_height, new_width]
```

Output size formula for one dimension:

```text
out = floor((in + 2*padding - dilation*(kernel - 1) - 1) / stride + 1)
```

For common case dilation 1:

```text
out = floor((in + 2*padding - kernel) / stride + 1)
```

Example:

```text
in = 32
kernel = 3
padding = 1
stride = 1

out = floor((32 + 2 - 3) / 1 + 1) = 32
```

## 15. Attention

Attention lets a model choose which parts of the input to focus on.

For each token:

- Query: what am I looking for?
- Key: what information do I contain?
- Value: what information should I pass along?

Scaled dot-product attention:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Interpretation:

1. Compare queries with keys using dot products.
2. Scale by `sqrt(d_k)` for stability.
3. Softmax to get attention weights.
4. Weighted average of values.

### Self-Attention

In self-attention, Q, K, and V all come from the same sequence.

Each token can attend to other tokens.

This is powerful for:

- Long-range dependencies.
- Language.
- Vision patches.
- Audio sequences.

### Multi-Head Attention

Instead of one attention operation, use multiple heads.

Each head can learn a different relation.

Examples:

- One head tracks subject-verb relation.
- One head tracks nearby tokens.
- One head tracks punctuation or boundaries.

Then heads are concatenated and projected.

## 16. Transformers

A transformer block usually contains:

1. Multi-head self-attention.
2. Residual connection.
3. Layer normalization.
4. Feed-forward network.
5. Residual connection.
6. Layer normalization.

Text transformer input:

```text
token IDs -> token embeddings + positional embeddings -> transformer blocks
```

Vision transformer input:

```text
image -> patches -> patch embeddings + positional embeddings -> transformer blocks
```

### Positional Information

Self-attention alone does not know token order.

Transformers add position information through:

- Learned positional embeddings.
- Sinusoidal positional encodings.
- Rotary position embeddings in many modern models.

For mastery, understand the concept:

```text
Tokens need content information plus position information.
```

### Encoder vs Decoder

Encoder:

- Reads entire input.
- Good for classification, embeddings, understanding.
- Example: BERT-style.

Decoder:

- Generates output autoregressively.
- Uses causal mask so token cannot see future tokens.
- Example: GPT-style.

Encoder-decoder:

- Encoder reads input.
- Decoder generates output.
- Example: translation, summarization.

## 17. PyTorch Core Concepts

### Tensor With Gradient

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)
```

This computes:

```text
dy/dx = 2x + 3
at x = 2 -> 7
```

### Module

Custom model:

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
```

`forward` defines computation.

Calling `model(x)` invokes `forward`.

### Parameters

```python
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
```

Only parameters with `requires_grad=True` are updated by optimizers.

### Training vs Evaluation Mode

```python
model.train()
model.eval()
```

This matters for:

- Dropout.
- Batch normalization.

During validation/inference:

```python
model.eval()
with torch.no_grad():
    pred = model(x)
```

`torch.no_grad()` saves memory and avoids building a gradient graph.

## 18. Datasets And Dataloaders

A dataset returns one example.

A dataloader batches examples.

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
```

Training data should usually be shuffled.

Validation/test data usually does not need shuffling.

## 19. Training Loop Template

Use this until it is muscle memory.

```python
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == yb).sum().item()
        total_count += xb.size(0)

    return total_loss / total_count, total_correct / total_count
```

Validation:

```python
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == yb).sum().item()
        total_count += xb.size(0)

    return total_loss / total_count, total_correct / total_count
```

Full loop:

```python
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
    print(epoch, train_loss, train_acc, val_loss, val_acc)
```

## 20. Optimizers

### SGD

Stochastic gradient descent:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

With momentum:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Momentum smooths updates by accumulating velocity.

### Adam

Adam adapts learning rates per parameter using estimates of first and second moments of gradients.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Often good default for many problems.

### AdamW

AdamW decouples weight decay from Adam's adaptive update.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

Often preferred for transformers and modern deep learning.

### Practical Defaults

MLP from scratch:

```text
Adam or AdamW, lr 1e-3
```

CNN:

```text
AdamW lr 1e-3 or SGD momentum lr 1e-2 to 1e-1 depending on setup
```

Fine-tuning pretrained model:

```text
AdamW, lr 1e-5 to 5e-5 for transformer-like models
```

These are starting points, not laws.

## 21. Learning Rates And Convergence

The learning rate is often the most important hyperparameter.

Too low:

```text
loss decreases slowly
training appears stuck but stable
```

Too high:

```text
loss jumps, diverges, or becomes NaN
validation unstable
```

Good:

```text
training loss decreases steadily
validation loss improves then plateaus
```

### Learning Rate Schedules

Common schedules:

- Step decay.
- Cosine decay.
- One-cycle policy.
- Warmup then decay.

Warmup is common for transformers:

```text
start with small lr -> increase -> decay
```

## 22. Regularization

Regularization reduces overfitting.

### Dropout

Dropout randomly zeroes activations during training.

```python
nn.Dropout(p=0.5)
```

Effect:

- Forces network not to depend too much on one unit.
- Acts like noisy ensemble.

Important:

- Dropout active in `model.train()`.
- Dropout disabled in `model.eval()`.

### Weight Decay

Weight decay penalizes large weights.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

Effect:

- Encourages simpler solutions.
- Often improves generalization.

### Early Stopping

Stop training when validation loss stops improving.

Basic logic:

```text
best_val_loss = infinity
patience = 5
if val_loss improves: save model
else: patience_counter += 1
if patience_counter >= patience: stop
```

Early stopping is simple and powerful.

### Data Augmentation

For images:

- Random crop.
- Flip.
- Color jitter.
- Rotation.

For text:

- Usually more delicate.
- Backtranslation or masking-style methods.

For audio:

- Noise.
- Time masking.
- Frequency masking.

Augmentation increases effective training diversity.

## 23. Weight Initialization

Bad initialization can kill learning.

If weights are too small:

- Signals vanish.

If weights are too large:

- Activations explode.

Common initializations:

- Xavier/Glorot: good for tanh/sigmoid-ish networks.
- Kaiming/He: good for ReLU networks.

PyTorch layers have reasonable defaults, but you should know the principle.

Manual initialization:

```python
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
```

## 24. Batch Normalization

Batch normalization normalizes activations within a mini-batch.

It can:

- Stabilize training.
- Allow higher learning rates.
- Reduce sensitivity to initialization.
- Add mild regularization.

PyTorch:

```python
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_channels)
```

Important:

- Behavior differs in train/eval mode.
- During training, uses batch statistics.
- During evaluation, uses running statistics.

## 25. Layer Normalization

Layer normalization normalizes across features for each example.

Common in transformers.

```python
nn.LayerNorm(normalized_shape)
```

Difference:

- BatchNorm depends on batch statistics.
- LayerNorm works per example.

Transformers typically use LayerNorm, not BatchNorm.

## 26. Autoencoders

An autoencoder learns to reconstruct its input.

Architecture:

```text
input -> encoder -> latent code -> decoder -> reconstruction
```

Loss:

```text
reconstruction loss
```

Uses:

- Dimensionality reduction.
- Denoising.
- Anomaly detection.
- Representation learning.

Simple PyTorch:

```python
class Autoencoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
```

## 27. Fine-Tuning

Fine-tuning starts from a pretrained model and adapts it to a new task.

Why it works:

- Pretrained model has learned general representations.
- Your task often needs less data.
- Training is faster than from scratch.

Common workflow:

1. Load pretrained model.
2. Replace final head.
3. Freeze most layers.
4. Train head.
5. Optionally unfreeze some layers.
6. Fine-tune with small learning rate.

### Full Fine-Tuning

Update all or most model parameters.

Pros:

- Maximum flexibility.

Cons:

- More compute.
- More memory.
- More risk of overfitting.
- Can forget pretrained knowledge.

### Parameter-Efficient Fine-Tuning

Update only a small number of parameters.

Examples:

- Linear probing: train only final classifier.
- Adapters: insert small trainable modules.
- LoRA: low-rank updates to selected weight matrices.
- Prompt/prefix tuning for language models.

For beginner mastery:

1. Understand linear probing.
2. Understand freezing/unfreezing.
3. Understand LoRA conceptually.
4. Implement LoRA later if needed.

## 28. Debugging Neural Networks

Most deep learning work is debugging.

### Sanity Checks

1. Can the model overfit a tiny batch?
2. Are labels correct?
3. Are input shapes correct?
4. Is loss decreasing?
5. Are gradients nonzero?
6. Are gradients exploding?
7. Is validation preprocessing same as training preprocessing, minus augmentation?
8. Is model in train mode for training and eval mode for validation?
9. Are you applying softmax twice?
10. Are labels the right dtype?

### Overfit One Batch Test

Take one batch of 32 examples.

Train on only that batch for many iterations.

Expected:

```text
training loss should go near zero
training accuracy should go near 100 percent
```

If not:

- Model may be wrong.
- Loss may be wrong.
- Labels may be wrong.
- Learning rate may be wrong.
- Gradients may not flow.

### Common Bugs

Bug:

```text
Using softmax before CrossEntropyLoss
```

Fix:

```text
Pass raw logits to CrossEntropyLoss
```

Bug:

```text
Forgetting optimizer.zero_grad()
```

Fix:

```text
Call optimizer.zero_grad() before backward
```

Bug:

```text
Using model.train() during validation
```

Fix:

```text
Use model.eval() and torch.no_grad()
```

Bug:

```text
Target dtype float for CrossEntropyLoss
```

Fix:

```text
Use long integer class labels
```

Bug:

```text
Learning rate too high
```

Fix:

```text
Lower lr by 10x
```

## 29. Reading Loss Curves

### Good Learning

```text
train loss decreases
validation loss decreases
validation accuracy improves
```

### Overfitting

```text
train loss keeps decreasing
validation loss starts increasing
```

Fix:

- More data.
- Data augmentation.
- Dropout.
- Weight decay.
- Smaller model.
- Early stopping.

### Underfitting

```text
train loss high
validation loss high
```

Fix:

- Larger model.
- Train longer.
- Better features/input.
- Lower regularization.
- Better optimizer/lr.

### Learning Rate Too High

```text
loss jumps wildly
loss becomes NaN
accuracy unstable
```

Fix:

- Lower learning rate.
- Add gradient clipping for RNN/transformer-like models.

### Learning Rate Too Low

```text
loss decreases extremely slowly
```

Fix:

- Increase learning rate.
- Use Adam/AdamW.
- Use scheduler.

## 30. Minimal PyTorch MLP Project

This template trains an MLP on tabular tensors.

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

X_train = torch.randn(1000, 20)
y_train = torch.randint(0, 3, (1000,))
X_val = torch.randn(200, 20)
y_val = torch.randint(0, 3, (200,))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

for epoch in range(10):
    model.train()
    train_loss = 0
    train_correct = 0
    train_count = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)
        train_correct += (logits.argmax(dim=1) == yb).sum().item()
        train_count += xb.size(0)

    model.eval()
    val_loss = 0
    val_correct = 0
    val_count = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            val_loss += loss.item() * xb.size(0)
            val_correct += (logits.argmax(dim=1) == yb).sum().item()
            val_count += xb.size(0)

    print(
        epoch,
        train_loss / train_count,
        train_correct / train_count,
        val_loss / val_count,
        val_correct / val_count
    )
```

This example uses random data, so accuracy will not become meaningful. Replace with real data for practice.

## 31. Problemsets Overview

Do the problemsets in order.

Levels:

```text
A: hand calculation
B: NumPy/from-scratch implementation
C: PyTorch implementation
D: debugging/explanation/project
```

Mastery means:

- You know why the code works.
- You can identify tensor shapes.
- You can fix training failures.
- You can write the loop without copying.

## Problemset 1: Tensors And Shapes

### A. Shape Reasoning

For each tensor, write what it could represent:

1. `[32, 10]`
2. `[64, 3, 224, 224]`
3. `[16, 128]`
4. `[8, 20, 768]`
5. `[1000]`

For each operation, give output shape:

1. `X [32, 20] @ W [20, 10]`
2. `A [4, 5, 6].mean(dim=1)`
3. `x [32].unsqueeze(1)`
4. `images [64, 3, 32, 32].flatten(start_dim=1)`
5. `emb [8, 20, 128].mean(dim=1)`

### B. PyTorch Drills

Create tensors and perform:

1. Matrix multiplication.
2. Reshape.
3. Unsqueeze and squeeze.
4. Move tensor to device.
5. Convert labels to `torch.long`.

### C. Debugging

Create a wrong matrix multiplication shape. Read the error message and explain it.

## Problemset 2: Perceptron

### A. Hand Computation

Given:

```text
x = [1, 2]
w = [3, -1]
b = -1
```

Compute:

1. `z = w dot x + b`
2. `step(z)`

Now use:

```text
y = 0
learning_rate = 0.1
```

Apply one perceptron update.

### B. Logic Gates

Find perceptron weights for:

1. AND.
2. OR.
3. NOT.

Explain why XOR cannot be solved by one perceptron.

### C. Code

Implement a perceptron from scratch in Python for AND and OR.

## Problemset 3: Activations

### A. Hand Computation

Compute sigmoid approximately:

1. `sigmoid(0)`
2. `sigmoid(2)`
3. `sigmoid(-2)`

Compute ReLU:

1. `ReLU(-3)`
2. `ReLU(0)`
3. `ReLU(5)`

Compute tanh signs:

1. `tanh(-10)`
2. `tanh(0)`
3. `tanh(10)`

### B. Plotting

Use Python to plot:

- Sigmoid.
- Tanh.
- ReLU.
- LeakyReLU.

Explain saturation.

### C. Gradient Intuition

For sigmoid, inspect gradient near:

```text
x = -10, 0, 10
```

Explain why this can slow learning.

## Problemset 4: Loss Functions

### A. Regression Losses

True values:

```text
y = [2, 4, 6]
pred = [1, 5, 10]
```

Compute:

1. MAE.
2. MSE.
3. RMSE.

### B. Cross Entropy

For true class 2 and logits:

```text
[1.0, 0.5, 3.0]
```

Compute softmax probabilities.

Then compute negative log probability of true class.

### C. PyTorch

Use:

```python
nn.MSELoss()
nn.L1Loss()
nn.CrossEntropyLoss()
nn.BCEWithLogitsLoss()
```

Create small tensors and verify expected shapes.

### D. Common Mistake

Show what happens if you pass one-hot labels into `CrossEntropyLoss`. Fix it.

## Problemset 5: Autograd

### A. Hand Gradient

For:

```text
y = x^2 + 3x
```

Compute derivative and evaluate at `x = 2`.

### B. PyTorch

Use autograd to compute the same derivative.

### C. Two Variables

For:

```text
z = x^2 + xy + y^2
```

Compute gradients by hand and with PyTorch.

### D. Gradient Accumulation

Call `.backward()` twice without zeroing or clearing gradients.

Observe what happens.

Explain why optimizers call `zero_grad()`.

## Problemset 6: Gradient Descent

### A. Hand Steps

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

### B. Learning Rate Experiment

Repeat with:

```text
0.01, 0.1, 0.5, 1.1
```

Describe behavior.

### C. PyTorch Parameter

Use a tensor `w` with `requires_grad=True`.

Minimize `(w - 4)^2` manually with:

```python
with torch.no_grad():
    w -= lr * w.grad
    w.grad.zero_()
```

## Problemset 7: Backpropagation By Hand

### A. Tiny Network

Network:

```text
x -> z = wx + b -> y_hat = z -> loss = (y_hat - y)^2
```

Given:

```text
x = 2
y = 5
w = 1
b = 0
```

Compute:

1. Forward pass.
2. Loss.
3. `dLoss/dy_hat`.
4. `dy_hat/dw`.
5. `dLoss/dw`.
6. `dLoss/db`.

### B. Add ReLU

Now:

```text
z = wx + b
h = ReLU(z)
y_hat = h
loss = (y_hat - y)^2
```

Compute gradients when:

1. `z > 0`.
2. `z < 0`.

Explain dead ReLU.

### C. PyTorch Check

Verify your gradients with autograd.

## Problemset 8: MLP For XOR

### A. Concept

Explain why XOR needs a hidden layer.

### B. PyTorch

Build an MLP:

```text
2 inputs -> 4 hidden ReLU -> 2 output logits
```

Train on XOR.

### C. Mastery

Overfit XOR to 100 percent accuracy.

Try:

- No hidden layer.
- Hidden layer with sigmoid.
- Hidden layer with ReLU.

Compare.

## Problemset 9: Training Loop

### A. Write From Memory

Write a training loop with:

```text
model.train()
optimizer.zero_grad()
forward
loss
backward
step
```

Write validation loop with:

```text
model.eval()
torch.no_grad()
```

### B. Dataset

Use scikit-learn `make_moons` or `make_classification`.

Train an MLP classifier.

### C. Track Curves

Store per epoch:

- Train loss.
- Train accuracy.
- Validation loss.
- Validation accuracy.

Plot curves.

### D. Explain

Identify whether model is underfitting, overfitting, or learning well.

## Problemset 10: PyTorch Dataset And DataLoader

### A. Custom Dataset

Create a custom `Dataset` for tensors.

### B. Dataloader

Use:

```text
batch_size = 32
shuffle = True
```

Print batch shapes.

### C. Bad Labels

Intentionally use labels with wrong dtype for `CrossEntropyLoss`.

Read error and fix it.

## Problemset 11: Optimizers

### A. Compare

Train same MLP with:

- SGD.
- SGD with momentum.
- Adam.
- AdamW.

Use same dataset and same initial seed if possible.

### B. Learning Rates

Try:

```text
1e-4, 1e-3, 1e-2, 1e-1
```

For each optimizer.

### C. Report

Make a table:

```text
optimizer | lr | best val acc | final train loss | notes
```

Explain convergence differences.

## Problemset 12: Regularization

### A. Dropout

Train an over-parameterized MLP on a small dataset.

Compare:

1. No dropout.
2. Dropout 0.2.
3. Dropout 0.5.

### B. Weight Decay

Compare:

```text
weight_decay = 0, 1e-4, 1e-2
```

### C. Early Stopping

Implement patience-based early stopping.

Save best model state.

### D. Explain

Use curves to show how regularization changed overfitting.

## Problemset 13: Initialization And BatchNorm

### A. Initialization

Build deep MLP with 8 hidden layers.

Try:

- Default initialization.
- Too-small initialization.
- Too-large initialization.
- Kaiming initialization.

Track gradient norms.

### B. BatchNorm

Add `BatchNorm1d` between linear and activation.

Compare training stability.

### C. Explain

Why can bad initialization make activations or gradients vanish/explode?

## Problemset 14: CNN And Pooling

### A. Shape Computation

Input:

```text
[batch=16, channels=3, height=32, width=32]
```

Layer:

```text
Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
```

Find output shape.

Then apply:

```text
MaxPool2d(2)
```

Find output shape.

### B. PyTorch CNN

Train a small CNN on MNIST or FashionMNIST if available.

Architecture:

```text
Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Linear
```

### C. Compare

Compare MLP vs CNN on image data.

Explain why CNN usually wins.

## Problemset 15: Embeddings

### A. Embedding Layer

Create:

```python
emb = nn.Embedding(100, 16)
ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
```

Find output shape.

### B. Text Classifier

Build a toy text classifier:

```text
token IDs -> embedding -> mean pooling -> linear classifier
```

Use synthetic token sequences.

### C. Similarity

After training, compute cosine similarity between embedding vectors.

Explain what can and cannot be inferred.

## Problemset 16: Autoencoder

### A. Build

Train an autoencoder on:

- MNIST flattened images, or
- synthetic vectors.

### B. Latent Size

Try latent dimensions:

```text
2, 8, 32, 128
```

Compare reconstruction loss.

### C. Denoising

Add noise to inputs and train model to reconstruct clean inputs.

### D. Use Case

Explain how reconstruction error can be used for anomaly detection.

## Problemset 17: Attention

### A. Hand Computation

Given:

```text
Q = [[1, 0]]
K = [[1, 0],
     [0, 1]]
V = [[10, 0],
     [0, 20]]
```

Compute:

1. `QK^T`
2. Softmax weights
3. Weighted sum of V

Ignore scaling for first attempt.

### B. Implement

Implement scaled dot-product attention in PyTorch:

```python
def attention(Q, K, V, mask=None):
    ...
```

### C. Mask

Create a causal mask.

Explain why decoder transformers need it.

## Problemset 18: Transformer Block

### A. Shapes

Input:

```text
x shape = [batch, seq_len, d_model]
```

For:

```text
d_model = 128
num_heads = 8
```

What is per-head dimension?

### B. PyTorch

Use:

```python
nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
```

Pass random input and confirm output shape.

### C. Mini Classifier

Build a tiny transformer encoder classifier for synthetic sequences.

Pipeline:

```text
token IDs -> embedding -> positional embedding -> transformer encoder -> mean pool -> classifier
```

## Problemset 19: Fine-Tuning

### A. Linear Probe

Use a pretrained image model if internet/packages are available, or simulate with a frozen feature extractor.

Steps:

1. Freeze backbone.
2. Replace head.
3. Train only head.
4. Evaluate.

### B. Partial Fine-Tuning

Unfreeze last block.

Use smaller learning rate.

Compare.

### C. Concept

Explain:

- Full fine-tuning.
- Linear probing.
- Adapters.
- LoRA.
- Why smaller learning rates are common.

## Problemset 20: Training Debugging

For each symptom, list likely causes and fixes.

1. Loss is NaN after first epoch.
2. Training loss decreases, validation loss increases.
3. Training loss does not decrease at all.
4. Accuracy is exactly random.
5. Validation accuracy is much higher than training accuracy.
6. Model works on CPU but device error happens on GPU.
7. `CrossEntropyLoss` raises dtype error.
8. Gradients are all zero.
9. Model cannot overfit one batch.
10. Results change wildly every run.

Then create one intentional bug in a small training script and fix it.

## Problemset 21: Full MLP Project

Dataset options:

- `make_moons`.
- `make_classification`.
- Iris.
- Wine.
- Breast cancer.
- Any small tabular dataset.

Requirements:

1. Build train/validation/test split.
2. Standardize features.
3. Build PyTorch Dataset/DataLoader.
4. Build MLP.
5. Train with AdamW.
6. Track train/validation loss.
7. Add dropout.
8. Add early stopping.
9. Tune hidden size and learning rate.
10. Write final report.

Report template:

```text
Task:
Dataset:
Input shape:
Output shape:
Architecture:
Loss:
Optimizer:
Best hyperparameters:
Train result:
Validation result:
Test result:
Overfitting/underfitting evidence:
Main errors:
Next improvement:
```

## Problemset 22: Full CNN Project

Dataset options:

- MNIST.
- FashionMNIST.
- CIFAR-10 if available.
- Synthetic image dataset.

Requirements:

1. Train MLP baseline.
2. Train CNN.
3. Compare parameter counts.
4. Compare validation accuracy.
5. Add dropout or weight decay.
6. Add simple augmentation if using images.
7. Save best model.
8. Load best model and evaluate test set.
9. Show 10 mistakes.
10. Explain why mistakes happened.

## Problemset 23: Full Autoencoder Project

Requirements:

1. Train autoencoder.
2. Plot reconstruction examples.
3. Try multiple latent dimensions.
4. Add noise and train denoising autoencoder.
5. Plot reconstruction loss distribution.
6. Mark high-loss examples as anomalies.
7. Explain limitations.

## Problemset 24: Final Exam

### Part A: Theory

Answer in writing:

1. What is a perceptron?
2. Why does XOR require a hidden layer?
3. Why do neural networks need nonlinear activations?
4. Compare sigmoid, tanh, and ReLU.
5. Explain vanishing gradients.
6. Explain cross entropy.
7. Explain backpropagation.
8. Explain mini-batch gradient descent.
9. Compare SGD, Adam, and AdamW.
10. Explain dropout.
11. Explain weight decay.
12. Explain batch normalization.
13. Explain embeddings.
14. Explain pooling.
15. Explain attention.
16. Explain transformers for text and images.
17. Explain autoencoders.
18. Explain full fine-tuning vs parameter-efficient fine-tuning.

### Part B: Hand Computation

1. Compute one perceptron forward pass.
2. Compute one perceptron update.
3. Compute sigmoid/ReLU/tanh outputs for simple values.
4. Compute MSE and cross entropy on tiny examples.
5. Compute gradients for one linear neuron.
6. Do three gradient descent updates by hand.
7. Compute convolution output shape.
8. Compute one attention output.

### Part C: PyTorch

Build from memory:

1. Tensor operations.
2. Custom `nn.Module`.
3. Dataset and DataLoader.
4. Training loop.
5. Validation loop.
6. Early stopping.
7. MLP classifier.
8. CNN classifier.
9. Autoencoder.
10. Tiny attention module.

### Part D: Debugging

Given a broken training script, identify:

1. Shape bug.
2. Dtype bug.
3. Device bug.
4. Loss misuse.
5. Missing `zero_grad`.
6. Missing `model.eval`.
7. Learning rate issue.
8. Overfitting or underfitting.

Passing standard:

- You can overfit one batch.
- You can explain every tensor shape.
- You can write a correct training loop.
- You can read loss curves.
- You can fix common failures.
- You can train an MLP and CNN without copying a full notebook.

## 32. External Practice Map

### IOAI Module 3

Use it for Indonesian IOAI alignment.

Focus:

- Perceptron.
- Activation functions.
- MLP and backpropagation.
- CNN.
- RNN/LSTM/GRU.
- Autoencoder.
- Transformer.
- PyTorch examples.

Link: https://ioai.toki.id/assets/modul/Modul_3_IOAI__Pengenalan_Deep_Learning.pdf

### Dive Into Deep Learning

Use it for structured theory plus implementation.

Recommended topics:

- Preliminaries.
- Linear neural networks.
- Multilayer perceptrons.
- Deep learning computation.
- Convolutional neural networks.
- Modern recurrent neural networks.
- Attention mechanisms and transformers.
- Optimization algorithms.

Link: https://d2l.ai/

### PyTorch Tutorials

Use it for official implementation practice.

Recommended:

- Learn the basics.
- 60-minute blitz.
- Learning PyTorch with examples.
- What is `torch.nn` really?
- Transfer learning for computer vision.
- Defining a neural network.
- Zeroing out gradients.

Link: https://docs.pytorch.org/tutorials/

### fast.ai

Use it when you want top-down practical training.

Focus:

- Training useful models quickly.
- Transfer learning.
- Vision and NLP examples.
- Practical experimentation.

Link: https://course.fast.ai/

### Deep Learning With Python

Use it for conceptual explanations and Keras-oriented examples.

Even if using PyTorch, the conceptual chapters are useful.

Link: https://deeplearningwithpython.io/chapters/

## 33. Eight-Week Mastery Plan

### Week 1: Tensors, Perceptron, Activations, Losses

Study:

- Tensors and shapes.
- Perceptron.
- Sigmoid, tanh, ReLU.
- MSE, MAE, cross entropy.

Do:

- Problemsets 1-4.
- PyTorch Learn the Basics.

Deliverable:

- A notebook showing tensor operations and loss functions.

### Week 2: Autograd, Gradient Descent, Backprop

Study:

- Autograd.
- Chain rule.
- Gradient descent.
- Backpropagation.

Do:

- Problemsets 5-7.
- Implement gradient descent manually.

Deliverable:

- A script comparing manual gradients and PyTorch autograd.

### Week 3: MLPs And Training Loops

Study:

- MLP.
- Dataset/DataLoader.
- Training loop.
- Validation loop.

Do:

- Problemsets 8-10.
- Train MLP on synthetic data.

Deliverable:

- Reusable PyTorch training loop.

### Week 4: Optimizers And Regularization

Study:

- SGD, momentum, Adam, AdamW.
- Learning rates.
- Dropout.
- Weight decay.
- Early stopping.

Do:

- Problemsets 11-12.

Deliverable:

- Experiment table comparing optimizers and regularization.

### Week 5: Initialization, BatchNorm, CNNs

Study:

- Initialization.
- Batch normalization.
- Convolution.
- Pooling.

Do:

- Problemsets 13-14.

Deliverable:

- CNN image classifier.

### Week 6: Embeddings, Autoencoders, Attention

Study:

- Embeddings.
- Autoencoders.
- Attention.

Do:

- Problemsets 15-17.

Deliverable:

- Autoencoder and attention implementation.

### Week 7: Transformers And Fine-Tuning

Study:

- Transformer encoder block.
- Positional embeddings.
- Linear probing.
- Fine-tuning.
- PEFT concepts.

Do:

- Problemsets 18-19.

Deliverable:

- Tiny transformer classifier or fine-tuning experiment.

### Week 8: Debugging And Final Projects

Study:

- Debugging checklist.
- Loss curve reading.
- Model reports.

Do:

- Problemsets 20-24.

Deliverable:

- Full MLP or CNN project with report.

## 34. Selected Answer Key

### Problemset 1, Shapes

```text
X [32, 20] @ W [20, 10] -> [32, 10]
A [4, 5, 6].mean(dim=1) -> [4, 6]
x [32].unsqueeze(1) -> [32, 1]
images [64, 3, 32, 32].flatten(start_dim=1) -> [64, 3072]
emb [8, 20, 128].mean(dim=1) -> [8, 128]
```

### Problemset 2, Perceptron

```text
x = [1, 2]
w = [3, -1]
b = -1

z = 3*1 + (-1)*2 - 1 = 0
step(z) = 1 if using z >= 0
```

If true `y = 0`:

```text
error = y - pred = -1
w_new = w + lr * error * x
      = [3, -1] + 0.1*(-1)*[1, 2]
      = [2.9, -1.2]
b_new = -1 + 0.1*(-1) = -1.1
```

### Problemset 3, Activations

```text
sigmoid(0) = 0.5
ReLU(-3) = 0
ReLU(0) = 0
ReLU(5) = 5
tanh(0) = 0
```

### Problemset 5, Autograd

For:

```text
y = x^2 + 3x
dy/dx = 2x + 3
at x = 2, gradient = 7
```

### Problemset 6, First Gradient Step

```text
f(w) = (w - 4)^2
f'(w) = 2(w - 4)
w0 = 0
lr = 0.25
grad = -8
w1 = 0 - 0.25*(-8) = 2
```

### Problemset 14, CNN Shape

Input:

```text
[16, 3, 32, 32]
```

Conv:

```text
Conv2d(3, 32, kernel=3, padding=1, stride=1)
```

Output height/width:

```text
floor((32 + 2*1 - 3) / 1 + 1) = 32
```

Conv output:

```text
[16, 32, 32, 32]
```

After `MaxPool2d(2)`:

```text
[16, 32, 16, 16]
```

### Problemset 17, Attention

```text
Q = [[1, 0]]
K = [[1, 0],
     [0, 1]]

QK^T = [[1, 0]]
```

Softmax over `[1, 0]`:

```text
[exp(1)/(exp(1)+exp(0)), exp(0)/(exp(1)+exp(0))]
approx [0.731, 0.269]
```

Weighted V:

```text
0.731*[10, 0] + 0.269*[0, 20]
= [7.31, 5.38]
```

## 35. Mistake Log

Use this for every failed experiment:

```text
Date:
Experiment:
Expected behavior:
Actual behavior:
Loss curve:
Train metric:
Validation metric:
Tensor shapes checked:
Bug or hypothesis:
Fix tried:
Result:
What I learned:
```

Deep learning mastery is mostly the discipline of turning mysterious failures into inspectable facts.

## 36. Final Mastery Checklist

Theory:

- I can explain perceptron learning.
- I can explain why nonlinear activations are necessary.
- I can compare sigmoid, tanh, and ReLU.
- I can compute MSE, MAE, and cross entropy.
- I can explain gradient descent and backpropagation.
- I can explain vanishing/exploding gradients.
- I can explain MLPs, CNNs, pooling, embeddings, attention, transformers, and autoencoders.

PyTorch:

- I can create tensors with correct dtype/device.
- I can write an `nn.Module`.
- I can use Dataset and DataLoader.
- I can write training and validation loops.
- I can use `CrossEntropyLoss` correctly.
- I can use SGD, Adam, and AdamW.
- I can save and load model weights.
- I can freeze and unfreeze parameters.

Training:

- I can overfit one batch.
- I can read loss curves.
- I can diagnose overfitting and underfitting.
- I can tune learning rate.
- I can add dropout, weight decay, and early stopping.
- I can compare optimizers.
- I can debug shape, dtype, device, and gradient problems.

Projects:

- I can train an MLP classifier.
- I can train a CNN classifier.
- I can train an autoencoder.
- I can implement scaled dot-product attention.
- I can build a tiny transformer classifier.
- I can fine-tune or linear-probe a pretrained model when hardware and packages allow.

If you can satisfy this checklist, you are ready to move into serious computer vision, NLP, and modern AI model work.
