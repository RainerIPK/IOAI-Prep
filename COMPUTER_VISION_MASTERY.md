# Computer Vision Mastery

This is a dense, self-contained course for learning computer vision for IOAI preparation, deep learning practice, and applied AI projects.

It covers:

- Image fundamentals: pixels, channels, RGB, grayscale, normalization, resizing, interpolation, histograms, filters, augmentation.
- CNN fundamentals: convolutional layers, padding, stride, dilation, receptive fields, pooling, normalization, residual connections.
- Image classification: MLP baselines, CNNs, ResNet, MobileNet, transfer learning, fine-tuning, evaluation.
- Object detection: bounding boxes, IoU, NMS, precision/recall, AP/mAP, YOLO, SSD, Faster R-CNN, DETR.
- Segmentation: semantic, instance, panoptic, U-Net, FCN, DeepLab, Mask R-CNN, Dice, IoU.
- Modern vision: pretrained encoders, self-supervised learning, vision transformers, CLIP, GANs, diffusion models.
- Practice: hand problems, PyTorch/TorchVision drills, debugging tasks, projects, final exam, and mastery checklist.

Verified resource map:

- IOAI Module 4: https://ioai.toki.id/assets/modul/Modul_4_IOAI__Pengenalan_Computer_Vision__CV_.pdf
- Stanford CS231n: https://cs231n.github.io/
- Computer Vision: Algorithms and Applications: https://szeliski.org/Book/
- TorchVision models and pretrained weights: https://docs.pytorch.org/vision/stable/models.html
- TorchVision transforms: https://docs.pytorch.org/vision/stable/transforms.html
- YOLO paper: https://arxiv.org/abs/1506.02640
- SSD paper: https://arxiv.org/abs/1512.02325
- ResNet paper: https://arxiv.org/abs/1512.03385
- U-Net paper: https://arxiv.org/abs/1505.04597
- DETR paper: https://arxiv.org/abs/2005.12872
- Vision Transformer paper: https://arxiv.org/abs/2010.11929
- GAN paper: https://papers.nips.cc/paper/5423-generative-adversarial-nets
- CLIP paper: https://arxiv.org/abs/2103.00020
- DDPM diffusion paper: https://arxiv.org/abs/2006.11239

## 0. How To Use This File

Computer vision has three levels:

1. Image representation: pixels, channels, geometry, annotations.
2. Model mechanics: convolution, pooling, residuals, attention, detection heads, masks.
3. Training practice: augmentation, transfer learning, metrics, error analysis, debugging.

Use this loop:

```text
read concept -> compute tiny example -> implement minimal version -> train small model -> inspect failures -> improve
```

You should be able to answer:

- What is the tensor shape?
- What does one pixel/channel represent?
- What is the output size after this convolution?
- What is the receptive field?
- What metric matches this task?
- Is this classification, detection, segmentation, retrieval, or generation?
- What annotation format does the dataset use?
- What augmentations preserve the label?
- What does the model get wrong?

If you can inspect images, annotations, predictions, losses, and metrics, computer vision becomes much less mysterious.

## 1. What Is Computer Vision?

Computer vision is the field of making machines interpret visual data.

Common tasks:

```text
Classification: What is in the image?
Detection: What objects are present and where are they?
Segmentation: Which pixels belong to which object/class?
Keypoints: Where are important body/pose/landmark points?
Depth: How far away is each point?
Tracking: Where did objects move over time?
Retrieval: Which images/texts are visually similar?
Generation: Create or edit images.
```

Examples:

```text
Image classification: dog vs cat.
Object detection: draw boxes around all cars and pedestrians.
Semantic segmentation: label every pixel as road, sky, person, car.
Instance segmentation: separate each individual person mask.
Vision-language: match "a red car on a street" to an image.
Diffusion: generate an image from a prompt.
```

## 2. Images As Arrays

A digital image is a grid of numbers.

Grayscale image:

```text
height x width
```

RGB image:

```text
height x width x 3
```

The 3 channels are usually:

```text
R = red
G = green
B = blue
```

Pixel values are often stored as integers:

```text
0 to 255
```

Neural networks usually use floating values:

```text
0.0 to 1.0
```

or standardized values:

```text
(x - mean) / std
```

### Channel Order

Common image array layout:

```text
HWC = height, width, channels
```

Common PyTorch tensor layout:

```text
CHW = channels, height, width
```

Batch of images in PyTorch:

```text
NCHW = batch, channels, height, width
```

Example:

```text
[32, 3, 224, 224]
```

means:

```text
32 images, 3 channels, 224 height, 224 width
```

### Common Beginner Bugs

Bug:

```text
Model expects [N, C, H, W] but input is [N, H, W, C].
```

Fix:

```python
x = x.permute(0, 3, 1, 2)
```

Bug:

```text
Pixel values are 0-255 but model expects normalized 0-1 or ImageNet normalization.
```

Fix:

```python
transforms.ToTensor()
```

and for pretrained ImageNet models use the weights' official transforms.

## 3. Image Preprocessing

Preprocessing prepares images for a model.

Common steps:

1. Decode image file.
2. Convert color mode.
3. Resize or crop.
4. Convert to tensor.
5. Normalize.
6. Batch.

### Resize

Resize changes image dimensions.

Why resize:

- CNNs can process variable spatial sizes in many cases, but batching requires same size.
- Classifier heads often expect a particular input size.
- Pretrained models expect specific transforms.

Risks:

- Distorts aspect ratio if width/height are forced independently.
- Shrinks small objects.
- Removes details.

### Center Crop

Center crop takes the middle region.

Good for:

- Evaluation with ImageNet-like classification.

Bad for:

- Objects off-center.
- Detection/segmentation if annotations are not transformed.

### Normalization

Common:

```text
x_normalized = (x - mean) / std
```

For pretrained models, use the exact preprocessing that matches their weights.

TorchVision pretrained weights provide transforms:

```python
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
model = resnet50(weights=weights)
```

## 4. Histograms And Color

An image histogram counts pixel intensities.

Grayscale histogram:

```text
count of pixels at intensity 0, 1, ..., 255
```

RGB histogram:

```text
one histogram per channel
```

Histograms help with:

- Brightness inspection.
- Contrast.
- Thresholding.
- Data drift.

### Color Spaces

RGB:

- Natural for displays and neural networks.

HSV:

- Separates hue, saturation, value.
- Useful for color thresholding.

Grayscale:

- One intensity channel.
- Smaller input.
- Loses color information.

In deep learning, RGB is the usual default.

## 5. Filters And Classical Image Operations

Before deep learning, many vision systems used hand-designed filters.

### Blur

Blur averages nearby pixels.

Use:

- Noise reduction.
- Smoothing.

### Edge Detection

Edges are rapid intensity changes.

Filters such as Sobel approximate derivatives:

```text
horizontal edge filter
vertical edge filter
```

CNNs learn filters like edges in early layers.

### Thresholding

Convert grayscale image to binary mask:

```text
pixel > threshold -> 1
else -> 0
```

Useful for simple segmentation when object/background are easy to separate.

## 6. Convolutional Layers

A convolutional layer slides learned kernels across an image.

Input:

```text
[batch, in_channels, height, width]
```

Layer:

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
```

Output:

```text
[batch, out_channels, out_height, out_width]
```

Each output channel is produced by learned filters.

### Why Convolution?

Convolution uses:

- Local connectivity: each output looks at a local patch.
- Weight sharing: same filter used across the image.
- Translation equivariance: if object shifts, feature map shifts.

This is much more efficient than fully connected layers on raw pixels.

### Parameter Count

Conv2d parameters:

```text
out_channels * in_channels * kernel_height * kernel_width + out_channels biases
```

Example:

```text
Conv2d(3, 32, kernel_size=3)
params = 32 * 3 * 3 * 3 + 32 = 896
```

A fully connected layer from `3 x 224 x 224` to 32 units:

```text
32 * 3 * 224 * 224 + 32 = 4,816,928
```

Convolution is far more parameter-efficient.

## 7. Convolution Output Size

For one spatial dimension:

```text
out = floor((in + 2*padding - dilation*(kernel - 1) - 1) / stride + 1)
```

For dilation 1:

```text
out = floor((in + 2*padding - kernel) / stride + 1)
```

Example:

```text
in = 32
kernel = 3
padding = 1
stride = 1

out = floor((32 + 2*1 - 3) / 1 + 1)
    = 32
```

This is called "same-ish" padding for stride 1.

Example:

```text
in = 32
kernel = 5
padding = 0
stride = 2

out = floor((32 - 5) / 2 + 1)
    = floor(14.5)
    = 14
```

### Shape Discipline

For every CNN layer, write:

```text
[N, C, H, W] -> [N, C_out, H_out, W_out]
```

This prevents most architecture bugs.

## 8. Padding, Stride, Dilation

### Padding

Padding adds pixels around image borders.

Why:

- Preserve spatial size.
- Let filters see border pixels.

Zero padding is common.

### Stride

Stride controls how far the filter moves.

```text
stride 1: move one pixel each step
stride 2: downsample by about 2
```

Stride reduces spatial resolution.

### Dilation

Dilation spaces out kernel elements.

It increases receptive field without increasing parameter count.

Useful in segmentation models such as DeepLab.

## 9. Pooling

Pooling reduces spatial dimensions.

### Max Pooling

Takes maximum value in each window.

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

Shape:

```text
[N, C, H, W] -> [N, C, H/2, W/2]
```

if H and W divisible by 2.

### Average Pooling

Takes average value in each window.

```python
nn.AvgPool2d(kernel_size=2)
```

### Global Average Pooling

Averages over all spatial positions:

```text
[N, C, H, W] -> [N, C]
```

Modern classifiers often use global average pooling before the final linear layer.

## 10. Receptive Field

The receptive field of a unit is the region of input image that can affect it.

Small early layers see:

```text
edges, corners, textures
```

Deeper layers see:

```text
parts, object shapes, semantic concepts
```

Stacking 3x3 convolutions increases receptive field while keeping parameter count low.

Two 3x3 convolutions roughly see a 5x5 region.

Three 3x3 convolutions roughly see a 7x7 region.

This is one reason VGG-style stacks of 3x3 filters are effective.

## 11. CNN Architecture Pattern

Common image classifier:

```text
input image
-> conv block
-> conv block
-> downsample
-> conv block
-> downsample
-> global average pool
-> linear classifier
```

Conv block:

```text
Conv2d -> BatchNorm -> ReLU
```

or:

```text
Conv2d -> ReLU -> Pool
```

Minimal PyTorch CNN:

```python
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

## 12. Image Classification

Image classification predicts one label for the whole image.

Input:

```text
image
```

Output:

```text
class probabilities or logits
```

Example:

```text
cat, dog, car, bird
```

Loss:

```python
nn.CrossEntropyLoss()
```

Metrics:

- Accuracy.
- Top-k accuracy.
- Confusion matrix.
- Precision/recall/F1 for imbalanced classes.

### Top-k Accuracy

Top-1:

```text
correct class is highest-scoring prediction
```

Top-5:

```text
correct class appears among top 5 predictions
```

ImageNet commonly reports top-1 and top-5.

## 13. Image Classification Training Loop

Typical workflow:

1. Split train/validation/test.
2. Define transforms.
3. Create datasets and dataloaders.
4. Build model.
5. Choose loss and optimizer.
6. Train.
7. Track loss and accuracy.
8. Save best validation checkpoint.
9. Evaluate test once.
10. Inspect mistakes.

Training transforms:

```python
from torchvision import transforms

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

Validation transforms:

```python
val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

Do not apply random augmentation to validation/test.

## 14. Image Augmentation

Augmentation creates label-preserving variations.

Common augmentations:

- Random crop.
- Horizontal flip.
- Rotation.
- Color jitter.
- Random resized crop.
- Gaussian blur.
- Random erasing.
- MixUp.
- CutMix.

### Label-Preserving Rule

An augmentation is valid only if the label remains correct.

Good:

```text
Flip cat image horizontally -> still cat.
```

Bad:

```text
Flip digit 6 vertically -> may become 9-like or invalid.
```

Detection and segmentation augmentations must transform annotations too:

```text
image crop -> boxes/masks must be cropped
image flip -> boxes/masks must be flipped
```

## 15. ResNet

ResNet introduced residual connections.

Instead of learning:

```text
H(x)
```

a residual block learns:

```text
F(x) = H(x) - x
```

and outputs:

```text
y = F(x) + x
```

Why it helps:

- Easier optimization.
- Gradients flow through skip paths.
- Allows much deeper networks.

Basic residual block:

```text
x -> Conv -> BN -> ReLU -> Conv -> BN -> + x -> ReLU
```

ResNet is a common pretrained backbone for classification, detection, and segmentation.

## 16. MobileNet And Efficient Models

MobileNet is designed for efficient inference.

Key idea:

```text
depthwise separable convolution
```

Standard convolution mixes:

- spatial filtering
- channel mixing

Depthwise separable convolution separates:

1. Depthwise convolution: spatial filtering per channel.
2. Pointwise 1x1 convolution: channel mixing.

This reduces computation.

Use MobileNet when:

- Device is small.
- Need low latency.
- Need fewer parameters.

Other efficient families:

- EfficientNet.
- ShuffleNet.
- SqueezeNet.
- ConvNeXt-Tiny for modern conv baseline.

## 17. Transfer Learning

Transfer learning uses a model pretrained on a large dataset.

Workflow:

1. Load pretrained model.
2. Replace final classification head.
3. Freeze backbone initially.
4. Train head.
5. Optionally unfreeze later layers.
6. Fine-tune with small learning rate.

TorchVision example:

```python
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

for p in model.parameters():
    p.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
```

Then train only the head.

Fine-tune later:

```python
for p in model.layer4.parameters():
    p.requires_grad = True
```

Use smaller learning rate for pretrained layers.

## 18. Vision Encoders

A vision encoder converts an image into a feature vector or feature map.

Examples:

- ResNet.
- MobileNet.
- EfficientNet.
- ConvNeXt.
- Vision Transformer.
- Swin Transformer.

Use cases:

- Classification head.
- Detection backbone.
- Segmentation encoder.
- Image retrieval.
- CLIP image encoder.

Feature vector:

```text
[batch, embedding_dim]
```

Feature map:

```text
[batch, channels, height, width]
```

## 19. Object Detection

Object detection predicts:

```text
class + bounding box
```

for each object.

Input:

```text
image
```

Output:

```text
boxes, labels, scores
```

Bounding box formats:

```text
xyxy: [x_min, y_min, x_max, y_max]
xywh: [x_min, y_min, width, height]
cxcywh: [center_x, center_y, width, height]
```

Always know the format.

### Detection Is Harder Than Classification

A detector must answer:

1. How many objects?
2. What class is each object?
3. Where is each object?
4. How confident is each prediction?
5. Which overlapping predictions are duplicates?

## 20. Intersection Over Union

IoU measures overlap between predicted and ground truth boxes.

```text
IoU = area(intersection) / area(union)
```

Range:

```text
0 to 1
```

`1` means perfect overlap.

`0` means no overlap.

### Example

Box A:

```text
[0, 0, 4, 4]
area = 16
```

Box B:

```text
[2, 2, 6, 6]
area = 16
```

Intersection:

```text
[2, 2, 4, 4]
area = 4
```

Union:

```text
16 + 16 - 4 = 28
```

IoU:

```text
4 / 28 = 1/7
```

## 21. Non-Maximum Suppression

Object detectors often produce many overlapping boxes.

NMS keeps high-confidence boxes and removes duplicates.

Algorithm:

1. Sort boxes by confidence score.
2. Pick highest score box.
3. Remove boxes with IoU above threshold with selected box.
4. Repeat.

NMS threshold:

```text
0.5 is common, but task-dependent
```

Too low:

- Suppresses nearby true objects.

Too high:

- Leaves duplicates.

DETR-style models reduce or remove need for NMS by predicting sets.

## 22. Detection Metrics: Precision, Recall, AP, mAP

A predicted box is usually considered correct if:

```text
class is correct and IoU >= threshold
```

Common threshold:

```text
IoU >= 0.5
```

Precision:

```text
TP / (TP + FP)
```

Recall:

```text
TP / (TP + FN)
```

Average Precision:

```text
area under precision-recall curve for one class
```

mAP:

```text
mean AP over classes
```

COCO-style mAP averages over multiple IoU thresholds, commonly 0.50 to 0.95.

This is stricter than AP at IoU 0.5.

## 23. YOLO

YOLO means You Only Look Once.

Core idea:

```text
single network predicts boxes and classes directly from full image
```

Original YOLO:

- Divides image into grid cells.
- Each cell predicts bounding boxes and class probabilities.
- Fast single-stage detection.

Strengths:

- Fast.
- Simple end-to-end pipeline.
- Good for real-time applications.

Weaknesses:

- Original versions struggled with small objects and localization precision.
- Modern YOLO variants improved a lot, but the core tradeoff remains speed vs precision.

Mastery target:

You do not need to memorize every YOLO version. Understand:

- Single-stage detector.
- Dense predictions.
- Confidence scores.
- Anchor or anchor-free variants.
- NMS postprocessing.
- Speed/accuracy tradeoff.

## 24. SSD

SSD means Single Shot MultiBox Detector.

Core idea:

- Single network.
- Default boxes at different scales and aspect ratios.
- Predictions from multiple feature maps.

Why multiple feature maps?

```text
early/high-resolution maps help smaller objects
later/low-resolution maps help larger objects
```

Strengths:

- Fast.
- Simpler than proposal-based detectors.
- Handles multiple scales better than original YOLO.

Key concepts:

- Default boxes.
- Class scores.
- Box offsets.
- Multi-scale features.
- NMS.

## 25. Faster R-CNN

Faster R-CNN is a two-stage detector.

Stage 1:

```text
Region Proposal Network proposes candidate boxes
```

Stage 2:

```text
Classify and refine proposals
```

Strengths:

- Often strong accuracy.
- Good localization.

Weaknesses:

- Slower than single-stage detectors.
- More complex pipeline.

Use it as a conceptual contrast:

```text
two-stage = proposals then classify/refine
single-stage = dense predictions directly
```

## 26. DETR

DETR means Detection Transformer.

Core idea:

```text
object detection as set prediction
```

DETR uses:

- CNN backbone.
- Transformer encoder-decoder.
- Learned object queries.
- Bipartite matching loss.

Unlike YOLO/SSD:

- Does not rely on anchor boxes in the same way.
- Does not require standard NMS in the original formulation.
- Predicts a fixed-size set of possible objects, including "no object".

Strengths:

- Elegant.
- Global reasoning via attention.
- Unified set prediction.

Weaknesses:

- Original DETR can train slowly.
- Small object performance can be challenging.

Modern DETR variants improve training speed and small-object handling.

## 27. Segmentation

Segmentation assigns labels at pixel level.

Types:

```text
Semantic segmentation: every pixel gets class label.
Instance segmentation: separate object instances.
Panoptic segmentation: semantic + instance segmentation.
```

Example:

Semantic:

```text
all person pixels are "person"
```

Instance:

```text
person 1 mask, person 2 mask, person 3 mask
```

Panoptic:

```text
road, sky, grass + each individual person/car
```

## 28. Segmentation Metrics

Pixel accuracy:

```text
correct pixels / all pixels
```

Problem:

- Can be misleading if background dominates.

IoU per class:

```text
intersection / union for class pixels
```

Mean IoU:

```text
mean IoU over classes
```

Dice coefficient:

```text
Dice = 2 * intersection / (prediction_area + target_area)
```

Dice is common in medical segmentation.

Dice loss:

```text
1 - Dice
```

## 29. U-Net

U-Net is a segmentation architecture.

Structure:

```text
encoder / contracting path
bottleneck
decoder / expanding path
skip connections
```

Encoder:

- Captures context.
- Downsamples spatial dimensions.

Decoder:

- Upsamples to pixel-level output.

Skip connections:

- Copy high-resolution features from encoder to decoder.
- Help precise localization.

Why U-Net is popular:

- Works well with limited data.
- Strong for biomedical segmentation.
- Simple and effective.

## 30. Mask R-CNN

Mask R-CNN extends Faster R-CNN.

It predicts:

- Class.
- Bounding box.
- Instance mask.

Use:

- Instance segmentation.

Key idea:

```text
Add a mask head to object detection pipeline.
```

## 31. Vision Transformers

Vision Transformer treats an image as a sequence of patches.

Pipeline:

```text
image -> patches -> patch embeddings -> positional embeddings -> transformer encoder -> classifier
```

Example:

```text
224 x 224 image
patch size 16 x 16
number of patches = 14 x 14 = 196
```

Each patch becomes a token.

Strengths:

- Global attention.
- Scales well with large data.
- Strong transfer with pretraining.

Weaknesses:

- Data hungry from scratch.
- Attention cost grows with number of patches.
- Less built-in locality bias than CNNs.

## 32. Self-Supervised Learning For Vision

Self-supervised learning learns from unlabeled images.

The model creates its own training signal.

Common ideas:

- Contrastive learning: make augmented views of same image close, different images far.
- Masked image modeling: hide patches and predict missing content.
- Teacher-student learning: train student to match teacher representations.

Examples:

- SimCLR.
- MoCo.
- BYOL.
- DINO.
- MAE.

Why it matters:

- Labels are expensive.
- Unlabeled images are abundant.
- Pretrained encoders transfer well.

Mastery target:

Understand the intuition, not every method detail:

```text
learn useful image representations without manual labels
```

## 33. CLIP And Vision-Text Encoders

CLIP trains an image encoder and text encoder together.

Training idea:

```text
given batch of image-text pairs,
make matching image/text embeddings similar,
make non-matching pairs dissimilar
```

This is contrastive learning across modalities.

At inference:

1. Encode image.
2. Encode text prompts such as "a photo of a cat".
3. Compare similarities.
4. Pick highest match.

Why CLIP matters:

- Zero-shot classification.
- Image-text retrieval.
- Vision-language grounding.
- Useful pretrained image representations.

Limitations:

- Prompt sensitivity.
- Dataset bias.
- Not guaranteed for fine-grained or domain-specific tasks.
- Similarity is not explanation.

## 34. GANs

GAN means Generative Adversarial Network.

Two networks:

```text
Generator G: creates fake images.
Discriminator D: predicts real vs fake.
```

Training game:

```text
G tries to fool D.
D tries to distinguish real from fake.
```

Strengths:

- Sharp image generation.
- Image-to-image translation.
- Super-resolution.

Weaknesses:

- Training instability.
- Mode collapse.
- Hard evaluation.

Mode collapse:

```text
Generator produces limited variety that fools discriminator.
```

GANs are historically important. Diffusion models now dominate many image generation tasks, but GAN concepts remain useful.

## 35. Diffusion Models

Diffusion models learn to remove noise.

Forward process:

```text
start with clean image
gradually add noise
```

Reverse process:

```text
start with noise
learn to denoise step by step
```

Training target:

- Predict noise.
- Or predict clean image.
- Or predict velocity-like parameterization in some variants.

Why diffusion works:

- Stable training compared with GANs.
- High-quality generation.
- Flexible conditioning on text, image, masks, depth, etc.

Costs:

- Sampling can require many denoising steps.
- Large models are compute-heavy.

Key terms:

- Noise schedule.
- Denoising network.
- U-Net backbone.
- Classifier-free guidance.
- Latent diffusion.

## 36. Interpretability And Error Analysis

Computer vision models can fail silently.

Always inspect examples.

For classification:

- Show correct high-confidence predictions.
- Show wrong high-confidence predictions.
- Show low-confidence predictions.
- Build confusion matrix.

For detection:

- Draw predicted boxes and ground truth boxes.
- Inspect false positives.
- Inspect false negatives.
- Inspect duplicate boxes.
- Inspect small objects.

For segmentation:

- Overlay predicted mask.
- Compare boundary quality.
- Check rare classes.
- Check holes, fragments, and class confusion.

Useful visualization:

- Grad-CAM for classification.
- Feature maps.
- Activation maximization.
- Embedding projection.
- Error galleries.

Interpretability is not proof. It is a debugging tool.

## 37. TorchVision Practical Patterns

### Pretrained Classification

```python
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
preprocess = weights.transforms()
model.eval()
```

### Replace Classification Head

```python
import torch.nn as nn

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
```

### Detection Model Output

TorchVision detection models often return list of dicts:

```python
outputs = model(images)
```

Each output may contain:

```text
boxes: [num_detections, 4]
labels: [num_detections]
scores: [num_detections]
masks: [num_detections, 1, H, W] for mask models
```

### Training Detection Models

Targets are usually list of dicts:

```python
target = {
    "boxes": boxes,
    "labels": labels,
}
```

Detection training is more annotation-sensitive than classification.

## 38. Dataset Annotation Formats

Common formats:

### Classification

Folder format:

```text
dataset/
  train/
    cats/
    dogs/
  val/
    cats/
    dogs/
```

### Detection

COCO format:

- JSON annotations.
- Images.
- Boxes.
- Categories.
- Segmentation masks optionally.

YOLO format:

```text
class_id x_center y_center width height
```

often normalized by image width/height.

Pascal VOC:

- XML files.
- Boxes usually in pixel coordinates.

### Segmentation

Mask image:

```text
each pixel value is class ID
```

or polygon annotations converted to masks.

Always verify:

- coordinate system
- box format
- normalization
- class ID mapping
- image size after transforms

## 39. Common Vision Failure Modes

### Classification

- Train/validation leakage.
- Wrong normalization.
- Too much augmentation.
- Label noise.
- Class imbalance.
- Spurious background correlations.
- Model learns watermark/camera/source instead of object.

### Detection

- Wrong box format.
- Boxes not transformed with images.
- Boxes outside image after crop.
- Class IDs off by one.
- NMS threshold wrong.
- Small objects lost after resizing.

### Segmentation

- Mask interpolation uses bilinear instead of nearest.
- Mask class IDs corrupted by normalization.
- Image/mask transforms out of sync.
- Background dominates loss.
- Boundary errors ignored by pixel accuracy.

### Generative Models

- GAN mode collapse.
- Diffusion over-smoothing.
- Prompt mismatch.
- Memorization risk.
- Bias in training data.

## 40. Problemsets Overview

Do the problemsets in order.

Levels:

```text
A: hand calculation
B: implementation drill
C: model training
D: debugging/report/project
```

Mastery means:

- You can compute output shapes by hand.
- You can build an image classifier.
- You can use pretrained models correctly.
- You can evaluate detection/segmentation outputs.
- You can explain modern vision models conceptually.
- You can inspect and fix annotation/transform bugs.

## Problemset 1: Images As Arrays

### A. Concepts

1. What is a pixel?
2. Difference between grayscale and RGB?
3. What does shape `[224, 224, 3]` mean?
4. What does shape `[32, 3, 224, 224]` mean?
5. Why do PyTorch models usually use NCHW?
6. Why convert pixel values from `0-255` to floats?
7. What can go wrong if channel order is BGR vs RGB?

### B. Coding

1. Load an image with PIL.
2. Print width, height, mode.
3. Convert to NumPy array.
4. Print shape, dtype, min, max.
5. Convert to PyTorch tensor in CHW format.
6. Normalize to `[0, 1]`.

### C. Visualization

Show:

- Original RGB image.
- Red channel only.
- Green channel only.
- Blue channel only.
- Grayscale version.

Explain what each channel captures.

## Problemset 2: Histograms, Thresholds, Filters

### A. Hand

Given grayscale pixels:

```text
[0, 0, 1, 1, 1, 2, 3, 3]
```

1. Compute histogram counts.
2. Threshold at `> 1`.
3. Count foreground pixels.

### B. Coding

1. Compute grayscale histogram of an image.
2. Apply threshold.
3. Apply blur.
4. Apply Sobel-like edge filters.
5. Display results.

### C. Report

Explain when classical filtering might solve a problem without deep learning.

## Problemset 3: Convolution Output Dimensions

### A. Hand Computation

Compute output height/width:

1. Input `32`, kernel `3`, padding `1`, stride `1`.
2. Input `32`, kernel `5`, padding `0`, stride `1`.
3. Input `32`, kernel `5`, padding `0`, stride `2`.
4. Input `64`, kernel `7`, padding `3`, stride `2`.
5. Input `128`, kernel `3`, padding `1`, stride `2`.

### B. Full Tensor Shapes

Input:

```text
[16, 3, 64, 64]
```

Layers:

```text
Conv2d(3, 32, 3, padding=1)
MaxPool2d(2)
Conv2d(32, 64, 3, padding=1)
MaxPool2d(2)
```

Write shape after each layer.

### C. PyTorch Verification

Build these layers and pass a random tensor through them.

Print shapes and compare with hand answers.

## Problemset 4: Convolution Parameters And Receptive Field

### A. Parameter Count

Compute parameter counts:

1. `Conv2d(3, 16, 3)`
2. `Conv2d(16, 32, 3)`
3. `Conv2d(32, 64, 1)`
4. Fully connected from `3 x 32 x 32` to 100 units.

### B. Receptive Field

Assume stride 1 and no dilation.

1. One 3x3 conv sees what receptive field?
2. Two stacked 3x3 convs?
3. Three stacked 3x3 convs?
4. Why might three 3x3 convs be preferable to one 7x7 conv?

### C. Implementation

Build a model with stacked 3x3 convs and compare parameter count against a single 7x7 conv with same channels.

## Problemset 5: Pooling

### A. Hand

For patch:

```text
[[1, 3],
 [2, 0]]
```

Compute:

1. Max pool.
2. Average pool.

For a `32 x 32` feature map, apply `MaxPool2d(2)` three times. What size remains?

### B. Coding

Use PyTorch pooling on a small tensor with known values.

Verify output by hand.

### C. Explanation

Explain why pooling can help classification but can hurt precise segmentation boundaries.

## Problemset 6: Build A Small CNN

### A. Model

Build:

```text
Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> GlobalAveragePool -> Linear
```

### B. Dataset

Use one:

- MNIST.
- FashionMNIST.
- CIFAR-10.
- Synthetic colored-shape dataset.

### C. Train

Track:

- train loss
- train accuracy
- validation loss
- validation accuracy

### D. Report

Show 10 mistakes and explain likely causes.

## Problemset 7: Augmentation

### A. Concepts

For each augmentation, say when it is valid or invalid:

1. Horizontal flip.
2. Vertical flip.
3. Rotation.
4. Color jitter.
5. Random crop.
6. Random erasing.

Consider:

- cats vs dogs
- digits
- medical scans
- traffic signs

### B. Experiment

Train a classifier:

1. No augmentation.
2. Basic augmentation.
3. Too aggressive augmentation.

Compare validation accuracy.

### C. Report

Explain how augmentation changed overfitting.

## Problemset 8: Transfer Learning With ResNet Or MobileNet

### A. Setup

Load pretrained ResNet18 or MobileNet.

Replace final head for your dataset.

### B. Linear Probe

Freeze backbone.

Train only head.

### C. Fine-Tune

Unfreeze last block.

Use smaller learning rate.

### D. Compare

Report:

```text
scratch CNN vs frozen pretrained vs fine-tuned pretrained
```

Explain compute, accuracy, and overfitting tradeoffs.

## Problemset 9: Confusion Matrix And Error Analysis

### A. Metrics

For classification predictions:

1. Compute confusion matrix.
2. Compute per-class precision.
3. Compute per-class recall.
4. Identify most confused class pair.

### B. Error Gallery

Show:

- correct high-confidence images
- wrong high-confidence images
- low-confidence images

### C. Report

Explain whether errors come from:

- blurry images
- ambiguous labels
- background bias
- class imbalance
- small object
- model weakness

## Problemset 10: Bounding Boxes And IoU

### A. Hand

Compute IoU:

```text
A = [0, 0, 4, 4]
B = [2, 2, 6, 6]
```

Then:

```text
C = [0, 0, 10, 10]
D = [1, 1, 9, 9]
```

### B. Coding

Implement:

```python
box_area(box)
iou(box1, box2)
pairwise_iou(boxes1, boxes2)
```

Use `xyxy` format.

### C. Format Conversion

Write converters:

```python
xywh_to_xyxy
xyxy_to_xywh
cxcywh_to_xyxy
```

Test with known boxes.

## Problemset 11: NMS

### A. Hand

Given boxes with scores:

```text
box1 score 0.95
box2 score 0.90, IoU with box1 = 0.8
box3 score 0.70, IoU with box1 = 0.2
```

NMS threshold `0.5`.

Which boxes remain?

### B. Coding

Implement NMS from scratch.

Compare with `torchvision.ops.nms` if available.

### C. Explanation

Explain how NMS threshold affects duplicate detections and nearby objects.

## Problemset 12: Detection Metrics

### A. Concepts

1. What is a true positive detection?
2. What is a false positive detection?
3. What is a false negative detection?
4. Why does confidence threshold affect precision and recall?
5. What is AP?
6. What is mAP?

### B. Tiny AP

Create 5 predictions sorted by confidence.

Mark each as TP or FP.

Compute precision and recall after each prediction.

Plot precision-recall points.

### C. Report

Explain why AP is better than accuracy for detection.

## Problemset 13: YOLO And SSD Concepts

### A. YOLO

Explain:

1. Why YOLO is called single-stage.
2. What a grid cell means in original YOLO.
3. What confidence score means.
4. Why small objects can be hard.
5. Why YOLO is fast.

### B. SSD

Explain:

1. Default boxes.
2. Multi-scale feature maps.
3. Box offsets.
4. Why SSD handles multiple object sizes.

### C. Practical

Use a pretrained detector if available.

Run on 10 images.

Draw boxes and scores.

Inspect false positives and missed objects.

## Problemset 14: DETR Concepts

### A. Concepts

1. What does "set prediction" mean?
2. What are object queries?
3. Why does DETR use bipartite matching?
4. Why can DETR avoid standard NMS?
5. Why can original DETR train slowly?

### B. Shapes

Suppose DETR outputs:

```text
num_queries = 100
num_classes = 91 plus no-object
```

For batch size 4, what are approximate shapes of:

```text
class logits
boxes
```

### C. Compare

Write a table:

```text
YOLO vs SSD vs Faster R-CNN vs DETR
```

Include:

- stage type
- anchors/proposals/queries
- speed
- strengths
- weaknesses

## Problemset 15: Segmentation Masks

### A. Concepts

1. Semantic vs instance segmentation.
2. What is a mask?
3. What shape is a class mask for an image?
4. Why should masks use nearest-neighbor interpolation?
5. Why can pixel accuracy be misleading?

### B. Coding

Create a toy mask:

```text
0 = background
1 = object
2 = border
```

Compute per-class pixel counts.

Create predicted mask and compute pixel accuracy.

### C. Visualization

Overlay mask on image with transparency.

## Problemset 16: Segmentation Metrics

### A. Hand IoU

Ground truth foreground:

```text
pixels {1, 2, 3, 4}
```

Prediction foreground:

```text
pixels {3, 4, 5, 6}
```

Compute:

1. Intersection.
2. Union.
3. IoU.
4. Dice.

### B. Coding

Implement:

```python
binary_iou(pred, target)
dice_score(pred, target)
mean_iou(pred, target, num_classes)
```

### C. Report

Compare Dice and IoU on small objects.

## Problemset 17: U-Net

### A. Architecture

Draw or describe:

```text
encoder -> bottleneck -> decoder
skip connections
```

### B. Build

Implement a small U-Net for `64 x 64` images.

### C. Synthetic Dataset

Generate images with random circles/rectangles and masks.

Train U-Net to segment shapes.

### D. Report

Show:

- input image
- ground truth mask
- predicted mask
- error mask

## Problemset 18: Mask R-CNN And Instance Segmentation

### A. Concepts

1. How is instance segmentation different from semantic segmentation?
2. What does Mask R-CNN add to Faster R-CNN?
3. Why do we need separate masks for each object?

### B. Practical

Run a pretrained Mask R-CNN if available.

Inspect outputs:

- boxes
- labels
- scores
- masks

### C. Failure Analysis

Find:

- duplicate instances
- missed small objects
- poor mask boundaries

## Problemset 19: Vision Transformers

### A. Patch Math

Image:

```text
224 x 224
```

Patch:

```text
16 x 16
```

Compute:

1. Patches per row.
2. Total number of patches.
3. Sequence length if using one class token.

### B. Implementation

Write code to split an image tensor into patches.

### C. Compare

CNN vs ViT:

- locality bias
- data needs
- compute
- transfer learning
- global context

## Problemset 20: CLIP And Retrieval

### A. Concepts

1. What are image and text encoders?
2. What is contrastive learning?
3. How does zero-shot classification work with text prompts?
4. Why does prompt wording matter?
5. What biases might CLIP inherit?

### B. Toy Contrastive Loss

Create 3 image embeddings and 3 text embeddings.

Compute similarity matrix.

Identify correct diagonal matches.

### C. Practical

If CLIP package/model is available:

1. Encode images.
2. Encode class prompts.
3. Run zero-shot classification.

If not, implement toy retrieval with manually created embeddings.

## Problemset 21: GANs

### A. Concepts

1. What does generator do?
2. What does discriminator do?
3. Why is GAN training a game?
4. What is mode collapse?
5. Why can GANs be hard to train?

### B. Toy GAN

Train a tiny GAN on 1D data:

```text
real data sampled from Normal(4, 1)
generator maps noise to scalar
discriminator predicts real/fake
```

Plot generated distribution over time.

### C. Report

List failure signs:

- discriminator too strong
- generator collapse
- unstable loss

## Problemset 22: Diffusion Models

### A. Concepts

1. What is the forward diffusion process?
2. What is the reverse denoising process?
3. What does the denoising network predict?
4. Why are U-Nets common in diffusion?
5. What is classifier-free guidance conceptually?

### B. Noise Schedule

Take a simple image tensor.

Add Gaussian noise at increasing levels.

Visualize:

```text
t = 0, 0.25, 0.5, 0.75, 1.0
```

### C. Tiny Denoiser

Train a small CNN to denoise synthetic noisy images.

This is not a full diffusion model, but it teaches the core denoising skill.

## Problemset 23: Self-Supervised Vision

### A. Concepts

1. What is self-supervised learning?
2. Why use two augmented views?
3. What is a positive pair?
4. What is a negative pair?
5. How is masked image modeling different from contrastive learning?

### B. Toy Contrastive

Take image embeddings.

Create augmented pairs.

Compute cosine similarities.

Explain desired pattern.

### C. Project

Train a simple encoder with an auxiliary pretext task:

- rotation prediction, or
- jigsaw patch order, or
- masked patch reconstruction.

Then fine-tune on classification.

## Problemset 24: Full Image Classification Project

Dataset options:

- CIFAR-10.
- FashionMNIST.
- Oxford Pets.
- custom folder dataset.
- synthetic shapes.

Requirements:

1. Build train/validation/test split.
2. Create transforms.
3. Train small CNN from scratch.
4. Train pretrained ResNet or MobileNet.
5. Compare augmentation settings.
6. Track curves.
7. Save best model.
8. Evaluate test once.
9. Build confusion matrix.
10. Make error gallery.
11. Write final report.

Report template:

```text
Task:
Dataset:
Input size:
Classes:
Model 1:
Model 2:
Transforms:
Optimizer:
Loss:
Best validation metric:
Test metric:
Most common errors:
What augmentation helped:
What failed:
Next improvement:
```

## Problemset 25: Full Detection Project

Dataset options:

- small COCO subset
- Pascal VOC subset
- synthetic shapes with bounding boxes
- custom object dataset

Requirements:

1. Inspect annotation format.
2. Draw ground truth boxes.
3. Convert boxes to one consistent format.
4. Train or run pretrained detector.
5. Draw predicted boxes.
6. Compute IoU for predictions.
7. Apply confidence threshold.
8. Apply NMS if needed.
9. Count false positives and false negatives.
10. Write detector error report.

Synthetic option:

Generate images with random colored rectangles/circles and known boxes. Train a tiny detector or use the dataset to practice box metrics and visualization.

## Problemset 26: Full Segmentation Project

Dataset options:

- synthetic shapes
- Oxford-IIIT Pet masks if available
- medical/toy masks
- any image-mask pair dataset

Requirements:

1. Verify image/mask alignment.
2. Use nearest-neighbor interpolation for masks.
3. Build U-Net.
4. Train with cross entropy or Dice/BCE.
5. Track mean IoU or Dice.
6. Visualize predictions.
7. Inspect boundary errors.
8. Compare with and without augmentation.
9. Write segmentation report.

## Problemset 27: Final Exam

### Part A: Theory

Answer in writing:

1. What is a digital image?
2. Explain RGB channels.
3. Explain convolution.
4. Explain padding, stride, and dilation.
5. Explain pooling.
6. Explain receptive field.
7. Explain why CNNs are parameter-efficient.
8. Explain residual connections.
9. Explain transfer learning.
10. Explain image augmentation.
11. Explain object detection.
12. Explain IoU.
13. Explain NMS.
14. Explain AP/mAP.
15. Compare YOLO, SSD, Faster R-CNN, and DETR.
16. Explain semantic, instance, and panoptic segmentation.
17. Explain U-Net.
18. Explain Vision Transformers.
19. Explain CLIP.
20. Explain GANs and diffusion models.

### Part B: Hand Computation

1. Compute convolution output size for 5 layers.
2. Compute parameter count for conv and fully connected layers.
3. Compute max pooling output.
4. Compute IoU for two boxes.
5. Run NMS on a small set of boxes.
6. Compute precision/recall from detections.
7. Compute Dice and IoU for masks.
8. Compute ViT patch sequence length.

### Part C: Coding

From memory:

1. Load image and convert to tensor.
2. Build CNN classifier.
3. Write training loop.
4. Use pretrained ResNet.
5. Draw bounding boxes.
6. Compute IoU.
7. Implement NMS.
8. Overlay segmentation mask.
9. Implement Dice score.
10. Split image into patches.

### Part D: Projects

Complete at least two:

1. Image classification project.
2. Detection project.
3. Segmentation project.
4. CLIP-style retrieval toy project.
5. Denoising/diffusion toy project.

Passing standard:

- You can compute shapes by hand.
- You can train and evaluate a classifier.
- You can use pretrained encoders correctly.
- You can visualize boxes and masks.
- You can explain detection and segmentation metrics.
- You can diagnose common vision bugs.

## 41. External Practice Map

### IOAI Module 4

Use for Indonesian IOAI alignment.

Focus:

- Digital images.
- Image reading/display.
- Basic image operations.
- Color and histograms.
- Preprocessing.
- CNN classification.
- ResNet.
- YOLO and detection metrics.
- U-Net and segmentation.
- Transfer learning and augmentation.

Link: https://ioai.toki.id/assets/modul/Modul_4_IOAI__Pengenalan_Computer_Vision__CV_.pdf

### Stanford CS231n

Use for deep conceptual understanding and assignments.

Focus:

- Image classification.
- Linear classifiers.
- Optimization.
- Backpropagation.
- CNN architectures.
- Training neural networks.
- Detection and segmentation lectures if available.

Link: https://cs231n.github.io/

### Szeliski Book

Use for broader computer vision beyond deep learning.

Focus:

- Image formation.
- Features and matching.
- Segmentation.
- Motion.
- 3D reconstruction.
- Recognition.

Link: https://szeliski.org/Book/

### TorchVision

Use for implementation.

Focus:

- Pretrained classification models.
- Detection models.
- Segmentation models.
- Transforms.
- Model weights and official preprocessing.

Links:

- https://docs.pytorch.org/vision/stable/models.html
- https://docs.pytorch.org/vision/stable/transforms.html

### Primary Papers

Read these after you can train a classifier and compute detection/segmentation metrics:

- ResNet: https://arxiv.org/abs/1512.03385
- YOLO: https://arxiv.org/abs/1506.02640
- SSD: https://arxiv.org/abs/1512.02325
- U-Net: https://arxiv.org/abs/1505.04597
- DETR: https://arxiv.org/abs/2005.12872
- Vision Transformer: https://arxiv.org/abs/2010.11929
- GAN: https://papers.nips.cc/paper/5423-generative-adversarial-nets
- CLIP: https://arxiv.org/abs/2103.00020
- DDPM: https://arxiv.org/abs/2006.11239

## 42. Eight-Week Mastery Plan

### Week 1: Images And CNN Math

Study:

- Images as arrays.
- RGB/grayscale.
- Transforms.
- Convolution.
- Padding, stride, pooling.

Do:

- Problemsets 1-5.

Deliverable:

- Notebook that loads images, computes histograms, applies filters, and verifies CNN output shapes.

### Week 2: Image Classification

Study:

- CNN classifier.
- Training loop.
- Augmentation.
- Confusion matrix.

Do:

- Problemsets 6-9.

Deliverable:

- Small CNN classifier with error gallery.

### Week 3: Transfer Learning

Study:

- ResNet.
- MobileNet.
- Pretrained weights.
- Linear probing.
- Fine-tuning.

Do:

- Problemset 8 in depth.
- Start Problemset 24.

Deliverable:

- From-scratch vs pretrained comparison.

### Week 4: Detection Basics

Study:

- Boxes.
- IoU.
- NMS.
- Precision/recall/AP.

Do:

- Problemsets 10-12.

Deliverable:

- Box toolkit: conversion, IoU, NMS, visualization.

### Week 5: Detection Models

Study:

- YOLO.
- SSD.
- Faster R-CNN.
- DETR.

Do:

- Problemsets 13-14.
- Start Problemset 25.

Deliverable:

- Detection report with visualized predictions and failure analysis.

### Week 6: Segmentation

Study:

- Semantic/instance segmentation.
- Segmentation metrics.
- U-Net.
- Mask R-CNN.

Do:

- Problemsets 15-18.
- Start Problemset 26.

Deliverable:

- U-Net on synthetic shapes with mask visualizations.

### Week 7: Modern Vision

Study:

- ViT.
- Self-supervised learning.
- CLIP.
- GANs.
- Diffusion.

Do:

- Problemsets 19-23.

Deliverable:

- Toy CLIP/retrieval or denoising project.

### Week 8: Final Projects And Exam

Do:

- Problemsets 24-27.

Deliverable:

- Two full projects and one written final exam.

## 43. Selected Answer Key

### Problemset 3, A

Formula:

```text
out = floor((in + 2*padding - kernel) / stride + 1)
```

1.

```text
in 32, k 3, p 1, s 1 -> 32
```

2.

```text
in 32, k 5, p 0, s 1 -> 28
```

3.

```text
in 32, k 5, p 0, s 2 -> floor((32 - 5) / 2 + 1) = 14
```

4.

```text
in 64, k 7, p 3, s 2 -> floor((64 + 6 - 7) / 2 + 1) = 32
```

5.

```text
in 128, k 3, p 1, s 2 -> floor((128 + 2 - 3) / 2 + 1) = 64
```

### Problemset 3, B

Input:

```text
[16, 3, 64, 64]
```

After `Conv2d(3, 32, 3, padding=1)`:

```text
[16, 32, 64, 64]
```

After `MaxPool2d(2)`:

```text
[16, 32, 32, 32]
```

After `Conv2d(32, 64, 3, padding=1)`:

```text
[16, 64, 32, 32]
```

After `MaxPool2d(2)`:

```text
[16, 64, 16, 16]
```

### Problemset 4, A

`Conv2d(3, 16, 3)`:

```text
16 * 3 * 3 * 3 + 16 = 448
```

`Conv2d(16, 32, 3)`:

```text
32 * 16 * 3 * 3 + 32 = 4640
```

`Conv2d(32, 64, 1)`:

```text
64 * 32 * 1 * 1 + 64 = 2112
```

Fully connected from `3 x 32 x 32` to 100:

```text
100 * 3072 + 100 = 307300
```

### Problemset 10, IoU

A:

```text
[0, 0, 4, 4], area 16
```

B:

```text
[2, 2, 6, 6], area 16
```

Intersection:

```text
[2, 2, 4, 4], area 4
```

Union:

```text
16 + 16 - 4 = 28
```

IoU:

```text
4 / 28 = 1/7
```

### Problemset 11, NMS

NMS threshold `0.5`.

Select box1 with score 0.95.

Box2 has IoU 0.8 with box1, so suppress it.

Box3 has IoU 0.2 with box1, so keep it.

Remaining:

```text
box1 and box3
```

### Problemset 16, Mask Metrics

Ground truth:

```text
{1, 2, 3, 4}
```

Prediction:

```text
{3, 4, 5, 6}
```

Intersection:

```text
{3, 4}, size 2
```

Union:

```text
{1, 2, 3, 4, 5, 6}, size 6
```

IoU:

```text
2 / 6 = 1/3
```

Dice:

```text
2 * 2 / (4 + 4) = 1/2
```

### Problemset 19, Patch Math

Image:

```text
224 x 224
```

Patch:

```text
16 x 16
```

Patches per row:

```text
224 / 16 = 14
```

Total patches:

```text
14 * 14 = 196
```

With one class token:

```text
sequence length = 197
```

## 44. Mistake Log

Use this after every exercise or experiment:

```text
Date:
Task:
Expected result:
Actual result:
Image shape:
Annotation format:
Transform pipeline:
Model:
Metric:
Bug or weakness:
Fix tried:
What changed:
What I learned:
```

Vision mistakes often hide in data and annotations. Always visualize before trusting metrics.

## 45. Final Mastery Checklist

Image fundamentals:

- I can explain pixels, channels, RGB, grayscale.
- I can load, display, resize, crop, and normalize images.
- I can convert between HWC and CHW/NCHW.
- I can compute histograms and apply simple filters.

CNNs:

- I can compute convolution output dimensions by hand.
- I can compute convolution parameter counts.
- I can explain padding, stride, dilation, pooling, and receptive field.
- I can build and train a small CNN.
- I can read training and validation curves.

Classification:

- I can train an image classifier.
- I can use augmentation correctly.
- I can use pretrained ResNet/MobileNet.
- I can fine-tune a model.
- I can build confusion matrices and error galleries.

Detection:

- I can convert bounding box formats.
- I can compute IoU.
- I can implement NMS.
- I can explain precision, recall, AP, and mAP.
- I can compare YOLO, SSD, Faster R-CNN, and DETR.
- I can visualize detection errors.

Segmentation:

- I can explain semantic, instance, and panoptic segmentation.
- I can compute pixel accuracy, IoU, mean IoU, and Dice.
- I can build a small U-Net.
- I can visualize masks and boundary errors.

Modern vision:

- I can explain ResNet and residual connections.
- I can explain ViT patch embeddings.
- I can explain self-supervised vision learning.
- I can explain CLIP zero-shot classification and retrieval.
- I can explain GANs and mode collapse.
- I can explain diffusion as iterative denoising.

If you can satisfy this checklist and complete two full projects, you have a strong computer vision foundation for IOAI, deep learning study, and practical model work.
