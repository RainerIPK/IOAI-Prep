# IOAI Problem Ladder

This is a problem-first training plan for going beyond IOAI selection level.

The goal is not only to pass IOAI-style selection. The goal is to become the kind of contestant who can:

1. Read a new AI problem and build a baseline quickly.
2. Diagnose why the baseline fails.
3. Improve the model with disciplined experiments.
4. Explain the method mathematically and practically.
5. Handle unfamiliar tasks in ML, CV, NLP, audio, probability, vectors, and algorithms.

Use this as a ladder. Do not try to do everything at once. Start at the lowest level where you cannot solve at least 80 percent of the tasks cleanly.

## 0. How To Use This File

For every problem you solve, write a short log:

```text
Problem:
Source:
Level:
Topic:
What I first thought:
Correct model:
Mistake:
Final method:
Could I solve it again in 7 days?
```

For every practical ML problem, also log:

```text
Metric:
Baseline score:
Best score:
What improved it:
What did not improve it:
Validation method:
Possible leakage:
One ablation:
One next experiment:
```

You are ready to move up a level only when:

1. You can solve most hand problems without looking at notes.
2. You can code small experiments without tutorial copying.
3. You can explain your answer in plain language.
4. You can debug your own wrong assumptions.
5. You can reproduce the solution one week later.

## 1. Source Registry

These are the main problem sources used in this ladder.

### Official IOAI And AI Olympiad Sources

- IOAI Education Hub: https://ioai-official.org/resources/
- IOAI 2026 syllabus page: https://ioai-official.org/republic-of-kazakhstan/syllabus-2026/
- IOAI 2025 official repository: https://github.com/IOAI-official/IOAI-2025
- IOAI 2024 official repository: https://github.com/IOAI-official/IOAI-2024
- IOAI 2024 tasks page: https://ioai-official.org/2024-tasks/
- Awesome IOAI tasks index: https://github.com/open-cu/awesome-ioai-tasks
- USAAIO past problems: https://www.usaaio.org/past-problems
- TLX OSN AI exhibition problems: https://tlx.toki.id/problems/ekshibisi-osn-ai-2025

### Math, Probability, Statistics, And Algorithms

- OpenIntro Statistics: https://www.openintro.org/book/os/
- Stanford CS109 problem sets: https://web.stanford.edu/class/archive/cs/cs109/cs109.1212/psets/index.html
- MIT 18.06 Linear Algebra assignments: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/assignments/
- Mathematics for Machine Learning: https://mml-book.github.io/book/mml-book.pdf
- Pen and Paper Exercises in Machine Learning: https://arxiv.org/abs/2206.13446
- CSES Problem Set: https://cses.fi/problemset/
- AtCoder Educational DP Contest: https://atcoder.jp/contests/dp/tasks
- Project Euler archives: https://projecteuler.net/archives

### Classical ML And Data Competitions

- scikit-learn MOOC: https://inria.github.io/scikit-learn-mooc/
- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- Kaggle competitions: https://www.kaggle.com/competitions
- Kaggle Playground Series: https://www.kaggle.com/competitions/playground-series
- DrivenData competitions: https://www.drivendata.org/competitions/

### Deep Learning, CV, NLP, And Audio

- Dive into Deep Learning: https://d2l.ai/
- Stanford CS231n assignments: https://cs231n.stanford.edu/assignments.html
- Stanford CS224n: https://web.stanford.edu/class/cs224n/
- Hugging Face NLP/LLM course: https://huggingface.co/learn/nlp-course
- Hugging Face Audio course: https://huggingface.co/learn/audio-course/chapter0/introduction
- PyTorch CIFAR-10 tutorial: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Kaggle Digit Recognizer: https://www.kaggle.com/competitions/digit-recognizer
- Kaggle BirdCLEF: https://www.kaggle.com/competitions/birdclef-2025
- TensorFlow Speech Commands dataset: https://www.tensorflow.org/datasets/catalog/speech_commands
- COCO dataset: https://cocodataset.org/
- Oxford-IIIT Pet dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/

## 2. Total Training Volume

This is the full ladder if you want to go beyond IOAI.

| Level | Target Volume | Main Purpose |
|---|---:|---|
| Beginner | 180 to 250 problems/tasks | Build fluency and stop fearing basic math/code/ML tasks. |
| Intermediate | 220 to 320 problems/tasks | Become reliable across probability, vectors, algorithms, and ML workflows. |
| Advanced | 120 to 200 problems/tasks | Handle official IOAI-like tasks and university assignments. |
| Expert | 50 to 100 problems/projects | Reproduce strong methods, write reports, and handle open-ended tasks. |

The word "problem" means different things depending on source:

- A hand-computation exercise counts as 1.
- A coding judge problem counts as 1.
- A Kaggle/DrivenData competition counts as 10 to 25, because it includes data reading, EDA, baseline, validation, feature work, modeling, error analysis, and write-up.
- A CS231n or CS224n assignment counts as 15 to 30.
- An IOAI official task counts as 20 to 40.

## 3. Beginner Ladder

Beginner goal: become comfortable with the basic shapes of IOAI selection questions.

You should finish this level if you are still shaky on mean, variance, Bayes rule, vectors, Python loops, simple ML, or convolution shapes.

### Beginner A: OSN AI Selection-Style Reasoning

Do these first because they are closest to the Indonesian selection style.

1. TLX OSN AI exhibition problem set: https://tlx.toki.id/problems/ekshibisi-osn-ai-2025
2. Re-solve the 40-topic table from `SELECTION_REASONING_MASTERY.md`.
3. Create your own variants for each question type.

Target:

- 40 official/selection questions.
- 40 self-made variants.
- 20 timed re-solves.

Checklist:

- Can model an interval game state.
- Can count small grid paths.
- Can compute variance after duplicating data.
- Can apply total probability and Bayes rule.
- Can fit a tiny least-squares line.
- Can compute Euclidean distances.
- Can compute convolution output shape.
- Can classify simple sequences by feature counts.

### Beginner B: Statistics And Probability

Source:

- OpenIntro Statistics: https://www.openintro.org/book/os/
- Stanford CS109: https://web.stanford.edu/class/archive/cs/cs109/cs109.1212/psets/index.html

Do:

1. OpenIntro Chapter 1 exercises: data, study design, visual summaries.
2. OpenIntro Chapter 2 exercises: probability.
3. OpenIntro Chapter 3 exercises: distributions.
4. OpenIntro Chapter 7 exercises: linear regression basics.
5. CS109 first two problem sets if you want a harder probability ramp.

Target:

- 60 OpenIntro exercises.
- 10 CS109 warm-up or pset problems.
- 20 custom probability table problems.

Must-master problem types:

1. Compute mean, median, variance, standard deviation.
2. Explain why duplicating a dataset changes sample variance but not population variance in the same way.
3. Combine two groups with different means.
4. Build a joint table from conditional probabilities.
5. Compute `P(A | B)` from `P(B | A)`.
6. Decide when independence is being assumed incorrectly.
7. Interpret regression slope and intercept.
8. Detect correlation-vs-causation traps.

### Beginner C: Linear Algebra And Vectors

Source:

- MIT 18.06 assignments: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/assignments/
- Mathematics for Machine Learning: https://mml-book.github.io/book/mml-book.pdf

Do:

1. MIT 18.06 Problem Sets 1 to 3.
2. MML exercises from chapters on vectors, matrices, and analytic geometry.
3. Write small NumPy implementations for each concept.

Target:

- 30 linear algebra exercises.
- 20 vector-similarity drills.
- 5 NumPy mini-implementations.

Must-master problem types:

1. Vector addition and subtraction.
2. Dot products and cosine similarity.
3. Euclidean distance.
4. Matrix multiplication shape checks.
5. Solving 2 by 2 and 3 by 3 systems.
6. Projection intuition.
7. PCA intuition through variance directions.
8. Embedding analogy questions such as `king - man + woman`.

### Beginner D: Python And Algorithms

Source:

- CSES Problem Set: https://cses.fi/problemset/
- Project Euler: https://projecteuler.net/archives

Do:

1. CSES Introductory Problems: all 19.
2. Project Euler problems 1 to 25.
3. At least 10 small grid/DFS problems you write yourself.

Target:

- 19 CSES introductory problems.
- 25 Project Euler problems.
- 10 grid path drills.
- 10 list/dictionary/string drills.

Must-master problem types:

1. Frequency counting.
2. Prefix sums.
3. Nested loops with bounds.
4. DFS on a grid.
5. Recursion with memoization.
6. Sorting and searching.
7. Basic combinatorics.
8. Simulation without off-by-one errors.

### Beginner E: Classical ML

Source:

- scikit-learn MOOC: https://inria.github.io/scikit-learn-mooc/
- UCI datasets: https://archive.ics.uci.edu/
- Kaggle Titanic: https://www.kaggle.com/competitions/titanic
- Kaggle House Prices: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- DrivenData Warm Up competitions: https://www.drivendata.org/competitions/

Do:

1. Train logistic regression on Iris.
2. Train kNN on Wine.
3. Train random forest on Breast Cancer.
4. Train linear regression on Diabetes.
5. Do Kaggle Titanic.
6. Do Kaggle House Prices.
7. Do DrivenData Warm Up: Heart.
8. Do DrivenData Warm Up: Blood Donations.

Target:

- 8 small datasets.
- 4 end-to-end notebooks.
- 2 competition submissions.
- 1 written comparison of 5 models.

Must-master problem types:

1. Train/validation/test split.
2. Cross-validation.
3. Accuracy, precision, recall, F1.
4. ROC AUC for binary classification.
5. MSE and MAE for regression.
6. Feature scaling.
7. One-hot encoding.
8. Missing-value imputation.
9. Data leakage detection.
10. Baseline vs tuned model comparison.

### Beginner F: Deep Learning, CV, NLP, Audio First Contact

Sources:

- Dive into Deep Learning: https://d2l.ai/
- PyTorch CIFAR-10 tutorial: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Kaggle Digit Recognizer: https://www.kaggle.com/competitions/digit-recognizer
- Hugging Face NLP/LLM course: https://huggingface.co/learn/nlp-course
- Hugging Face Audio course: https://huggingface.co/learn/audio-course/chapter0/introduction

Do:

1. D2L preliminaries exercises.
2. D2L linear neural networks exercises.
3. PyTorch CIFAR-10 tutorial.
4. Kaggle Digit Recognizer baseline.
5. Hugging Face pipeline task for text classification.
6. Hugging Face pipeline task for question answering.
7. Hugging Face audio classification demo.

Target:

- 20 D2L exercises.
- 3 PyTorch notebooks.
- 1 image classifier.
- 1 text classifier.
- 1 audio classifier demo.

Must-master problem types:

1. Tensor shape tracing.
2. Training loop structure.
3. Loss decreasing vs overfitting.
4. Batch size effects.
5. Learning rate effects.
6. CNN output shape.
7. Tokenization and padding.
8. Audio waveform vs spectrogram.

## 4. Intermediate Ladder

Intermediate goal: become reliable across all core IOAI domains, not only selection-style MCQ problems.

### Intermediate A: Algorithms For AI Problems

Source:

- CSES Problem Set: https://cses.fi/problemset/
- AtCoder Educational DP Contest: https://atcoder.jp/contests/dp/tasks
- Project Euler: https://projecteuler.net/archives

Do:

1. AtCoder Educational DP Contest: all 26.
2. CSES Sorting and Searching: at least 20.
3. CSES Dynamic Programming: all 19.
4. CSES Graph Algorithms: at least 15.
5. Project Euler problems 26 to 75.

Target:

- 26 AtCoder DP problems.
- 54 CSES problems.
- 50 Project Euler problems.

Why this matters for IOAI:

- You need DP for sequence tasks.
- You need graph thinking for grid, clustering, and nearest-neighbor problems.
- You need complexity awareness when brute force is too slow.
- You need clean implementation under time pressure.

Must-master problem types:

1. 1D DP.
2. 2D grid DP.
3. Knapsack.
4. Longest increasing subsequence.
5. Edit distance.
6. Shortest paths.
7. Connected components.
8. Binary search on answer.
9. Sliding windows.
10. Coordinate compression.

### Intermediate B: Math For ML

Sources:

- MIT 18.06: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/assignments/
- CS109: https://web.stanford.edu/class/archive/cs/cs109/cs109.1212/psets/index.html
- MML: https://mml-book.github.io/book/mml-book.pdf

Do:

1. MIT 18.06 Problem Sets 4 to 8.
2. CS109 Problem Sets 3 to 6.
3. MML exercises on optimization, probability, and PCA.
4. Derive linear regression normal equations once.
5. Derive logistic regression gradient once.

Target:

- 50 linear algebra problems.
- 50 probability/statistics problems.
- 20 calculus/optimization problems.
- 5 derivation write-ups.

Must-master problem types:

1. Matrix rank.
2. Least squares projection.
3. Eigenvalues and eigenvectors.
4. PCA by covariance matrix.
5. Conditional expectation.
6. Variance of sums.
7. Maximum likelihood.
8. Gradient descent on quadratic functions.
9. Convex vs non-convex losses.
10. Bias-variance tradeoff.

### Intermediate C: Classical ML Workflow

Sources:

- scikit-learn MOOC: https://inria.github.io/scikit-learn-mooc/
- UCI: https://archive.ics.uci.edu/
- Kaggle Playground: https://www.kaggle.com/competitions/playground-series
- DrivenData: https://www.drivendata.org/competitions/

Do:

1. Complete the scikit-learn MOOC exercises.
2. Solve 15 UCI datasets.
3. Enter 3 Kaggle Playground competitions.
4. Enter 2 DrivenData intermediate practice competitions.
5. Build one reusable tabular pipeline.

Target:

- 30 to 50 scikit-learn exercises.
- 15 dataset notebooks.
- 5 competition-style submissions.
- 1 reusable pipeline.

Required dataset types:

1. Binary classification.
2. Multiclass classification.
3. Regression.
4. Imbalanced classification.
5. Missing data.
6. Categorical-heavy tabular data.
7. Time-like split.
8. Dataset with possible leakage.

Must-master problem types:

1. Stratified split.
2. Pipeline with preprocessing.
3. ColumnTransformer.
4. GridSearchCV and RandomizedSearchCV.
5. Calibration.
6. Class weighting.
7. Feature importance.
8. Permutation importance.
9. Error analysis by subgroup.
10. Reproducible experiment table.

### Intermediate D: Deep Learning

Sources:

- D2L: https://d2l.ai/
- CS231n assignment page: https://cs231n.stanford.edu/assignments.html
- CS224n: https://web.stanford.edu/class/cs224n/

Do:

1. D2L chapters on MLPs, CNNs, optimization, regularization.
2. CS231n Assignment 1.
3. CS224n Assignment 1.
4. CS224n Assignment 2 if you are ready for derivatives and parsing.
5. Build a PyTorch training template from scratch.

Target:

- 40 D2L exercises.
- 1 full CS231n assignment.
- 1 to 2 CS224n assignments.
- 5 PyTorch models trained from scratch.

Must-master problem types:

1. Manual backprop for a tiny network.
2. Softmax and cross-entropy.
3. Initialization effects.
4. Vanishing gradients.
5. Overfitting control.
6. Batch normalization.
7. Dropout.
8. Adam vs SGD.
9. Learning-rate scheduling.
10. Validation curve interpretation.

### Intermediate E: Computer Vision

Sources:

- CS231n: https://cs231n.github.io/
- CS231n assignments: https://cs231n.stanford.edu/assignments.html
- PyTorch CIFAR-10: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Oxford-IIIT Pets: https://www.robots.ox.ac.uk/~vgg/data/pets/
- COCO: https://cocodataset.org/

Do:

1. Train MNIST from scratch.
2. Train CIFAR-10 from scratch.
3. Fine-tune a pretrained ResNet.
4. Train a simple segmentation model on Oxford-IIIT Pets.
5. Run inference with an object detector on COCO-like images.
6. Write a convolution-shape calculator.

Target:

- 4 image classification notebooks.
- 1 segmentation notebook.
- 1 object detection demo.
- 50 convolution-shape drills.

Must-master problem types:

1. CNN shape calculation.
2. Receptive field intuition.
3. Augmentation effects.
4. Transfer learning.
5. Freezing vs fine-tuning.
6. Confusion matrix for image classes.
7. Segmentation mask interpretation.
8. Object detection boxes.
9. IoU.
10. mAP intuition.

### Intermediate F: NLP And Audio

Sources:

- CS224n: https://web.stanford.edu/class/cs224n/
- Hugging Face NLP/LLM course: https://huggingface.co/learn/nlp-course
- Hugging Face Audio course: https://huggingface.co/learn/audio-course/chapter0/introduction
- TensorFlow Speech Commands: https://www.tensorflow.org/datasets/catalog/speech_commands

Do:

1. Train bag-of-words text classifier.
2. Train TF-IDF logistic regression classifier.
3. Fine-tune a small transformer for classification.
4. Use a pretrained QA model.
5. Implement nearest-neighbor retrieval with embeddings.
6. Train or fine-tune a small speech-command classifier.
7. Try an ASR pipeline.

Target:

- 4 NLP notebooks.
- 2 transformer fine-tunes.
- 2 audio notebooks.
- 30 tokenization/attention drills.

Must-master problem types:

1. Tokenization.
2. Vocabulary and unknown tokens.
3. Padding and attention masks.
4. Word embeddings.
5. Cosine similarity retrieval.
6. Encoder-only vs decoder-only vs encoder-decoder.
7. Transformer attention shapes.
8. Sequence classification.
9. Question answering span outputs.
10. Audio classification from spectrogram or pretrained encoder.

## 5. Advanced Ladder

Advanced goal: become IOAI-ready.

At this level, you stop only doing textbook exercises. You start solving full tasks where the hard part is choosing the right experiment.

### Advanced A: Official IOAI Tasks

Sources:

- IOAI 2024 tasks page: https://ioai-official.org/2024-tasks/
- IOAI 2024 repo: https://github.com/IOAI-official/IOAI-2024
- IOAI 2025 repo: https://github.com/IOAI-official/IOAI-2025

Do all official tasks you can run locally or in Colab.

IOAI 2024:

1. Scientific at-home ML task.
2. Scientific at-home NLP task.
3. Scientific at-home CV task.
4. Scientific on-site ML task.
5. Scientific on-site NLP task.
6. Scientific on-site CV task.
7. Practical generative image/video task if you want broad coverage.

IOAI 2025:

1. Radar.
2. Chicken Counting.
3. Concepts.
4. Restroom Icon Matching.
5. Antique Painting Authentication.
6. Pixel Efficiency.
7. GAITE tasks if available and runnable.

Target:

- At least 8 official IOAI tasks.
- At least 3 full write-ups.
- At least 1 reimplementation of an official solution.

For each task, do:

1. Read statement.
2. Build dumb baseline.
3. Build reasonable baseline.
4. Improve with 5 experiments.
5. Compare to official/reference solution.
6. Write a report.
7. Re-solve after 2 weeks.

### Advanced B: National And Regional AI Olympiad Tasks

Source:

- Awesome IOAI tasks: https://github.com/open-cu/awesome-ioai-tasks
- USAAIO past problems: https://www.usaaio.org/past-problems

Do:

1. USAAIO 2025 Round 1.
2. USAAIO 2025 Round 2.
3. USAAIO 2026 Round 1.
4. USAAIO 2026 Round 2 Day 1.
5. USAAIO 2026 Round 2 Day 2.
6. NEOAI 2025 tasks listed in the awesome IOAI index.
7. Kazakhstan selection tasks.
8. Polish AI Olympiad tasks.
9. Hungarian AI Olympiad tasks.
10. Romania ONIA practice tasks if accessible.

Target:

- 10 national/regional tasks.
- 4 timed attempts.
- 4 detailed write-ups.
- 2 tasks solved twice using a better method.

Recommended order:

1. USAAIO Round 1 style written/data problems.
2. USAAIO Round 2.
3. Kazakhstan task set.
4. NEOAI tasks.
5. Polish OAI tasks.
6. Hungarian HAIO tasks.

Why this matters:

- IOAI tasks are young, so national repositories are a major source of realistic practice.
- They cover feature engineering, embeddings, fine-tuning, CV, NLP, audio, and robustness.
- They often include unusual constraints, which is exactly the point.

### Advanced C: University Assignments

Sources:

- CS231n assignments: https://cs231n.stanford.edu/assignments.html
- CS224n: https://web.stanford.edu/class/cs224n/
- D2L: https://d2l.ai/

Do:

1. CS231n Assignment 2.
2. CS231n Assignment 3.
3. CS224n Assignment 3.
4. CS224n Assignment 4 if publicly accessible.
5. D2L exercises on modern CNNs, attention, transformers, optimization, and generative models.

Target:

- 2 CS231n assignments.
- 2 CS224n assignments.
- 50 D2L advanced exercises.
- 1 final project-style model.

Must-master problem types:

1. Implement layers.
2. Debug tensor shapes.
3. Implement attention.
4. Evaluate pretrained models.
5. Fine-tune models.
6. Use self-supervised representations.
7. Interpret embeddings.
8. Analyze failure cases.

### Advanced D: Serious Competitions

Sources:

- Kaggle competitions: https://www.kaggle.com/competitions
- Kaggle Playground: https://www.kaggle.com/competitions/playground-series
- DrivenData: https://www.drivendata.org/competitions/

Do:

1. One Kaggle Playground competition to completion.
2. One tabular competition beyond beginner level.
3. One computer vision competition.
4. One NLP competition.
5. One audio or multimodal competition.
6. One DrivenData competition with social-impact data.

Suggested tasks:

1. Kaggle Titanic, if not done.
2. Kaggle House Prices, if not done.
3. Kaggle Digit Recognizer.
4. Kaggle NLP Getting Started: disaster tweets.
5. Kaggle BirdCLEF for audio.
6. DrivenData Reboot: Box-Plots for Education.
7. DrivenData United Nations Millennium Development Goals.
8. DrivenData Warm Up tasks for quick refreshers.

Target:

- 6 competition-style projects.
- 1 top 50 percent finish.
- 1 top 25 percent finish.
- 6 public/private leaderboard reflection notes.

Competition rule:

Do not just chase leaderboard score. For each competition, produce:

1. Baseline notebook.
2. EDA notebook.
3. Validation explanation.
4. Model comparison table.
5. Error analysis.
6. Final write-up.

### Advanced E: Full Mock IOAI

Do four mock contests.

Mock 1: Selection reasoning

- 40 questions.
- 120 minutes.
- No notes.
- Topics: stats, probability, vectors, convolution, simple classifiers, Python reasoning.

Mock 2: Tabular ML

- 5 hours.
- New dataset.
- Submit predictions.
- Write 1-page report.

Mock 3: CV or NLP

- 6 hours.
- Use pretrained model if allowed.
- Must include baseline and improvement.

Mock 4: Official IOAI style

- 8 hours.
- Choose one official IOAI task or national selection task.
- Produce final notebook and report.

Pass condition:

- You understand the problem within 30 minutes.
- You build a baseline within 90 minutes.
- You improve the score at least twice.
- You can explain your final method.

## 6. Expert Ladder

Expert goal: surpass IOAI requirements.

At this level, you should become comfortable with open-ended, ambiguous, research-like tasks.

### Expert A: Reproduce Strong Solutions

Sources:

- IOAI 2025 repo: https://github.com/IOAI-official/IOAI-2025
- IOAI 2024 repo: https://github.com/IOAI-official/IOAI-2024
- Awesome IOAI tasks: https://github.com/open-cu/awesome-ioai-tasks
- Kaggle competitions and write-ups: https://www.kaggle.com/competitions

Do:

1. Pick one official IOAI 2025 task.
2. Reproduce the reference solution.
3. Replace one component with your own idea.
4. Compare scores.
5. Write a technical report.

Then repeat with:

1. One NLP task.
2. One CV task.
3. One tabular ML task.
4. One audio or multimodal task.

Target:

- 5 reproduced strong solutions.
- 5 ablation reports.
- 1 public portfolio repository or clean local archive.

### Expert B: Paper And Theory Problems

Source:

- Pen and Paper Exercises in Machine Learning: https://arxiv.org/abs/2206.13446
- MML: https://mml-book.github.io/book/mml-book.pdf
- D2L: https://d2l.ai/

Do:

1. 20 linear algebra/theory exercises.
2. 20 optimization exercises.
3. 20 probabilistic modeling exercises.
4. 10 graphical model exercises.
5. 10 variational inference or Monte Carlo exercises.

Target:

- 80 high-density theory problems.

Why this matters:

- IOAI does not require full research-level ML theory.
- But going beyond IOAI means you can reason when libraries stop being obvious.
- You will learn to recognize assumptions behind methods.

### Expert C: Build Your Own Mini-Benchmarks

Create your own tasks:

1. Tabular challenge with hidden test split.
2. Image classification challenge with distribution shift.
3. Text classification challenge with adversarial examples.
4. Audio classification challenge with noise shift.
5. Embedding retrieval challenge.
6. Multimodal classification challenge.

For each benchmark, create:

1. Dataset.
2. Baseline.
3. Metric.
4. Leaderboard script.
5. Failure cases.
6. Strong solution.

Target:

- 3 complete mini-benchmarks.

This is beyond IOAI because it forces you to think like a task author.

### Expert D: Model Robustness And Evaluation

Do projects:

1. Train a classifier and attack it with corrupted inputs.
2. Compare calibration before and after temperature scaling.
3. Test a text classifier on paraphrases.
4. Test an image classifier on blur, crop, color shift, and noise.
5. Test an audio model on background noise.
6. Compare in-distribution and out-of-distribution performance.
7. Build an error taxonomy.
8. Write a "model card" for your system.

Target:

- 8 robustness projects.

Must-master problem types:

1. Distribution shift.
2. Data leakage.
3. Overfitting to public leaderboard.
4. Calibration.
5. OOD detection.
6. Adversarial examples.
7. Fairness and subgroup error.
8. Reproducibility.

### Expert E: Capstone Projects

Choose 3:

1. Build an end-to-end image classifier with deployment-ready inference script.
2. Build an object detector baseline on a small custom dataset.
3. Build a U-Net segmentation model.
4. Fine-tune a small language model with LoRA.
5. Build a retrieval-augmented QA system over your own notes.
6. Build a speech-command classifier.
7. Build a multimodal classifier combining image and text.
8. Reimplement a simplified transformer from scratch.
9. Reimplement logistic regression, decision tree, and random forest from scratch.
10. Create an IOAI-style mock contest for another student.

Target:

- 3 capstones.
- 3 reports.
- 3 reusable repositories or clean folders.

## 7. Original Problem Bank

These are custom problems to fill the gaps between official sources.

### Beginner Original Problems

1. Given `[2, 4, 4, 4, 5, 5, 7, 9]`, compute mean, population variance, sample variance, and standard deviation.
2. Duplicate the dataset from problem 1 exactly. Which quantities change?
3. Group A has 10 students with mean 70. Group B has 20 students with mean 85. Find the combined mean.
4. A dataset has 5 numbers with mean 12. Four numbers are `8, 10, 15, 20`. Find the fifth.
5. A model predicts `[1, 0, 1, 1, 0]`; truth is `[1, 0, 0, 1, 1]`. Compute accuracy, precision, recall, F1.
6. A disease affects 1 percent of people. A test is 95 percent sensitive and 90 percent specific. If positive, what is the probability of disease?
7. Bag A has 3 red and 2 blue balls. Bag B has 1 red and 4 blue balls. Pick bag A with probability 0.7. What is `P(red)`?
8. If `P(A)=0.3`, `P(B)=0.5`, and `P(A and B)=0.2`, compute `P(A | B)` and decide whether A and B are independent.
9. Compute Euclidean distance between `(1, 2, 3)` and `(4, 6, 3)`.
10. Compute cosine similarity between `(1, 0, 1)` and `(0, 1, 1)`.
11. Given `a=(2,1)`, `b=(5,5)`, and `c=(1,4)`, which point is nearest to `a`?
12. Given embeddings `king=(3,4)`, `man=(1,2)`, `woman=(2,3)`, compute `king - man + woman`.
13. Input image is `32 x 32`, kernel `3 x 3`, stride 1, padding 0. Find output size.
14. Input image is `28 x 28`, kernel `5 x 5`, stride 1, padding 2. Find output size.
15. Input image is `64 x 64`, kernel `3 x 3`, stride 2, padding 1. Find output size.
16. A classifier counts letters A and B. Class 0 prototype is `(A=5,B=1)`, class 1 prototype is `(A=1,B=5)`. Classify `AABABB`.
17. A text classifier uses word counts for `good`, `bad`, `fast`, `slow`. Build feature vector for `"good fast fast bad"`.
18. Count all paths of length 2 on a `2 x 2` grid using 4-neighbor moves, starting at top-left, revisits allowed.
19. Count all paths of length 2 on the same grid, revisits forbidden.
20. In an interval game on `[0,1]`, one move forbids radius `0.1` around a chosen point. How much length can one move cover away from boundaries?
21. Fit model `y = alpha x` for points `(1,2)`, `(2,4)`, `(3,6)` by minimizing MSE. Find alpha.
22. Fit model `y = alpha x` for `(1,1)`, `(2,2)`, `(3,5)`. Try alpha values 1, 1.5, 2 and compare MSE.
23. For confusion matrix `TP=12, FP=3, FN=5, TN=80`, compute all metrics.
24. A random variable has values 0, 1, 2 with probabilities 0.2, 0.5, 0.3. Compute expectation and variance.
25. Write Python to brute-force nearest neighbor among 5 vectors.

### Intermediate Original Problems

1. Prove that duplicating a dataset keeps the same population variance.
2. Show how exact duplication changes sample variance denominator from `n-1` to `2n-1`.
3. Derive the MSE-minimizing alpha for `y = alpha x`.
4. Derive the MSE-minimizing alpha and beta for `y = alpha x + beta`.
5. Given 6 points, compute linear regression by hand using covariance and variance.
6. Build a probability tree with three stages and compute a posterior probability.
7. Given a `5 x 5` grid with obstacles, count paths of length 4 using 8-neighbor movement.
8. Solve the same grid problem with DP over time.
9. Solve the same grid problem with matrix exponentiation.
10. Given 10 two-dimensional vectors, cluster them manually into 2 groups with k-means for 2 iterations.
11. Given a covariance matrix, compute the first principal component direction if the matrix is diagonal.
12. Given train error decreasing and validation error increasing, diagnose overfitting and propose fixes.
13. Given imbalanced labels 95 percent negative, explain why accuracy is misleading.
14. Given a ROC-like threshold table, choose threshold for high recall.
15. Compute output sizes for three convolution/pooling layers in sequence.
16. Compute number of parameters in a convolution layer.
17. Compute number of parameters in a dense layer after flattening a CNN feature map.
18. Given token IDs and attention mask, identify padded positions.
19. Given query/key/value dimensions, compute attention score matrix shape.
20. Build a TF-IDF vector for three short documents.
21. Use Naive Bayes by hand on two classes with word counts.
22. Compare kNN with Euclidean distance vs cosine similarity on the same vectors.
23. Write a `GridSearchCV` plan for kNN, SVM, random forest.
24. Design a validation split for time-ordered data.
25. Find three possible sources of leakage in a Kaggle notebook.

### Advanced Original Problems

1. Create a synthetic dataset where train accuracy is high but validation accuracy is random. Explain why.
2. Create a synthetic dataset where a leakage feature gives perfect validation score. Remove it.
3. Implement logistic regression from scratch with gradient descent.
4. Implement k-means from scratch and test it on blobs.
5. Implement PCA from scratch using covariance and eigendecomposition.
6. Implement a small MLP in PyTorch without using high-level trainer code.
7. Train a CNN on a small image dataset and run 5 augmentation experiments.
8. Fine-tune a pretrained ResNet and compare frozen vs unfrozen backbone.
9. Build a bag-of-words text classifier and compare against a transformer.
10. Build an embedding retrieval system and evaluate top-k accuracy.
11. Create an adversarial text perturbation set for a classifier.
12. Train an audio classifier on clean audio and evaluate noisy audio.
13. Build a segmentation baseline and compute IoU.
14. Build an object detection inference demo and manually evaluate 20 images.
15. Reproduce an official IOAI baseline from a repository.
16. Improve an IOAI baseline with only preprocessing changes.
17. Improve an IOAI baseline with only model changes.
18. Improve an IOAI baseline with only validation changes.
19. Write a 2-page technical report for a model you trained.
20. Take a failed model and identify the top 5 failure modes.

### Expert Original Problems

1. Create a new IOAI-style tabular task with hidden leakage traps.
2. Create a new IOAI-style NLP task where labels require semantic similarity, not keyword matching.
3. Create a new IOAI-style CV task with train/test distribution shift.
4. Create a new IOAI-style audio task with background-noise shift.
5. Build a mini-leaderboard script for one of your tasks.
6. Write a reference solution and a weak baseline for one of your tasks.
7. Reproduce a top Kaggle solution idea on a smaller dataset.
8. Write ablations proving which component mattered.
9. Build a model card for one trained model.
10. Give your task to another student and revise it based on their confusion.

## 8. Domain-Specific Problem Menus

Use these menus when you feel weak in a particular area.

### Probability And Statistics Menu

Beginner:

1. OpenIntro Chapter 1: 20 exercises.
2. OpenIntro Chapter 2: 20 exercises.
3. OpenIntro Chapter 3: 20 exercises.
4. Custom Bayes table drills: 20.

Intermediate:

1. CS109 psets 1 to 4.
2. OpenIntro regression exercises.
3. Simulate probability problems in Python.

Advanced:

1. CS109 psets 5 to 8.
2. Derive MLE for Bernoulli, Gaussian mean, logistic regression.
3. Write simulation checks for all derivations.

Expert:

1. Pen and Paper Exercises in ML: probabilistic modeling sections.
2. Build hidden Markov model inference by hand and in code.

### Linear Algebra And Optimization Menu

Beginner:

1. Vectors, dot product, norms.
2. Matrix multiplication.
3. Solving systems.
4. Euclidean nearest-neighbor drills.

Intermediate:

1. MIT 18.06 psets 1 to 5.
2. Least squares.
3. Eigenvectors.
4. PCA.

Advanced:

1. MIT 18.06 psets 6 to 10.
2. MML optimization chapters.
3. Derive gradients for MSE, logistic loss, softmax loss.

Expert:

1. Prove projection theorem.
2. Derive backprop for a two-layer network.
3. Analyze conditioning and learning-rate stability.

### Algorithms Menu

Beginner:

1. CSES Introductory Problems.
2. Project Euler 1 to 25.
3. Grid DFS drills.

Intermediate:

1. AtCoder DP all 26.
2. CSES Dynamic Programming all 19.
3. CSES Sorting and Searching 20.
4. CSES Graph Algorithms 15.

Advanced:

1. CSES remaining graph problems.
2. CSES Range Queries.
3. CSES Mathematics.
4. Project Euler 76 to 150.

Expert:

1. Implement efficient nearest-neighbor search.
2. Implement beam search.
3. Implement dynamic programming for sequence alignment.
4. Implement Viterbi decoding.

### Classical ML Menu

Beginner:

1. Iris classification.
2. Wine classification.
3. Breast cancer classification.
4. Diabetes regression.
5. Titanic.

Intermediate:

1. scikit-learn MOOC exercises.
2. 15 UCI datasets.
3. 3 Kaggle Playground competitions.

Advanced:

1. Imbalanced dataset with PR AUC.
2. Time-like split.
3. Feature leakage challenge.
4. Model calibration challenge.
5. Interpretability report.

Expert:

1. Build an AutoML-lite search loop.
2. Build a robust validation framework.
3. Reproduce a tabular competition write-up.

### Deep Learning Menu

Beginner:

1. MLP on MNIST.
2. Manual training loop.
3. Track train/validation loss.
4. Add dropout.

Intermediate:

1. CNN on CIFAR-10.
2. ResNet fine-tuning.
3. CS231n Assignment 1.
4. D2L optimization exercises.

Advanced:

1. CS231n Assignments 2 and 3.
2. Transformer attention implementation.
3. Self-supervised learning demo.
4. Diffusion or CLIP experiment.

Expert:

1. Reimplement a mini-transformer.
2. Fine-tune with LoRA.
3. Reproduce a paper result on a small dataset.

### Computer Vision Menu

Beginner:

1. Convolution output shape drills.
2. MNIST.
3. CIFAR-10.
4. Digit Recognizer.

Intermediate:

1. Transfer learning with ResNet.
2. Augmentation experiments.
3. Oxford-IIIT Pets segmentation.
4. Object detector inference.

Advanced:

1. Train segmentation model.
2. Train/fine-tune detector.
3. Interpret misclassified images.
4. Run robustness tests.

Expert:

1. Reproduce an IOAI CV task.
2. Build distribution-shift image challenge.
3. Compare CNN, ViT, CLIP embeddings.

### NLP Menu

Beginner:

1. Bag-of-words classifier.
2. TF-IDF classifier.
3. Word-vector similarity.
4. Hugging Face pipeline tasks.

Intermediate:

1. Fine-tune transformer classifier.
2. Question answering with pretrained model.
3. Embedding retrieval.
4. CS224n Assignment 1.

Advanced:

1. CS224n Assignments 2 to 4.
2. Fine-tune a small LM.
3. Evaluate hallucination detection.
4. Build RAG over notes.

Expert:

1. Reproduce an IOAI NLP task.
2. Build adversarial NLP benchmark.
3. Evaluate calibration and refusal/failure cases.

### Audio Menu

Beginner:

1. Load waveform.
2. Plot waveform.
3. Plot spectrogram.
4. Run pretrained classifier.

Intermediate:

1. Speech Commands classifier.
2. Music genre classifier.
3. Fine-tune HuBERT-like audio encoder if compute allows.

Advanced:

1. BirdCLEF-style audio classification.
2. Noise robustness experiments.
3. Audio embedding retrieval.

Expert:

1. Build noisy-speech benchmark.
2. Compare Whisper-style ASR vs audio classification features.
3. Build multimodal audio + text classifier.

## 9. The Four Promotion Gates

### Gate 1: Beginner To Intermediate

You pass if:

1. You solve 150 beginner problems.
2. You finish Titanic and Digit Recognizer baselines.
3. You can compute convolution shapes without notes.
4. You can explain Bayes rule with a table.
5. You can train and evaluate a scikit-learn classifier.

Suggested exam:

- 20 math/stat/prob questions.
- 10 vector questions.
- 10 convolution questions.
- 1 Python grid-counting problem.
- 1 scikit-learn mini-notebook.

### Gate 2: Intermediate To Advanced

You pass if:

1. You solve AtCoder DP A to Z.
2. You complete at least 3 Kaggle/DrivenData submissions.
3. You finish CS231n Assignment 1 or equivalent.
4. You finish one transformer fine-tuning task.
5. You write one serious model comparison report.

Suggested exam:

- 2-hour written math exam.
- 2-hour coding algorithm exam.
- 4-hour ML dataset task.

### Gate 3: Advanced To Expert

You pass if:

1. You finish at least 8 official/national AI olympiad tasks.
2. You produce 3 full write-ups.
3. You can build a baseline in 90 minutes.
4. You can improve the baseline with disciplined experiments.
5. You can explain why your validation method is trustworthy.

Suggested exam:

- One official IOAI or national task.
- 8 hours.
- Final notebook plus report.

### Gate 4: Beyond IOAI

You pass if:

1. You reproduce a strong official or competition solution.
2. You perform meaningful ablations.
3. You build your own benchmark.
4. You can critique your own model honestly.
5. You can teach the task to someone else.

Suggested exam:

- Create an IOAI-style task.
- Provide baseline, dataset, metric, and reference solution.
- Let another person attempt it.
- Improve the statement based on their errors.

## 10. Weekly Structure

Use this if you want a sustainable schedule.

### Light Week: 8 to 10 hours

- 2 hours math.
- 2 hours algorithms.
- 2 hours ML notebook.
- 1 hour error log.
- 1 to 3 hours reading or videos.

### Serious Week: 15 to 20 hours

- 4 hours math.
- 4 hours algorithms.
- 6 hours ML/CV/NLP/audio.
- 2 hours write-up.
- 2 to 4 hours official task or competition.

### Camp Week: 30 to 40 hours

- 2 mock contests.
- 1 official task.
- 1 Kaggle/DrivenData push.
- Daily write-up.
- Daily review of mistakes.

## 11. Minimum Path If Time Is Short

If you have only 6 to 8 weeks:

1. TLX OSN AI exhibition set.
2. OpenIntro probability/statistics selected exercises.
3. MIT 18.06 psets 1 to 3.
4. AtCoder DP A to Z.
5. scikit-learn MOOC core exercises.
6. Titanic, House Prices, Digit Recognizer.
7. PyTorch CIFAR-10 tutorial.
8. Hugging Face text classification.
9. 2 official IOAI 2025 tasks.
10. 1 USAAIO round.

This is enough to become selection-competitive.

## 12. Maximum Path If You Want To Go Beyond

If you have 6 to 12 months:

1. Finish beginner ladder.
2. Finish intermediate ladder.
3. Do all official IOAI 2024 and 2025 tasks you can access.
4. Do USAAIO 2025 and 2026.
5. Do 10 tasks from the awesome IOAI index.
6. Finish CS231n Assignments 1 to 3.
7. Finish CS224n Assignments 1 to 4 or equivalent archived tasks.
8. Do 5 Kaggle/DrivenData competitions seriously.
9. Reproduce 3 strong solutions.
10. Build 1 original IOAI-style benchmark.

This is beyond normal IOAI preparation.

## 13. Final Mastery Checklist

You are ready for IOAI-level tasks when:

- You can solve selection-style statistics/probability/vectors/convolution questions quickly.
- You can implement grid search, DFS, DP, and nearest-neighbor logic.
- You can train classical ML models with proper validation.
- You can explain model metrics and errors.
- You can write PyTorch training loops.
- You can use pretrained models for CV, NLP, and audio.
- You can debug tensor shapes.
- You can prevent leakage.
- You can write clear reports.
- You can handle a new dataset without panic.

You are beyond IOAI when:

- You can reproduce official solutions.
- You can design ablation studies.
- You can build your own benchmark.
- You can reason about robustness and distribution shift.
- You can teach the problem to someone else.
- You can read a new task and create a useful baseline in under 90 minutes.

