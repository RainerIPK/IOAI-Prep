# IOAI / OSN AI Study Guide

Sources scraped:

- IOAI Indonesia syllabus: https://ioai.toki.id/silabus.html
- IOAI Indonesia preparation materials: https://ioai.toki.id/materi.html
- OSN 2025 AI exhibition page: https://ioai.toki.id/ekshibisi.html
- Online selection results/context: https://ioai.toki.id/hasil_seleksi_online.html
- TLX OSN AI 2025 archive/problem A: https://tlx.toki.id/problems/ekshibisi-osn-ai-2025/A
- Official IOAI syllabus PDF: https://ioai-official.org/wp-content/uploads/2025/10/Syllabus.pdf

## How To Use This Guide

There are two overlapping targets:

1. **Ekshibisi / selection readiness**: problem solving, Python, math, probability, statistics, linear algebra, calculus, optimization, and basic AI-style reasoning. This is the fastest path if you are preparing for an Indonesian selection/exhibition format.
2. **Full IOAI readiness**: everything above, plus classical ML, deep learning, computer vision, NLP, audio, and practical model implementation in Python/PyTorch/scikit-learn.

Recommended order:

1. Finish the **Core Foundation Track**.
2. Solve the **Ekshibisi/TLX Practice Track**.
3. Move into the **Full IOAI AI/ML Track**.
4. Regularly revisit the **Mastery Checklist**.

## Priority Map

| Priority | Area | Why it matters |
|---|---|---|
| 1 | Python problem solving | Needed for both selection-style problems and final programming tasks. |
| 1 | Computational thinking and logic | Many selection tasks are story problems requiring modeling, counting, or reasoning. |
| 1 | Probability and statistics | Appears directly in Ekshibisi selection and is foundational for ML. |
| 1 | Linear algebra, vectors, matrices | Needed for embeddings, regression, neural networks, PCA, and CV. |
| 1 | Calculus and optimization | Needed for loss minimization, gradient descent, and model training. |
| 2 | Data handling and visualization | Needed for practical ML work with NumPy, pandas, Matplotlib, and Seaborn. |
| 2 | Classical ML | Regression, classification, clustering, evaluation, and model selection. |
| 3 | Deep learning | Neural networks, backpropagation, PyTorch, training loops, regularization. |
| 3 | Computer vision | CNNs, convolution shapes, image classification, detection, segmentation, transfer learning. |
| 3 | NLP and audio | Text classification, embeddings, transformers, language models, tokenization, audio encoders. |

## Core Foundation Track

### 1. Computational Thinking And Logic

Learn:

- Pattern recognition and rule discovery.
- Logical puzzles and proof-style reasoning.
- Translating story problems into mathematical/programming models.
- Game reasoning: optimal play, winning/losing states, invariants.
- Counting paths and state transitions.

Practice:

- Solve grid/path counting problems by hand, then with code.
- Write small brute-force solvers to test hypotheses.
- For games, compute small cases first and look for a pattern.

Official practice connection:

- TLX Problem A asks about optimal play on a number interval, grid path counting, vector matching, convolution dimensions, and simple classification.

### 2. Python Programming

Learn:

- Variables, data types, expressions, input/output.
- Control flow: `if`, `for`, `while`.
- Functions and recursion.
- Lists, tuples, dictionaries, maps/sets.
- Searching and sorting.
- Writing clean scripts that produce exact numeric/text output.
- Offline-friendly programming: no dependency on online notebooks during a contest unless allowed.

Core resources:

- IOAI Module 1: https://ioai.toki.id/assets/modul/Modul_1_IOAI__Materi_Dasar.pdf
- Python Official Tutorial: https://docs.python.org/3/tutorial/
- Python Data Science Handbook: https://jakevdp.github.io/PythonDataScienceHandbook/
- 100 NumPy Exercises: https://github.com/rougier/numpy-100

Practice goals:

- Implement path counting on a grid.
- Implement linear regression loss calculation.
- Implement Euclidean distance and vector arithmetic.
- Implement simple probability calculations.
- Implement convolution output-size formulas.

### 3. Data Representation And Visualization

Learn:

- Base-n encoding and number representation.
- Reading tables, hierarchies, graphs, entity-relation data.
- Bar charts, histograms, pie charts, scatter plots.
- Recognizing trends, clusters, outliers, and correlation.
- Difference between correlation and causation.

Resources:

- IOAI Module 1: https://ioai.toki.id/assets/modul/Modul_1_IOAI__Materi_Dasar.pdf
- Matplotlib Tutorials: https://matplotlib.org/stable/tutorials/index.html
- Seaborn Tutorial: https://seaborn.pydata.org/tutorial.html
- Google Data Prep: https://developers.google.com/machine-learning/data-prep

Practice goals:

- Load small CSV data with pandas.
- Plot simple histograms/scatter plots.
- Explain what a plot says in one or two sentences.

### 4. Algebra

Learn:

- Functions and function notation.
- Polynomial roots and properties.
- Quadratic equations.
- Arithmetic and geometric sequences.
- Sigma notation and summation.

Resources:

- Khan Academy Math: https://www.khanacademy.org/math
- Mathematics for Machine Learning: https://mml-book.github.io/book/mml-book.pdf

Practice goals:

- Convert a story problem into equations.
- Solve for unknowns symbolically.
- Recognize arithmetic/geometric growth.

### 5. Probability And Combinatorics

Learn:

- Multiplication and addition rules.
- Permutations and combinations.
- Independent and dependent events.
- Conditional probability.
- Bayes theorem.
- Expected-value style reasoning.

Resources:

- IOAI Module 1: https://ioai.toki.id/assets/modul/Modul_1_IOAI__Materi_Dasar.pdf
- Khan Academy probability: https://www.khanacademy.org/math
- Think Bayes: https://greenteapress.com/wp/think-bayes/

Practice goals:

- Build a probability table from conditional data.
- Compute `P(A and B)`, `P(A or B)`, and `P(A | B)`.
- Solve selection problems with rounding rules.

### 6. Statistics

Learn:

- Mean, median, mode.
- Variance and standard deviation.
- Percentiles.
- Random variables and distributions.
- Sample variance intuition.
- Correlation vs causation.

Resources:

- OpenIntro Statistics: https://www.openintro.org/book/os/
- Think Stats: https://allendowney.github.io/ThinkStats2/
- Harvard CS109: https://cs109.github.io/2015/

Practice goals:

- Compute mean/variance by hand and with Python.
- Understand what happens when a dataset is duplicated.
- Combine groups with different means.

### 7. Linear Algebra And Vectors

Learn:

- Vectors and vector arithmetic.
- Euclidean distance.
- Dot product.
- Matrix addition, multiplication, inverse.
- Solving systems of linear equations.
- Embeddings as vector representations.
- PCA intuition.

Resources:

- Mathematics for Machine Learning: https://mml-book.github.io/book/mml-book.pdf
- 3Blue1Brown Linear Algebra: https://www.3blue1brown.com/topics/linear-algebra
- MIT OCW 18.06 Linear Algebra: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

Practice goals:

- Compute distances and nearest neighbors.
- Match two vector spaces using distance preservation.
- Implement matrix operations in NumPy.

### 8. Calculus And Optimization

Learn:

- Derivatives.
- Critical points.
- Minima and maxima.
- Loss functions.
- Gradient descent intuition.
- Learning rate intuition.
- Least squares minimization.

Resources:

- 3Blue1Brown Calculus: https://www.3blue1brown.com/topics/calculus
- Khan Academy Calculus: https://www.khanacademy.org/math
- Google ML Crash Course: https://developers.google.com/machine-learning/crash-course

Practice goals:

- Minimize a quadratic function.
- Fit a line by minimizing mean squared error.
- Explain why gradients point toward faster increase and `-gradient` lowers loss.

## Ekshibisi / Selection Practice Track

The OSN 2025 AI exhibition page says the selection emphasized story problems that can be solved with logic, math, statistics, probability, linear algebra, calculus, optimization, paper/calculator/spreadsheet, or Python. The online selection used a 3-hour format with 5-8 description-based problems, short answers, or output-only style answers.

### Topics In The Ekshibisi Syllabus

Learn all of these before going deep into full ML:

- Computational and logical thinking.
- Python fundamentals.
- Data representation and visualization.
- Algebra.
- Probability.
- Statistics.
- Linear algebra and matrices.
- Calculus and optimization.

### TLX Problem A Topic Breakdown

The provided TLX archive contains one bundle task:

- Problem set: `Ekshibisi OSN AI 2025`
- Problem A title: `Seleksi Ekshibisi OSN AI 2025`
- API-scraped type: `BUNDLE`
- Public archive: https://tlx.toki.id/problems/ekshibisi-osn-ai-2025/A

The worksheet covers:

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

### Official Ekshibisi Practice Resources

Use these in this order:

1. Example story problems: https://ioai.toki.id/assets/contohsoal/Contoh_Soal_Eksebisi.pdf
2. Example programming problem 1: https://colab.research.google.com/drive/1vbYcD1L62w9LCSJaV4SKrqST7PLOH0mY?usp=sharing
3. Example programming problem 2: https://colab.research.google.com/drive/1TnVKxpKUzYwRMvrwW-kiy_c5hcEyZVbJ?usp=sharing
4. Example programming problem 3: https://colab.research.google.com/drive/1z7zv6KebbKEwb4CzxZ0Sn2MzzZFy7aeL?usp=sharing
5. TLX selection archive: https://tlx.toki.id/problems/ekshibisi-osn-ai-2025
6. Final essay problems: https://ioai.toki.id/assets/soal2025/Ekshibisi_Essay_Final_AI.pdf
7. Final coding problems: https://ioai.toki.id/assets/soal2025/EKKA_2025_coding_problems.pdf

Reference/context resources:

- Ekshibisi socialization slides: https://ioai.toki.id/assets/sosialisasi/Materi_Sosialisasi_Ekshibisi_AI_2025.pdf
- Socialization recording: https://youtu.be/q1euPgRfXH8?si=FeRWTh17WAi3GeQd
- Technical meeting video: https://www.youtube.com/live/oBD0ym-xQzM?si=Xs_M9Cy0B5ngPVM-
- Technical meeting slides: https://docs.google.com/presentation/d/1pQ7wEC9CFC9pcboy_VkhcZF2iPhMSQ7s/edit?usp=sharing&ouid=115345245319095023060&rtpof=true&sd=true
- Selection result context: https://ioai.toki.id/hasil_seleksi_online.html

## Full IOAI AI/ML Track

Use the official IOAI 2026 syllabus PDF as the main source for the full international target. It includes topics in four major areas:

1. Foundational skills and classical machine learning.
2. Neural networks and deep learning.
3. Computer vision.
4. Natural language processing and audio.

### 1. Programming And Data Science Fundamentals

Learn:

- Python basics.
- NumPy and pandas.
- Matplotlib and Seaborn.
- scikit-learn.
- PyTorch basics.
- Tensor/multidimensional array manipulation.
- CPU/GPU model training.
- Feature engineering.
- Data processing: missing data, irregular data, imputation, padding, normalization, standardization, train/validation/test splits, basic augmentation, tokenization, vocabulary building, image patching.

Resources:

- IOAI Module 1: https://ioai.toki.id/assets/modul/Modul_1_IOAI__Materi_Dasar.pdf
- NumPy Quickstart: https://numpy.org/doc/stable/user/quickstart.html
- Pandas User Guide: https://pandas.pydata.org/docs/user_guide/index.html
- Pandas Intro Tutorials: https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html
- Python Data Science Handbook: https://jakevdp.github.io/PythonDataScienceHandbook/
- Kaggle Learn: https://www.kaggle.com/learn

### 2. Classical Machine Learning

Learn supervised learning:

- Linear regression.
- Logistic regression.
- L1 and L2 regularization.
- K-nearest neighbors.
- Decision trees.
- Random forests, bagging, gradient boosting.
- Support vector machines.

Learn unsupervised learning:

- K-means clustering.
- PCA.
- t-SNE and UMAP.
- DBSCAN, hierarchical clustering, spectral clustering.

Learn evaluation:

- Accuracy, precision, recall, F1.
- Confusion matrix.
- ROC curves.
- Underfitting and overfitting.
- Hyperparameter tuning.
- Cross-validation.

Resources:

- IOAI Module 2: https://ioai.toki.id/assets/modul/Modul_2_IOAI__Pengenalan_Machine_Learning__ML_.pdf
- Google ML Crash Course: https://developers.google.com/machine-learning/crash-course
- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- scikit-learn MOOC: https://inria.github.io/scikit-learn-mooc/
- ISLR: https://www.statlearning.com/
- ISLR Python labs: https://www.statlearning.com/resources-python
- StatQuest: https://statquest.org/

Practice goals:

- Train a regression and classification model in scikit-learn.
- Explain metrics from a confusion matrix.
- Compare train vs validation error.
- Tune a model with cross-validation.
- Try at least one small dataset from UCI or Kaggle.

Dataset resources:

- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php
- Kaggle Datasets: https://www.kaggle.com/datasets

### 3. Neural Networks And Deep Learning

Learn theory:

- Perceptron.
- Gradient descent.
- Backpropagation.
- Activation functions: ReLU, sigmoid, tanh.
- Loss functions: MSE, MAE, cross entropy.
- Multi-layer perceptrons.
- Data embeddings for text, image, and audio.
- Pooling.
- Attention mechanism.
- Transformers for text and image.

Learn practice:

- PyTorch tensors and modules.
- SGD and mini-batch gradient descent.
- Adam and AdamW.
- Learning rates and convergence.
- Dropout, early stopping, weight decay.
- Weight initialization.
- Batch normalization.
- Autoencoders.
- Full and parameter-efficient fine-tuning.

Resources:

- IOAI Module 3: https://ioai.toki.id/assets/modul/Modul_3_IOAI__Pengenalan_Deep_Learning.pdf
- Dive into Deep Learning: https://d2l.ai
- fast.ai Practical Deep Learning: https://course.fast.ai/
- Deep Learning with Python: https://deeplearningwithpython.io/chapters/

Practice goals:

- Build an MLP in PyTorch.
- Write a training loop.
- Track train and validation loss.
- Add early stopping/dropout.
- Fine-tune a small pretrained model if hardware allows.

### 4. Computer Vision

Learn:

- Convolutional layers.
- Pooling.
- Image classification.
- Object detection: YOLO, SSD, DETR.
- Image segmentation: U-Net.
- Pretrained vision encoders such as ResNet.
- Transfer learning.
- Image augmentation.
- GANs.
- Self-supervised learning for vision.
- Vision-text encoders such as CLIP.
- Diffusion models.

Resources:

- IOAI Module 4: https://ioai.toki.id/assets/modul/Modul_4_IOAI__Pengenalan_Computer_Vision__CV_.pdf
- Stanford CS231n: https://cs231n.github.io/
- Computer Vision: Algorithms and Applications: https://szeliski.org/Book/

Practice goals:

- Compute convolution output dimensions by hand.
- Train/fine-tune an image classifier.
- Use a pretrained ResNet/MobileNet.
- Try basic augmentation and compare validation accuracy.
- Understand what detection and segmentation outputs look like.

### 5. NLP And Audio

Learn:

- Text classification.
- Word embeddings and vector similarity.
- Pretrained text encoders such as BERT.
- Language modeling.
- Encoder-decoder models.
- Transformers and attention.
- Pretrained language models, open-source and API-based.
- Question answering with pretrained models.
- Chatbot basics.
- Fine-tuning methods and limitations: LoRA, adapters, parameter-efficient tuning.
- Basic LLM agents.
- Audio processing with pretrained audio encoders such as HuBERT.
- Audio models such as Qwen-Audio, Whisper, Voxtral.

Resources:

- IOAI Module 5: https://ioai.toki.id/assets/modul/Modul_5_IOAI__Pengenalan_Pengolahan_Bahasa_Alami__NLP_.pdf
- Stanford CS224n: https://web.stanford.edu/class/cs224n/
- Hugging Face NLP Course: https://huggingface.co/learn/nlp-course

Practice goals:

- Compute and compare word/vector similarities.
- Train a simple text classifier.
- Use a pretrained transformer for classification or question answering.
- Tokenize text and understand vocabulary/padding.
- Try a small speech/audio model demo if compute is available.

## Official IOAI Indonesia Modules

Take these first, because they are aligned to the Indonesian IOAI preparation page:

| Module | Topic | Link |
|---|---|---|
| 1 | Math, statistics, probability, optimization, Python basics | https://ioai.toki.id/assets/modul/Modul_1_IOAI__Materi_Dasar.pdf |
| 2 | Machine learning: preprocessing, supervised and unsupervised learning | https://ioai.toki.id/assets/modul/Modul_2_IOAI__Pengenalan_Machine_Learning__ML_.pdf |
| 3 | Deep learning: MLP, CNN, RNN | https://ioai.toki.id/assets/modul/Modul_3_IOAI__Pengenalan_Deep_Learning.pdf |
| 4 | Computer vision | https://ioai.toki.id/assets/modul/Modul_4_IOAI__Pengenalan_Computer_Vision__CV_.pdf |
| 5 | Natural language processing | https://ioai.toki.id/assets/modul/Modul_5_IOAI__Pengenalan_Pengolahan_Bahasa_Alami__NLP_.pdf |

## Extra Resource List From IOAI Indonesia

Math and statistics:

- Mathematics for Machine Learning: https://mml-book.github.io/book/mml-book.pdf
- Khan Academy Math: https://www.khanacademy.org/math
- 3Blue1Brown Linear Algebra: https://www.3blue1brown.com/topics/linear-algebra
- 3Blue1Brown Calculus: https://www.3blue1brown.com/topics/calculus
- MIT OCW 18.06 Linear Algebra: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- OpenIntro Statistics: https://www.openintro.org/book/os/
- Think Stats: https://allendowney.github.io/ThinkStats2/
- Think Bayes: https://greenteapress.com/wp/think-bayes/

Python and data science:

- Python Official Tutorial: https://docs.python.org/3/tutorial/
- TutorialsPoint Python: https://www.tutorialspoint.com/python/index.htm
- Python Data Science Handbook: https://jakevdp.github.io/PythonDataScienceHandbook/
- NumPy Quickstart: https://numpy.org/doc/stable/user/quickstart.html
- 100 NumPy Exercises: https://github.com/rougier/numpy-100
- Pandas User Guide: https://pandas.pydata.org/docs/user_guide/index.html
- Pandas Intro Tutorials: https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html

EDA and visualization:

- Matplotlib Tutorials: https://matplotlib.org/stable/tutorials/index.html
- Seaborn Tutorial: https://seaborn.pydata.org/tutorial.html
- Google Data Prep: https://developers.google.com/machine-learning/data-prep

Machine learning:

- Google ML Crash Course: https://developers.google.com/machine-learning/crash-course
- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- Kaggle Learn: https://www.kaggle.com/learn
- scikit-learn MOOC: https://inria.github.io/scikit-learn-mooc/
- ISLR: https://www.statlearning.com/
- ISLR Python labs: https://www.statlearning.com/resources-python
- StatQuest: https://statquest.org/
- Harvard CS109: https://cs109.github.io/2015/

Datasets and practice:

- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php
- Kaggle Datasets: https://www.kaggle.com/datasets
- IOAI 2024 official tasks archive: https://ioai-official.org/2024-tasks/

Deep learning:

- Dive into Deep Learning: https://d2l.ai
- fast.ai Practical Deep Learning: https://course.fast.ai/
- Deep Learning with Python: https://deeplearningwithpython.io/chapters/

Computer vision:

- Stanford CS231n: https://cs231n.github.io/
- Szeliski computer vision book: https://szeliski.org/Book/

NLP:

- Stanford CS224n: https://web.stanford.edu/class/cs224n/
- Hugging Face NLP Course: https://huggingface.co/learn/nlp-course

## Suggested 12-Week Plan

### Weeks 1-2: Python, Logic, And Math Basics

- Finish IOAI Module 1.
- Review Python basics, lists, dictionaries, functions, recursion.
- Study algebra, combinatorics, probability, descriptive statistics.
- Solve small brute-force/counting problems every day.

Deliverables:

- 20 short Python scripts.
- 1 notebook/script for probability tables.
- 1 script for grid path counting.

### Weeks 3-4: Linear Algebra, Calculus, And Selection Practice

- Study vectors, matrices, dot products, distances.
- Study derivatives and simple optimization.
- Solve the Ekshibisi example story problems.
- Recreate TLX Problem A-style subproblems locally.

Deliverables:

- Vector similarity solver.
- Linear regression loss/minimizer script.
- Convolution output-size calculator.

### Weeks 5-6: Classical ML

- Finish IOAI Module 2.
- Take Google ML Crash Course or scikit-learn MOOC sections.
- Train regression, classification, and clustering models.
- Learn evaluation metrics deeply.

Deliverables:

- 3 scikit-learn mini-projects: regression, classification, clustering.
- A short writeup comparing accuracy, precision, recall, F1.

### Weeks 7-8: Deep Learning

- Finish IOAI Module 3.
- Learn PyTorch basics.
- Build MLPs and CNN basics.
- Practice training loops and validation.

Deliverables:

- PyTorch MLP classifier.
- Training loop with early stopping.
- Experiment showing overfitting vs regularization.

### Weeks 9-10: Computer Vision

- Finish IOAI Module 4.
- Study CS231n notes for CNNs and transfer learning.
- Train or fine-tune an image classifier.
- Practice convolution/stride/padding calculations.

Deliverables:

- Image classifier with transfer learning.
- Convolution dimension cheat sheet.

### Weeks 11-12: NLP, Transformers, Audio Overview, And Mock Contests

- Finish IOAI Module 5.
- Take Hugging Face NLP Course basics.
- Study embeddings, attention, BERT/GPT-style models.
- Do timed practice with TLX/archive/final tasks.

Deliverables:

- Text classifier or question-answering demo.
- One 3-hour mock selection.
- One final-style programming practice session.

## Mastery Checklist

You are ready for selection-style problems when you can:

- Convert story problems into equations, cases, or code.
- Write Python scripts quickly without searching syntax.
- Use lists/dicts/sets comfortably.
- Brute-force small cases and generalize.
- Compute probability and conditional probability from tables.
- Compute mean, variance, and combined means.
- Work with vectors, distances, matrices, and dot products.
- Minimize simple loss functions.
- Calculate convolution output shapes.
- Explain your reasoning clearly under time pressure.

You are ready for full IOAI-style AI/ML when you can:

- Load, clean, split, and visualize data.
- Train and evaluate scikit-learn models.
- Explain underfitting, overfitting, and validation.
- Use cross-validation and hyperparameter tuning.
- Implement a PyTorch training loop.
- Explain gradient descent and backpropagation at a high level.
- Use pretrained models for vision or NLP.
- Understand CNN, transformer, embedding, and attention basics.
- Choose a reasonable model for tabular, text, image, audio, or time-series data.

## What To Do First

If starting today:

1. Read IOAI Module 1.
2. Learn/review Python basics from the official tutorial.
3. Solve the example story problems PDF.
4. Work through TLX Problem A and recreate each subtopic locally.
5. Move to IOAI Module 2 and Google ML Crash Course.

Keep one running notebook or folder of scripts. For every topic, write a tiny implementation:

- Probability table calculator.
- Grid path counter.
- Linear regression fitter.
- Vector nearest-neighbor matcher.
- Convolution shape calculator.
- Simple classifier.

This will prepare you better than passive reading alone.
