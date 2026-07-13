# NLP And Audio Mastery

This is a dense, self-contained course for natural language processing, language models, and audio understanding for IOAI preparation and practical AI work.

It covers:

- Text fundamentals: preprocessing, normalization, tokenization, vocabulary, padding, attention masks.
- Classical NLP: bag-of-words, TF-IDF, Naive Bayes, logistic regression, text classification.
- Embeddings: word vectors, sentence vectors, cosine similarity, nearest neighbors, analogies, retrieval.
- Sequence models: RNNs, LSTMs, encoder-decoder models, attention.
- Transformers: self-attention, multi-head attention, positional encodings, encoder-only, decoder-only, encoder-decoder.
- Pretrained text encoders: BERT-style models, sentence transformers, classification and question answering.
- Language modeling: next-token prediction, decoding, sampling, perplexity, instruction tuning.
- LLM use: open-source models, API-based models, prompt design, limitations, safety, evaluation.
- Fine-tuning: full fine-tuning, adapters, LoRA, PEFT, when not to fine-tune.
- Basic agents: tools, retrieval, memory, planning loops, failure modes.
- Audio: waveform, sampling rate, spectrograms, mel features, ASR, WER/CER, HuBERT, Whisper, Qwen-Audio/Qwen2-Audio, Voxtral.
- Practice: hand problems, coding drills, Hugging Face tasks, PyTorch tasks, audio demos, projects, final exam.

Verified resource map:

- IOAI Module 5: https://ioai.toki.id/assets/modul/Modul_5_IOAI__Pengenalan_Pengolahan_Bahasa_Alami__NLP_.pdf
- Stanford CS224n: https://web.stanford.edu/class/cs224n/
- Hugging Face NLP/LLM Course: https://huggingface.co/learn/nlp-course
- Hugging Face Transformers docs: https://huggingface.co/docs/transformers/index
- Hugging Face PEFT docs: https://huggingface.co/docs/peft/index
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- BERT: https://arxiv.org/abs/1810.04805
- LoRA: https://arxiv.org/abs/2106.09685
- HuBERT: https://arxiv.org/abs/2106.07447
- Whisper: https://arxiv.org/abs/2212.04356
- Qwen-Audio: https://arxiv.org/abs/2311.07919
- Qwen2-Audio: https://arxiv.org/abs/2407.10759
- Voxtral: https://arxiv.org/html/2507.13264v1
- Mistral Voxtral audio docs: https://docs.mistral.ai/studio-api/audio/overview

## 0. How To Use This File

NLP and audio are sequence problems.

Text:

```text
characters -> tokens -> embeddings -> sequence model -> task output
```

Audio:

```text
waveform -> frames/features -> encoder -> text or semantic output
```

Use this study cycle:

```text
read concept -> compute tiny example -> implement small version -> use pretrained model -> inspect errors -> improve
```

You should be able to answer:

- What are the tokens?
- What is the vocabulary?
- What shape is the tensor?
- What does the attention mask hide?
- What task is this model trained for?
- Is the model encoder-only, decoder-only, or encoder-decoder?
- Is the output a class, span, sequence, embedding, transcript, or generated response?
- What metric matches the task?
- What failure mode should I expect?

## 1. What NLP Is

Natural language processing is the field of making computers process human language.

Common tasks:

```text
Text classification: classify sentiment, topic, spam, intent.
Token classification: named entity recognition, part-of-speech tagging.
Question answering: answer from context or generate answer.
Summarization: shorten a document.
Translation: map text from one language to another.
Language modeling: predict next token.
Retrieval: find relevant documents.
Chatbots: maintain multi-turn conversation.
Agents: use tools and external state to complete tasks.
```

Language is difficult because:

- Words are ambiguous.
- Context changes meaning.
- Long-range dependencies matter.
- Syntax and semantics interact.
- Text has spelling, style, slang, code-mixing, and noise.
- A correct answer may require world knowledge.

## 2. What Audio NLP Is

Audio language work connects speech and sound to text or meaning.

Common tasks:

```text
Automatic speech recognition: audio -> transcript.
Speech translation: speech in one language -> text in another.
Speaker diarization: who spoke when?
Audio classification: speech, music, dog bark, alarm.
Speech emotion recognition: happy, angry, neutral.
Audio question answering: answer questions about an audio clip.
Speech-to-speech agents: listen, reason, respond.
Text-to-speech: text -> audio.
```

Audio is difficult because:

- Speech is continuous, not discrete like text.
- Accents vary.
- Background noise matters.
- Speakers overlap.
- Audio can be long.
- Timing matters.
- Transcripts may not capture tone, emotion, or environment.

## 3. Text Preprocessing

Preprocessing converts raw text into a consistent form.

Examples:

```text
"HELLO!!!" -> "hello"
"can't" -> "can not" or keep as "can't"
"I loved it :)" -> tokens including sentiment marker
```

Common steps:

- Lowercasing.
- Unicode normalization.
- Removing or preserving punctuation.
- Expanding contractions.
- Removing HTML.
- Handling URLs, emails, usernames.
- Sentence splitting.
- Tokenization.
- Stopword handling.
- Stemming or lemmatization.

### Important Rule

Do not blindly remove information.

For sentiment:

```text
"not good"
```

If you remove "not", the meaning flips.

For spam:

```text
URLs, dollar signs, all caps
```

may be useful.

For authorship:

```text
punctuation and casing
```

may be useful.

## 4. Regular Expressions

Regex is a pattern language for strings.

Examples:

```text
\d+        one or more digits
\w+        word characters
\s+        whitespace
[A-Z]+     uppercase letters
https?://\S+   URL-ish pattern
```

Use regex for:

- Cleaning HTML snippets.
- Finding numbers.
- Replacing URLs.
- Extracting hashtags.
- Validating simple formats.

Do not use regex to understand grammar or meaning. It is a surface pattern tool.

## 5. Tokenization

Tokenization splits text into units.

Possible token units:

- Characters.
- Words.
- Subwords.
- Bytes.

Example:

```text
"I love NLP!"
word tokens: ["I", "love", "NLP", "!"]
character tokens: ["I", " ", "l", "o", "v", "e", ...]
subword tokens: ["I", "love", "NL", "P", "!"] depending on tokenizer
```

### Why Subwords?

Word-level vocabulary can explode.

Problems:

- Rare words.
- Misspellings.
- Morphology.
- Names.
- New terms.

Subwords solve out-of-vocabulary problems by breaking rare words into pieces.

Example:

```text
"unhappiness" -> "un", "happi", "ness"
```

Tokenizers used in modern models:

- BPE.
- WordPiece.
- SentencePiece.
- Unigram language model tokenization.

### Token IDs

Models do not read strings. They read integer IDs.

```text
"hello world" -> ["hello", "world"] -> [7592, 2088]
```

Token IDs are then mapped to embeddings.

## 6. Vocabulary, Padding, Truncation, Attention Masks

### Vocabulary

A vocabulary maps tokens to IDs.

```text
{"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
```

Special tokens:

- `[PAD]`: padding.
- `[UNK]`: unknown token.
- `[CLS]`: classification token in BERT-like models.
- `[SEP]`: separator token.
- `<bos>`: beginning of sequence.
- `<eos>`: end of sequence.

### Padding

Batches need same length.

Example:

```text
["I like cats"] -> [10, 20, 30]
["Great"] -> [40]
```

Pad second sequence:

```text
[40, 0, 0]
```

### Attention Mask

Attention mask tells model which tokens are real.

```text
tokens:         [40, 0, 0]
attention_mask: [1, 0, 0]
```

Without a mask, the model may attend to padding.

### Truncation

Models have maximum context length.

If text is too long:

- truncate
- chunk
- summarize
- retrieve relevant parts

For QA and RAG, truncating blindly can remove the answer.

## 7. Bag-Of-Words

Bag-of-words represents text by word counts.

Example vocabulary:

```text
["cat", "dog", "love"]
```

Sentence:

```text
"cat cat love"
```

Vector:

```text
[2, 0, 1]
```

BoW ignores order.

These two can have same BoW:

```text
"dog bites man"
"man bites dog"
```

But BoW is still a strong baseline for many classification tasks.

## 8. TF-IDF

TF-IDF means term frequency inverse document frequency.

It increases weight for words frequent in one document but rare across documents.

Term frequency:

```text
how often term appears in document
```

Inverse document frequency:

```text
how rare term is across corpus
```

Common intuition:

```text
"the" appears everywhere -> low value
"photosynthesis" appears in few docs -> high value
```

TF-IDF is useful for:

- Search.
- Text classification baselines.
- Keyword extraction.

## 9. Text Classification

Text classification predicts a label for text.

Examples:

- sentiment: positive/negative/neutral
- spam: spam/not spam
- topic: sports/politics/science
- intent: book_flight/cancel_order/greet

Classical pipeline:

```text
text -> preprocessing -> BoW/TF-IDF -> classifier
```

Strong baselines:

- Naive Bayes.
- Logistic regression.
- Linear SVM.

Transformer pipeline:

```text
text -> tokenizer -> pretrained encoder -> classification head
```

Metrics:

- Accuracy for balanced classes.
- Precision/recall/F1 for imbalanced classes.
- Confusion matrix.

## 10. Naive Bayes For Text

Naive Bayes uses Bayes rule with a simplifying independence assumption.

For class `c` and document words:

```text
P(c | words) proportional to P(c) * product P(word_i | c)
```

It is "naive" because it assumes words are conditionally independent given class.

This assumption is false, but the model often works surprisingly well.

Why useful:

- Fast.
- Good baseline.
- Works well with word counts.
- Interpretable word likelihoods.

Important:

Use smoothing so unseen words do not produce zero probability.

Laplace smoothing:

```text
P(word | class) = (count(word, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

## 11. Logistic Regression For Text

Logistic regression with TF-IDF is a strong baseline.

Model:

```text
score = w dot x + b
probability = sigmoid(score) for binary
```

For multiclass:

```text
softmax over class scores
```

Advantages:

- Fast.
- Interpretable coefficients.
- Strong on small/medium datasets.
- Hard to beat if data is mostly keyword-driven.

Use before fine-tuning a transformer. If transformer cannot beat a strong TF-IDF baseline, inspect your setup.

## 12. Word Embeddings

An embedding maps a word/token to a vector.

Goal:

```text
similar meaning -> nearby vectors
```

Example:

```text
king, queen, man, woman
```

may have meaningful geometric relationships.

Operations:

- Euclidean distance.
- Cosine similarity.
- Nearest neighbors.
- Vector arithmetic.

Cosine similarity:

```text
cos(a, b) = (a dot b) / (||a|| ||b||)
```

### Static vs Contextual Embeddings

Static embedding:

```text
"bank" has one vector
```

Contextual embedding:

```text
"bank" in "river bank" differs from "bank account"
```

Models like BERT produce contextual embeddings.

## 13. Word2Vec

Word2Vec learns word vectors from context.

Two main forms:

### CBOW

Predict center word from surrounding context.

```text
context -> target word
```

### Skip-Gram

Predict surrounding context from center word.

```text
target word -> context
```

Important idea:

Words appearing in similar contexts get similar vectors.

This is distributional semantics:

```text
You shall know a word by the company it keeps.
```

## 14. GloVe

GloVe uses global word co-occurrence statistics.

Instead of predicting local context only, it factorizes information from a word co-occurrence matrix.

Intuition:

```text
meaning is encoded in ratios of co-occurrence probabilities
```

For mastery:

- Know that Word2Vec is predictive/local-context oriented.
- Know that GloVe uses global co-occurrence statistics.
- Know both produce static word vectors.

## 15. Sentence Embeddings And Retrieval

Sentence embeddings map a sentence/document to a vector.

Use cases:

- Semantic search.
- Clustering.
- Duplicate detection.
- RAG retrieval.
- Recommendation.

Simple methods:

- Average word embeddings.
- TF-IDF weighted average.

Modern methods:

- Sentence transformers.
- Contrastively trained encoders.
- Embeddings from API or open-source encoders.

Retrieval workflow:

```text
documents -> embeddings -> vector index
query -> embedding -> nearest neighbors
```

Always evaluate retrieval by inspecting top results.

## 16. RNNs And LSTMs

RNNs process sequences step by step.

```text
h_t = f(x_t, h_{t-1})
```

RNN remembers previous state in hidden vector.

Problems:

- Slow because sequential.
- Hard to learn long dependencies.
- Vanishing/exploding gradients.

LSTM improves RNNs with gates:

- input gate
- forget gate
- output gate
- cell state

LSTMs were important before transformers and still appear in some lightweight or time-series settings.

## 17. Encoder-Decoder Models

Encoder-decoder models map one sequence to another.

Examples:

- translation
- summarization
- question answering generation
- speech recognition

Architecture:

```text
encoder reads input sequence
decoder generates output sequence
```

Before transformers, encoder-decoder often meant RNN/LSTM with attention.

Modern encoder-decoder:

- T5-style text-to-text transformers.
- BART-style denoising seq2seq models.
- Whisper-style audio encoder plus text decoder.

## 18. Attention

Attention lets a model focus on relevant positions.

Scaled dot-product attention:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Query:

```text
what am I looking for?
```

Key:

```text
what do I contain?
```

Value:

```text
what information do I pass?
```

Self-attention:

```text
Q, K, V come from same sequence
```

Cross-attention:

```text
decoder queries attend to encoder keys/values
```

## 19. Transformers

A transformer block contains:

1. Multi-head attention.
2. Residual connection.
3. Layer normalization.
4. Feed-forward network.
5. Residual connection.
6. Layer normalization.

Why transformers became dominant:

- Parallel training over sequence positions.
- Strong long-range modeling.
- Scales well with data and compute.
- Works across text, images, audio, code, and multimodal data.

### Positional Information

Self-attention alone is permutation-invariant. It needs position.

Methods:

- sinusoidal positional encodings
- learned position embeddings
- rotary position embeddings
- relative position bias

For mastery:

```text
content embedding + position information = usable token representation
```

## 20. Transformer Families

### Encoder-Only

Examples:

- BERT.
- RoBERTa.

Good for:

- classification
- token classification
- extractive QA
- embeddings
- reranking

Uses bidirectional context.

### Decoder-Only

Examples:

- GPT-style models.
- Llama-style models.
- Mistral-style models.
- Qwen text models.

Good for:

- generation
- chat
- code completion
- agents

Uses causal mask:

```text
token can only attend to previous tokens
```

### Encoder-Decoder

Examples:

- T5.
- BART.
- Whisper.

Good for:

- translation
- summarization
- seq2seq QA
- speech recognition

## 21. BERT

BERT stands for Bidirectional Encoder Representations from Transformers.

Core idea:

```text
pretrain deep bidirectional representations from unlabeled text
```

Pretraining tasks in original BERT:

- masked language modeling
- next sentence prediction

Masked language modeling:

```text
"The cat sat on the [MASK]." -> predict "mat"
```

Fine-tuning:

```text
pretrained BERT + small task head -> train on labeled task
```

BERT is good for:

- text classification
- NER
- extractive QA
- sentence pair classification

It is not naturally a free-form text generator.

## 22. Language Modeling

Language modeling predicts tokens.

Autoregressive language model:

```text
P(text) = product P(token_t | previous tokens)
```

Training target:

```text
predict next token
```

Example:

```text
Input: "The capital of France is"
Target: "Paris"
```

Loss:

```text
cross entropy over vocabulary
```

Perplexity:

```text
exp(average cross entropy)
```

Lower perplexity means the model assigns higher probability to the true text, but lower perplexity does not always mean better answers for every task.

## 23. Decoding And Sampling

A language model outputs logits over vocabulary.

Decoding chooses next token.

### Greedy Decoding

Always choose highest probability token.

Pros:

- deterministic
- simple

Cons:

- can be repetitive
- may miss better sequence-level output

### Beam Search

Keeps several candidate sequences.

Useful for:

- translation
- summarization in some settings

Can produce generic outputs.

### Sampling

Sample from probability distribution.

Parameters:

- temperature
- top-k
- top-p / nucleus sampling

Temperature:

```text
lower -> more deterministic
higher -> more random
```

Top-k:

```text
sample from k most likely tokens
```

Top-p:

```text
sample from smallest set whose cumulative probability >= p
```

## 24. Question Answering

### Extractive QA

Given context and question, predict answer span inside context.

Example:

```text
Context: "Paris is the capital of France."
Question: "What is the capital of France?"
Answer span: "Paris"
```

BERT-style models can predict:

- start token
- end token

### Generative QA

Model generates answer text.

Useful when:

- answer is not exact span
- multiple sources
- reasoning/summarization needed

Risk:

- hallucination

Always ground answers in provided context when factual accuracy matters.

## 25. Chatbot Basics

A chatbot is a system for multi-turn interaction.

Basic components:

- conversation history
- system/developer instruction
- user message
- model response
- optional tools
- optional memory
- safety filters

Key problems:

- context window limits
- ambiguity
- hallucination
- prompt injection
- stale knowledge
- user intent tracking
- tool errors

Good chatbot design:

- clear system purpose
- grounded data
- refusal boundaries
- tool verification
- concise memory
- evaluation set

## 26. LLMs: Open-Source And API-Based

Open-source/open-weights models:

- run locally or on your servers if hardware allows
- inspect weights depending on license
- fine-tune or adapt
- manage deployment yourself

API-based models:

- hosted by provider
- easier to use
- often strong and regularly updated
- cost per token/request
- less control over weights
- data/privacy terms matter

Use open-source when:

- you need local control
- you need customization
- you need predictable deployment
- licensing fits

Use APIs when:

- you need strong models quickly
- you do not want infrastructure
- you need multimodal or voice capabilities now

For high-stakes tasks, evaluate both on your actual data.

## 27. Fine-Tuning

Fine-tuning updates model weights on task data.

Types:

### Full Fine-Tuning

Update all model parameters.

Pros:

- maximum flexibility

Cons:

- expensive
- more memory
- higher overfitting risk
- can degrade general ability

### Linear Probing

Freeze encoder and train a small head.

Good for:

- classification
- small datasets
- quick baselines

### Adapters

Insert small trainable modules into transformer layers.

Freeze main model.

Pros:

- fewer trainable parameters
- multiple task adapters possible

Cons:

- may add inference latency
- architecture changes

### LoRA

LoRA freezes pretrained weights and trains low-rank update matrices.

For a weight matrix `W`, learn:

```text
W + BA
```

where `A` and `B` are low-rank matrices.

Pros:

- much fewer trainable parameters
- lower memory
- common for LLM adaptation
- can merge updates in some cases

Cons:

- rank and target modules matter
- can overfit bad data
- does not magically add missing knowledge if data is poor

### When Not To Fine-Tune

Do not fine-tune if:

- prompt + retrieval solves the task
- you have tiny low-quality data
- task needs fresh facts
- evaluation is unclear
- safety requirements are not understood

Try first:

1. better prompting
2. retrieval
3. reranking
4. tool use
5. small classifier
6. PEFT
7. full fine-tuning

## 28. Basic LLM Agents

An agent is a model-driven loop that can use tools or take actions.

Simple agent loop:

```text
observe -> reason -> choose tool/action -> execute -> observe result -> continue
```

Tools:

- search
- calculator
- code execution
- database lookup
- file reading
- APIs

Agent design issues:

- tool selection
- tool result parsing
- hallucinated tool calls
- infinite loops
- prompt injection
- stale memory
- hidden state drift
- evaluation difficulty

Safer agent pattern:

```text
plan briefly
use one tool
check result
continue only if needed
stop with evidence
```

Agents should be evaluated on complete tasks, not just single responses.

## 29. Retrieval-Augmented Generation

RAG combines retrieval with generation.

Pipeline:

1. Split documents into chunks.
2. Embed chunks.
3. Store vectors.
4. Embed query.
5. Retrieve top chunks.
6. Put chunks into prompt.
7. Generate answer grounded in chunks.

Failure modes:

- bad chunking
- poor embeddings
- retrieval misses answer
- too many irrelevant chunks
- model ignores context
- model hallucinates beyond context

Evaluate:

- retrieval recall
- answer faithfulness
- citation correctness
- end-to-end task success

## 30. Audio Fundamentals

Sound is pressure variation over time.

Digital audio is sampled waveform.

Waveform:

```text
array of amplitude values
```

Sampling rate:

```text
samples per second
```

Examples:

```text
16000 Hz = 16000 samples per second
44100 Hz = CD-quality common rate
```

Mono:

```text
one channel
```

Stereo:

```text
two channels
```

Duration:

```text
num_samples / sample_rate
```

## 31. Spectrograms And Mel Features

Neural audio models may use raw waveform or spectral features.

### STFT

Short-time Fourier transform splits audio into frames and computes frequency content.

Spectrogram axes:

```text
time x frequency
```

### Mel Spectrogram

Mel scale approximates human pitch perception.

Common ASR input:

```text
log-mel spectrogram
```

Why useful:

- compresses waveform
- highlights frequency patterns
- easier for CNN/transformer encoders

## 32. Speech Recognition

Automatic speech recognition maps:

```text
audio -> transcript
```

Challenges:

- accents
- noise
- multiple speakers
- domain terms
- punctuation
- timestamps
- language switching

Metrics:

### WER

Word error rate:

```text
WER = (substitutions + deletions + insertions) / reference_words
```

### CER

Character error rate:

```text
CER = character-level edit distance / reference_characters
```

WER is common for speech transcripts.

CER is useful for languages or settings where word segmentation is difficult.

## 33. HuBERT

HuBERT means Hidden-Unit BERT.

It is a self-supervised speech representation method.

Core idea:

- Cluster speech features to create hidden unit targets.
- Mask parts of continuous speech input.
- Predict hidden unit labels for masked regions.

Why it matters:

- Speech labels are expensive.
- Audio is continuous.
- Self-supervised pretraining learns useful acoustic representations.

Use HuBERT-like encoders for:

- speech recognition features
- speaker tasks
- emotion tasks
- audio classification

## 34. Whisper

Whisper is a speech recognition model trained on large-scale weak supervision.

Key ideas:

- encoder-decoder transformer
- audio input converted to log-mel features
- decoder generates text tokens
- multilingual and multitask training

Tasks:

- transcription
- translation
- language identification
- timestamps depending on usage

Strengths:

- robust general-purpose ASR
- multilingual support
- works zero-shot on many domains

Limitations:

- can hallucinate in silence/noise
- domain vocabulary may need prompting or correction
- long audio needs chunking/segmentation
- timestamps can be imperfect

## 35. Qwen-Audio And Qwen2-Audio

Qwen-Audio is an audio-language model family for universal audio understanding.

It targets diverse audio:

- speech
- natural sounds
- music
- songs

Qwen2-Audio extends the line with audio and text input modes.

Capabilities:

- audio analysis
- speech instruction following
- audio question answering
- voice chat style interaction

Conceptual architecture:

```text
audio encoder -> language model interface -> text response
```

For mastery, understand:

- audio-language models are not just ASR
- they can answer questions about audio content
- they combine acoustic understanding and language generation
- evaluation must cover more than transcript accuracy

## 36. Voxtral

Voxtral is a Mistral audio model family covering transcription, speech understanding, and audio chat workflows.

Public references describe:

- Voxtral Mini and Voxtral Small audio chat models.
- Speech transcription and translation.
- Audio understanding.
- Long context for audio.
- Open-weight models in parts of the family.
- Realtime and TTS products in Mistral docs.

Conceptual takeaway:

Modern audio systems increasingly combine:

```text
speech recognition + language understanding + tool/chat behavior + speech generation
```

Do not evaluate them only by transcript WER if your task is audio reasoning or voice-agent behavior.

## 37. Audio Agents

A voice agent pipeline may look like:

```text
microphone -> VAD -> ASR/audio model -> LLM/agent -> tool calls -> response text -> TTS -> speaker
```

Components:

- VAD: voice activity detection
- ASR: speech to text
- LLM: reasoning and response
- tools: APIs, search, database
- TTS: text to speech

Important constraints:

- latency
- interruptions/barge-in
- noise robustness
- diarization
- privacy
- logging
- safety
- user confirmation for actions

## 38. Evaluation Cheat Sheet

Text classification:

- accuracy
- precision
- recall
- F1
- confusion matrix

Embeddings/retrieval:

- recall@k
- precision@k
- MRR
- nDCG
- manual top-k inspection

Language modeling:

- cross entropy
- perplexity
- task-specific human/eval tests

QA:

- exact match
- F1 overlap
- faithfulness
- source attribution

Chatbots:

- task success
- helpfulness
- safety
- hallucination rate
- latency
- user satisfaction

ASR:

- WER
- CER
- latency
- robustness by noise/accent/domain

Audio understanding:

- task accuracy
- QA correctness
- retrieval/grounding quality
- transcript quality if relevant

## 39. Common Failure Modes

### Text

- Tokenization mismatch.
- Truncation removes key evidence.
- Padding mask bug.
- Label leakage.
- Class imbalance.
- Spurious keyword learning.
- Domain shift.
- Hallucination.
- Prompt injection.

### Embeddings

- Similarity does not imply truth.
- Poor chunking.
- Wrong distance metric.
- Embedding model not suited to language/domain.
- Top result is semantically close but not answer-bearing.

### Fine-Tuning

- Data too small.
- Data low quality.
- Catastrophic forgetting.
- Overfitting style instead of skill.
- Bad evaluation.
- Train/test contamination.

### Audio

- Wrong sample rate.
- Clipping.
- Background noise.
- Speaker overlap.
- Bad segmentation.
- Long audio exceeds context.
- Transcript hallucination.
- Domain terms misrecognized.

## 40. Practical Hugging Face Patterns

### Text Classification Pipeline

```python
from transformers import pipeline

clf = pipeline("sentiment-analysis")
print(clf("I loved this movie."))
```

### Tokenizer And Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)

batch = tokenizer(
    ["hello world", "this is longer"],
    padding=True,
    truncation=True,
    return_tensors="pt"
)

outputs = model(**batch)
print(outputs.logits.shape)
```

### Question Answering

```python
qa = pipeline("question-answering")
qa(
    question="What is the capital of France?",
    context="Paris is the capital of France."
)
```

### Speech Recognition

```python
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
result = asr("audio.wav")
print(result["text"])
```

These snippets require installed packages and possibly internet/model downloads.

## 41. Problemsets Overview

Levels:

```text
A: hand calculation
B: from-scratch implementation
C: pretrained model / fine-tuning practice
D: project, debugging, report
```

Mastery means:

- You can tokenize and inspect text.
- You can train classical baselines.
- You can compute vector similarities.
- You can use transformer models correctly.
- You can explain attention and model families.
- You can evaluate QA/chat/retrieval.
- You can process audio and run ASR/audio demos.

## Problemset 1: Text Cleaning And Tokenization

### A. Concepts

Answer:

1. What is tokenization?
2. Why are subword tokenizers useful?
3. What is a vocabulary?
4. What is padding?
5. What is truncation?
6. What is an attention mask?
7. Why can removing stopwords be dangerous?

### B. Hand Tokenization

Given:

```text
"I don't love this movie!!!"
```

Create three tokenizations:

1. whitespace tokens
2. punctuation-aware word tokens
3. character tokens

Explain which information each keeps or loses.

### C. Coding

Write Python functions:

```python
lowercase(text)
replace_urls(text)
simple_word_tokenize(text)
build_vocab(texts, min_count=1)
encode(tokens, vocab)
pad_batch(encoded_sequences, pad_id=0)
```

### D. Report

Take 20 real sentences. Show how your tokenizer handles:

- punctuation
- numbers
- URLs
- repeated spaces
- mixed case
- unknown words

## Problemset 2: Bag-Of-Words And TF-IDF

### A. Hand BoW

Vocabulary:

```text
["cat", "dog", "love", "hate"]
```

Documents:

```text
d1 = "cat love cat"
d2 = "dog hate"
d3 = "cat dog love"
```

Compute BoW vectors.

### B. TF-IDF Intuition

Which words should have lower IDF?

```text
"the", "quantum", "movie", "and"
```

Explain.

### C. Coding

Implement:

```python
count_vectorize(texts)
tfidf_vectorize(texts)
cosine_similarity(a, b)
```

Compare documents by cosine similarity.

### D. Baseline

Use scikit-learn `TfidfVectorizer` and logistic regression for a tiny sentiment dataset.

## Problemset 3: Naive Bayes Text Classifier

### A. Hand

Training data:

```text
positive: "good fun", "good nice"
negative: "bad boring", "bad dull"
```

Use Laplace smoothing.

Classify:

```text
"good boring"
```

Show class priors and word likelihoods.

### B. Coding

Implement multinomial Naive Bayes from scratch.

### C. Compare

Compare your implementation with scikit-learn `MultinomialNB`.

### D. Error Analysis

Find examples where Naive Bayes fails because word independence is unrealistic.

## Problemset 4: Logistic Regression For Text

### A. Concepts

1. Why is TF-IDF + logistic regression a strong baseline?
2. What do positive coefficients mean?
3. What do negative coefficients mean?
4. Why use regularization?
5. Why evaluate class imbalance with F1?

### B. Coding

Train logistic regression on a small text dataset.

Report:

- accuracy
- precision
- recall
- F1
- confusion matrix

### C. Interpret

Print top positive and negative words by coefficient.

### D. Compare

Compare Naive Bayes vs logistic regression.

## Problemset 5: Word Embeddings And Similarity

### A. Hand

Vectors:

```text
cat = [1, 2]
dog = [1, 3]
car = [10, 0]
bus = [11, 1]
```

Compute:

1. Euclidean distance cat-dog.
2. Euclidean distance cat-car.
3. Cosine similarity cat-dog.
4. Cosine similarity car-bus.

### B. Coding

Write:

```python
cosine(a, b)
nearest_neighbors(embeddings, query, k)
analogy(a, b, c)
```

### C. Experiment

Use pretrained word vectors if available, or create toy vectors.

Test:

```text
king - man + woman
paris - france + italy
```

### D. Report

Explain why vector similarity is useful but not proof of factual correctness.

## Problemset 6: Sentence Embeddings And Retrieval

### A. Concepts

1. What is a sentence embedding?
2. How does semantic search differ from keyword search?
3. What is recall@k?
4. What can go wrong with chunking?
5. Why inspect retrieved documents manually?

### B. Build Toy Retriever

Create 20 short documents.

Implement retrieval using:

1. TF-IDF cosine.
2. Sentence embeddings if model available.

### C. Evaluate

Create 10 queries with known relevant documents.

Compute recall@1 and recall@3.

### D. Report

Show failures and explain whether the issue is query wording, chunking, embeddings, or missing data.

## Problemset 7: RNN And LSTM Concepts

### A. Concepts

1. What is hidden state?
2. Why are RNNs sequential?
3. What is vanishing gradient?
4. What problem do LSTM gates address?
5. Why did transformers replace RNNs for many NLP tasks?

### B. Hand

Given recurrence:

```text
h_t = tanh(Wx_t + Uh_{t-1})
```

with scalar:

```text
W = 1
U = 0.5
h_0 = 0
x = [1, 1]
```

Compute `h_1` and `h_2` approximately.

### C. Coding

Build a character-level RNN or LSTM for tiny text generation.

### D. Report

Compare generated text at different temperatures.

## Problemset 8: Encoder-Decoder And Attention

### A. Concepts

1. What does an encoder do?
2. What does a decoder do?
3. What is cross-attention?
4. Why did attention improve seq2seq?
5. What tasks use encoder-decoder models?

### B. Hand Attention

Given:

```text
Q = [[1, 0]]
K = [[1, 0], [0, 1]]
V = [[10, 0], [0, 20]]
```

Compute:

1. `QK^T`
2. softmax weights
3. weighted value output

### C. Coding

Implement scaled dot-product attention.

### D. Explain

Explain the difference between self-attention and cross-attention using translation.

## Problemset 9: Transformer Shapes

### A. Shapes

Input:

```text
batch = 4
seq_len = 16
d_model = 128
num_heads = 8
```

Answer:

1. input embedding shape
2. per-head dimension
3. attention score shape per head
4. output shape after multi-head attention

### B. Causal Mask

Create a 5-token causal mask by hand.

Explain why decoder-only models need it.

### C. Coding

Use PyTorch `nn.MultiheadAttention` or manual attention.

Print all shapes.

## Problemset 10: BERT-Style Encoders

### A. Concepts

1. Why is BERT bidirectional?
2. What is masked language modeling?
3. Why is BERT good for classification?
4. Why is BERT not naturally a chatbot generator?
5. What is `[CLS]` used for?

### B. Use A Pretrained Encoder

Use Hugging Face tokenizer/model.

Tokenize sentence pairs.

Inspect:

- input IDs
- tokens
- attention mask
- logits shape

### C. Classification

Fine-tune or linear-probe a small encoder for text classification if compute allows.

### D. Report

Compare transformer classifier with TF-IDF baseline.

## Problemset 11: Language Modeling

### A. Concepts

1. What is next-token prediction?
2. What is perplexity?
3. What is teacher forcing?
4. What is context length?
5. Why does lower perplexity not guarantee better chatbot behavior?

### B. Tiny Character LM

Train a character-level language model on a small text.

### C. Decoding

Generate with:

- greedy
- temperature 0.5
- temperature 1.0
- temperature 1.5

### D. Report

Explain repetition, randomness, and coherence differences.

## Problemset 12: Decoding Methods

### A. Hand

Vocabulary probabilities:

```text
A: 0.50
B: 0.20
C: 0.15
D: 0.10
E: 0.05
```

1. Greedy token?
2. Top-2 candidates?
3. Top-p set for `p=0.8`?
4. What happens if temperature increases?

### B. Coding

Implement:

```python
softmax(logits)
temperature_scale(logits, temperature)
top_k_filter(probs, k)
top_p_filter(probs, p)
sample_next(probs)
```

## Problemset 13: Question Answering

### A. Concepts

1. Extractive vs generative QA.
2. What are start and end logits?
3. What is hallucination?
4. Why should factual QA be grounded?
5. How can truncation remove the answer?

### B. Use QA Pipeline

Use a pretrained QA pipeline on 10 custom contexts/questions.

### C. Stress Test

Create cases:

- answer absent
- answer appears twice
- misleading sentence
- long context

### D. Report

Describe model failures and how to reduce them.

## Problemset 14: Chatbot Design

### A. Concepts

1. What is conversation history?
2. What is a system instruction?
3. What is context window?
4. What is hallucination?
5. What is prompt injection?

### B. Build Rule-Based Chatbot

Implement a tiny intent chatbot:

- greet
- ask_weather
- ask_time
- fallback

### C. LLM Chat Evaluation

If using an LLM/API/local model, create 20 test prompts.

Score:

- helpfulness
- factuality
- refusal correctness
- consistency

### D. Report

Write a chatbot failure taxonomy.

## Problemset 15: Fine-Tuning And PEFT

### A. Concepts

1. Full fine-tuning vs linear probing.
2. What are adapters?
3. What is LoRA?
4. Why does LoRA reduce trainable parameters?
5. When should you avoid fine-tuning?

### B. Parameter Count

Suppose a weight matrix is:

```text
W shape = 4096 x 4096
```

Full update parameters:

```text
4096 * 4096
```

LoRA rank:

```text
r = 8
```

LoRA parameters:

```text
4096*r + r*4096
```

Compute both and the reduction ratio.

### C. Practical

Use PEFT/LoRA on a small model if hardware allows.

If not, simulate LoRA on a small linear layer.

### D. Report

Explain data quality requirements for fine-tuning.

## Problemset 16: RAG

### A. Concepts

1. What is chunking?
2. What is embedding?
3. What is retrieval recall?
4. What is reranking?
5. Why can RAG still hallucinate?

### B. Build Toy RAG

Use 20 local text snippets.

Implement:

1. chunking
2. TF-IDF retrieval
3. answer prompt construction

### C. Evaluate

Create 10 questions with known supporting chunks.

Compute retrieval recall@3.

### D. Report

Show a case where retrieval succeeds but generation would still need caution.

## Problemset 17: Basic Agents

### A. Concepts

1. What is a tool?
2. What is an observation?
3. What is a planning loop?
4. Why can agents loop forever?
5. What is prompt injection through tool output?

### B. Build Toy Agent

Implement a simple loop with tools:

- calculator
- dictionary lookup
- string length

The "model" can be rule-based.

### C. Failure Test

Create tool outputs that try to inject instructions.

Write rules for ignoring untrusted tool instructions.

### D. Report

Explain when an agent is overkill compared with a normal function call.

## Problemset 18: Audio Basics

### A. Hand

Audio has:

```text
sample_rate = 16000
num_samples = 48000
```

Compute duration.

If stereo has 2 channels, what is the shape if stored as:

```text
[channels, samples]
```

### B. Coding

Load or synthesize audio:

- sine wave
- silence
- white noise

Plot waveform.

### C. Report

Explain sampling rate, amplitude, clipping, mono/stereo.

## Problemset 19: Spectrograms

### A. Concepts

1. What is STFT?
2. What does a spectrogram show?
3. Why use mel scale?
4. What is a log-mel spectrogram?
5. Why are spectrograms useful for neural networks?

### B. Coding

Generate:

- low-frequency sine
- high-frequency sine
- chirp if possible

Compute and plot spectrograms.

### C. Report

Describe how frequency patterns appear visually.

## Problemset 20: ASR And WER

### A. Hand WER

Reference:

```text
the cat sat on the mat
```

Prediction:

```text
the cat sit on mat
```

Compute substitutions, deletions, insertions, and WER.

### B. Use ASR

Run Whisper or another ASR model if available.

Test:

- clean speech
- noisy speech
- accented speech if available
- domain terms

### C. Report

Analyze errors:

- substitutions
- deletions
- insertions
- punctuation
- hallucinations

## Problemset 21: HuBERT And Audio Encoders

### A. Concepts

1. What is self-supervised speech learning?
2. Why is speech harder than text?
3. What are hidden units in HuBERT?
4. Why mask regions?
5. What downstream tasks can use speech encoders?

### B. Embeddings

Use a pretrained audio encoder if available.

Extract embeddings for several clips.

Compare similarities.

### C. Report

Explain why audio embedding similarity may reflect speaker, noise, language, or content.

## Problemset 22: Audio-Language Models

### A. Concepts

1. How is audio QA different from ASR?
2. What can Qwen-Audio/Qwen2-Audio do beyond transcription?
3. What can Voxtral-style models do in audio chat?
4. Why evaluate more than WER?
5. What is long-audio context?

### B. Task Design

Design 20 audio-understanding prompts:

- "What language is spoken?"
- "How many speakers?"
- "What event happens?"
- "Summarize the meeting."
- "What command did the speaker give?"

### C. Evaluation

Create rubric:

- transcript correctness
- answer correctness
- grounding in audio
- refusal when audio is unclear
- latency

## Problemset 23: Text Classification Project

Dataset options:

- sentiment dataset
- spam dataset
- news/topic dataset
- custom Indonesian/English mini dataset

Requirements:

1. Clean text.
2. Build train/validation/test split.
3. Train Naive Bayes baseline.
4. Train TF-IDF logistic regression.
5. Use a pretrained transformer if possible.
6. Compare metrics.
7. Print confusion matrix.
8. Inspect 20 errors.
9. Write report.

## Problemset 24: Embedding Retrieval Project

Requirements:

1. Collect 50-200 short documents.
2. Build TF-IDF retrieval.
3. Build embedding retrieval if possible.
4. Create 20 queries.
5. Evaluate recall@k.
6. Inspect failures.
7. Improve chunking or query handling.
8. Write report.

## Problemset 25: QA/RAG Project

Requirements:

1. Use local notes or small documents.
2. Chunk documents.
3. Retrieve relevant chunks.
4. Use extractive QA or generative QA.
5. Require answer citation/source.
6. Create 30 test questions.
7. Score answer correctness and faithfulness.
8. Write report.

## Problemset 26: Chatbot Or Agent Project

Requirements:

1. Define domain.
2. Define allowed tasks.
3. Define refusal/clarification rules.
4. Implement memory or conversation history.
5. Add one tool if building an agent.
6. Create 30-turn test set.
7. Evaluate task success, safety, hallucination.
8. Write failure analysis.

## Problemset 27: Audio Demo Project

Options:

- ASR with Whisper.
- Audio classification with pretrained encoder.
- Spectrogram classifier.
- Audio QA with Qwen-Audio/Qwen2-Audio or Voxtral if available.

Requirements:

1. Load audio.
2. Inspect waveform and duration.
3. Compute spectrogram.
4. Run model.
5. Evaluate with WER or task metric.
6. Test noise sensitivity.
7. Write report.

## Problemset 28: Final Exam

### Part A: NLP Theory

Answer:

1. What is tokenization?
2. Why use subwords?
3. What is an attention mask?
4. Explain BoW and TF-IDF.
5. Explain Naive Bayes for text.
6. Explain word embeddings and cosine similarity.
7. Static vs contextual embeddings.
8. Explain RNN/LSTM limitations.
9. Explain encoder-decoder models.
10. Explain attention.
11. Explain transformer encoder, decoder, encoder-decoder.
12. Explain BERT.
13. Explain next-token language modeling.
14. Explain decoding methods.
15. Explain extractive vs generative QA.
16. Explain LoRA and adapters.
17. Explain RAG.
18. Explain basic agents.

### Part B: Audio Theory

Answer:

1. What is sampling rate?
2. What is waveform amplitude?
3. What is a spectrogram?
4. What is a mel spectrogram?
5. What is ASR?
6. What is WER?
7. Explain HuBERT.
8. Explain Whisper.
9. Explain Qwen-Audio/Qwen2-Audio.
10. Explain Voxtral-style audio models.

### Part C: Hand Computation

1. Build BoW vectors.
2. Compute TF-IDF intuition.
3. Compute cosine similarity.
4. Compute attention weights.
5. Compute transformer head dimension.
6. Compute LoRA parameter reduction.
7. Compute audio duration from samples.
8. Compute WER.

### Part D: Implementation

From memory:

1. Tokenize text.
2. Build vocabulary.
3. Pad sequences.
4. Train TF-IDF classifier.
5. Use Hugging Face tokenizer/model.
6. Run QA pipeline.
7. Build simple retrieval.
8. Plot waveform.
9. Compute spectrogram.
10. Run ASR demo.

Passing standard:

- You can build classical NLP baselines.
- You can inspect tokenizer outputs.
- You can compute vector similarity.
- You can explain transformer families.
- You can use pretrained encoders and QA models.
- You can explain fine-tuning tradeoffs.
- You can process audio basics and evaluate ASR.

## 42. External Practice Map

### IOAI Module 5

Focus:

- NLP intro and LLMs.
- Regex and tokenization.
- Text normalization.
- BoW and TF-IDF.
- Sentiment classification.
- Word2Vec and GloVe.
- RNN/LSTM.
- Fine-tuning.
- Transformers.
- BERT/GPT-style models.
- Chatbots, QA, agents.
- Audio basics and audio models.

Link: https://ioai.toki.id/assets/modul/Modul_5_IOAI__Pengenalan_Pengolahan_Bahasa_Alami__NLP_.pdf

### Stanford CS224n

Focus:

- Word vectors.
- Neural classifiers.
- Backpropagation.
- Dependency parsing.
- RNNs.
- Attention.
- Transformers.
- Pretraining.
- NLP applications.

Link: https://web.stanford.edu/class/cs224n/

### Hugging Face Course

Focus:

- Transformer tasks.
- Tokenizers.
- Datasets.
- Fine-tuning.
- Sharing models.
- LLM topics.
- NLP task workflows.

Link: https://huggingface.co/learn/nlp-course

### Hugging Face Docs

Use for implementation:

- Transformers: https://huggingface.co/docs/transformers/index
- PEFT: https://huggingface.co/docs/peft/index

### Primary Papers And Reports

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- BERT: https://arxiv.org/abs/1810.04805
- LoRA: https://arxiv.org/abs/2106.09685
- HuBERT: https://arxiv.org/abs/2106.07447
- Whisper: https://arxiv.org/abs/2212.04356
- Qwen-Audio: https://arxiv.org/abs/2311.07919
- Qwen2-Audio: https://arxiv.org/abs/2407.10759
- Voxtral: https://arxiv.org/html/2507.13264v1

## 43. Ten-Week Mastery Plan

### Week 1: Text Preprocessing

Study:

- regex
- normalization
- tokenization
- vocabulary
- padding/masks

Do:

- Problemset 1

Deliverable:

- tokenizer and padding utilities from scratch

### Week 2: Classical Text Classification

Study:

- BoW
- TF-IDF
- Naive Bayes
- logistic regression

Do:

- Problemsets 2-4

Deliverable:

- text classifier baseline report

### Week 3: Embeddings And Retrieval

Study:

- word embeddings
- sentence embeddings
- cosine similarity
- retrieval metrics

Do:

- Problemsets 5-6

Deliverable:

- semantic search toy system

### Week 4: Sequence Models And Attention

Study:

- RNN
- LSTM
- encoder-decoder
- attention

Do:

- Problemsets 7-8

Deliverable:

- character language model or attention implementation

### Week 5: Transformers And BERT

Study:

- transformer shapes
- encoder-only models
- BERT
- classification

Do:

- Problemsets 9-10

Deliverable:

- pretrained encoder classification experiment

### Week 6: Language Modeling And QA

Study:

- next-token prediction
- decoding
- perplexity
- extractive/generative QA

Do:

- Problemsets 11-13

Deliverable:

- QA stress-test report

### Week 7: Chatbots, Fine-Tuning, PEFT, RAG

Study:

- chatbot design
- LoRA/adapters
- RAG
- evaluation

Do:

- Problemsets 14-16

Deliverable:

- RAG toy system or chatbot test suite

### Week 8: Agents

Study:

- tools
- planning loops
- observations
- prompt injection
- agent evaluation

Do:

- Problemset 17

Deliverable:

- toy tool-using agent

### Week 9: Audio

Study:

- waveform
- sampling rate
- spectrogram
- ASR
- WER
- HuBERT
- Whisper
- audio-language models

Do:

- Problemsets 18-22

Deliverable:

- ASR or spectrogram demo report

### Week 10: Final Projects

Do:

- Problemsets 23-28

Deliverable:

- two projects and final exam

## 44. Selected Answer Key

### Problemset 5, Cosine Example

```text
cat = [1, 2]
dog = [1, 3]

dot = 1*1 + 2*3 = 7
||cat|| = sqrt(5)
||dog|| = sqrt(10)
cos = 7 / sqrt(50)
```

### Problemset 8, Attention

```text
QK^T = [[1, 0]]
```

Softmax:

```text
[exp(1)/(exp(1)+exp(0)), exp(0)/(exp(1)+exp(0))]
approx [0.731, 0.269]
```

Output:

```text
0.731*[10, 0] + 0.269*[0, 20] = [7.31, 5.38]
```

### Problemset 9, Transformer Shapes

```text
batch = 4
seq_len = 16
d_model = 128
num_heads = 8
```

Input embedding:

```text
[4, 16, 128]
```

Per-head dimension:

```text
128 / 8 = 16
```

Attention score shape per head:

```text
[4, 16, 16]
```

Output shape:

```text
[4, 16, 128]
```

### Problemset 15, LoRA Count

Full matrix:

```text
4096 * 4096 = 16777216
```

LoRA rank 8:

```text
4096*8 + 8*4096 = 65536
```

Reduction:

```text
16777216 / 65536 = 256x fewer trainable parameters for that matrix
```

### Problemset 18, Audio Duration

```text
sample_rate = 16000
num_samples = 48000
duration = 48000 / 16000 = 3 seconds
```

Stereo shape as `[channels, samples]`:

```text
[2, 48000]
```

### Problemset 20, WER

Reference:

```text
the cat sat on the mat
```

Prediction:

```text
the cat sit on mat
```

One possible alignment:

```text
the cat sat on the mat
the cat sit on --- mat
```

Errors:

```text
sat -> sit: 1 substitution
the deleted: 1 deletion
insertions: 0
```

Reference words:

```text
6
```

WER:

```text
(1 + 1 + 0) / 6 = 0.3333
```

## 45. Mistake Log

Use this after every exercise:

```text
Date:
Task:
Expected behavior:
Actual behavior:
Text/audio input:
Tokenizer or feature extractor:
Model:
Metric:
Failure:
Root cause hypothesis:
Fix tried:
Result:
What I learned:
```

NLP and audio failures often hide in preprocessing, tokenization, truncation, retrieval, and evaluation. Inspect the inputs before blaming the model.

## 46. Final Mastery Checklist

Text fundamentals:

- I can clean and tokenize text.
- I can build a vocabulary.
- I can pad and mask sequences.
- I can explain subword tokenization.
- I can inspect tokenizer outputs.

Classical NLP:

- I can build BoW and TF-IDF vectors.
- I can train Naive Bayes.
- I can train logistic regression for text.
- I can evaluate precision, recall, F1, and confusion matrix.

Embeddings:

- I can compute cosine similarity.
- I can find nearest neighbors.
- I can explain static vs contextual embeddings.
- I can build a small retrieval system.

Transformers:

- I can compute attention on a tiny example.
- I can explain encoder-only, decoder-only, and encoder-decoder models.
- I can explain BERT.
- I can explain language modeling and decoding.
- I can use Hugging Face tokenizers and models.

LLMs and applications:

- I can explain chatbots, QA, RAG, and agents.
- I can explain full fine-tuning, adapters, and LoRA.
- I can explain when not to fine-tune.
- I can evaluate hallucination and grounding.

Audio:

- I can explain waveform, sampling rate, spectrogram, and mel features.
- I can compute audio duration.
- I can explain ASR and WER.
- I can explain HuBERT, Whisper, Qwen-Audio/Qwen2-Audio, and Voxtral-style audio models.
- I can run or design a small ASR/audio understanding demo.

If you can complete the problemsets and at least two projects, you have a strong NLP and audio foundation for IOAI, LLM work, and multimodal AI systems.
