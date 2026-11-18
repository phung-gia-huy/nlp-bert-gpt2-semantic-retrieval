# NLP Lab 8 — BERT, GPT-2 & Semantic Retrieval

Individual academic coursework exploring three NLP tasks: **sentiment classification with BERT**, **text generation with GPT-2**, and **semantic search with sentence embeddings and FAISS**.

| | |
|---|---|
| **Course** | Machine Learning and Artificial Intelligence Laboratory |
| **University** | Ho Chi Minh City University of Technology and Engineering (HCM-UTE) |
| **Completed** | June 2026 |
| **Type** | Individual academic lab |
| **Documentation** | [Full Lab 8 report (PDF)](Lab8_BERT_GPT2_FAISS_Report.pdf) |

## What I implemented

| Experiment | Implementation | What was evaluated |
|---|---|---|
| **BERT sentiment classification** | IMDB dataset preprocessing, BERT tokenization and fine-tuning with Hugging Face `Trainer` | Classification accuracy and different learning rates |
| **GPT-2 text generation** | Pretrained GPT-2 inference, prompt experiments and a simple command-line chatbot | Example generations, repetition and factual errors |
| **Semantic retrieval** | SentenceTransformer embeddings and a FAISS `IndexFlatL2` index | Top-2 nearest-neighbor retrieval from a five-sentence knowledge base |

## 1. BERT sentiment classification

**Task:** Classify IMDB movie reviews as positive or negative.

**Documented experiment:**

- Dataset: IMDB movie reviews via Hugging Face Datasets.
- Tokenizer and pretrained model: `bert-base-uncased`.
- Text processing: padding/truncation to `max_length=128`.
- Baseline fine-tuning: **2,000 training examples**, **1 epoch**, **batch size 8**, **learning rate `2e-5`**.
- Evaluation: predictions on a **500-example subset of the IMDB test split**; additional experiments with learning rates `5e-5`, `2e-5`, and `1e-5`.

**Result recorded in the report: 83.40% accuracy** for the baseline 500-example evaluation. This is a **reported coursework result**, not a new independent reproduction. The 500 examples overlap a 1,000-example subset of the same IMDB test split configured for evaluation in the reported training code; therefore, **the 83.40% figure should not be treated as a fully independent final test result**.

## 2. GPT-2 text generation

- Loaded pretrained `gpt2` through the Hugging Face `pipeline("text-generation")` interface.
- Generated text from multiple prompts and built a basic command-line chatbot loop.
- Inspected sample responses and discussed observed issues such as repetition, off-topic text, and unsupported factual claims.

The chatbot was a **coursework demonstration**, not a production conversational assistant.

## 3. Sentence embeddings & FAISS semantic retrieval

The report documents the following retrieval pipeline:

```text
Five-sentence knowledge base
          |
          v
SentenceTransformer (all-MiniLM-L6-v2)
          |
          v
Sentence embeddings -> FAISS IndexFlatL2
                              ^
                              |
User question -> Query embedding
                              |
                              v
                       Top-2 matched entries
```

The example query is **"What is machine learning?"**. The code embeds the question, searches a FAISS index, and prints the two nearest knowledge-base entries.

**Important distinction:** This demonstrates **semantic retrieval, a component commonly used in RAG**. Although the report discusses RAG, the implemented example **does not pass retrieved text to a generative model to produce a final answer**. It is therefore **not an end-to-end RAG system**. Retrieval can help ground generated responses, but it does not guarantee the elimination of hallucinations.

## Technologies practiced

**Python**, PyTorch, Hugging Face Transformers and Datasets, BERT, GPT-2, SentenceTransformers, FAISS, NumPy, scikit-learn.

Key concepts: tokenization, transfer learning, model evaluation, learning-rate experiments, prompt testing, text embeddings, and vector similarity search.

## Repository contents

| File | Description |
|---|---|
| [`Lab8_BERT_GPT2_FAISS_Report.pdf`](Lab8_BERT_GPT2_FAISS_Report.pdf) | Academic lab report documenting BERT fine-tuning, GPT-2 text generation, sentence embeddings, and FAISS semantic retrieval experiments. |
| `README.md` | Project overview and accurate scope of the experiments |

This repository currently contains **documentation only**. It does **not** include an executable notebook, model checkpoints, or a deployed application, so the experiments cannot be reproduced directly from this repository alone.

## Potential next steps

- Add a cleaned, rerunnable notebook and environment requirements after validation.
- Evaluate embedding retrieval with more documents and queries.
- Connect retrieval to an LLM and evaluate a complete retrieve-then-generate pipeline.

---

*Academic coursework, June 2026. The descriptions and metrics above refer to experiments documented in the accompanying report.*
