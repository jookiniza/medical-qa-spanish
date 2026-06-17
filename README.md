# 🏥 Extractive Medical QA in Spanish — Transfer Learning on Casimedicos-SQuAD

Fine-tuning **mBERT** for extractive Question Answering on the Spanish medical domain using the Casimedicos-SQuAD corpus (MIR exam questions). Five transfer learning experiments comparing domain specialization vs. data volume strategies.

---

## 📊 Results

| Model | Training Data | F1 (%) |
|---|---|---|
| Baseline | SQuAD v2 (~10k, EN) | 20.78 |
| Exp. 1b | SQuAD v2 + Casimedicos | 25.62 |
| Exp. 3 | emrQA → Casimedicos (sequential) | 46.73 |
| Exp. 1 | Casimedicos only | 48.43 |
| **Exp. 2** | **emrQA + Casimedicos (joint)** | **49.00** |

**Key finding:** Domain specialization beats data volume. A model trained on just ~400 domain-specific examples (F1=48.43%) outperforms one trained on 10,000 generic English examples (F1=20.78%).

---

## 🔍 What This Does

Given a clinical context and a question (in Spanish), the model predicts the exact text span from the context that answers it — no generation, pure extraction.

```
Context:  "Mujer de 34 años con clínica de tos seca, disnea y febrícula...
           El diagnóstico más probable es una neumonía atípica por Mycoplasma pneumoniae."

Question: "¿Cuál es el tratamiento empírico de elección?
           1) Amoxicilina-clavulánico. 2) Azitromicina. 3) Ceftriaxona..."

Prediction: "neumonía atípica por Mycoplasma pneumoniae"
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)

- **Model:** [`bert-base-multilingual-cased`](https://huggingface.co/bert-base-multilingual-cased) (mBERT)
- **Task head:** `AutoModelForQuestionAnswering`
- **Dataset:** [Casimedicos-SQuAD](https://huggingface.co/datasets/HiTZ/casimedicos-squad) + emrQA + SQuAD v2
- **Evaluation:** HuggingFace `evaluate` (SQuAD metrics: EM + F1)

---

## 🧪 Experiments

| # | Training Data | Purpose |
|---|---|---|
| Baseline | SQuAD v2 (~10k EN) | Cross-lingual transfer, no domain |
| Exp. 1 | Casimedicos (~400 ES) | Domain only, scarce data |
| Exp. 1b | SQuAD v2 + Casimedicos | Generic volume + domain |
| Exp. 2 | emrQA + Casimedicos (joint) | Medical volume + domain — **best** |
| Exp. 3 | emrQA → Casimedicos (sequential) | Domain adaptation, avoids catastrophic forgetting |

All experiments evaluated on the same Casimedicos-SQuAD validation split (56 examples).

---

## ⚙️ Key Implementation Details

- **Sliding window:** `max_length=384`, `doc_stride=128` — handles long clinical contexts without truncating answers
- **Question truncation:** MIR questions include 5 answer options inline, truncated to 60 words pre-tokenization to avoid tokenizer errors
- **Span decoder fix:** argmax over start/end logits ignoring position 0 (`[CLS]`) — without this, models abstain 100% of the time
- **Sequential LR:** `1e-5` for Exp. 3 (vs. `2e-5` elsewhere) to mitigate catastrophic forgetting

---

## 🚀 How to Run

```bash
git clone https://github.com/jookiniza/medical-qa-spanish
cd medical-qa-spanish
pip install -r requirements.txt
```

Open and run `TrabajoIndividual_JokinIzaguirre_codigo.ipynb` in Jupyter or Google Colab.

> ⚠️ GPU recommended. Experiments were run on a single GPU with batch size 8 + gradient accumulation (effective batch 16 for Exp. 1b).

---

## 📁 Repository Structure

```
medical-qa-spanish/
├── TrabajoIndividual_JokinIzaguirre_codigo.ipynb   # Full pipeline
├── TrabajoIndividual_JokinIzaguirre_memoria.pdf    # Technical report (Spanish)
├── requirements.txt
└── README.md
```

---

## 📌 Research Questions Answered

- **RQ1:** Yes — ~400 examples are enough for a functional system (F1≈48%), but scaling with classic combination methods shows diminishing returns.
- **RQ2:** Domain specialization > data volume. Medical domain in another language still beats 25x more generic data.
- **RQ3:** Joint training > sequential adaptation in this low-resource scenario.

---

## 👤 Author

**Jokin Izaguirre Perez** — AI Student @ UPV/EHU | Incoming Exchange @ University of Padova  
[LinkedIn](https://www.linkedin.com/in/jokin-izaguirre-perez/) · [GitHub](https://github.com/jokin-izaguirre-perez)

---

*Course project — Text Data Mining (MDT), UPV/EHU, 2024–25*
