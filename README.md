# Flipkart Order Intelligence & Support Assistant

A connected system with three parts: a return-risk prediction model, a product-image
categoriser built with transfer learning, and a LangGraph-based support agent that uses
both saved models as tools on top of a retrieval-augmented policy knowledge base.

## Project structure

```
├── generate_orders.py          # Part 1 dataset generator
├── orders_dataset.csv          # Part 1 generated dataset (6,000 rows)
├── models/
│   ├── return_risk_model.pkl   # Part 1 saved pipeline (preprocessing + tuned RandomForest)
│   └── product_classifier.pt   # Part 2 saved classifier head weights
├── data/
│   ├── FashionMNIST/           # auto-downloaded by torchvision
│   └── sample_images/          # 5 real exported test-split images (Part 2 Task 8)
├── notebooks/
│   ├── 01-returnrisks.ipynb    # Part 1: return-risk pipeline
│   ├── 02-imageclassifier.ipynb# Part 2: image classifier (transfer learning)
│   └── 03-supportagent.ipynb   # Part 3: LangGraph RAG support agent
├── transcripts/
│   └── all_test_conversations.txt  # 8+ required test conversations + retrieval eval
└── README.md
```

## How to run each part

### Part 1 — Return-Risk Pipeline

1. From the project root, regenerate the dataset (already committed, but reproducible):
   ```bash
   python3 generate_orders.py
   ```
   This is deterministic (`np.random.default_rng(42)`), producing `orders_dataset.csv`
   with 6,000 rows and 13 columns.
2. Open `notebooks/01-returnrisks.ipynb` and run all cells top to bottom. This:
   - Verifies the dataset (return rate, missingness pattern — diagnosed as MAR, conditional
     on `payment_method`: COD orders are missing ratings ~22.8% of the time vs ~6% for other
     payment methods)
   - Builds a leak-free preprocessing pipeline (median/mode imputation, one-hot encoding,
     scaling — fit only on the training split)
   - Trains a DummyClassifier baseline (77.25% accuracy, 0.0 F1 — the "high accuracy, zero
     recall" trap)
   - Trains and threshold-tunes a Logistic Regression (ROC-AUC 0.625, F1 0.39 at default
     threshold; best threshold 0.44 raises recall from 58% to 76%)
   - Trains and tunes a Random Forest via GridSearchCV (best params: 100 trees, max_depth 6;
     CV ROC-AUC 0.618, test ROC-AUC 0.614 — no overfitting)
   - Compares built-in vs. permutation feature importance (payment_method_COD is genuinely
     important under both; delivery_distance_km is overrated by built-in importance and drops
     out under permutation)
   - Runs subgroup analysis by category and payment method (found: non-COD orders are a
     recall blind spot — 0–5% recall vs 87.7% for COD; proposed fix: per-payment-method
     decision thresholds)
   - Saves the final combined pipeline (preprocessing + tuned Random Forest) to
     `models/return_risk_model.pkl`, with `t*_rf = 0.46`

### Part 2 — Product Image Classifier

1. Open `notebooks/02-imageclassifier.ipynb` and run all cells top to bottom.
2. Fashion-MNIST downloads automatically (no login required) into `data/FashionMNIST/`.
3. The notebook:
   - Splits 60k training images into 55k train / 5k validation, keeps the 10k test split
     untouched until final evaluation
   - Preprocesses images for ResNet-18 (grayscale → 3-channel, resized to 128×128 —
     reduced from the standard 224×224 due to CPU-only hardware overheating during feature
     extraction on a MacBook Air M4; documented, valid workaround), normalized with
     ImageNet mean/std
   - Loads a pretrained ResNet-18, freezes the backbone, caches its extracted features,
     and trains only a new linear classifier head (Adam optimizer, lr=0.001, batch size 32,
     10 epochs)
   - Reaches 88.24% validation accuracy from feature extraction alone — above the 80% bar,
     so no fine-tuning of backbone layers was required
   - Evaluates on the untouched test set: **88.6% test accuracy**, full confusion matrix,
     per-class precision/recall
   - Names two visually-plausible confusion pairs: Shirt↔T-shirt/top and Shirt↔Coat (Shirt
     is the weakest class — 0.68 F1 — due to overlapping silhouettes at low resolution)
   - Saves the trained head to `models/product_classifier.pt`, with a documented
     `load_classifier()` / `predict_image()` pair
   - Exports 5 real test-split images as `.png` files to `data/sample_images/`

### Part 3 — Support Agent (runs in MOCK_LLM mode by default — no API key, no network)

1. Open `notebooks/03-supportagent.ipynb` and run all cells top to bottom.
2. The notebook:
   - Authors 12 short policy documents (return windows, COD refund timelines, delivery
     SLAs, reverse-pickup eligibility, etc.), chunked sentence-wise into 36 chunks with
     chunk → parent-document mapping preserved
   - Embeds all chunks locally with `all-MiniLM-L6-v2` and indexes them in ChromaDB
   - Implements `check_return_risk()`, which loads `models/return_risk_model.pkl` and
     buckets risk as **Low** if probability < `t*_rf` (0.46), **Medium** if between 0.46
     and 0.61, **High** if ≥ 0.61 — anchored to Part 1's own F1-maximizing threshold
   - Implements `classify_product_image()`, which loads `models/product_classifier.pt`
     and classifies real files in `data/sample_images/`
   - Builds a 4-node LangGraph graph (intent → [RAG retrieval | tool-calling, branched by
     a conditional edge] → response generation), with `last_order_id` carried in shared
     state across turns for multi-turn follow-ups, and correctly absent in a fresh
     conversation (see transcript Tests 5 and 6 below)
   - Runs entirely in **MOCK_LLM** mode: rule-based intent classification and
     template-based response generation, zero API keys, zero network calls
   - Adds guardrails: an input-side prompt-injection filter (blocks phrases like "ignore
     previous instructions") and an output-side groundedness check that refuses to answer
     ungrounded policy questions, printing the retrieved similarity score against the
     0.46 threshold (Note: threshold value here refers to `SIMILARITY_THRESHOLD = 1.0`,
     distinct from Part 1's `t*_rf`)
   - Computes Precision@3 / Recall@3 at the document level across 5 test queries
     (average Precision@3 = 0.40, average Recall@3 = 1.00)

## System prompt design (4S + role prompting)

The Part 3 intent-classification and response-generation logic is designed against:
- **Specific** — the intent node classifies into exactly 3 categories (policy,
  return_risk, product_category) plus a blocked state for injection attempts, rather than
  open-ended classification
- **Short** — response templates are single, direct sentences per intent type, avoiding
  verbose preambles
- **Surround** — retrieved policy chunks and tool outputs are always injected directly
  into the response, so the model (or in mock mode, the template) is grounded in real
  retrieved/computed content, not free-form generation
- **Single** — each response returns exactly one structured JSON answer
  (`answer`, `source`, `confidence`), never multiple competing answers
- **Role prompting** — the agent is scoped throughout as "Flipkart's support assistant,"
  answering only from its policy knowledge base or its two tools
- **Few-shot examples** — two example query→intent pairs are documented directly in the
  intent node's logic (e.g. "Is order 4521 likely to be returned?" → `return_risk`;
  "What category is this product image?" → `product_category`), and both are shown
  driving correct routing in transcript Tests 3 and 4

## Example transcript (see `transcripts/all_test_conversations.txt` for the full set)

**Multi-turn state carried:**
```
Turn 1: "Is order 4521 likely to be returned?"
→ 59.0% return probability, Medium risk bucket | last_order_id: order_4521

Turn 2: "What about the order I just asked about?"
→ correctly re-routes to return_risk using the carried last_order_id,
  re-reports the same order's risk | last_order_id still: order_4521
```

**Fresh conversation, state correctly absent:**
```
"What about the order I just asked about?" (new conversation, no prior turns)
→ last_order_id: None → falls through to policy intent → no relevant chunk found
  → groundedness refusal
```

## Notes on hardware constraints

Development was done on a MacBook Air M4 (CPU only, no GPU) inside a single virtual
environment. Part 2's feature extraction initially caused thermal throttling at the
standard 224×224 ResNet input size; switching to 128×128 with batch size 32 resolved this
while still comfortably clearing the 80% test-accuracy requirement (88.6% achieved).

## Optional live-LLM extension

Not implemented in this submission. The agent runs fully and passes every acceptance
criterion via `MOCK_LLM` mode with `USE_LIVE_LLM` unset.
