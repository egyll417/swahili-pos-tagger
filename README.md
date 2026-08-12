# Swahili POS Tagger — Feature Ablation Study

Feature-ablation study of morphology vs. word identity vs. context for Swahili POS tagging on MasakhaPOS.

## Requirements

- Python 3.10
- `numpy` — perceptron, BiLSTM helpers, results aggregation
- `torch` — character BiLSTM (`src/bilstm.py`, `src/aggregate_bilstm.py`)
- `scipy` — Wilcoxon signed-rank test (`src/wilcoxon_test.py`)
- `scikit-learn` — optional; only used as an independent cross-check in `verify_results.py` (skipped automatically if not installed)
- `matplotlib` — optional; only used by `make_figure.py` to render the per-tag F1 figure

```bash
pip install numpy torch scipy scikit-learn matplotlib
```

## Repo layout

```
.
├── data/                     # MasakhaPOS swa splits, 2-column CoNLL. Read-only, never regenerate.
│   ├── train.txt              # 693 sentences / 20898 tokens
│   ├── dev.txt                 # 138 sentences / 3817 tokens
│   └── test.txt                # 553 sentences / 16074 tokens
├── src/
│   ├── reader.py               # CoNLL reader
│   ├── featurecore.py          # Shared feature extractor; nolex/morph configs are built on this
│   ├── features.py             # Condition 1 — lexical: target word + affixes + neighbour words
│   ├── delex_features.py       # Condition 2 — delex: affixes + neighbour words (no target word)
│   ├── nolex_features.py       # Condition 3 — nolex: target affixes + neighbour affixes
│   ├── morph_features.py       # Condition 4 — morph: target affixes only, no context
│   ├── vocab.py                # Char/tag vocabularies for the BiLSTM
│   ├── baseline.py             # Majority-class baseline
│   ├── perceptron.py           # Multi-class averaged perceptron (Collins 2002)
│   ├── bilstm.py                # Character BiLSTM tagger
│   ├── evaluate.py             # Precision/recall/F1/accuracy, macro-F1 (X excluded)
│   ├── run_experiments.py      # 5-seed x {vanilla, averaged} x 4 conditions -> output/results.json
│   ├── aggregate_bilstm.py     # 5-seed BiLSTM aggregation, merges 'bilstm' entry into results.json
│   ├── wilcoxon_test.py        # Paired Wilcoxon on adjacent condition pairs, Bonferroni-corrected
│   ├── confusion_matrix.py     # Per-tag confusion matrix from a prediction file
│   └── run_all.sh              # Full pipeline (run from inside src/)
├── tests/                     # 10 tests covering feature-core equivalence and metric computation
├── verify_results.py           # Independent re-implementation of metrics; cross-checks output/ against expected numbers
├── make_figure.py              # Renders output/fig_pertag_f1.{pdf,png} from results.json
└── output/                     # Gitignored. All predictions, results.json, figures — fully regenerable.
```

## How to run

Full pipeline (perceptron on all 4 conditions, 5-seed BiLSTM aggregation, Wilcoxon tests, confusion matrices):

```bash
cd src
bash run_all.sh
```

This produces, in `output/`:
- `avg_{lex,delex,nolex,morph}_{dev,test}_pred.txt` — seed-1 averaged-perceptron predictions per condition
- `results.json` — 5-seed mean ± std (accuracy, macro F1, per-tag F1) for vanilla/averaged perceptron x 4 conditions, the majority baseline, and BiLSTM
- `per_sentence_acc.tsv` — per-sentence accuracy for all 4 conditions, aligned
- `confusion_{avg_lexical,avg_delex,avg_nolex,avg_morph,bilstm}.csv` — confusion matrices
- `bilstm_s{1..5}_{dev,test}_pred.txt`, `bilstm_s{1..5}_model.pt` — per-seed BiLSTM predictions/checkpoints
- `bilstm_{dev,test}_pred.txt` — canonical (seed-1) copy of the BiLSTM predictions

Optional figure for the report (run from the repo root, after `run_all.sh` has produced `results.json`):

```bash
python3 make_figure.py
```

Run the test suite (from the repo root):

```bash
python3 -m pytest tests/ -q
```

Independently verify the reported numbers (from the repo root, after `run_all.sh`). This re-implements metric computation from scratch with no shared code, checks token alignment between gold and predictions, and cross-checks against scikit-learn if installed:

```bash
python3 verify_results.py
```

## Expected results

Test set, 5 seeds, mean ± std. Macro F1 excludes tag `X` (support 1). DET has only 16 test tokens — treat its numbers as a supervision-scarcity case, not a tuning target.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline | 0.2980 | 0.0328 |
| Perceptron — morph only | 0.8759 ± 0.0009 | 0.7408 ± 0.0009 |
| Perceptron — nolex (+ neighbour affixes) | 0.8901 ± 0.0006 | 0.7985 ± 0.0028 |
| Perceptron — delex (+ neighbour words) | 0.8997 ± 0.0009 | 0.8052 ± 0.0041 |
| Perceptron — lexical (full) | 0.9019 ± 0.0008 | 0.8074 ± 0.0020 |
| Character BiLSTM | 0.8926 ± 0.0027 | 0.7898 ± 0.0076 |

(All perceptron rows use the averaged perceptron; numbers are from `output/results.json`, verified independently by `verify_results.py`.)

## Note on `output/`

`output/` is gitignored — it holds only regenerable predictions, results, and figures. Nothing in it is committed; re-running `run_all.sh` reproduces it from scratch.
