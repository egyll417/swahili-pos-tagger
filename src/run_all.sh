#!/bin/bash
# Full reproducible pipeline. Run from src/.
set -e
mkdir -p ../output
echo "== 1. Averaged perceptron (lexical) =="
python3 perceptron.py --features features --seed 1 --out-prefix ../output/avg_lex
echo "== 2. Averaged perceptron (delexicalized) =="
python3 perceptron.py --features delex_features --seed 1 --out-prefix ../output/avg_delex
echo "== 3. Averaged perceptron (nolex) =="
python3 perceptron.py --features nolex_features --seed 1 --out-prefix ../output/avg_nolex
echo "== 4. Averaged perceptron (morph) =="
python3 perceptron.py --features morph_features --seed 1 --out-prefix ../output/avg_morph
echo "== 5. Results table (5 seeds) =="
python3 run_experiments.py
echo "== 6. BiLSTM (5 seeds, merges into results.json) =="
python3 aggregate_bilstm.py
echo "== 7. Wilcoxon =="
python3 wilcoxon_test.py
echo "== 8. Confusion matrices =="
python3 confusion_matrix.py ../data/test.txt ../output/avg_lex_test_pred.txt avg_lexical
python3 confusion_matrix.py ../data/test.txt ../output/avg_delex_test_pred.txt avg_delex
python3 confusion_matrix.py ../data/test.txt ../output/avg_nolex_test_pred.txt avg_nolex
python3 confusion_matrix.py ../data/test.txt ../output/avg_morph_test_pred.txt avg_morph
python3 confusion_matrix.py ../data/test.txt ../output/bilstm_test_pred.txt bilstm
