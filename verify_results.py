"""
Independent verification of the reported results table.

Deliberately shares NO code with src/. Re-implements metric computation from
scratch, and cross-checks against scikit-learn where available. Also verifies
that each prediction file is token-aligned with the gold file, which is the
failure mode that silently corrupts every downstream number.

Run from the repo root:
    python3 verify_results.py
"""
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(REPO, 'data', 'test.txt')
OUT = os.path.join(REPO, 'output')

# Expected values as printed by run_all.sh (seed 1, X excluded from macro F1)
EXPECTED = {
    'avg_lex_test_pred.txt':   (0.9019, 0.8074, 'lexical perceptron'),
    'avg_delex_test_pred.txt': (0.8990, 0.8011, 'delexicalized perceptron'),
    'avg_nolex_test_pred.txt': (0.8911, 0.7989, 'nolex perceptron'),
    'avg_morph_test_pred.txt': (0.8743, 0.7394, 'morphology-only perceptron'),
    'bilstm_test_pred.txt':    (0.8945, 0.7979, 'character BiLSTM'),
}

EXPECTED_SENTENCES = 553
EXPECTED_TOKENS = 16074


def load(path):
    """Return list of (word, tag) and sentence lengths. No shared code with src/."""
    pairs, lengths, n = [], [], 0
    with open(path, encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith('#'):
                continue
            if not line:
                if n:
                    lengths.append(n)
                    n = 0
                continue
            cols = line.split()
            if len(cols) < 2:
                continue
            pairs.append((cols[0], cols[-1]))
            n += 1
    if n:
        lengths.append(n)
    return pairs, lengths


def metrics(gold, pred, exclude=('X',)):
    """Accuracy and macro F1 over tags with non-zero gold support, minus `exclude`."""
    assert len(gold) == len(pred)
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    gold_tags = sorted(set(gold) - set(exclude))
    f1s = {}
    for t in gold_tags:
        prec = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) else 0.0
        rec = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) else 0.0
        f1s[t] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    return acc, sum(f1s.values()) / len(f1s), f1s


def sklearn_check(gold, pred):
    try:
        from sklearn.metrics import f1_score, accuracy_score
    except ImportError:
        return None
    labels = sorted(set(gold) - {'X'})
    return (accuracy_score(gold, pred),
            f1_score(gold, pred, average='macro', labels=labels, zero_division=0))


def main():
    if not os.path.exists(GOLD):
        sys.exit(f"FAIL: gold file not found at {GOLD}. Run from the repo root.")

    gold_pairs, gold_lens = load(GOLD)
    gold_tags = [t for _, t in gold_pairs]

    print("=" * 66)
    print("GOLD DATA")
    print("=" * 66)
    ok_sent = len(gold_lens) == EXPECTED_SENTENCES
    ok_tok = len(gold_pairs) == EXPECTED_TOKENS
    print(f"  sentences : {len(gold_lens):>6}  expected {EXPECTED_SENTENCES}  "
          f"{'OK' if ok_sent else '*** MISMATCH ***'}")
    print(f"  tokens    : {len(gold_pairs):>6}  expected {EXPECTED_TOKENS}  "
          f"{'OK' if ok_tok else '*** MISMATCH ***'}")
    dist = Counter(gold_tags)
    print(f"  tag types : {len(dist)}")
    print(f"  DET support in test: {dist.get('DET', 0)}  "
          f"(should be 16 — the rare-class story)")

    failures = []
    print()
    print("=" * 66)
    print("PREDICTION FILES")
    print("=" * 66)

    for fname, (exp_acc, exp_mf1, label) in EXPECTED.items():
        path = os.path.join(OUT, fname)
        print(f"\n{label}  [{fname}]")
        if not os.path.exists(path):
            print("  *** FILE MISSING — pipeline did not produce this ***")
            failures.append(f"{fname}: missing")
            continue

        pred_pairs, pred_lens = load(path)

        if len(pred_pairs) != len(gold_pairs):
            print(f"  *** TOKEN COUNT MISMATCH: {len(pred_pairs)} vs {len(gold_pairs)} ***")
            failures.append(f"{fname}: token count")
            continue

        misaligned = sum(1 for (gw, _), (pw, _) in zip(gold_pairs, pred_pairs) if gw != pw)
        if misaligned:
            print(f"  *** {misaligned} TOKENS MISALIGNED WITH GOLD — results invalid ***")
            failures.append(f"{fname}: alignment")
            continue
        print(f"  alignment : all {len(pred_pairs)} tokens match gold word-for-word  OK")

        if pred_lens != gold_lens:
            print("  *** sentence boundaries differ from gold ***")
            failures.append(f"{fname}: sentence boundaries")

        acc, mf1, f1s = metrics(gold_tags, [t for _, t in pred_pairs])
        a_ok = abs(acc - exp_acc) < 5e-4
        f_ok = abs(mf1 - exp_mf1) < 5e-4
        print(f"  accuracy  : {acc:.4f}   reported {exp_acc:.4f}   "
              f"{'OK' if a_ok else '*** MISMATCH ***'}")
        print(f"  macro F1  : {mf1:.4f}   reported {exp_mf1:.4f}   "
              f"{'OK' if f_ok else '*** MISMATCH ***'}")
        if not a_ok:
            failures.append(f"{fname}: accuracy")
        if not f_ok:
            failures.append(f"{fname}: macro F1")

        sk = sklearn_check(gold_tags, [t for _, t in pred_pairs])
        if sk:
            s_ok = abs(sk[0] - acc) < 1e-9 and abs(sk[1] - mf1) < 1e-9
            print(f"  sklearn   : acc {sk[0]:.4f}  macroF1 {sk[1]:.4f}   "
                  f"{'agrees' if s_ok else '*** DISAGREES ***'}")
            if not s_ok:
                failures.append(f"{fname}: sklearn disagreement")
        else:
            print("  sklearn   : not installed (optional cross-check skipped)")

        print(f"  PART F1   : {f1s.get('PART', 0.0):.3f}    "
              f"DET F1: {f1s.get('DET', 0.0):.3f}")

    print()
    print("=" * 66)
    if failures:
        print("RESULT: NOT VERIFIED")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: VERIFIED — every reported number reproduced independently")
    print("=" * 66)


if __name__ == '__main__':
    main()
