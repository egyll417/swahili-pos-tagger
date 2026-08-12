"""BiLSTM, 5 seeds, test, X excluded from macro F1 -- mean +/- std.

Reuses per-seed prediction files on disk (../output/bilstm_s{seed}_{dev,test}_pred.txt)
if present. If a prediction file is missing but a checkpoint (bilstm_s{seed}_model.pt)
is already saved, loads it and just re-runs inference instead of retraining. Only
trains from scratch when neither exists.

Merges a 'bilstm' entry into ../output/results.json alongside the perceptron entries
written by run_experiments.py. Must run AFTER run_experiments.py in run_all.sh: that
script overwrites results.json wholesale, so running before it would get clobbered.

Also (re)writes ../output/bilstm_{dev,test}_pred.txt as a canonical copy of the
seed=1 predictions, since confusion_matrix.py and verify_results.py expect those
fixed filenames.
"""
import os, json, shutil, numpy as np, torch
from reader import read_conll
from evaluate import read_tags, compute_metrics
from vocab import build_char_vocab, build_tag_vocab
from bilstm import set_seed, CharBiLSTMTagger, train_model, generate_predictions_file

SEEDS = [1, 2, 3, 4, 5]
OUT_DIR = '../output'
RESULTS_PATH = f'{OUT_DIR}/results.json'


def seed_paths(seed):
    prefix = f'{OUT_DIR}/bilstm_s{seed}'
    return f'{prefix}_dev_pred.txt', f'{prefix}_test_pred.txt', f'{prefix}_model.pt'


def ensure_predictions(seed, train_sents, char2idx, tag2idx):
    dev_pred, test_pred, model_path = seed_paths(seed)
    if os.path.exists(dev_pred) and os.path.exists(test_pred):
        print(f"[seed {seed}] reusing existing prediction files")
        return dev_pred, test_pred

    model = CharBiLSTMTagger(len(char2idx), len(tag2idx))
    if os.path.exists(model_path):
        print(f"[seed {seed}] predictions missing but checkpoint found -- loading weights, skipping retrain")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print(f"[seed {seed}] no predictions or checkpoint on disk -- training from scratch")
        set_seed(seed)
        train_model(model, train_sents, char2idx, tag2idx)
        torch.save(model.state_dict(), model_path)

    for split, out in [('dev', dev_pred), ('test', test_pred)]:
        generate_predictions_file(read_conll(f'../data/{split}.txt'), char2idx, tag2idx, model, out)
    return dev_pred, test_pred


def main():
    train_sents = read_conll('../data/train.txt')
    char2idx = build_char_vocab(train_sents)
    tag2idx = build_tag_vocab(train_sents)
    gold_test = read_tags('../data/test.txt')

    accs, macs, per_label_by_seed = [], [], []
    for seed in SEEDS:
        _, test_pred_path = ensure_predictions(seed, train_sents, char2idx, tag2idx)
        pred_test = read_tags(test_pred_path)
        per_label, acc, mac = compute_metrics(gold_test, pred_test, exclude_tags=('X',))
        accs.append(acc); macs.append(mac); per_label_by_seed.append(per_label)
        print(f"  seed{seed}: acc={acc:.4f} macroF1(excl X)={mac:.4f}")

    acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
    mf1_mean, mf1_std = float(np.mean(macs)), float(np.std(macs))
    print(f"{'bilstm':20} acc {acc_mean:.4f}±{acc_std:.4f}   macroF1 {mf1_mean:.4f}±{mf1_std:.4f}")

    per_tag = {}
    for tag in sorted(per_label_by_seed[0]):
        f1s = [pl[tag]['f1'] for pl in per_label_by_seed]
        per_tag[tag] = {'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
                         'support': per_label_by_seed[0][tag]['support']}

    if not os.path.exists(RESULTS_PATH):
        raise SystemExit(f"{RESULTS_PATH} not found -- run run_experiments.py first")
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    results['bilstm'] = {'acc_mean': acc_mean, 'acc_std': acc_std,
                         'mf1_mean': mf1_mean, 'mf1_std': mf1_std, 'per_tag': per_tag}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"merged 'bilstm' entry into {RESULTS_PATH}")

    seed1_dev, seed1_test, _ = seed_paths(1)
    shutil.copyfile(seed1_dev, f'{OUT_DIR}/bilstm_dev_pred.txt')
    shutil.copyfile(seed1_test, f'{OUT_DIR}/bilstm_test_pred.txt')
    print(f"wrote {OUT_DIR}/bilstm_dev_pred.txt and {OUT_DIR}/bilstm_test_pred.txt (seed=1 canonical copy)")


if __name__ == '__main__':
    main()
