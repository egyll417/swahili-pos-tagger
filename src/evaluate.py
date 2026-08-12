import sys
from collections import defaultdict

def read_tags(filepath):
    tags = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tags.append(line.split()[-1])
    return tags

def compute_metrics(gold, pred, exclude_tags=()):
    """macro-F1 averages over tags PRESENT IN GOLD, minus exclude_tags."""
    assert len(gold) == len(pred)
    labels = sorted(set(gold) | set(pred))
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for g, p in zip(gold, pred):
        if g == p: tp[g] += 1
        else: fp[p] += 1; fn[g] += 1
    per_label = {}
    for l in labels:
        P = tp[l]/(tp[l]+fp[l]) if tp[l]+fp[l] else 0.0
        R = tp[l]/(tp[l]+fn[l]) if tp[l]+fn[l] else 0.0
        F = 2*P*R/(P+R) if P+R else 0.0
        per_label[l] = {'p':P,'r':R,'f1':F,'support':tp[l]+fn[l]}
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    gold_labels = [l for l in labels if per_label[l]['support'] > 0 and l not in exclude_tags]
    macro_f1 = sum(per_label[l]['f1'] for l in gold_labels) / len(gold_labels)
    return per_label, acc, macro_f1

def print_results(per_label, acc, macro_f1, excluded=()):
    print(f"{'Label':<8}{'P':>8}{'R':>8}{'F1':>8}{'Supp':>7}")
    print("-"*39)
    for l in sorted(per_label, key=lambda x: -per_label[x]['support']):
        m = per_label[l]
        if m['support'] == 0: continue
        mark = " (excl)" if l in excluded else ""
        print(f"{l:<8}{m['p']:>8.3f}{m['r']:>8.3f}{m['f1']:>8.3f}{m['support']:>7}{mark}")
    print("-"*39)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}" + (f"  (excluding {','.join(excluded)})" if excluded else ""))

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    excl = ('X',) if '--exclude-X' in sys.argv else ()
    gold, pred = read_tags(args[0]), read_tags(args[1])
    if len(gold) != len(pred):
        print(f"Token mismatch: {len(gold)} vs {len(pred)}"); sys.exit(1)
    pl, acc, mf = compute_metrics(gold, pred, exclude_tags=excl)
    print_results(pl, acc, mf, excl)
