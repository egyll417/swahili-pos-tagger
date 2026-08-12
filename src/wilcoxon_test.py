"""Paired Wilcoxon signed-rank on adjacent feature-condition pairs, per-sentence accuracy.
Pairs: lexical-delex, delex-nolex, nolex-morph. Bonferroni-corrected alpha for 3 comparisons.
Per-sentence accuracies for all four conditions are written to one aligned file
(../output/per_sentence_acc.tsv) since a supervisor asked for a single aligned file;
the pairwise tests below are computed from those same columns."""
import numpy as np, importlib, warnings, contextlib, io
from scipy.stats import wilcoxon
from reader import read_conll
from evaluate import compute_metrics
from perceptron import train, predict_sentences
warnings.filterwarnings("ignore")

def per_sentence_acc(feat_mod, seed, averaged=True):
    fm = importlib.import_module(feat_mod)
    tr=read_conll('../data/train.txt'); te=read_conll('../data/test.txt')
    f2i,t2i=fm.index_mapping(tr); data=fm.sentences_and_tags_to_indices(tr,f2i,t2i)
    with contextlib.redirect_stdout(io.StringIO()):
        W,b=train(data,len(t2i),len(f2i),seed=seed,epochs=25,averaged=averaged)
    preds=predict_sentences(fm,te,f2i,t2i,W,b)
    return np.array([np.mean([p==g for p,(_,g) in zip(pt,s)]) for pt,s in zip(preds,te)]), te

CONDITIONS = [('features','lexical'),('delex_features','delex'),
              ('nolex_features','nolex'),('morph_features','morph')]
PAIRS = [('lexical','delex'),('delex','nolex'),('nolex','morph')]
ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / len(PAIRS)
SEED = 1
SEEDS = [1,2,3,4,5]

# cache keyed on (feature module, seed) so no two seeds/conditions collide
_cache = {}
def cached_acc(feat_mod, seed):
    key = (feat_mod, seed)
    if key not in _cache:
        _cache[key] = per_sentence_acc(feat_mod, seed)
    return _cache[key]

feat_mod_by_name = {name: mod for mod, name in CONDITIONS}

# single aligned per-sentence file across all four conditions (supervisor asked for this)
accs_seed1 = {}
te = None
for mod, name in CONDITIONS:
    accs_seed1[name], te = cached_acc(mod, SEED)

with open('../output/per_sentence_acc.tsv','w') as f:
    f.write("sent_id\tn_tokens\tlexical\tdelex\tnolex\tmorph\n")
    for i, s in enumerate(te):
        row = [f"{accs_seed1[name][i]:.4f}" for _, name in CONDITIONS]
        f.write(f"{i}\t{len(s)}\t" + "\t".join(row) + "\n")
print("wrote ../output/per_sentence_acc.tsv")

print(f"\nBonferroni-corrected alpha for {len(PAIRS)} comparisons: {BONFERRONI_ALPHA:.4f} (uncorrected 0.05)\n")

for name_a, name_b in PAIRS:
    a_acc = accs_seed1[name_a]; b_acc = accs_seed1[name_b]
    d = b_acc - a_acc
    stat, p = wilcoxon(b_acc, a_acc, zero_method='wilcox')
    print(f"=== AVERAGED perceptron, seed={SEED}, n={len(a_acc)} test sentences: {name_a} vs {name_b} ===")
    print(f"  {name_a:8} mean per-sent acc : {a_acc.mean():.4f}")
    print(f"  {name_b:8} mean per-sent acc : {b_acc.mean():.4f}")
    print(f"  mean diff ({name_b}-{name_a})  : {d.mean():+.4f}")
    print(f"  {name_b} better: {(d>0).sum()} | {name_a} better: {(d<0).sum()} | tied: {(d==0).sum()}")
    print(f"  Wilcoxon W={stat:.1f}  p={p:.4f}  -> "
          f"{'SIGNIFICANT' if p<ALPHA else 'NOT significant'} at a=0.05, "
          f"{'SIGNIFICANT' if p<BONFERRONI_ALPHA else 'NOT significant'} at Bonferroni a={BONFERRONI_ALPHA:.4f}")

print("\n=== ROBUSTNESS across 5 seeds (averaged) ===")
for name_a, name_b in PAIRS:
    mod_a = feat_mod_by_name[name_a]; mod_b = feat_mod_by_name[name_b]
    print(f" -- {name_a} vs {name_b} --")
    sig = 0
    for sd in SEEDS:
        l,_ = cached_acc(mod_a,sd); dd,_ = cached_acc(mod_b,sd)
        _,pv = wilcoxon(dd,l,zero_method='wilcox'); sig += pv<BONFERRONI_ALPHA
        print(f"  seed{sd}: {name_a}={l.mean():.4f} {name_b}={dd.mean():.4f} diff={dd.mean()-l.mean():+.4f} p={pv:.3f}{' *' if pv<BONFERRONI_ALPHA else ''}")
    print(f"  significant (Bonferroni a={BONFERRONI_ALPHA:.4f}) in {sig}/5 seeds")
