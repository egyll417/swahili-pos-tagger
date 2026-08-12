"""Locked results table: 5 seeds x {vanilla,averaged} x {lexical,delex,nolex,morph}."""
import numpy as np, importlib, warnings, json
from reader import read_conll
from evaluate import compute_metrics
from perceptron import train, predict_sentences
warnings.filterwarnings("ignore")

TRAIN='../data/train.txt'; TEST='../data/test.txt'
SEEDS=[1,2,3,4,5]

def run(feat_mod, seed, averaged):
    fm = importlib.import_module(feat_mod)
    tr = read_conll(TRAIN); te = read_conll(TEST)
    f2i,t2i = fm.index_mapping(tr)
    data = fm.sentences_and_tags_to_indices(tr,f2i,t2i)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        W,b = train(data,len(t2i),len(f2i),seed=seed,epochs=25,averaged=averaged)
    preds = predict_sentences(fm,te,f2i,t2i,W,b)
    gold=[t for s in te for _,t in s]; flat=[t for p in preds for t in p]
    per_label,acc,mac = compute_metrics(gold,flat,exclude_tags=('X',))
    psa = np.array([np.mean([p==g for p,(_,g) in zip(pt,s)]) for pt,s in zip(preds,te)])
    return acc,mac,psa,per_label

results={}
CONDITIONS = [('features','lexical'),('delex_features','delex'),
              ('nolex_features','nolex'),('morph_features','morph')]
for feat,fname in CONDITIONS:
    for avg,aname in [(False,'vanilla'),(True,'averaged')]:
        accs,macs,per_label_by_seed=[],[],[]
        for sd in SEEDS:
            a,m,_,pl=run(feat,sd,avg); accs.append(a); macs.append(m); per_label_by_seed.append(pl)
        key=f"{aname}_{fname}"
        results[key]={'acc_mean':float(np.mean(accs)),'acc_std':float(np.std(accs)),
                      'mf1_mean':float(np.mean(macs)),'mf1_std':float(np.std(macs))}
        if avg:
            # per-tag F1 mean/std over 5 seeds; support is invariant across seeds (gold-derived)
            per_tag={}
            for tag in sorted(per_label_by_seed[0]):
                f1s=[pl[tag]['f1'] for pl in per_label_by_seed]
                per_tag[tag]={'f1_mean':float(np.mean(f1s)),'f1_std':float(np.std(f1s)),
                              'support':per_label_by_seed[0][tag]['support']}
            results[key]['per_tag']=per_tag
        print(f"{key:20} acc {np.mean(accs):.4f}±{np.std(accs):.4f}   macroF1 {np.mean(macs):.4f}±{np.std(macs):.4f}")

# majority baseline
te=read_conll(TEST); gold=[t for s in te for _,t in s]
_,acc,mac = compute_metrics(gold,['NOUN']*len(gold),exclude_tags=('X',))
results['majority']={'acc_mean':acc,'acc_std':0.0,'mf1_mean':mac,'mf1_std':0.0}
print(f"{'majority(NOUN)':20} acc {acc:.4f}          macroF1 {mac:.4f}")
json.dump(results,open('../output/results.json','w'),indent=2)
