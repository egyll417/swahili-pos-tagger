"""Confusion matrix + per-tag report from a prediction file."""
import sys
from collections import Counter, defaultdict
from evaluate import read_tags, compute_metrics

gold = read_tags(sys.argv[1]); pred = read_tags(sys.argv[2])
name = sys.argv[3] if len(sys.argv)>3 else "model"
tags = sorted(set(gold)|set(pred))
conf = defaultdict(Counter)
for g,p in zip(gold,pred): conf[g][p]+=1

pl, acc, mf = compute_metrics(gold,pred,exclude_tags=('X',))
print(f"### {name}   acc={acc:.4f}  macroF1(excl X)={mf:.4f}\n")
print(f"{'tag':<7}{'P':>7}{'R':>7}{'F1':>7}{'supp':>6}   top confusions (gold -> pred)")
print("-"*72)
for t in sorted(tags,key=lambda x:-pl[x]['support']):
    if pl[t]['support']==0: continue
    m=pl[t]
    errs=[f"{k}:{v}" for k,v in conf[t].most_common() if k!=t][:3]
    print(f"{t:<7}{m['p']:>7.3f}{m['r']:>7.3f}{m['f1']:>7.3f}{m['support']:>6}   {', '.join(errs) if errs else '-'}")

# CSV matrix
with open(f"../output/confusion_{name}.csv","w") as f:
    f.write("gold\\pred,"+",".join(tags)+"\n")
    for g in tags:
        f.write(g+","+",".join(str(conf[g][p]) for p in tags)+"\n")
print(f"\nwrote ../output/confusion_{name}.csv")
