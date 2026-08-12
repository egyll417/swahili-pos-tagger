"""Multi-class perceptron (vanilla + averaged, Collins 2002). Seeded & reproducible."""
import numpy as np, random, argparse, importlib
from reader import read_conll

def train(data, n_tags, n_feats, seed=1, epochs=25, lr=1.0, averaged=True):
    random.seed(seed); np.random.seed(seed)
    W = np.zeros((n_tags, n_feats)); b = np.zeros(n_tags)
    U = np.zeros((n_tags, n_feats)); ub = np.zeros(n_tags); q = 1
    data = list(data)
    for ep in range(epochs):
        random.shuffle(data); errs = 0
        for fidx, gold in data:
            pred = int(np.argmax(W[:, fidx].sum(axis=1) + b))
            if pred != gold:
                errs += 1
                W[gold, fidx] += lr; b[gold] += lr
                W[pred, fidx] -= lr; b[pred] -= lr
                if averaged:
                    U[gold, fidx] += q*lr; ub[gold] += q*lr
                    U[pred, fidx] -= q*lr; ub[pred] -= q*lr
            q += 1
        print(f"  epoch {ep}: errors={errs}")
    if averaged:
        return W - U/q, (b - ub/q).reshape(-1, 1)
    return W, b.reshape(-1, 1)

def predict_sentences(fm, sentences, f2i, t2i, W, b, oov_tag="NOUN"):
    id2tag = {v: k for k, v in t2i.items()}
    out = []
    for s in sentences:
        tags = []
        for i in range(len(s)):
            fidx = [f2i[f] for f in fm.word_feature_extractor(s, i) if f in f2i]
            tags.append(oov_tag if not fidx
                        else id2tag[int(np.argmax(W[:, fidx].sum(axis=1).reshape(-1,1) + b))])
        out.append(tags)
    return out

def write_predictions(sentences, pred_tags, path):
    with open(path, 'w', encoding='utf-8') as f:
        for s, tags in zip(sentences, pred_tags):
            for (w, _), t in zip(s, tags):
                f.write(f"{w} {t}\n")
            f.write("\n")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='features',
                     choices=['features','delex_features','nolex_features','morph_features'])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--vanilla', action='store_true', help='disable averaging')
    ap.add_argument('--train', default='../data/train.txt')
    ap.add_argument('--eval', nargs='+', default=['../data/dev.txt','../data/test.txt'])
    ap.add_argument('--out-prefix', default='../output/perc')
    a = ap.parse_args()

    fm = importlib.import_module(a.features)
    train_s = read_conll(a.train)
    f2i, t2i = fm.index_mapping(train_s)
    print(f"[{a.features}] feats={len(f2i)} tags={len(t2i)} averaged={not a.vanilla} seed={a.seed}")
    data = fm.sentences_and_tags_to_indices(train_s, f2i, t2i)
    W, b = train(data, len(t2i), len(f2i), seed=a.seed, epochs=a.epochs, averaged=not a.vanilla)
    for path in a.eval:
        sents = read_conll(path)
        preds = predict_sentences(fm, sents, f2i, t2i, W, b)
        name = path.split('/')[-1].replace('.txt','')
        out = f"{a.out_prefix}_{name}_pred.txt"
        write_predictions(sents, preds, out)
        print(f"  wrote {out}")
