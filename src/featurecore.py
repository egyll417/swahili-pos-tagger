"""Shared feature core. Every feature set is a configuration of this one
extractor, so no two feature modules can silently diverge or converge."""
BOS, EOS = "<s>", "</s>"

def extract(sentence, i, target_word, context):
    """context: 'form' (full neighbour word) | 'affix' | 'none'"""
    word = sentence[i][0]
    feats = []
    if target_word:
        feats.append(f"WORD_{word.lower()}")
    feats += [
        f"CAP_{word[0].isupper()}",
        f"PREF-2_{word[:2].lower()}",
        f"PREF-4_{word[:4].lower()}",
        f"SUF-2_{word[-2:].lower()}",
        f"SUF-4_{word[-4:].lower()}",
    ]
    if context == 'none':
        return feats
    prev_w = sentence[i-1][0].lower() if i > 0 else BOS
    next_w = sentence[i+1][0].lower() if i < len(sentence)-1 else EOS
    if context == 'form':
        feats.append(f"PREV={prev_w}")
        feats.append(f"NEXT={next_w}")
    elif context == 'affix':
        for side, w in (("PREV", prev_w), ("NEXT", next_w)):
            if w in (BOS, EOS):
                feats.append(f"{side}-BOUND={w}")
            else:
                feats.append(f"{side}-PREF-2={w[:2]}")
                feats.append(f"{side}-SUF-2={w[-2:]}")
    else:
        raise ValueError(f"unknown context: {context}")
    return feats

def make(target_word, context):
    def wfe(sentence, i):
        return extract(sentence, i, target_word, context)
    def index_mapping(train_sentences):
        f2i, t2i = {}, {}
        for s in train_sentences:
            for i in range(len(s)):
                for f in wfe(s, i):
                    if f not in f2i:
                        f2i[f] = len(f2i)
                t = s[i][1]
                if t not in t2i:
                    t2i[t] = len(t2i)
        return f2i, t2i
    def to_indices(sentences, f2i, t2i):
        data = []
        for s in sentences:
            for i in range(len(s)):
                fidx = [f2i[f] for f in wfe(s, i) if f in f2i]
                data.append((fidx, t2i[s[i][1]]))
        return data
    return wfe, index_mapping, to_indices
