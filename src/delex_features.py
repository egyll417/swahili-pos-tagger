"""Delexicalized extractor: NO word identity (WORD_ removed). Morphology + context only."""
DELEX = True

def word_feature_extractor(sentence, i):
    word = sentence[i][0]
    features = []
    if not DELEX:
        features.append(f"WORD_{word.lower()}")
    features += [
        f"CAP_{word[0].isupper()}",
        f"PREF-2_{word[:2].lower()}",
        f"PREF-4_{word[:4].lower()}",
        f"SUF-2_{word[-2:].lower()}",
        f"SUF-4_{word[-4:].lower()}",
    ]
    prev_w = sentence[i-1][0].lower() if i > 0 else "<s>"
    features.append(f"PREV={prev_w}")
    next_w = sentence[i+1][0].lower() if i < len(sentence)-1 else "</s>"
    features.append(f"NEXT={next_w}")
    return features

def index_mapping(train_sentences):
    f2i, t2i = {}, {}
    for sentence in train_sentences:
        for i in range(len(sentence)):
            for f in word_feature_extractor(sentence, i):
                if f not in f2i:
                    f2i[f] = len(f2i)
            tag = sentence[i][1]
            if tag not in t2i:
                t2i[tag] = len(t2i)
    return f2i, t2i

def sentences_and_tags_to_indices(sentences, f2i, t2i):
    data = []
    for sentence in sentences:
        for i in range(len(sentence)):
            fidx = [f2i[f] for f in word_feature_extractor(sentence, i) if f in f2i]
            data.append((fidx, t2i[sentence[i][1]]))
    return data
