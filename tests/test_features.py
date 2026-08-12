import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import features, delex_features, nolex_features, morph_features, featurecore
from reader import read_conll

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.txt')
SENTS = read_conll(DATA)

def _all_feats(mod):
    return [f for s in SENTS for i in range(len(s))
            for f in mod.word_feature_extractor(s, i)]

def test_core_reproduces_lexical_exactly():
    wfe, _, _ = featurecore.make(target_word=True, context='form')
    for s in SENTS:
        for i in range(len(s)):
            assert wfe(s, i) == features.word_feature_extractor(s, i)

def test_core_reproduces_delex_exactly():
    wfe, _, _ = featurecore.make(target_word=False, context='form')
    for s in SENTS:
        for i in range(len(s)):
            assert wfe(s, i) == delex_features.word_feature_extractor(s, i)

def test_lexical_and_delex_are_not_identical():
    assert _all_feats(features) != _all_feats(delex_features)

def test_delex_drops_word_identity():
    assert not any(f.startswith('WORD_') for f in _all_feats(delex_features))

def test_nolex_drops_neighbour_forms():
    fs = _all_feats(nolex_features)
    assert not any(f.startswith('WORD_') for f in fs)
    assert not any(f.startswith('PREV=') or f.startswith('NEXT=') for f in fs)

def test_morph_has_no_context():
    assert not any(f.startswith(('WORD_', 'PREV', 'NEXT')) for f in _all_feats(morph_features))

def test_feature_space_shrinks_monotonically():
    n = [len(featurecore.make(tw, c)[1](SENTS)[0]) for tw, c in
         [(True, 'form'), (False, 'form'), (False, 'affix'), (False, 'none')]]
    assert n[0] > n[1] > n[2] > n[3], n
    print("feature counts lexical/delex/nolex/morph:", n)
