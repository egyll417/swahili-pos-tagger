import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evaluate import compute_metrics
from reader import read_conll

def test_metrics_hand_computed():
    gold = ['NOUN', 'VERB', 'NOUN', 'ADJ']
    pred = ['NOUN', 'NOUN', 'NOUN', 'ADJ']
    pl, acc, mf = compute_metrics(gold, pred)
    assert acc == 0.75
    assert abs(pl['NOUN']['f1'] - 0.8) < 1e-9
    assert pl['VERB']['f1'] == 0.0
    assert pl['ADJ']['f1'] == 1.0
    assert abs(mf - (0.8 + 0.0 + 1.0) / 3) < 1e-9

def test_exclude_tag_changes_macro():
    gold = ['NOUN', 'X']
    pred = ['NOUN', 'NOUN']
    _, _, with_x = compute_metrics(gold, pred)
    _, _, no_x = compute_metrics(gold, pred, exclude_tags=('X',))
    assert no_x > with_x

def test_reader_splits_sentences():
    d = os.path.join(os.path.dirname(__file__), '..', 'data', 'test.txt')
    s = read_conll(d)
    assert len(s) == 553
    assert sum(len(x) for x in s) == 16074
    assert all(len(tok) == 2 for sent in s for tok in sent)
