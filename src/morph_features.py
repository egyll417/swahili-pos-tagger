"""Morphology only: target affixes + CAP. No context of any kind."""
from featurecore import make
word_feature_extractor, index_mapping, sentences_and_tags_to_indices = \
    make(target_word=False, context='none')
