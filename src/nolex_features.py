"""No explicit word identity anywhere: target affixes + CAP + neighbour affixes."""
from featurecore import make
word_feature_extractor, index_mapping, sentences_and_tags_to_indices = \
    make(target_word=False, context='affix')
