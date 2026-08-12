from reader import read_conll

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

def build_char_vocab(sentences):
    chars = set()
    for sentence in sentences:
        for word, tag in sentence:
            for ch in word:
                chars.add(ch)

    char2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for ch in sorted(chars):
        char2idx[ch] = len(char2idx)

    return char2idx

def build_tag_vocab(sentences):
    tags = set()
    for sentence in sentences:
        for word, tag in sentence:
            tags.add(tag)

    tag2idx = {}
    for tag in sorted(tags):
        tag2idx[tag] = len(tag2idx)

    return tag2idx

def word_to_char_indices(word, char2idx):
    return [char2idx.get(ch, char2idx[UNK_TOKEN]) for ch in word]

if __name__ == '__main__':
    train = read_conll('../data/train.txt')
    char2idx = build_char_vocab(train)
    tag2idx = build_tag_vocab(train)

    print(f"Char vocab size: {len(char2idx)}")
    print(f"Tag vocab size: {len(tag2idx)}")
    print(f"Tags: {tag2idx}")

    # quick test: convert one word to indices
    test_word = train[0][0][0]
    print(f"\nWord: {test_word}")
    print(f"As char indices: {word_to_char_indices(test_word, char2idx)}")
    
