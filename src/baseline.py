import sys
from collections import Counter
from reader import read_conll


def get_majority_tag(train_file):
    sentences = read_conll(train_file)
    counts = Counter()
    for sentence in sentences:
        for token, tag in sentence:
            counts[tag] += 1
    return counts.most_common(1)[0][0]


def predict(input_file, majority_tag):
    predictions = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                predictions.append('')
            else:
                predictions.append(majority_tag)
    return predictions


def write_predictions(predictions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for p in predictions:
            f.write(p + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python baseline.py <train_file> <input_file> <output_file>")
        sys.exit(1)

    train_file, input_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    majority_tag = get_majority_tag(train_file)
    print(f"Majority tag: {majority_tag}")

    predictions = predict(input_file, majority_tag)
    write_predictions(predictions, output_file)
    print(f"Predictions written to {output_file}")
