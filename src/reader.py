def read_conll(filepath):
    """Read 2-column CoNLL (token TAG). Robust to comments and extra columns."""
    sentences, current = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line == '':
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                current.append((parts[0], parts[-1]))
        if current:
            sentences.append(current)
    return sentences
