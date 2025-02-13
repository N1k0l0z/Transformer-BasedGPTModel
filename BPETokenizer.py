class BPETOKENIZER:
    def __init__(self, train_text_path, vocab_size):
        self.train_tokens = self.get_tokens(train_text_path)
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        self.ids = list(self.train_tokens)
        self.merges = {}
        self.vocab = None

    def get_tokens(self, train_text_path):
        with open(train_text_path, 'r', encoding='utf-8') as f:
            train_text = f.read()
        train_tokens = train_text.encode('utf-8')
        train_tokens = list(map(int, train_tokens))
        return train_tokens

    def get_stats(self, ids):
        """gets ids and returns dictionary where keys are all pairs and values amount of each pair."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        """gets ids which are token ids, pair which is pair with highest frequency and idx by which this pair will be replaced."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def get_vocab(self):
        if self.vocab is None:
            self.vocab = {idx: bytes([idx]) for idx in range(256)}
            for (p0, p1), idx in self.merges.items():
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        return self.vocab

    def decode(self, ids):
        vocab = self.get_vocab()
        tokens = b"".join(vocab[idx] for idx in ids)
        return tokens.decode('utf-8', errors='replace')

    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def train(self):
        self.merges = {}
        for i in range(self.num_merges):
            stats = self.get_stats(self.ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"Merging {pair} into a new token {idx}")
            self.ids = self.merge(self.ids, pair, idx)
            self.merges[pair] = idx
        
