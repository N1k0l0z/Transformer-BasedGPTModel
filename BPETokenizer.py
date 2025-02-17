import regex as re

class BPETOKENIZER:
    def __init__(self, data, vocab_size):
        self.vocab_size = vocab_size
        self.train_tokens, self.vocab = self.get_tokens(data)
        self.num_merges = vocab_size - len(self.vocab)
        self.ids = list(self.train_tokens)
        self.merges = {}

    def get_tokens(self, train_text, language='ge'):
        """Reads text and assigns unique token IDs to characters, extending the vocabulary with 255 additional byte values."""
        if language == 'en':
            pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE)
        elif language == 'ge':
            pat = re.compile(r""" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE)

        chars = sorted(set(train_text))

        byte_tokens = [chr(i) for i in range(256)]
        chars.extend(byte_tokens)
        chars = sorted(set(chars))

        stoi = {ch: i for i, ch in enumerate(chars)}
        vocab = {idx: char for idx, char in stoi.items()}

        train_tokens = [stoi[c] for c in train_text]

        train_tokens.insert(0, (vocab.get(' ')))

        reg_chars = re.findall(pat, train_text)
        self.reg_tokens = [[stoi[token] for token in i if token in stoi] for i in reg_chars]
        return train_tokens, vocab
    
    def get_stats(self, ids):
        """Compute frequency of adjacent pairs in tokenized data."""
        counts = {}
        for j in ids:
            if isinstance(j, list):
                for pair in zip(j, j[1:]):
                    counts[tuple(pair)] = counts.get(tuple(pair), 0) + 1
        return counts if counts else {}

    def merge(self, ids, pair, idx):
        """Merge a token pair in a sequence of token IDs."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)  
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def decode(self, ids):
        """Convert token IDs back to text, handling multi-byte characters and characters not in vocab."""
        reverse_merges = {v: k for k, v in self.merges.items()}
        reverse_vocab = {v: k for k, v in self.vocab.items()}

        while any(id_ in reverse_merges for id_ in ids):
            new_ids = []
            for id_ in ids:
                if id_ in reverse_merges:
                    new_ids.extend(reverse_merges[id_])
                else:
                    new_ids.append(id_)
            ids = new_ids

        decoded_text = []

        for i in ids:
            if i in reverse_vocab:
                decoded_text.append(reverse_vocab[i])
            else:
                try:
                    decoded_text.append(chr(i)) 
                except ValueError:
                    decoded_text.append(f"[UNKNOWN_{i}]") 

        return ''.join(decoded_text)


    def encode(self, text):
        """Convert text into token IDs using learned merges."""
        tokens = [self.vocab.get(char, list(char.encode('utf-8'))) for char in text]
        tokens = [t if isinstance(t, list) else [t] for t in tokens]
        tokens = [item for sublist in tokens for item in sublist]  

        while len(tokens) >= 2:
            stats = self.get_stats([tokens])
            if not stats:
                break
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)

        return tokens

    def train(self):
        self.merges = {}
        for i in range(self.num_merges):
            stats = self.get_stats(self.reg_tokens)
        
            pair = max(stats, key=stats.get)
            
            items = [(value) for key, value in self.vocab.items()]

            idx = max(items) + 1 + i
            print(f"Merging {pair} into a new token {idx}")
            self.ids = self.merge(self.ids, pair, idx)

            for j in range(len(self.reg_tokens)):
                self.reg_tokens[j] = self.merge(self.reg_tokens[j], pair, idx)

            self.merges[pair] = idx