import regex as re


class BPETOKENIZER:
    def __init__(self, train_text_path, vocab_size):
        self.train_tokens, self.vocab = self.get_tokens(train_text_path)
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - len(set(self.train_tokens))
        self.ids = list(self.train_tokens)
        self.merges = {}  

    def get_tokens(self, train_text_path = 'C:\\Users\\NiKordzakhia\\Desktop\\Transformer-BasedGPTModel\\vefxistyaosani.txt', language = 'ge'):
        """Reads text and assigns unique token IDs to characters."""
        with open(train_text_path, 'r', encoding='utf-8') as f:
            train_text = f.read()
        
        if language == 'en':
            pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        elif language == 'ge':
            pat = re.compile(r""" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    
        chars = sorted(set(train_text))
        stoi = {ch: i for i, ch in enumerate(chars)} 

        train_tokens = [stoi[c] for c in train_text]  
        vocab = {idx: char for idx, char in stoi.items()}  

        reg_chars = re.findall(pat, train_text)
        self.reg_tokens = []
        for i in reg_chars:
            self.reg_tokens.append([vocab.get(token) for token in i])
        return train_tokens, vocab

    def get_stats(self, ids, stats_status='train'):
        """Compute frequency of adjacent pairs in tokenized data."""
        counts = {}
        if stats_status == 'encoder':
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        else:
            for j in ids:
                if isinstance(j, list): 
                    for pair in zip(j, j[1:]):
                        counts[tuple(pair)] = counts.get(tuple(pair), 0) + 1
        return counts if counts else {}

    def merge(self, ids, pair, idx):
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

    def decode(self, ids):
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

        decoded = [reverse_vocab.get(i, f"[UNK:{i}]") for i in ids]  
        return "".join(decoded)  

    def encode(self, text):
        tokens = []
        for i in text:
            tokens.append(self.vocab.get(i))
        
        while len(tokens) >= 2:
            stats = self.get_stats(tokens, stats_status = 'encoder')
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

            idx = len(set(self.train_tokens)) + i
            print(f"Merging {pair} into a new token {idx}")
            self.ids = self.merge(self.ids, pair, idx)

            for j in range(len(self.reg_tokens)):
                self.reg_tokens[j] = self.merge(self.reg_tokens[j], pair, idx)

            self.merges[pair] = idx  


from BPETokenizer import BPETOKENIZER
tokenizer = BPETOKENIZER('C:\\Users\\NiKordzakhia\\Desktop\\Transformer-BasedGPTModel\\vefxistyaosani.txt', 51)
tokenizer.train()
#print(tokenizer.merges)