class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        words = set()
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
            
            
        for text in texts:
            words.update(text.lower().split())
        for word in sorted(words):
            if word not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[word]=idx
                self.id_to_word[idx] = word
        self.vocab_size = len(self.word_to_id)
                
        

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        words = text.lower().split()
        return [self.word_to_id.get(word, self.word_to_id[self.unk_token]) for word in words]
        

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        return " ".join(self.id_to_word.get(idx, self.unk_token) for idx in ids)