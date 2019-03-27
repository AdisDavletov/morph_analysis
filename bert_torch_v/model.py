class BertMorphAnalyzer:
    def __init__(self):
        self.from_model
        pass

    def prepare_data(self, df):
        df = df.copy()
        df.tokens.apply(lambda x: ' '.join(x))
        self.tokenizer = BertTokenizer.from_pretrained(self.from_model, do_lower_case='uncased' in self.from_model)
