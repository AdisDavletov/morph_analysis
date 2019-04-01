import jsonpickle


class EndingsVectorizer(object):

    def __init__(self, n_endings=3, lower=True):
        self.all_endings = set()
        self.name_to_index = {}
        self.n_endings = n_endings
        self.lower = lower

    def _convert_ending(self, ending: str) -> str:
        ending = ending.lower() if self.lower else ending
        return ending

    def collect_endings(self, filename: str) -> None:
        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                _, word, _ = line.split('\t')[:3]
                self.add_ending(word[-self.n_endings:])

    def add_ending(self, ending: str) -> int:
        ending = self._convert_ending(ending)
        if ending not in self.name_to_index:
            self.name_to_index[ending] = len(self.name_to_index)
            self.all_endings.add(ending)

        return self.name_to_index[ending]

    def get_index(self, word: str) -> None:
        ending = self._convert_ending(word[-self.n_endings:])
        if ending in self.name_to_index:
            index = self.name_to_index[ending]
        else:
            index = len(self.name_to_index)
        return index

    def get_size(self) -> int:
        return len(self.name_to_index) + 1

    def is_empty(self):
        return len(self.name_to_index) == 0

    def load(self, filename: str) -> None:
        with open(filename, 'r') as f:
            vectorizer = jsonpickle.decode(f.read())
            self.__dict__.update(vectorizer.__dict__)

    def save(self, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write(jsonpickle.encode(self, f))



# if __name__ == '__main__':
#     v = EndingsVectorizer(lower=False)
#     v.collect_endings('datasets/gikrya_new_train.out')
#     print('endings size:', v.get_size())
#     print('voc:')
#     print(v.name_to_index)