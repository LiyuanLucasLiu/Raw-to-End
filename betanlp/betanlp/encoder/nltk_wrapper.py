import nltk
import re
import emoji

class NLTK_wrapper:
    def __init__(self, emb_path):
        self.ccef_vocab = {}
        vocab_size = 0
        self.unk_index = -1
        UNK = '__unk__'
        with open(emb_path, 'r') as f:
            for line in f.readlines():
                line = line.split()
                if len(line) == 2:
                    continue # vocab_size, emb_size
                self.ccef_vocab[line[0]] = vocab_size
                if line[0] == UNK:
                    self.unk_index = vocab_size
                vocab_size += 1
        assert self.unk_index != -1, f'unk:{UNK} is not found in the corpus'
        self.tokenizer = nltk.word_tokenize

    def word_ccef_preprocess(self, word):
        if word in emoji.UNICODE_EMOJI:
            word = word.replace(':', '___')
            word = re.sub(r'[^\w\s]', '_', word)
            return word.lower()
        if word == '#':
            return '__ht__'
        if re.match(r'^[-+]?\d+$', word):
            return '__number__'
        elif re.match(r'^@[\da-zA-Z_]+$', word):
            return '__at__'
        elif re.match(r'^https?://[^\s]+$', word) or re.match(r'^pic.twitter.com/[\da-zA-Z_]+$', word):
            return '__url__'
        return word.lower() if word.isalpha() else word

    def word_lookup(self, word):
        return self.ccef_vocab.get(word, self.unk_index)

    def index_words_in_sentence(self, sentence, words):
        offset = 0
        start_indices = []
        end_indices = []
        cleaned_words = []
        for word in words:
            index = sentence.find(word, offset)
            # with special handling of '' and ``
            if word == '\'\'' or word == '``':
                best_index = 12345
                best_word = ''
                for tmp_word in ['\"', '\'\'', word]:
                    tmp_index = sentence.find(tmp_word, offset)
                    if tmp_index != -1 and tmp_index < best_index:
                        best_index = tmp_index
                        best_word = tmp_word
                if best_index == 12345:
                    assert False, f'Did not find appropriate quotes at index {offset} in sentences: [{sentence}]'
                index = best_index
                word = best_word
            assert index != -1, f'Did not find word: [{word}] in sentence: [{sentence}], starting from index {offset}'
            start_indices.append(index)
            end_indices.append(index + len(word))
            offset = index + len(word)
            cleaned_words.append(word)
        return start_indices, end_indices, cleaned_words

    def sentence_mapping(self, sentence):
        words = self.tokenizer(sentence)
        # print(words)
        start_indices, end_indices, words = self.index_words_in_sentence(sentence, words)
        words = [self.word_ccef_preprocess(word) for word in words]
        ccef_indices = [self.word_lookup(word) for word in words]
        indexized_sent = [-1] * len(sentence)
        for start, end, idx in zip(start_indices, end_indices, ccef_indices):
            for i in range(start, end):
                indexized_sent[i] = idx
        return indexized_sent

    def forward(self, sentences):
        return [{'text': self.sentence_mapping(sentence), 'len': len(sentence)} for sentence in sentences]


if __name__ == '__main__':
    print('Testing')
    '''
    8 2
    i   1.0 2.2
    __unk__ 3.3 3.3
    what    5.5 3.3
    how 3.1 4.4
    this    6.6 0.5
    ''  1 1
    "   2 3
    ``  3 44
    '''
    sentences = ['I what  how @Alice. #this', 'I', '#', '@', ' see \'\' wpw I " \'']
    expected = [[0, -1, 2, 2, 2, 2, -1, -1, 3, 3, 3, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 4, 4, 4, 4], [0], [1], [1], [-1, 1, 1, 1, -1, 5, 5, -1, 1, 1, 1, -1, 0, -1, 6, -1, 1]]
    sm = NLTK_wrapper('simulate_ccef_emb.txt')
    result = sm.forward(sentences)
    assert str(expected) == str(result)
    for x in result:
        print(x)

