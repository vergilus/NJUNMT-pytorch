from .bpe import Bpe
import sentencepiece


class _Tokenizer(object):
    """The abstract class of Tokenizer

    Implement ```tokenize``` method to split a string of sentence into tokens.
    Implement ```detokenize``` method to combine tokens into a whole sentence.
    ```special_tokens``` stores some helper tokens to describe and restore the tokenizing.
    """

    def __init__(self, **kwargs):
        pass

    def tokenize(self, sent, special=None):
        raise NotImplementedError

    def detokenize(self, tokens):
        raise NotImplementedError


class WordTokenizer(_Tokenizer):
    # tokenize word with white space
    def __init__(self, **kwargs):
        super(WordTokenizer, self).__init__(**kwargs)

    def tokenize(self, sent, special=None):
        return sent.strip().split()

    def detokenize(self, tokens):
        return ' '.join(tokens)


class BPETokenizer(_Tokenizer):
    # tokenize by BPE
    def __init__(self, codes=None, **kwargs):
        """ Byte-Pair-Encoding (BPE) Tokenizer

        Args:
            codes: Path to bpe codes. Default to None, which means the text has already been segmented  into
                bpe tokens.
        """
        super(BPETokenizer, self).__init__(**kwargs)

        if codes is not None:
            self.bpe = Bpe(codes=codes)
        else:
            self.bpe = None

    def tokenize(self, sent, special=["<UNK>"]):

        if self.bpe is None:
            return sent.strip().split()
        else:
            # return sum([self.bpe.segment_word(w) for w in sent.strip().split()], [])
            results = []
            for w in sent.strip().split():
                if w not in special:
                    results.append(self.bpe.segment_word(w))
                else:
                    results.append([w])
            return sum(results, [])

    def detokenize(self, tokens):

        return ' '.join(tokens).replace("@@ ", "")


class SPMTokenizer(_Tokenizer):
    def __init__(self, codes=None, **kwargs):
        """sentencepiece encoder: required a regulation model (unigram-language model)

        :param codes: model file
        :param kwargs:
        """
        super(SPMTokenizer, self).__init__(**kwargs)
        assert codes is not None, "model for sentencepiece must be provided!"
        self.spm = sentencepiece.SentencePieceProcessor()
        print(codes)  # model directories
        self.spm.load(codes)

    def tokenize(self, sent, special=None):
        return self.spm.encode_as_pieces(sent)

    # def tokenize(self, sent):
    #     # best-64 sampling with times-0.1 sharpening
    #     return self.spm.SampleEncodeAsPieces(sent, 64, 0.1)

    def detokenize(self, tokens):
        return ''.join(tokens).replace("▁", " ")


class Tokenizer(object):

    def __new__(cls, type, **kwargs):
        if type == "word":
            return WordTokenizer(**kwargs)
        elif type == "bpe":
            return BPETokenizer(**kwargs)
        elif type == "spm":
            return SPMTokenizer(**kwargs)
        else:
            print("Unknown tokenizer type {0}".format(type))
            raise ValueError
