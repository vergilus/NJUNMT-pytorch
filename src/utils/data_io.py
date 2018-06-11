import numpy as np
from itertools import islice
import random
import tempfile
import os

from .bpe import Bpe
from .common_utils import Vocab, INFO, GlobalNames

__all__ = [
    'TextDataset',
    'ZipDatasets',
    'DataIterator'
]

random.seed(GlobalNames.SEED)

class accumulate_takewhile(object):
    """
    This is the combination of ```itertools.takewhile``` and ```itertools.accumulate```
    >>> my_iter = accumulate_takewhile(range(10), 3)
    >>> list(my_iter) # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """

    def __init__(self, iterable, stop, func=lambda item : 1):

        self.iter = iter(iterable)
        self.func = func
        self.size = stop

    def __iter__(self):
        return self

    def __next__(self):

        out = []
        count = 0

        while True:
            try:
                item = next(self.iter)
            except StopIteration:
                if len(out) > 0:
                    return out
                else:
                    raise StopIteration

            out.append(item)
            count += self.func(item)

            if count >= self.size:
                return out

def accumulate_slicewhilce(data_iter, stop, key_func=lambda _: 1):
    """Slicing data according to key function

    Accumulate data into one batch until the accumulated value of key function
    reach stop criterion.
    """

    lines = []
    count = 0
    while True:
        try:
            line = next(data_iter)
        except StopIteration:
            break

        lines.append(line)
        count += key_func(line)

        if count >= stop:
            break

    return lines

def shuffle(*path):

    f_handles = [open(p) for p in path]

    # Read all the data
    lines = []
    for l in f_handles[0]:
        line = [l.strip()] + [ff.readline().strip() for ff in f_handles[1:]]
        lines.append(line)

    # close file handles
    [f.close() for f in f_handles]

    # random shuffle the data
    INFO('Shuffling data...')
    random.shuffle(lines)
    INFO('Done.')

    # Set up temp files
    f_handles = []
    for p in path:
        _, filename = os.path.split(p)
        f_handles.append(tempfile.TemporaryFile(prefix=filename + '.shuf', dir="/tmp/", mode="a+"))

    for line in lines:
        for ii, f in enumerate(f_handles):
            print(line[ii], file=f)

    # release memory
    lines = []

    # Reset file handles
    [f.seek(0) for f in f_handles]

    return tuple(f_handles)

class Dataset(object):
    def __init__(self, *args, **kwargs):
        pass

    @property
    def num_datasets(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _apply(self, lines):
        """ Do some processing on the raw input of the dataset.

        Return ```None``` when you don't want to output this line.

        Args:
            lines: A tuple representing one line of the dataset, where ```len(lines) == self.num_datasets```

        Returns:
            A tuple representing the processed output of one line, whose length equals ```self.num_datasets```
        """
        raise NotImplementedError

    def _data_iter(self, shuffle):
        """ Generate file handles of datasets.

        Always return a tuple of handles.
        """
        raise NotImplementedError

    def _not_empty(self, lines):

        if len([1 for l in lines if l is None]) == 0:
            return True
        else:
            return False

    def data_iter(self, shuffle=False):

        f_handles = self._data_iter(shuffle=shuffle)

        for lines in zip(*f_handles):

            lines = self._apply(lines)

            if self._not_empty(lines):
                yield lines

        [f.close() for f in f_handles]


class TextDataset(Dataset):

    def __init__(self,
                 data_path,
                 vocab,
                 bpe_codes=None,
                 use_char=False,
                 max_len=-1,
                 shuffle=False
                 ):

        super(TextDataset, self).__init__()

        if bpe_codes is not None and use_char is True:
            raise ValueError("BPE and character tokenizer could not use simultaneously!")

        if not isinstance(vocab, Vocab):
            raise ValueError("vocab must be an instance of Vocab.")

        self._data_path = data_path
        self._vocab = vocab
        self._use_char = use_char
        self._max_len = max_len
        self.shuffle = shuffle

        if bpe_codes is not None and len(bpe_codes) > 0:
            self._bpe = Bpe(codes=bpe_codes) # type: Bpe
        else:
            self._bpe = None

        with open(self._data_path) as f:
            self.num_lines = sum(1 for _ in f)

    @property
    def num_datasets(self):
        return 1

    def __len__(self):
        return self.num_lines

    def _data_iter(self, shuffle):
        if shuffle:
            return shuffle(self._data_path)
        else:
            return [open(self._data_path)]

    def _apply(self, lines):
        """
        Process one line

        :type line: str
        """

        line = lines[0].strip().split()

        if self._bpe is not None:
            line = sum([self._bpe.segment_word(w) for w in line], [])

        if self._use_char is True:
            line = sum([list(w) for w in line], [])

        line = [self._vocab.token2id(w) for w in line]

        if self._max_len > 0 and len(line) > self._max_len:
            return (None, )

        return (line, )

class ZipDatasets(Dataset):

    def __init__(self, *datasets, shuffle=False):
        """
        """
        super(ZipDatasets, self).__init__()
        self.shuffle = shuffle
        self.datasets = datasets

    @property
    def num_datasets(self):
        return len(self.datasets)

    def __len__(self):
        return len(self.datasets[0])

    def _data_iter(self, shuffle):

        if shuffle:
            return shuffle(*[ds._data_path for ds in self.datasets])
        else:
            return [open(ds._data_path) for ds in self.datasets]

    def _apply(self, lines):
        """
        :type dataset: TextDataset
        """

        outs = [d._apply((l,)) for d, l in zip(self.datasets, lines)]

        return sum(outs, ()) # (line_1, line_2, ..., line_n)

class DataIterator(object):

    def __init__(self,
                 dataset,
                 batch_size,
                 buffer_size=None,
                 use_bucket=True,
                 batching_key="samples"):

        """ Build data iterator given a dataset

        Args:
            dataset: An Dataset Object
            batch_size: Integer. Size of a batch. When batching_key is "samples", it represents the
                the number of samples. When batching_key is "tokens", it represents the tokens in a batch.
            use_bucket: Boolean value. Whether to use bucket.
            batching_key: Criterion to allocate a batch. Can only be "samples" or "tokens"
        """
        self.dataset = dataset

        self.batch_size = batch_size

        if batching_key not in {"samples", "tokens"}:
            print("Unknown batching key {0}".format(batching_key))
            raise ValueError

        self.batching_key = batching_key

        # Batching Key
        #
        # We have two kinds of batching key, ```tokens``` and ```samples```.
        # For tokens, we allocate a batch according to the number of tokens in it. For example,
        # in machine translation, if we use "tokens" as the key and set the batch_size as 4096,
        # we allocate a batch when the number of tokens at source or target side reach 4096.
        # For samples, we allocate a batch according to the number of samples in it. In machine
        # translation, 50 batch size with "samples" as key means 50 bi-text sentences.

        if self.batching_key == "samples":
            self.batching_key_func = lambda line : 1
        else:
            self.batching_key_func = lambda line : max(len(l) for l in line)

        # buffer size for bucketing
        # buffer size is the max number of batches in a buffer
        # if batching key is 'samples', buffer size is 100 times of batch size,
        # else we suppose that their are 50 tokens in one sample and then estimate
        # the number of samples in one batch as self.batch_size // 50

        if buffer_size is None:
            buffer_size = self.batch_size * 10

        self._buffer_size = buffer_size

        self.use_bucket = use_bucket

        self.reset()

    def __len__(self):
        return len(self.dataset)

    @property
    def n_datasets(self):
        return self.dataset.num_datasets

    def _fill_buffer(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        # 1. Allocate a new buffer
        inc_buffer = accumulate_slicewhilce(self.data_iter, self._buffer_size, key_func=self.batching_key_func)

        if len(inc_buffer) <= 0:
            # data_iter reach the end of the dataset
            self._end = True
            return

        # 2. Merge the residual samples in previous buffer (if any) into the inc_buffer

        if len(self.buffer) > 0:
            new_buffer = self.buffer[0] + inc_buffer
        else:
            new_buffer = inc_buffer


        # 3. Split buffer into batches. If ues_bucket is enable,
        # we sort the whole buffer according to the length of the sentence.

        if self.use_bucket:

            scores = np.array([max(len(s) for s in sample) for sample in new_buffer])
            sorted_indices = np.argsort(scores).tolist()
            new_buffer = [new_buffer[i] for i in sorted_indices]

        new_batch_buffer = list(accumulate_takewhile(new_buffer, stop=batch_size, func=self.batching_key_func))
        del new_buffer # release memory

        # 4. If use_bucket is enable, we shuffle the order of batches.
        if self.use_bucket and len(new_batch_buffer) > 1:
            new_batch_buffer_full = new_batch_buffer[:-1]
            random.shuffle(new_batch_buffer_full)
            new_batch_buffer[:-1] = new_batch_buffer_full

        # FIFO
        new_batch_buffer.reverse()
        self.buffer = new_batch_buffer

    @property
    def is_end(self):
        return self._end

    def reset(self):
        self.buffer = []
        self.data_iter = self.dataset.data_iter()
        self._end = False

    def build_generator(self, batch_size=None):

        while True:

            # We re-allocate the buffer when there at most on batch.
            # Usually this batch is not full.

            if len(self.buffer) <= 1:
                self._fill_buffer(batch_size=batch_size)

            if len(self.buffer) == 0:
                """Reach the end of the dataset, exit.
                """
                self.reset()
                break

            # Accumulated batches until reach the batch_size

            try:
                batch_ = self.buffer.pop()
            except IndexError:
                break

            if len(batch_) == 0:
                self.reset()
                break
            else:
                batch = [list(d) for d in zip(*batch_)]

                yield batch



