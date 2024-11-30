import requests
import os
import zipfile
import re
from collections import Counter, OrderedDict, namedtuple
from nltk import Tree
from torch.utils.data import Dataset


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []
        self.i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
        self.t2i = OrderedDict({p : i for p, i in zip(self.i2t, range(len(self.i2t)))})

    def count_token(self, t):
        self.freqs[t] += 1

    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq=0):
        """
        min_freq: minimum number of occurrences for a word to be included
                  in the vocabulary
        """
        self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
        self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)


class SentimentDataset(Dataset):
    def __init__(self, split="train", lower=False, supervise_nodes=False):
        self.split = split
        self.lower = lower
        self.supervise_nodes = supervise_nodes

        # Download and extract data if needed
        self._maybe_download_and_extract()

        # Define constants
        self.SHIFT = 0
        self.REDUCE = 1

        # Load data
        self.Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])
        self.data = list(self._examplereader(f"trees/{split}.txt", lower=lower, supervise_nodes=supervise_nodes))

        # Build vocabulary
        self.vocab = self._build_vocabulary()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _maybe_download_and_extract(self):
        url = "http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"
        zip_path = "trainDevTestTrees_PTB.zip"
        extract_path = "trees"

        if not os.path.exists(extract_path):
            # Download if not exists
            if not os.path.exists(zip_path):
                print("Downloading sentiment dataset...")
                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Extract
                print("Extracting files...")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_path)

    def _filereader(self, path):
        with open(path, mode="r", encoding="utf-8") as f:
            for line in f:
                yield line.strip().replace("\\", "")

    def _tokens_from_treestring(self, s):
        """Extract the tokens from a sentiment tree"""
        return re.sub(r"\([0-9] |\)", "", s).split()

    def _transitions_from_treestring(self, s):
        s = re.sub("\([0-5] ([^)]+)\)", "0", s)
        s = re.sub("\)", " )", s)
        s = re.sub("\([0-4] ", "", s)
        s = re.sub("\([0-4] ", "", s)
        s = re.sub("\)", "1", s)
        return list(map(int, s.split()))
    
    def _extract_subtrees(self, tree):
        subtrees_with_labels = []
        for subtree in tree.subtrees():
            subtrees_with_labels.append((subtree.label(), subtree))
        return subtrees_with_labels

    def _examplereader(self, path, lower=False, supervise_nodes=False):
        """Returns all examples in a file one by one."""
        for line in self._filereader(path):
            line = line.lower() if lower else line
            tokens = self._tokens_from_treestring(line)
            tree = Tree.fromstring(line)
            label = int(line[1])
            trans = self._transitions_from_treestring(line)
            yield self.Example(tokens=tokens, tree=tree, label=label, transitions=trans)
            
            if supervise_nodes:
                subtrees_with_labels = self._extract_subtrees(tree)
                for label, subtree in subtrees_with_labels:
                    line_repr = str(subtree).replace('\n', '')
                    yield self.Example(
                        tokens=self._tokens_from_treestring(line_repr),
                        tree=subtree,
                        label=int(label), 
                        transitions=self._transitions_from_treestring(line_repr)
                    )

    def _build_vocabulary(self):
        vocab = Vocabulary()
        for example in self.data:
            for token in example.tokens:
                vocab.count_token(token)
        vocab.build()
        return vocab
