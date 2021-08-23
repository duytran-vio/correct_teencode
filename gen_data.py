from datasets import load_dataset
from functools import reduce
import string
import numpy as np

# file_data = "data_wiki.txt"
# dataset = load_dataset("./open_sub_lm.py", lang="vi",
#                        min_len=1, eos="", bos="", split="train")
# dataset = dataset.map(lambda x: {"doc": list(
#     map(lambda y: y + "\n", x["doc"]))}, batched=True)
# dataset = dataset["doc"]


class Embed():
    def __init__(self) -> None:
        self.build_alphabet()

    def __len__(self):
        return len(self.alphabet)

    def build_alphabet(self):
        latin_to_accented_char = {
            'a': ['á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', ],
            'o': ['ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', ],
            'e': ['é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', ],
            'u': ['ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', ],
            'i': ['í', 'ì', 'ỉ', 'ĩ', 'ị', ],
            'y': ['ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', ],
            'd': ['đ', ]
        }

        accented_chars = [
            'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
            'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
            'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
            'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
            'í', 'ì', 'ỉ', 'ĩ', 'ị',
            'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
            'đ',
        ]
        accented_chars.extend([c.upper() for c in accented_chars])

        alphabet = list(chr(0)
                        + chr(1)
                        + string.printable
                        + ''.join(accented_chars))

        self.alphabet = alphabet
        self.i2c = dict(enumerate(alphabet))
        self.c2i = dict((c, i) for i, c in enumerate(alphabet))

    def text_to_sequence(self, text):
        seq = [self.c2i.get(c) for c in text]
        seq = [i if i is not None else 1 for i in seq]
        return seq

    def sequence_to_text(self, seq):
        return [self.i2c[i] for i in seq]

    def one_hot(self, seq, one_hot_length):
        seq_len = len(seq)
        result = np.zeros((seq_len, one_hot_length))
        result[np.arange(seq_len), seq] = 1
        return result

    def one_hot_scalar(self, value, one_hot_length):
        result = np.zeros((one_hot_length, ))
        result[value] = 1
        return result

embed = Embed()

class AccentStripper:

    def __init__(self):
        self.build_accent_stripped_map()

    def build_accent_stripped_map(self):
        latin_to_accented_char = {
            'a': ['á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', ],
            'o': ['ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', ],
            'e': ['é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', ],
            'u': ['ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', ],
            'i': ['í', 'ì', 'ỉ', 'ĩ', 'ị', ],
            'y': ['ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', ],
            'd': ['đ', ]

        }

        map = dict()
        for k, cs in latin_to_accented_char.items():
            for c in cs:
                map[c] = k

            k = k.upper()
            cs = [c.upper() for c in cs]
            for c in cs:
                map[c] = k

        accented_chars = set()
        accented_chars.update(map.keys())
        accented_chars.update(map.values())

        self.accented_char_map = map
        self.accented_chars = accented_chars

    def is_target_char(self, c):
        return c in self.accented_chars

    def strip_accent(self, text):
        chars = [c if self.accented_char_map.get(
            c) is None else self.accented_char_map.get(c) for c in text]
        return ''.join(chars)


accent_stripper = AccentStripper()


def split_sen(exm: dict) -> dict:
    splits = exm["text"].replace("\n", "")
    # splits = [x.strip() for x in splits]
    # splits = [x for x in splits if x != ""]
    # exm["splits"] = splits
    # exm["stripped"] = accent_stripper.strip_accent(splits)
    exm["pair"] = [splits, accent_stripper.strip_accent(splits)]
    return exm

def get_data(split="train[:1%]"):
    dataset = load_dataset("wikipedia", language="vi",
                        date="20210701", split=split)
    dataset = dataset.map(split_sen)
    return dataset
