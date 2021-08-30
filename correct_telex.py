# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import unicodedata
import re
import json

class TelexErrorCorrector:
    '''
      Fix telex typing errors by regexs by function fix_telex_sentence

      Step 1: Use regex to fix characters such as aw => ă, aa => â
      Step 2: Use regex to fix accent such as af => à, ar => ả
      Step 3: Use regex to fix complex telex fault ừơng -> ường
    '''
    fi = open('telex/complex_telex_fault.json', encoding='utf-8')
    complex_telex = json.load(fi)

    def __init__(self):
        self.build_character_regexs()
        self.build_accent_regexs()

    def fix_telex_sentence(self, sentence):
        sentence = unicodedata.normalize('NFC', sentence)
        words = [self.fix_telex_word(word) for word in sentence.split()]
        return ' '.join(words)

    def fix_telex_word(self, word):
        for key, value in self.char_telex_errors.items():
            word = re.sub(key, value, word)

        word = re.sub('ưo', 'ươ', word)

        for key, value in self.accent_telex_errors.items():
            word = re.sub(key, value, word)

        for key, value in self.complex_telex.items():
            word = re.sub(key, value, word)

        return word

    def build_character_regexs(self):
        chars = ['ư', 'â', 'ă', 'ô', 'ơ', 'ê']
        additional_keystrokes = ['w', 'a', 'w', 'o', 'w', 'e']

        char_telex_errors = dict()

        for i, c in enumerate(chars):
            parts = unicodedata.normalize('NFKD', c)
            base_c = parts[0]
            keystroke = additional_keystrokes[i]
            pattern = f'{base_c}(.*){keystroke}'
            char_telex_errors[pattern] = c + '\\1'

        char_telex_errors['d(.*)d'] = 'đ\\1'

        self.char_telex_errors = char_telex_errors

    def build_accent_regexs(self):
        chars = ['ơ', 'ô', 'ê', 'e', 'ă', 'â', 'ư', 'a', 'o', 'i', 'u', 'y']
        accents = ['í', 'ỉ', 'ĩ', 'ì', 'ị']
        accents = [unicodedata.normalize('NFKD', a)[1] for a in accents]
        additional_keystrokes = ['s', 'r', 'x', 'f', 'j']

        accent_telex_errors = dict()

        for c in chars:
            for i, a in enumerate(accents):
                text = ''.join([c, a])
                merged = unicodedata.normalize('NFC', text)

                keystroke = additional_keystrokes[i]
                pattern = f'{c}(.*){keystroke}'
                accent_telex_errors[pattern] = merged + '\\1'

        self.accent_telex_errors = accent_telex_errors


# %%
if __name__ == "__main__":
    corrector = TelexErrorCorrector()
    fixed = corrector.fix_telex_sentence('khuir')
    print(fixed)
