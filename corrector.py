from correct_teencode import correct_teencode, correct_short_word_sent, correct_teencode_word
from correct_close_character import correct_close_character_sent
from correct_telex import TelexErrorCorrector
import json
import re

telexCorrector = TelexErrorCorrector()
fi = open('data/tudien_don.json', 'r', encoding='utf-8')
dictionary = json.load(fi)

def preprocess(sent):
    sent = sent.lower()
    sent = re.sub(r'[;,.?:\=\+\|!]', r' ', sent)
    sent = re.sub('\n', '', sent)
    sent = re.sub(r'\s+', r' ', sent)
    sent = re.sub(r'^\s', '', sent)
    sent = re.sub(r'\s$', '', sent)
    return sent

def in_dictionary(word):
    try:
        return dictionary[word] != 0
    except:
        return False

def use_correct_func(sent, func):
    words = sent.split()
    ls = []
    for word in words:
        new_word = func(word)
        if not in_dictionary(new_word):
            new_word = word
        ls.append(new_word)
    return ' '.join(ls)

def correct_sent(sent):
    sent = preprocess(sent)
    sent = correct_short_word_sent(sent)
    sent = use_correct_func(sent, telexCorrector.fix_telex_word)
    sent = use_correct_func(sent, correct_teencode_word)
    sent = use_correct_func(sent, telexCorrector.fix_telex_word)
    sent = correct_close_character_sent(sent)
    return sent

if __name__ == "__main__":
    sent = 'Có hỗ trơk đổi k shop ơi'
    print(correct_sent(sent))
    # print(in_dictionary('facebook'))