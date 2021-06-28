import json
import re
import itertools

vowel_file = open('vietnamese_vowel.json', encoding='utf-8')
vowel_dic = json.load(vowel_file)

short_word_file = open('short_word.json', encoding='utf-8')
short_word_dic = json.load(short_word_file)

teencode_re_file = open('teencode_regex.json', encoding='utf-8')
teencode_re_dic = json.load(teencode_re_file)

def preprocess(sent):
    '''
    preprocess
        multi space, characters
        after a comma, semi-comma has space
    '''
    sent = sent.lower()
    sent = re.sub(r'(?<=[;,])(?=[^\s])', r' ', sent)
    sent = re.sub(r'\s+', r' ', sent)
    sent = re.sub(r'^\s', '', sent)
    sent = re.sub(r'\s$', '', sent)
    sent = ''.join(c[0] for c in itertools.groupby(sent))
    return sent

def replace_one_one(word, dictionary):
    return dictionary.get(word, word)

def replace_with_regex(word, regex_list):
    new_word = word
    for pattern in regex_list.keys():
        if re.search(pattern, word):
            new_word = re.sub(pattern, regex_list[pattern], new_word)
    return new_word

def correct_vowel(sent, dictionary):
    words = sent.split()
    pattern = r'[aăâeêuưiyoôơ][.`~?\']'
    sent = ""
    for word in words:
        p = re.search(pattern, word)
        new_word = word
        if p:
            idx = p.span()
            replace_vowel = dictionary[word[idx[0]]][word[idx[0] + 1]]
            new_word = re.sub(pattern, replace_vowel, new_word)
        sent += new_word + ' '
    return sent

def correct_teencode(sent):
    sent = preprocess(sent)
    sent = correct_vowel(sent, vowel_dic)
    words = sent.split()
    sent = ""
    for word in words:
        new_word = ""
        if word[-1] == ',' or word[-1] == ';':
            new_word = replace_one_one(word[:-1], short_word_dic)
            if word[:-1] == new_word:
                new_word = replace_with_regex(new_word, teencode_re_dic)
            sent += new_word + word[-1]
        else:
            new_word = replace_one_one(word, short_word_dic)
            if word == new_word:
                new_word = replace_with_regex(new_word, teencode_re_dic)
            sent += new_word
        sent += ' '
    return sent

if __name__ == '__main__':
    sent = "vơ'i e thi` la` va.y đo' đó"
    print(correct_teencode(sent))
            