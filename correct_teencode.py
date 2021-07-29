import json
import re
import itertools
import unidecode

def read_file(file_path):
    fi = open(file_path, 'r', encoding='utf-8')
    ls = fi.readlines()
    return ls

vowel_file = open('vietnamese_vowel.json', encoding='utf-8')
vowel_dic = json.load(vowel_file)

short_word_file = open('short_word.json', encoding='utf-8')
short_word_dic = json.load(short_word_file)

teencode_re_file = open('teencode_regex.json', encoding='utf-8')
teencode_re_dic = json.load(teencode_re_file)

single_word_dic = read_file('unidecode_vietnamese_dic.txt')
single_word_dic = [re.sub('\n','', s) for s in single_word_dic]

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
    '''
    replace teencode with correct one by using dictionary
    Input: 
        word        :str - teencode word 
        dictionary  : pd.Dataframe - 1-1 dictionary
    return: 
        new_word    :str - correct word
    '''
    new_word = dictionary.get(word, word)
    if new_word == word:
        uni_word = unidecode.unidecode(word)
        uni_word = replace_with_regex(uni_word, teencode_re_dic, dictionary)
        new_word = dictionary.get(uni_word, word)
    return new_word

def replace_with_regex(word, regex_list, dic_one_one, check = 0):
    '''
    replace teencode with correct one by using rule (regex)
    Input:
        word        : str - teencode word
        regex_list  : pd.DataFrame - teencode regex 
        dic_one_one : pd.DataFrame - 1-1 dictionary
        check       : boolean - number of times using this method
    return: 
        new_word    : str - correct word
    '''
    new_word = unidecode.unidecode(word)
    for pattern in regex_list.keys():
        if re.search(pattern, new_word):
            new_word = re.sub(pattern, regex_list[pattern], new_word)
            break
    if dic_one_one.get(new_word, new_word) != new_word: return dic_one_one.get(new_word, new_word)
    if check == 2 or new_word in single_word_dic: return new_word
    new_word = replace_with_regex(new_word, teencode_re_dic, short_word_dic, check + 1)
    return new_word

def correct_vowel(sent, vowel_dictionary):
    '''
    correct sentence has vowel next to symbol by rule. Ex: a~ -> ã
    Input:
        sent    : str - teencode sentence
        vowel_dictionary: pd.DataFrame - vietnamese_vowel dictionary
    return:
        sent    : str - correct sentence
    '''
    words = sent.split()
    pattern = r'[aăâeêuưiyoôơ][.`~?\']'
    sent = ""
    for word in words:
        p = re.search(pattern, word)
        new_word = word
        if p:
            idx = p.span()
            replace_vowel = vowel_dictionary[word[idx[0]]][word[idx[0] + 1]]
            new_word = re.sub(pattern, replace_vowel, new_word)
        sent += new_word + ' '
    return sent

def correct_teencode(sent):
    '''
    correct teencode sentence
    Input: 
        sent    : str - teencode sent
    Return:
        correct sent 
    '''
    sent = preprocess(sent)
    sent = correct_vowel(sent, vowel_dic)
    words = sent.split()
    sent = ""
    for word in words:
        new_word = ""
        if word[-1] == ',' or word[-1] == ';':
            new_word = replace_one_one(word[:-1], short_word_dic)
            if word[:-1] == new_word:
                new_word = replace_with_regex(new_word, teencode_re_dic, short_word_dic)
            sent += new_word + word[-1]
        else:
            new_word = replace_one_one(word, short_word_dic)
            if word == new_word:
                new_word = replace_with_regex(new_word, teencode_re_dic, short_word_dic)
            sent += new_word
        sent += ' '
    return sent[:-1]

if __name__ == '__main__':
    print(correct_teencode("okiii"))

            