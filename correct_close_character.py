from correct_telex import TelexErrorCorrector
import unidecode
import json
import re

telexCorrector = TelexErrorCorrector()

fi = open('close_character/single_word_dict.json', 'r', encoding='utf-8')
single_dict = json.load(fi)

fi = open('close_character/vietnamese_dict.json', 'r', encoding='utf-8')
vietnamese_dict = json.load(fi)

def read_close_charater_json():
    fi = open('close_character/close_character.json')
    close_char = json.load(fi)
    for character in close_char.keys():
        close_char[character] = close_char[character].split('|')
    return close_char

close_char = read_close_charater_json()

def preprocess(sent):
    '''
    preprocess
        multi space, characters
        after a comma, semi-comma has space
    '''
    sent = sent.lower()
    # sent = re.sub(r'(?<=[;,.])(?=[^\s])', r' ', sent)
    sent = re.sub(r'[;,.]', r' ', sent)
    sent = re.sub(r'\s+', r' ', sent)
    sent = re.sub(r'^\s', '', sent)
    sent = re.sub(r'\s$', '', sent)
    return sent

def in_single_dict(word):
    try:
        return single_dict[word]
    except:
        return 0

def in_vietnamese_dict(word):
    try:
        return vietnamese_dict[word] == 1
    except:
        return False

def gen_correct_word(word):
    ls_correct_word = []
    new_word = word
    new_word = telexCorrector.fix_telex_word(new_word)
    if in_single_dict(new_word): 
        ls_correct_word.append(new_word)
        return ls_correct_word
    for j in range(len(word)):
        try:
            i = len(word) - j - 1
            replace_char = close_char[word[i]]
            for new_ch in replace_char:
                new_word = word[:i] + new_ch + word[i + 1:]
                new_word = telexCorrector.fix_telex_word(new_word)
                if in_single_dict(new_word): 
                    ls_correct_word.append(new_word)
        except: 
            pass
    ls_correct_word.sort(key=lambda w: in_single_dict(w), reverse=True)
    return ls_correct_word

def find_correct_phrase(ls_1st, ls_2nd):
    for word_1st in ls_1st:
        for word_2nd in ls_2nd:
            new_word = word_1st + ' ' + word_2nd
            if in_vietnamese_dict(new_word):
                return word_1st, word_2nd
    return ls_1st[0], None
    
def correct_close_character_sent(sent):
    sent = preprocess(sent)
    words = sent.split(' ')
    ls_cor_word = []
    for word in words:
        ls_cor_word.append(gen_correct_word(word))
    is_fix = [False] * len(words)
    for i in range(len(words)):
        if is_fix[i]: continue
        if len(ls_cor_word[i]) == 0: 
            is_fix[i] = True
            continue
        if len(ls_cor_word[i]) == 1 or i == len(words) - 1:
            is_fix[i] = True
            words[i] = ls_cor_word[i][0]
            continue
        
        new_word_1st, new_word_2nd = find_correct_phrase(ls_cor_word[i], ls_cor_word[i + 1])
        words[i] = new_word_1st
        is_fix[i] = True
        if new_word_2nd is not None:
            words[i + 1] = new_word_2nd
            is_fix[i + 1] = True
    return ' '.join(words)

def read_file(file_path):
    fi = open(file_path, 'r', encoding='utf-8')
    ls = fi.readlines()
    return ls

if __name__ == '__main__':
    # wrong = read_file('data/phimgan_wrong.txt')
    # truth = read_file('data/phimgan_truth.txt')
    # ls = []
    # for i in range(len(wrong)):
    #     if wrong[i] == None or truth[i] == None: continue
    #     fixed = correct_close_character_sent(wrong[i])
    #     truth[i] = preprocess(truth[i])
    #     unicode_truth = unidecode.unidecode(truth[i])
    #     if fixed != truth[i] and fixed != unicode_truth:
    #         ls.append({'wrong': wrong[i], 'true': truth[i], 'fixed': fixed})
    # print(len(ls))  
    # import pandas as pd
    # pd.DataFrame(ls).to_excel('data/phimgan.xlsx', index = False, engine='xlsxwriter')
    print(correct_close_character_sent('bitnh thuofngw'), telexCorrector.fix_telex_word('tòw'))