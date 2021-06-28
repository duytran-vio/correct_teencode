import json

fi = open('short_word.json', encoding='utf-8')
dic = json.load(fi)

new_dic={}
k = list(dic.items())
k.sort(key=lambda x:len(x[0]),reverse=True)

for i in k :
    new_dic.update({i[0]:i[1]})

print(new_dic)

fo = open('short_word_sorted.json', 'w', encoding='utf-8')
json.dump(new_dic, fo)