import pandas as pd


# bep-eng.dicから辞書作成
dic_file = 'data/bep-eng.dic'
english_list = []
katakana_list = []
with open(dic_file, mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i >= 6:
            line_list = line.replace('\n', '').split(' ')
            english_list.append(line_list[0])
            katakana_list.append(line_list[1])
df = pd.DataFrame(dict(english=english_list, katakana=katakana_list))
print(df.shape)
df.to_csv('data/bep-eng.csv', sep=',', index=False)