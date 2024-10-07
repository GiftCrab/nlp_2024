# pip3 install zhconv

import zhconv

with open('../../datasets/wiki/zhwiki.txt', 'r', encoding='utf-8') as fin, open('../../datasets/wiki/simplified_chinese_wiki.txt', 'w', encoding='utf-8') as fout:
    for line in fin:
        simplified_line = zhconv.convert(line, 'zh-cn')
        fout.write(simplified_line)
