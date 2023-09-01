import re

with open('myfunction/陈立恒-人工智能科普.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

chinese_lines = []
for line in lines:
    chinese = re.findall(u'[\u4e00-\u9fff]+', line)
    if chinese:
        chinese_lines.append(''.join(chinese))

with open('myfunction/output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(chinese_lines))