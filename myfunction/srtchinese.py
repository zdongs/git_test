import re

with open('git_test\myfunction\CHS_AI方向基础讲解.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

chinese_lines = []
for line in lines:
    chinese = re.findall(u'[\u4e00-\u9fff]+', line)
    if chinese:
        chinese_lines.append(''.join(chinese))

with open('git_test\myfunction/output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(chinese_lines))