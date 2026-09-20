# -*- coding: utf-8 -*-
"""Fix misaligned merged-cell writes in table 0 rows 2-3."""
import docx

PATH = r'work\apply\申请书_TARGET.docx'
d = docx.Document(PATH)
t0 = d.tables[0]

def unique_cells(row):
    seen, out = set(), []
    for c in row.cells:
        k = id(c._tc)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out

def set_by_label(row, label, value):
    """Fill value cell that follows a label cell; restore label if missing."""
    cells = unique_cells(row)
    labels = [c.text.strip() for c in cells]
    for i, lab in enumerate(labels):
        if lab == label and i + 1 < len(cells):
            cells[i + 1].text = ''
            set_cell(cells[i + 1], value)
            return True
    return False

def set_cell(cell, text):
    paras = cell.paragraphs
    p0 = paras[0]
    if p0.runs:
        p0.runs[0].text = text
        for r in p0.runs[1:]:
            r.text = ''
    else:
        p0.add_run(text)
    for extra in paras[1:]:
        extra.text = ''

# R2: 学号标签可能被覆盖 → 恢复标签并填值
r2 = t0.rows[2]
r2cells = unique_cells(r2)
r2labels = [c.text.strip() for c in r2cells]
# 找"性别"与"出生年月"之间的格子：应为 学号标签 + 学号值
for i, lab in enumerate(r2labels):
    if lab in ('（待填）',) and i > 0 and r2labels[i-1] == '性别':
        # 学号标签位置（被误覆盖）
        set_cell(r2cells[i], '学号')
    if lab == '学号':
        if i + 1 < len(r2cells):
            set_cell(r2cells[i + 1], '（待填）')
# 若标签仍未出现（被当成值），按位置修复：性别之后第2格为学号标签
r2labels2 = [c.text.strip() for c in unique_cells(r2)]
if '学号' not in r2labels2:
    cells = unique_cells(r2)
    for i, lab in enumerate(cells):
        if lab.text.strip() == '性别' and i + 2 < len(cells):
            set_cell(cells[i + 2], '学号')
            set_cell(cells[i + 3], '（待填）')
            break

# R3: 专业标签可能被覆盖 → 恢复标签并填值
r3 = t0.rows[3]
r3cells = unique_cells(r3)
r3labels = [c.text.strip() for c in r3cells]
if '专业' not in r3labels:
    for i, c in enumerate(r3cells):
        if c.text.strip() == '学院' and i + 1 < len(r3cells):
            # 学院值已填；其后应为 专业标签 + 专业值
            set_cell(r3cells[i + 2], '专业')
            set_cell(r3cells[i + 3], '（待填）')
            break
else:
    for i, lab in enumerate(r3labels):
        if lab == '专业' and i + 1 < len(r3cells):
            set_cell(r3cells[i + 1], '（待填）')

d.save(PATH)
print('fixed')

# 复查
d2 = docx.Document(PATH)
for ri in (2, 3, 4):
    seen, cells = set(), []
    for c in d2.tables[0].rows[ri].cells:
        k = id(c._tc)
        if k not in seen:
            seen.add(k)
            cells.append(c.text.strip())
    print(f'R{ri}: {cells}')
