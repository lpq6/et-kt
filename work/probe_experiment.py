lines = open('src/a2g/experiment.py', encoding='utf-8').readlines()
print('=== train loader call ===')
for i, l in enumerate(lines):
    if 'bundles["train"]' in l:
        for j in range(max(0, i - 6), min(len(lines), i + 6)):
            print(j + 1, lines[j].rstrip())
        break
print()
print('=== epoch loop / evaluate / early stop ===')
for i, l in enumerate(lines):
    if 'best_auc' in l:
        for j in range(max(0, i - 4), min(len(lines), i + 12)):
            print(j + 1, lines[j].rstrip())
        break
print()
print('=== epochs loop head ===')
for i, l in enumerate(lines):
    if 'for epoch in range' in l:
        for j in range(max(0, i - 2), min(len(lines), i + 4)):
            print(j + 1, lines[j].rstrip())
        break
