path = 'src/a2g/data.py'
src = open(path, encoding='utf-8').read()

old = """            "batch_count": len(self),
            "schedule_sha256": self.signature,"""
new = """            "batch_count": len(self),
            "shuffle": self.shuffle,
            "schedule_sha256": self.signature,"""
assert old in src, 'state_dict pattern not found'
src = src.replace(old, new)

# import random（shuffle 用）
if 'import random\n' not in src:
    anchor = 'import hashlib\n'
    assert anchor in src
    src = src.replace(anchor, anchor + 'import random\n', 1)

open(path, 'w', encoding='utf-8', newline='\n').write(src)
print('OK: state_dict + import random applied')
