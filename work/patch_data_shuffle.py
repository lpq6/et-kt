path = 'src/a2g/data.py'
src = open(path, encoding='utf-8').read()

old = """    def __init__(self, metadata, batch_size, seed, epoch):
        self.size = len(metadata)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.signature = hashlib.sha256(
            json.dumps(
                {
                    \"batch_size\": batch_size,
                    \"order\": [
                        [entry.csv_row_index, entry.learner_uid] for entry in metadata
                    ],
                },
                separators=(\",\", \":\"),
            ).encode()
        ).hexdigest()

    def __iter__(self):
        for start in range(0, self.size, self.batch_size):
            yield list(range(start, min(start + self.batch_size, self.size)))"""

new = """    def __init__(self, metadata, batch_size, seed, epoch, shuffle=False):
        self.size = len(metadata)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.shuffle = bool(shuffle)
        self.signature = hashlib.sha256(
            json.dumps(
                {
                    \"batch_size\": batch_size,
                    \"shuffle\": bool(shuffle),
                    \"order\": [
                        [entry.csv_row_index, entry.learner_uid] for entry in metadata
                    ],
                },
                separators=(\",\", \":\"),
            ).encode()
        ).hexdigest()

    def __iter__(self):
        # v46: seeded per-epoch shuffle for training (deterministic, reproducible).
        indices = list(range(self.size))
        if self.shuffle:
            rng = random.Random((self.seed * 1000003 + self.epoch) & 0xFFFFFFFF)
            rng.shuffle(indices)
        for start in range(0, self.size, self.batch_size):
            yield indices[start : min(start + self.batch_size, self.size)]"""

assert old in src, 'sampler pattern not found'
src = src.replace(old, new)

old2 = """            "batch_count": len(self),
            "schedule_sha256": self.signature,"""
new2 = """            "batch_count": len(self),
            "shuffle": self.shuffle,
            "schedule_sha256": self.signature,"""
assert old2 in src, 'state_dict pattern not found'
src = src.replace(old2, new2)

open(path, 'w', encoding='utf-8', newline='\n').write(src)
print('OK: shuffle support added')
