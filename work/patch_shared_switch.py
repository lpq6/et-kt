path = 'src/a2g/_initialization.py'
src = open(path, encoding='utf-8').read()

# 1) 签名加 use_shared_embeddings（默认 False → v43 逐位兼容）
old_sig = """        use_split_boundary=1,
        use_umk_ssm=0,
        umk_lambda0=0.1,
        **kwargs,
    ):"""
new_sig = """        use_split_boundary=1,
        use_shared_embeddings=0,
        use_umk_ssm=0,
        umk_lambda0=0.1,
        **kwargs,
    ):"""
assert old_sig in src, 'init signature not found'
src = src.replace(old_sig, new_sig)

# 2) 条件绑定（默认独立表）
old_emb = """        self.item_emb = nn.Embedding(self.n_pid + 1, d_model, padding_idx=0)
        self.concept_emb = nn.Embedding(self.n_question + 1, d_model, padding_idx=0)
        self.hist_item_emb = self.item_emb
        self.hist_concept_emb = self.concept_emb"""
new_emb = """        self.item_emb = nn.Embedding(self.n_pid + 1, d_model, padding_idx=0)
        self.concept_emb = nn.Embedding(self.n_question + 1, d_model, padding_idx=0)
        if bool(int(use_shared_embeddings)):
            # v46: weight tying —— target/history 双流复用同一词表
            # （跨域借鉴 Transformer 权重共享：同一概念空间在不同时间位置用同一向量）
            self.hist_item_emb = self.item_emb
            self.hist_concept_emb = self.concept_emb
        else:
            self.hist_item_emb = nn.Embedding(self.n_pid + 1, d_model, padding_idx=0)
            self.hist_concept_emb = nn.Embedding(
                self.n_question + 1, d_model, padding_idx=0
            )"""
assert old_emb in src, 'embedding block not found'
src = src.replace(old_emb, new_emb)

open(path, 'w', encoding='utf-8', newline='\n').write(src)
print('OK: use_shared_embeddings switch added (default off)')
