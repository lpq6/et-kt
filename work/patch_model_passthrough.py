path = 'src/a2g/model.py'
src = open(path, encoding='utf-8').read()

# 签名加参数
old1 = """        boundary_local_floor=0.5,
        use_split_boundary=1,
        normalization_topology=\"full_postnorm\","""
new1 = """        boundary_local_floor=0.5,
        use_split_boundary=1,
        use_shared_embeddings=0,
        normalization_topology=\"full_postnorm\","""
assert old1 in src, 'model sig not found'
src = src.replace(old1, new1)

# super 透传
old2 = """            boundary_local_floor=boundary_local_floor,
            use_split_boundary=use_split_boundary,
            use_umk_ssm=use_umk_ssm,"""
new2 = """            boundary_local_floor=boundary_local_floor,
            use_split_boundary=use_split_boundary,
            use_shared_embeddings=use_shared_embeddings,
            use_umk_ssm=use_umk_ssm,"""
assert old2 in src, 'super call not found'
src = src.replace(old2, new2)

open(path, 'w', encoding='utf-8', newline='\n').write(src)
print('OK: model.py passthrough added')
