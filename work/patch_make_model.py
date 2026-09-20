path = 'src/a2g/experiment.py'
src = open(path, encoding='utf-8').read()

old = """    elif architecture == "v43":
        from .candidate import A2GModal

        model_class = A2GModal
    elif config.get("architecture", "v33") == "v33":
        model_class = A2G
    else:
        raise ValueError("unknown architecture")"""

new = """    elif architecture in ("v43", "v46"):
        from .candidate import A2GModal

        model_class = A2GModal
    elif config.get("architecture", "v33") == "v33":
        model_class = A2G
    else:
        raise ValueError("unknown architecture")"""

assert old in src, 'make_model branch not found'
src = src.replace(old, new)
open(path, 'w', encoding='utf-8', newline='\n').write(src)
print('OK: v46 architecture routed to A2GModal')
