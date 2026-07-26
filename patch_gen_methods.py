p = "scripts/run_generation.py"
s = open(p, encoding="utf-8").read()
if "captioning" in s:
    print("already patched"); raise SystemExit
old1 = 'choices=["gram", "fusion", "text-only"]'
new1 = 'choices=["gram", "fusion", "text-only", "random", "captioning", "reranking"]'
old2 = "        retriever = make_baseline(args.method, mkg, index, clip)"
new2 = """        if args.method in ("captioning", "reranking"):
            from gram.baselines import make_extended_baseline
            retriever = make_extended_baseline(args.method, mkg, index, clip, args.data)
        else:
            retriever = make_baseline(args.method, mkg, index, clip)"""
assert old1 in s and old2 in s, "anchor not found - paste run_generation.py to Claude"
s = s.replace(old1, new1, 1).replace(old2, new2, 1)
open(p, "w", encoding="utf-8").write(s)
print("patched OK")