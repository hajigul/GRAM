"""Route 'captioning' and 'reranking' methods in run_retrieval.py to the
extended baselines. Run once from the repo root: python patch_baselines.py"""
p = "scripts/run_retrieval.py"
s = open(p, encoding="utf-8").read()
if "make_extended_baseline" in s:
    print("already patched"); raise SystemExit
old = "retriever = make_baseline(method, mkg, index, clip)"
new = """if method in ("captioning", "reranking"):
                    from gram.baselines import make_extended_baseline
                    retriever = make_extended_baseline(method, mkg, index, clip, args.data)
                else:
                    retriever = make_baseline(method, mkg, index, clip)"""
assert old in s, "anchor line not found - paste your run_retrieval.py lines 75-95 to Claude"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("patched OK - run_retrieval.py now accepts --methods captioning reranking")
