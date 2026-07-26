p = "scripts/run_generation.py"
s = open(p, encoding="utf-8").read()

if "--mix-text" in s:
    print("already patched"); raise SystemExit

s = s.replace(
    'ap.add_argument("--save", default=None)',
    'ap.add_argument("--mix-text", type=int, default=0,\n'
    '                help="replace last N of top-K with top text-similarity triplets")\n'
    '    ap.add_argument("--save", default=None)'
)

s = s.replace(
    "rankings = retriever.retrieve(qs, cand_mask, topk=args.K)",
    """rankings = retriever.retrieve(qs, cand_mask, topk=args.K)
    if args.mix_text > 0:
        text_r = make_baseline("text-only", mkg, index, clip)
        t_rankings = text_r.retrieve(qs, cand_mask, topk=args.K + 5)
        merged = []
        for main, tr in zip(rankings, t_rankings):
            keep = main[: args.K - args.mix_text]
            for j in tr:
                if len(keep) >= args.K:
                    break
                if j not in keep:
                    keep.append(j)
            merged.append(keep)
        rankings = merged"""
)

open(p, "w", encoding="utf-8").write(s)
print("patched OK")