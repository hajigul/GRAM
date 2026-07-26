p = "scripts/run_generation.py"
s = open(p, encoding="utf-8").read()
if "--beta" in s:
    print("already patched"); raise SystemExit
old1 = 'ap.add_argument("--K", type=int, default=5)'
new1 = (old1 + '\n    ap.add_argument("--beta", type=float, default=5.0)'
             '\n    ap.add_argument("--gamma", type=float, default=0.0)'
             '\n    ap.add_argument("--lam", type=float, default=1.0)')
old2 = "GramConfig(K=args.K)"
new2 = ("GramConfig(K=args.K, beta=args.beta, gamma=args.gamma, lam=args.lam, "
        "use_scv=args.gamma > 0, use_gcr=args.lam < 1.0)")
assert old1 in s and old2 in s, "anchor not found - paste run_generation.py to Claude"
s = s.replace(old1, new1, 1).replace(old2, new2, 1)
open(p, "w", encoding="utf-8").write(s)
print("patched OK")