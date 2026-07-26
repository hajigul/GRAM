"""Run retrieval evaluation (Tables 3/4 style).

Examples:
  # toy smoke test, all settings, GRAM + Fusion baseline
  python scripts/run_retrieval.py --data data/toy --split test --settings S1 S2 S3 S4 S5

  # real benchmark subset
  python scripts/run_retrieval.py --data data/bench_g --split test --settings S1 S4 \
      --methods gram fusion text-only --save outputs/bench_g_retrieval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from gram.data import MKG, filter_setting, load_queries
from gram.encoders import FrozenCLIP
from gram.index import GramIndex, build_index
from gram.metrics import evaluate_retrieval
from gram.retriever import GramConfig, GramRetriever, make_baseline


def get_index(mkg: MKG, clip: FrozenCLIP, data_root: str, m: int, rebuild: bool) -> GramIndex:
    cache = os.path.join(data_root, f"index_m{m}.pt")
    if os.path.exists(cache) and not rebuild:
        print(f"Loading cached index: {cache}")
        return GramIndex.load(cache)
    print("Building index (offline embeddings, cached for reuse)...")
    idx = build_index(mkg, clip, m=m)
    idx.save(cache)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root directory")
    ap.add_argument("--split", default="test")
    ap.add_argument("--settings", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    ap.add_argument("--methods", nargs="+", default=["gram", "fusion", "text-only"])
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--save", default=None)
    # GRAM hyperparameters (paper defaults)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--kappa", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=0.0) # added 
    args = ap.parse_args()

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    mkg = MKG.load(args.data)
    print(f"MKG: {len(mkg)} triplets, {len(mkg.entities)} entities, {len(mkg.relations)} relations")
    queries = load_queries(args.data, args.split)
    print(f"Queries ({args.split}): {len(queries)} "
          f"({sum(q.is_multimodal for q in queries)} multimodal)")

    clip = FrozenCLIP()
    index = get_index(mkg, clip, args.data, args.m, args.rebuild_index)

    all_results = {}
    for setting in args.settings:
        qs, cand_mask = filter_setting(queries, mkg, setting)
        if not qs:
            print(f"[{setting}] no queries in this setting, skipping")
            continue
        print(f"\n===== Setting {setting}: {len(qs)} queries, "
              f"{sum(cand_mask)} candidate triplets =====")
        gold_sets = [set(mkg.tid2idx[t] for t in q.gold_triplets if t in mkg.tid2idx) for q in qs]

        for method in args.methods:
            if method == "gram":
                cfg = GramConfig(m=args.m, gamma=args.gamma, beta=args.beta,
                                 N=args.N, kappa=args.kappa, lam=args.lam, tau=args.tau)
                retriever = GramRetriever(mkg, index, clip, cfg)
            else:
                if method in ("captioning", "reranking"):
                    from gram.baselines import make_extended_baseline
                    retriever = make_extended_baseline(method, mkg, index, clip, args.data)
                else:
                    retriever = make_baseline(method, mkg, index, clip)
            t0 = time.time()
            rankings = retriever.retrieve_ranked(qs, cand_mask, depth=args.depth)
            dt = time.time() - t0
            metrics = evaluate_retrieval(rankings, gold_sets)
            qps = len(qs) / dt if dt > 0 else float("inf")
            print(f"\n[{setting}] {method.upper()}  (QPS: {qps:.2f})")
            for k in (5, 10, 20, 50, 100):
                print(f"  NDCG@{k:<3} {metrics[f'NDCG@{k}']:6.2f}   "
                      f"P@{k:<3} {metrics[f'P@{k}']:6.2f}   "
                      f"R@{k:<3} {metrics[f'R@{k}']:6.2f}")
            all_results[f"{setting}/{method}"] = {**metrics, "QPS": qps}

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {args.save}")


if __name__ == "__main__":
    main()
