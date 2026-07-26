"""Component ablation (Table 6 style) under a chosen setting (default S4).

  python scripts/run_ablation.py --data data/bench_m --split test --setting S4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gram.data import MKG, filter_setting, load_queries
from gram.encoders import FrozenCLIP
from gram.index import GramIndex, build_index
from gram.metrics import evaluate_retrieval
from gram.retriever import GramConfig, GramRetriever


VARIANTS = {
    "Fusion baseline": dict(use_scv=False, use_rca=False, use_gcr=False, gamma=0.0),
    "GRAM w/o SCV":    dict(use_scv=False),
    "GRAM w/o RCA":    dict(use_rca=False),
    "GRAM w/o GCR":    dict(use_gcr=False),
    "GRAM (full)":     dict(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--setting", default="S4")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    mkg = MKG.load(args.data)
    queries = load_queries(args.data, args.split)
    qs, cand_mask = filter_setting(queries, mkg, args.setting)
    gold_sets = [set(mkg.tid2idx[t] for t in q.gold_triplets if t in mkg.tid2idx) for q in qs]
    print(f"{len(qs)} queries | setting {args.setting}")

    clip = FrozenCLIP()
    cache = os.path.join(args.data, "index_m8.pt")
    index = GramIndex.load(cache) if os.path.exists(cache) else build_index(mkg, clip, m=8)
    if not os.path.exists(cache):
        index.save(cache)

    results = {}
    print(f"\n{'Variant':<22} {'NDCG@5':>8} {'R@100':>8}")
    for name, overrides in VARIANTS.items():
        cfg = GramConfig(**overrides)
        retriever = GramRetriever(mkg, index, clip, cfg)
        rankings = retriever.retrieve_ranked(qs, cand_mask, depth=100)
        m = evaluate_retrieval(rankings, gold_sets)
        results[name] = m
        print(f"{name:<22} {m['NDCG@5']:8.2f} {m['R@100']:8.2f}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {args.save}")


if __name__ == "__main__":
    main()
