"""Offline smoke test: runs the FULL GRAM pipeline (index -> stage1 with
RCA+SCV -> GCR -> metrics -> ablations -> all 5 settings) using a mock
encoder, so it needs no network or GPU. Real runs use FrozenCLIP instead.

NOTE: the printed numbers only verify the code paths run and metrics compute;
mock hash embeddings do NOT reflect CLIP semantics, so relative method
quality here is meaningless. Use run_retrieval.py with real CLIP for that.

  python tests/test_pipeline_mock.py
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from gram.data import MKG, filter_setting, load_queries
from gram.index import build_index
from gram.metrics import evaluate_retrieval
from gram.retriever import GramConfig, GramRetriever, make_baseline

DIM = 128


def _text_vec(s: str) -> torch.Tensor:
    """Character-trigram hashing embedding: similar strings -> similar vectors."""
    v = torch.zeros(DIM)
    s = s.lower()
    for i in range(len(s) - 2):
        h = int(hashlib.md5(s[i:i+3].encode()).hexdigest(), 16) % DIM
        v[h] += 1.0
    return torch.nn.functional.normalize(v + 1e-8, dim=-1)


def _img_vec(path: str) -> torch.Tensor:
    """Mock 'visual semantics': embed the dominant color + rough shape as text,
    placed in the same space as text vectors (so cross-modal matching works)."""
    img = Image.open(path).convert("RGB").resize((16, 16))
    px = list(img.getdata())
    r = sum(p[0] for p in px) / len(px)
    g = sum(p[1] for p in px) / len(px)
    b = sum(p[2] for p in px) / len(px)
    color = max([("red", r - (g + b) / 2), ("green", g - (r + b) / 2), ("blue", b - (r + g) / 2)],
                key=lambda x: x[1])[0]
    return _text_vec(f"image of a {color} object")


class MockCLIP:
    dim = DIM

    def encode_text(self, texts, batch_size=256):
        return torch.stack([_text_vec(t) for t in texts])

    def encode_images(self, paths, batch_size=128):
        return torch.stack([_img_vec(p) for p in paths])


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "toy")
    assert os.path.exists(root), "run scripts/make_toy_data.py first"

    mkg = MKG.load(root)
    queries = load_queries(root, "test")
    clip = MockCLIP()
    index = build_index(mkg, clip, m=8)
    print(f"index: e_txt {tuple(index.e_txt.shape)}, images: {int(index.has_img.sum())}")

    for setting in ["S1", "S2", "S3", "S4", "S5"]:
        qs, mask = filter_setting(queries, mkg, setting)
        if not qs:
            print(f"[{setting}] no queries, skipped")
            continue
        gold = [set(mkg.tid2idx[t] for t in q.gold_triplets) for q in qs]
        rows = {}
        for method in ["gram", "fusion", "text-only"]:
            if method == "gram":
                r = GramRetriever(mkg, index, clip, GramConfig())
            else:
                r = make_baseline(method, mkg, index, clip)
            ranks = r.retrieve_ranked(qs, mask, depth=100)
            rows[method] = evaluate_retrieval(ranks, gold)
        print(f"[{setting}] ({len(qs)} q) " + "  ".join(
            f"{m}: NDCG@5={rows[m]['NDCG@5']:.2f} R@100={rows[m]['R@100']:.2f}" for m in rows))

    # ablation sanity: every variant runs, full GRAM produced
    qs, mask = filter_setting(queries, mkg, "S1")
    gold = [set(mkg.tid2idx[t] for t in q.gold_triplets) for q in qs]
    for name, ov in [("w/o SCV", dict(use_scv=False)), ("w/o RCA", dict(use_rca=False)),
                     ("w/o GCR", dict(use_gcr=False))]:
        r = GramRetriever(mkg, index, clip, GramConfig(**ov))
        m = evaluate_retrieval(r.retrieve_ranked(qs, mask, depth=100), gold)
        print(f"[ablation {name}] NDCG@5={m['NDCG@5']:.2f}")

    # invariants
    cfg = GramConfig()
    r = GramRetriever(mkg, index, clip, cfg)
    qbar = r.embed_queries(qs[:4])
    s1 = r.stage1(qbar)
    B = max(cfg.a, cfg.b) + cfg.gamma
    assert torch.isfinite(s1).all() and s1.abs().max() <= B + 1e-4, "P3 boundedness violated"
    # P1: text-only query on image-less candidates == pure text matching path
    print("P3 boundedness OK; pipeline smoke test PASSED")


if __name__ == "__main__":
    main()
