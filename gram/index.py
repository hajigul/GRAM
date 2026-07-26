"""Offline indexing (Algorithm 1, line 1).

For every candidate triplet c we cache:
  e_txt[c] = fT(verb(c))
  e_img[c] = fI(v_h)              (zero vector + mask=False if no image)
  e_rel[c] = fT(r)
  e_ctx[c] = fT( ⊕_{c' in N_m(c)} verb(c') )   (SCV, Eq. 2)

Concatenate-then-encode: neighbors are joined into ONE context string and
encoded jointly (not averaged), per the paper.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from tqdm import tqdm

from .data import MKG
from .encoders import FrozenCLIP


@dataclass
class GramIndex:
    e_txt: torch.Tensor     # [T, d]
    e_img: torch.Tensor     # [T, d] (zeros where has_img is False)
    e_rel: torch.Tensor     # [T, d]
    e_ctx: torch.Tensor     # [T, d]
    has_img: torch.Tensor   # [T] bool

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "e_txt": self.e_txt,
                "e_img": self.e_img,
                "e_rel": self.e_rel,
                "e_ctx": self.e_ctx,
                "has_img": self.has_img,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "GramIndex":
        d = torch.load(path, map_location="cpu")
        return cls(**d)


def build_index(mkg: MKG, clip: FrozenCLIP, m: int = 8, batch_size: int = 256) -> GramIndex:
    T = len(mkg)
    d = clip.dim

    # --- triplet text embeddings ------------------------------------------------
    verb = [c.verbalize() for c in mkg.triplets]
    e_txt = clip.encode_text(verb, batch_size)

    # --- relation embeddings (encode unique relations once) ----------------------
    rel_names = sorted({c.r_name for c in mkg.triplets})
    rel_emb = clip.encode_text(rel_names, batch_size)
    rel2row = {r: i for i, r in enumerate(rel_names)}
    e_rel = torch.stack([rel_emb[rel2row[c.r_name]] for c in mkg.triplets])

    # --- image embeddings ---------------------------------------------------------
    e_img = torch.zeros(T, d)
    has_img = torch.zeros(T, dtype=torch.bool)
    img_idx = [i for i, c in enumerate(mkg.triplets) if c.image and os.path.exists(c.image)]
    if img_idx:
        # encode each unique image once
        uniq = sorted({mkg.triplets[i].image for i in img_idx})
        uniq_emb = clip.encode_images(list(uniq))
        path2row = {p: j for j, p in enumerate(uniq)}
        for i in tqdm(img_idx, desc="assign image embeddings", leave=False):
            e_img[i] = uniq_emb[path2row[mkg.triplets[i].image]]
            has_img[i] = True

    # --- SCV context embeddings (Eq. 2) ------------------------------------------
    ctx_strings = []
    for i in range(T):
        nbrs = mkg.neighbors.get(i, [])[:m]
        if nbrs:
            ctx_strings.append(" ; ".join(mkg.triplets[j].verbalize() for j in nbrs))
        else:
            # isolated candidate: fall back to its own verbalization
            ctx_strings.append(mkg.triplets[i].verbalize())
    e_ctx = clip.encode_text(ctx_strings, batch_size)

    return GramIndex(e_txt=e_txt, e_img=e_img, e_rel=e_rel, e_ctx=e_ctx, has_img=has_img)
