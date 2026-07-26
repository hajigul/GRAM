"""Data structures and loaders for multimodal knowledge graphs (MKGs) and QA sets.

Expected on-disk format (one directory per benchmark subset, e.g. data/bench_g/):

  entities.json   {entity_id: {"name": str, "image": relative/path.jpg or null}}
  relations.json  {relation_id: "relation surface name"}
  triplets.json   [{"id": str, "h": entity_id, "r": relation_id, "t": entity_id}]
  queries_{split}.json  [
      {"id": str,
       "text": str,                      # query text
       "image": path or null,            # optional query image
       "gold_triplets": [triplet_id,...],# retrieval supervision
       "question": str,                  # generation question (often == text)
       "answers": [str, ...]}            # gold answers
  ]
  images/         image files referenced above

Use scripts/convert_markg.py / scripts/convert_medmkg.py to produce this
format from the original dataset releases, or scripts/make_toy_data.py to
generate a synthetic MKG for smoke-testing the pipeline.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Triplet:
    tid: str
    h: str
    r: str
    t: str
    # filled by MKG
    h_name: str = ""
    r_name: str = ""
    t_name: str = ""
    image: Optional[str] = None  # absolute path to head-entity image, if any
    is_mm: bool = False          # multimodal-pool membership (corpus flag)

    def verbalize(self) -> str:
        """verb(c): delimiter-joined surface form 'h | r | t'."""
        return f"{self.h_name} | {self.r_name} | {self.t_name}"


@dataclass
class Query:
    qid: str
    text: str
    image: Optional[str]
    gold_triplets: List[str]
    question: str
    answers: List[str]
    mm: bool = False  # multimodal flag from the benchmark (survives missing image files)

    @property
    def is_multimodal(self) -> bool:
        return self.mm or self.image is not None


@dataclass
class MKG:
    entities: Dict[str, dict]
    relations: Dict[str, str]
    triplets: List[Triplet]
    root: str
    tid2idx: Dict[str, int] = field(default_factory=dict)
    # neighborhoods: triplet index -> list of neighbor triplet indices,
    # sorted by shared-entity co-occurrence frequency (descending)
    neighbors: Dict[int, List[int]] = field(default_factory=dict)

    @classmethod
    def load(cls, root: str, max_neighbors: int = 16) -> "MKG":
        #with open(os.path.join(root, "entities.json")) as f:
        with open(os.path.join(root, "entities.json"), encoding="utf-8") as f:    
            entities = json.load(f)
        with open(os.path.join(root, "relations.json"), encoding="utf-8") as f:
            relations = json.load(f)
        with open(os.path.join(root, "triplets.json"), encoding="utf-8") as f:
            raw = json.load(f)

        triplets: List[Triplet] = []
        for item in raw:
            h, r, t = item["h"], item["r"], item["t"]
            ent_h = entities.get(h, {"name": h, "image": None})
            ent_t = entities.get(t, {"name": t, "image": None})
            # triplet-level image (if present) overrides the head-entity image;
            # this lets the same entity appear in both text-only and multimodal pools
            img = item.get("image", ent_h.get("image"))
            # pool membership: explicit "mm" flag (real benchmark) or image presence (toy)
            is_mm = bool(item.get("mm", img is not None))
            triplets.append(
                Triplet(
                    tid=item["id"],
                    h=h,
                    r=r,
                    t=t,
                    h_name=ent_h.get("name", h),
                    r_name=relations.get(r, r),
                    t_name=ent_t.get("name", t),
                    image=os.path.join(root, img) if img else None,
                    is_mm=is_mm,
                )
            )
        g = cls(entities=entities, relations=relations, triplets=triplets, root=root)
        g.tid2idx = {c.tid: i for i, c in enumerate(triplets)}
        g._build_neighborhoods(cap=max_neighbors)
        return g

    def _build_neighborhoods(self, cap: int = 16) -> None:
        """N(c): triplets sharing c's head or tail entity, capped at `cap`.

        Selection follows the paper's rule — descending shared-entity
        co-occurrence frequency — implemented without materializing full
        neighborhoods (dense 1:n medical entities can have thousands of
        triplets, so we draw neighbors from the higher-frequency shared
        entity first and stop at `cap`). Only the top-m (SCV) and top-κ
        (GCR) neighbors are ever used, so cap >= max(m, κ) is lossless.
        """
        ent2trip: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(self.triplets):
            ent2trip[c.h].append(i)
            ent2trip[c.t].append(i)
        ent_freq = {e: len(v) for e, v in ent2trip.items()}

        for i, c in enumerate(self.triplets):
            ents = sorted({c.h, c.t}, key=lambda e: -ent_freq.get(e, 0))
            out: List[int] = []
            seen = {i}
            for e in ents:
                for j in ent2trip[e]:
                    if j not in seen:
                        out.append(j)
                        seen.add(j)
                        if len(out) >= cap:
                            break
                if len(out) >= cap:
                    break
            self.neighbors[i] = out

    def __len__(self) -> int:
        return len(self.triplets)


#def load_queries(root: str, split: str) -> List[Query]:
#    with open(os.path.join(root, f"queries_{split}.json")) as f:
#        raw = json.load(f)

def load_queries(root: str, split: str) -> List[Query]:
    with open(os.path.join(root, f"queries_{split}.json"), encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for item in raw:
        img = item.get("image")
        out.append(
            Query(
                qid=item["id"],
                text=item["text"],
                image=os.path.join(root, img) if img else None,
                gold_triplets=item.get("gold_triplets", []),
                question=item.get("question", item["text"]),
                answers=item.get("answers", []),
                mm=bool(item.get("mm", img is not None)),
            )
        )
    return out


def filter_setting(queries: List[Query], mkg: MKG, setting: str):
    """Apply one of the five official modality settings.

    S1: all queries x all triplets
    S2: text-only queries x text-only triplets
    S3: text-only queries x all triplets
    S4: multimodal queries x multimodal triplets
    S5: multimodal queries x all triplets

    Returns (queries, candidate_mask) where candidate_mask is a boolean
    list over mkg.triplets (True = candidate is in the pool).
    """
    setting = setting.upper()
    n = len(mkg)
    all_mask = [True] * n
    txt_mask = [not c.is_mm for c in mkg.triplets]
    mm_mask = [c.is_mm for c in mkg.triplets]

    if setting == "S1":
        return queries, all_mask
    if setting == "S2":
        return [q for q in queries if not q.is_multimodal], txt_mask
    if setting == "S3":
        return [q for q in queries if not q.is_multimodal], all_mask
    if setting == "S4":
        return [q for q in queries if q.is_multimodal], mm_mask
    if setting == "S5":
        return [q for q in queries if q.is_multimodal], all_mask
    raise ValueError(f"Unknown setting {setting}")
