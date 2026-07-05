#!/usr/bin/env python3
"""
Convert AmbigQA (AmbigNQ light) multipleQAs items into single-gold-illusion
instances that mirror the FinInteract schema, for the general-domain replication.

For each genuinely ambiguous question we take >=2 readings with DISTINCT answers.
We designate one reading as the *intended* target (deterministic, disclosed rule:
the reading whose disambiguated question is most specific = longest; ties broken by
sorted answer string) and treat the remaining valid readings as *defaults* (the
non-target readings a lenient single-gold benchmark would also accept).

Emitted per instance:
  question          : the ambiguous question Q
  context           : the intended reading's disambiguated question (the resolver C)
  answer            : intended reading's answer (primary alias)
  answer_aliases    : all aliases of the intended answer
  default_answers   : list of {answer, aliases, disambig} for every OTHER reading
  all_readings      : every reading (for any-valid grading)
No passages: the general-domain replication runs answer-only / interact / oracle,
none of which need a retrieval corpus.
"""
import json, hashlib, argparse
from pathlib import Path


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def readings_of(item):
    """Return list of readings [{q, answers(list of aliases)}] with distinct answers."""
    readings = []
    seen_ans = set()
    for ann in item["annotations"]:
        if ann["type"] != "multipleQAs":
            continue
        for qa in ann["qaPairs"]:
            aliases = qa["answer"] if isinstance(qa["answer"], list) else [qa["answer"]]
            aliases = [a for a in aliases if a and a.strip()]
            if not aliases:
                continue
            key = tuple(sorted(norm(a) for a in aliases))
            # distinct if it shares no alias with an already-taken reading
            flat = set(norm(a) for a in aliases)
            if flat & seen_ans:
                continue
            seen_ans |= flat
            readings.append({"q": qa["question"].strip(), "answers": aliases})
    return readings


def main(src, out, limit, intended_rule="longest"):
    data = json.load(open(src))
    instances = []
    for item in data:
        rs = readings_of(item)
        if len(rs) < 2:
            continue
        h = int(hashlib.md5(item["id"].encode()).hexdigest(), 16)
        if intended_rule == "hash":
            # reproducible pseudo-random target: robustness control vs "longest"
            intended = rs[h % len(rs)]
        else:
            # deterministic intended pick: most specific (longest disambig q)
            intended = max(rs, key=lambda r: (len(r["q"]), norm(r["answers"][0]), h))
        others = [r for r in rs if r is not intended]
        instances.append({
            "instance_id": f"ambig_{item['id']}",
            "language": "en",
            "source": "ambigqa",
            "question": item["question"].strip(),
            "context": intended["q"],                       # the resolver C
            "answer": intended["answers"][0],
            "answer_aliases": intended["answers"],
            "default_answers": [{"answer": r["answers"][0],
                                 "aliases": r["answers"],
                                 "disambig": r["q"]} for r in others],
            "all_readings": [{"answer": r["answers"][0], "aliases": r["answers"],
                              "disambig": r["q"]} for r in rs],
            "n_readings": len(rs),
            "axes": ["general_ambiguity"],
        })
    # stable order, then cap
    instances.sort(key=lambda x: x["instance_id"])
    if limit:
        instances = instances[:limit]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ins in instances:
            f.write(json.dumps(ins, ensure_ascii=False) + "\n")
    n_read = sum(x["n_readings"] for x in instances) / max(1, len(instances))
    print(f"wrote {len(instances)} instances -> {out}  (avg readings/inst = {n_read:.2f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/general_domain/raw/dev_light.json")
    p.add_argument("--out", default="data/general_domain/ambig_instances.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--intended", choices=["longest", "hash"], default="longest")
    a = p.parse_args()
    main(a.src, a.out, a.limit, a.intended)
