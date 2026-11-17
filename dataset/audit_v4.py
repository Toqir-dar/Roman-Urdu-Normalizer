#!/usr/bin/env python3
"""How much does each v4 test set test generalization? -> data/v4/generalization_report.json + a markdown table.

For every evaluation set: how much of it is unseen in train (gold words, noisy spellings, exact corrections,
English words, names, sentence frames), and WER/CER of two reference systems:
  * unchanged   - the input copied to the output
  * dictionary  - most frequent train correction per noisy word, then the train-safe lexicon
Standard library only.
"""
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.join(HERE, "data", "v4")
TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+(?:[.:,/]\d+)*(?:%|[^\W\d_]+)?|([^\w\s])\1*", re.UNICODE)
SETS = ("test_real", "test_social", "test_social_uci", "test_synth_iid", "test_synth_ood", "dev")


def read(name):
    p = os.path.join(V4, name + ".jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def words(text):
    out = []
    for m in TOKEN_RE.finditer(text.lower()):
        t = m.group(0)
        if not re.match(r"^[^\w\s]+$", t):        # punctuation is not scored
            out.append(t)
    return out


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer_cer(hyps, refs):
    we = wn = ce = cn = 0
    for h, r in zip(hyps, refs):
        hw, rw = words(h), words(r)
        we += lev(hw, rw)
        wn += len(rw)
        hs, rs = " ".join(hw), " ".join(rw)
        ce += lev(hs, rs)
        cn += len(rs)
    return 100 * we / max(wn, 1), 100 * ce / max(cn, 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    train = read("train")
    sets = {k: read(k) for k in SETS}
    sets = {k: v for k, v in sets.items() if v}
    # split dev by source so the social / templated / real parts can be read separately
    for src in ("dakshina", "templated", "social_new"):
        sub = [r for r in sets.get("dev", []) if r["source"] == src]
        if sub:
            sets["dev_" + src] = sub
    sets.pop("dev", None)

    def toks(r):
        return [t for t in r["tokens"] if t.get("norm")]
    tr_gold = Counter(t["norm"].lower() for r in train for t in toks(r) if t["lang"] == "UR")
    tr_noisy = set(t["noisy"].lower() for r in train for t in toks(r) if t.get("noisy"))
    tr_pairs = set((t["noisy"].lower(), t["norm"].lower()) for r in train for t in toks(r) if t.get("noisy"))
    tr_en = set(t["norm"].lower() for r in train for t in toks(r) if t["lang"] == "EN")
    tr_ne = set(t["norm"].lower() for r in train for t in toks(r) if t["lang"] == "NE")
    tr_frames = set(r.get("frame") for r in train if r.get("frame"))

    safe_canon, safe_inv = set(), {}
    with open(os.path.join(V4, "lexicon_trainsafe.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            safe_canon.add(r["canonical"])
            if r["variant"]:
                safe_inv.setdefault(r["variant"], r["canonical"])

    votes = defaultdict(Counter)
    for r in train:
        if r.get("tier") == "silver":
            continue
        for t in toks(r):
            if t.get("noisy") and " " not in t["noisy"] and not t.get("merged"):
                votes[t["noisy"].lower()][t["norm"].lower()] += 1
    dic = {k: v.most_common(1)[0][0] for k, v in votes.items()}

    def dict_norm(text):
        out = []
        for w in words(text):
            l = re.sub(r"(.)\1{2,}", r"\1", w)
            out.append(dic.get(w) or dic.get(l) or (l if l in safe_canon else safe_inv.get(l, l)))
        return " ".join(out)

    report = {}
    for name, rows in sets.items():
        ur = [t for r in rows for t in toks(r) if t["lang"] == "UR"]
        withnoisy = [t for r in rows for t in toks(r) if t.get("noisy")]
        changed = [t for t in withnoisy if t["noisy"].lower() != t["norm"].lower()]
        en = set(t["norm"].lower() for r in rows for t in toks(r) if t["lang"] == "EN")
        ne = set(t["norm"].lower() for r in rows for t in toks(r) if t["lang"] == "NE")
        frames = [r.get("frame") for r in rows if r.get("frame")]
        refs = [r["normalized"] for r in rows]
        w0, c0 = wer_cer([r["noisy"] for r in rows], refs)
        w1, c1 = wer_cer([dict_norm(r["noisy"]) for r in rows], refs)
        pct = lambda a, b: round(100 * a / max(b, 1), 1)
        report[name] = {
            "sentences": len(rows),
            "words": sum(len(words(r["normalized"])) for r in rows),
            "gold_urdu_unseen_in_train_pct": pct(sum(t["norm"].lower() not in tr_gold for t in ur), len(ur)),
            "gold_urdu_unseen_in_train_and_trainsafe_lexicon_pct": pct(sum(
                t["norm"].lower() not in tr_gold and t["norm"].lower() not in safe_canon for t in ur), len(ur)),
            "noisy_spelling_unseen_in_train_pct": pct(sum(t["noisy"].lower() not in tr_noisy for t in withnoisy),
                                                      len(withnoisy)),
            "tokens_needing_change_pct": pct(len(changed), len(withnoisy)),
            "changed_tokens_with_seen_pair_pct": pct(sum((t["noisy"].lower(), t["norm"].lower()) in tr_pairs
                                                         for t in changed), len(changed)),
            "english_types": len(en), "english_types_unseen_pct": pct(len(en - tr_en), len(en)),
            "name_types": len(ne), "name_types_unseen_pct": pct(len(ne - tr_ne), len(ne)),
            "frames_unseen_pct": pct(sum(f not in tr_frames for f in frames), len(frames)) if frames else None,
            "WER_unchanged": round(w0, 2), "CER_unchanged": round(c0, 2),
            "WER_dictionary": round(w1, 2), "CER_dictionary": round(c1, 2),
        }
    with open(os.path.join(V4, "generalization_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cols = [("sentences", "sentences"), ("words", "words"),
            ("tokens_needing_change_pct", "tokens needing a change %"),
            ("gold_urdu_unseen_in_train_pct", "gold Urdu word unseen in train %"),
            ("gold_urdu_unseen_in_train_and_trainsafe_lexicon_pct", "…and not in train-safe lexicon %"),
            ("noisy_spelling_unseen_in_train_pct", "noisy spelling unseen in train %"),
            ("changed_tokens_with_seen_pair_pct", "corrections whose exact pair is in train %"),
            ("english_types_unseen_pct", "English types unseen %"), ("name_types_unseen_pct", "name types unseen %"),
            ("frames_unseen_pct", "sentence frames unseen %"), ("WER_unchanged", "WER, input left unchanged"),
            ("WER_dictionary", "WER, train dictionary + train-safe lexicon")]
    names = list(report)
    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for key, label in cols:
        print("| %s | %s |" % (label, " | ".join("–" if report[n][key] is None else str(report[n][key])
                                                for n in names)))


if __name__ == "__main__":
    main()
