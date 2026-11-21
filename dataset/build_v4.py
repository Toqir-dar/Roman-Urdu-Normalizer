#!/usr/bin/env python3
"""Build the Roman Urdu normalization dataset v4 from the raw sources (full rebuild).

    python build_v4.py build      # everything -> data/v4/ (+ annotation/social_test_sheet.tsv on first run)
    python build_v4.py review     # type-level review lists for the social normalizer -> annotation/review_*.tsv
    python audit_v4.py            # generalization report for the v4 test sets

Sources
  ../external_raw/dataset.csv             74,860 (noisy, normalized) pairs: a templated part and a real social part
  ../external_raw/Roman Urdu DataSet.csv  UCI Roman Urdu corpus (Sharf & Rahman 2018): real comments + sentiment
  sources/dakshina_*.jsonl                Google Dakshina real sentences (cleaned) with Urdu script
  sources/uci_test_draft.jsonl            186 UCI comments with drafted gold
  sources/lexicon_dakshina.tsv            Dakshina-derived spelling lexicon with Urdu script
  annotation/social_test_*.tsv            the hand-checked social test sample and its gold

Standard library only. Deterministic (SEED).
"""
import argparse
import csv
import difflib
import functools
import hashlib
import html
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import v4_resources as R
import v4_review as V

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "external_raw")
NEW_CSV = os.path.join(RAW, "dataset.csv")
UCI_CSV = os.path.join(RAW, "Roman Urdu DataSet.csv")
SRC = os.path.join(HERE, "sources")
V4 = os.path.join(HERE, "data", "v4")
ANN = os.path.join(HERE, "annotation")
SEED = 2026

TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+(?:[.:,/]\d+)*(?:%|[^\W\d_]+)?|([^\w\s])\1*", re.UNICODE)
WORD_RE = re.compile(r"^[^\W\d_]+(?:'[^\W\d_]+)*$", re.UNICODE)
LAUGH_RE = re.compile(r"^(?:h+a+)+h*$|^(?:h+e+)+h*$|^(?:a*ha){2,}h*$|^l+o+l+$|^(?:x+d+)$")
URDU_PUNCT = {"?": "؟", ",": "،", ".": "۔"}


# ================================================================ helpers
def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def tokenize(text):
    return [m.group(0) for m in TOKEN_RE.finditer(text)]


def is_word(t):
    return bool(WORD_RE.match(t))


def detok(parts):
    s = " ".join(p for p in parts if p)
    s = re.sub(r"\s+([.,!?;:%)\]}؟،۔])", r"\1", s)
    s = re.sub(r"([(\[{#@])\s+", r"\1", s)
    return s


def collapse_punct(t):
    """'...' -> '.', '!!!' -> '!' ; other tokens unchanged."""
    return t[0] if len(t) > 1 and not is_word(t) and len(set(t)) == 1 and not t[0].isdigit() else t


@functools.lru_cache(maxsize=500000)
def sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def squash(w, n=1):
    return re.sub(r"(.)\1{2,}", lambda m: m.group(1) * n, w)


def key_of(text):
    """case / punctuation / spacing insensitive key used for duplicate and leakage checks"""
    return " ".join(re.findall(r"[^\W_]+", text.lower()))


def skeleton(w):
    """first letter + consonants, doubled letters collapsed: behen / bhn -> bhn"""
    sk = w[0] + re.sub(r"[aeiouy]", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", sk)


def stable_hash(s):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


# ================================================================ alignment (noisy tokens <-> gold tokens)
def align(src, tgt):
    """Monotone alignment allowing 1-1, 1-2 (noisy merged), 2-1 (noisy split), 1-0, 0-1.
    Returns a list of (src_tokens, tgt_tokens) groups."""
    n, m = len(src), len(tgt)
    s = [x.lower() for x in src]
    t = [x.lower() for x in tgt]
    if n == m and all(a == b or (is_word(a) == is_word(b) and sim(squash(a, 2), b) >= 0.5) for a, b in zip(s, t)):
        return [([a], [b]) for a, b in zip(src, tgt)]   # fast path: one-to-one
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0

    def sub(a, b):
        if a == b:
            return 0.0
        return 1.0 - sim(squash(a, 2), b)

    band = 4 + abs(n - m)
    for i in range(n + 1):
        lo = max(0, int(i * m / max(n, 1)) - band)
        hi = min(m, int(i * m / max(n, 1)) + band)
        for j in range(lo, hi + 1):
            c = cost[i][j]
            if c == INF:
                continue
            steps = []
            if i < n and j < m:
                steps.append((1, 1, sub(s[i], t[j])))
            if i < n and j + 1 < m:
                steps.append((1, 2, sub(s[i], t[j] + t[j + 1]) + 0.15))
            if i + 1 < n and j < m:
                steps.append((2, 1, sub(s[i] + s[i + 1], t[j]) + 0.15))
            if i < n:
                steps.append((1, 0, 1.0))
            if j < m:
                steps.append((0, 1, 1.0))
            for di, dj, w in steps:
                if c + w < cost[i + di][j + dj]:
                    cost[i + di][j + dj] = c + w
                    back[i + di][j + dj] = (di, dj)
    groups, i, j = [], n, m
    while i or j:
        di, dj = back[i][j]
        groups.append((src[i - di:i], tgt[j - dj:j]))
        i, j = i - di, j - dj
    return groups[::-1]


# ================================================================ lexicon
def load_v3_lexicon():
    lex = {}
    with open(os.path.join(SRC, "lexicon_dakshina.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            c = r["canonical"]
            if c in R.DROP_V3_CANON:
                continue
            c = R.CONVENTION.get(c, c)
            e = lex.setdefault(c, {"urdu": r["urdu"], "type": r["type"], "variants": set()})
            v = r["variant"]
            if v and v != c:
                e["variants"].add(v)
    # v4 additions: homograph readings and canonicals that v3 lacked or had as junk
    extra = {"main": "میں", "nahi": "نہیں", "kiya": "کیا", "kya": "کیا", "ke": "کے", "yeh": "یہ", "woh": "وہ",
             "to": "تو", "hoon": "ہوں", "ammi": "امی", "abbu": "ابو", "haan": "ہاں", "hi": "ہی"}
    for c, u in extra.items():
        lex.setdefault(c, {"urdu": u, "type": "urdu", "variants": set()})
    return lex


def parse_social_map():
    """noisy -> canonical from v4_resources.SOCIAL_MAP_RAW, then the reviewed v4_review.SOCIAL_MAP_2 (wins)"""
    m = {}
    for line in R.SOCIAL_MAP_RAW.strip().splitlines():
        if ":" not in line:
            continue
        canon, vs = line.split(":", 1)
        canon = canon.strip()
        for v in vs.split():
            if "_" not in v and v != canon:
                m[v] = canon
    for item in V.SOCIAL_MAP_2.split():
        v, c = item.split(">")
        m[v] = c.replace("+", " ")
    return m


class Normalizer:
    """Rule + lexicon normalizer used to (re)write gold for real social comments.

    Every decision is labelled (`how`) so that rows can be graded: 'gold' rows contain only tokens whose
    normalization is known (canonical word, curated variant, context rule, English, name, symbol);
    'silver' rows contain at least one word the normalizer does not know (kept as written) or only guessed
    with a fuzzy chat-spelling rule."""

    CTX = set("""main mai mein me may ma mn mei mey men he han ha or the thay bad chor day pay key say both bt gay kia
    kiya kya ta wha yea den lie jan ku ana sara hotai mian per pr ty bar zada kise raah see too mare ata ur""".split())

    def __init__(self, lex, core_freq, names):
        self.lex = lex
        self.english = set(R.ENGLISH_SYN) | set(R.ENGLISH_COMMON) | set(V.ENGLISH_MORE)
        self.names = names | set(V.NAMES_MORE)
        self.curated = parse_social_map()
        self.ok = set(V.SOCIAL_OK.split())
        self.canon = set(lex) | {w for c in self.curated.values() for w in c.split()} | self.ok | set(core_freq)
        # strictly English words (not also common Roman Urdu spellings) for context detection and EN tags
        self.en_only = {w for w in self.english if w not in R.URDU_HOMOGRAPHS and w not in self.curated
                        and w not in self.CTX}
        # fuzzy targets: frequent gold words (templated + Dakshina gold, curated outputs, reviewed social words)
        self.core = Counter(core_freq)
        for c in {w for c in self.curated.values() for w in c.split()} | self.ok:
            self.core[c] += 50
        self.skel = defaultdict(list)
        for c in self.core:
            if c not in self.english:
                self.skel[skeleton(c)].append(c)

    def fuzzy(self, w):
        """Conservative chat-spelling rules -> a frequent gold word, or None."""
        cands = []
        if len(w) >= 3:
            if w.endswith(("ay", "ey")):
                cands.append(w[:-2] + "e")
            elif w.endswith("y") and not w.endswith(("oy", "uy")):
                cands.append(w[:-1] + "e")
            if w.endswith("ain"):
                cands.append(w[:-3] + "ein")
        d = re.sub(r"([^aeiou])\1", r"\1", w)
        if d != w:
            cands.append(d)
        for c in cands:
            if c in self.core and c not in self.en_only:
                return c
        if len(w) >= 2 and not re.search(r"[aeiou]", w[1:]):
            end = {"y": "e", "i": "i", "a": "a", "e": "e"}.get(w[-1])
            opts = [c for c in self.skel.get(skeleton(w), []) if self.core[c] >= 20 and c[0] == w[0]
                    and len(c) <= len(w) + 3 and len(w) >= 0.6 * len(c) and sim(w, c) >= 0.5
                    and (end is None or c.endswith(end))
                    and (end is not None or c[-1] not in "aeiou")]
            if opts:
                best = max(opts, key=lambda c: self.core[c])
                if self.core[best] >= 0.8 * sum(self.core[c] for c in opts):
                    return best
        return None

    # ---------------------------------------------------------------- word lookup (no context)
    def lookup(self, w):
        """-> (norm, how) or None"""
        if w in self.curated:
            return self.curated[w], "curated"
        if w in V.UNCLEAR:
            return None
        if w in R.SPLIT_MAP:
            return R.SPLIT_MAP[w], "split"
        if w in R.KEEP or LAUGH_RE.match(w):
            return w, "keep"
        if w in self.canon:
            return w, "canon"
        if w in self.names:
            return w, "name"
        if w in self.english:
            return w, "english"
        f = self.fuzzy(w)
        if f:
            return f, "fuzzy"
        return None

    def lang_of(self, w, how):
        if how == "english" or w in self.en_only or (w in self.english and w in R.LOANS):
            return "EN"
        if how == "name":
            return "NE"
        return "UR"

    # ---------------------------------------------------------------- context rules
    def _ctx(self, w, prev, nxt, en_ctx, clause_start, prev_tok=None, ne_before=False):
        """Context-dependent words. Returns (norm, ambiguous_flag)."""
        if w in ("main", "mai", "mein", "me", "may", "ma", "mn", "mei", "mey", "men"):
            if w == "may" and nxt in ("allah", "god"):
                return "may", True                        # may allah bless ...
            if w in ("me", "may", "men") and (self._sent_en or prev in R.EN_BEFORE_ME):
                return w, True
            if w == "may" and prev_tok is not None and prev_tok[:1].isdigit():
                return "may", False                       # 9 may (the month)
            if w == "ma" and nxt in ("baap", "ki", "ka", "ke", "ko", "behen", "behin", "bhen", "bhai",
                                     "beti", "bete", "beta", "bache", "bachay", "bacche", "jaan"):
                return "maa", True
            if clause_start or prev in R.CONJ_START:
                return "main", True
            if nxt in ("ne", "nay", "ny", "khud", "hoon", "hun", "hu"):
                return "main", True
            if nxt is not None and (nxt.endswith("unga") or nxt.endswith("ungi") or nxt in ("hoon", "hun", "hu")):
                return "main", True
            return "mein", True
        if w == "he":
            if en_ctx or nxt in ("is", "was", "has", "will", "can", "does", "did", "should", "would", "said", "says")                     or prev in ("is", "was", "did", "does", "can", "will", "would", "should", "has", "have", "and"):
                return "he", True
            if nxt in R.AUX_AFTER_HI or nxt in ("nahi", "nhi", "nh", "ni", "na", "nai"):
                return "hi", True                         # daku he hongy -> daku hi honge
            if nxt is None or nxt in R.NEW_CLAUSE or prev in R.VERB_AUX:
                return "hai", True
            if prev in R.HI_PREV:
                return "hi", True
            if nxt in R.POSSESSIVE:
                return "hai", True                        # yeh he mera ghar -> yeh hai mera ghar
            return "hi", True                             # apas me he ladte -> apas mein hi ladte
        if w in ("han", "ha"):
            if w == "ha" and en_ctx:
                return "ha", True
            if clause_start and w == "han" or (clause_start and nxt in ("ji", "jee", "g", "bhai", "yaar", "yar")):
                return "haan", True
            return ("hain", True) if w == "han" else ("hai", True)
        if w == "or":
            return ("or", True) if en_ctx else ("aur", True)
        if w in ("the", "thay"):
            return "the", en_ctx is False
        if w == "bad":
            if self._sent_en or prev in ("very", "so", "too", "not", "is", "was", "a", "its", "it's", "how") or \
                    nxt in ("luck", "news", "boy", "girl", "day", "time", "behaviour", "habit"):
                return "bad", True
            return "baad", True
        if w == "chor":
            return ("chhor", True) if nxt in R.CHHOR_NEXT else ("chor", True)
        if w in ("day", "pay", "key", "say", "per", "pr", "bar"):
            if en_ctx:
                return w, True
            return {"day": "de", "pay": "pe", "key": "ke", "say": "se", "per": "par", "pr": "par",
                    "bar": "baar"}[w], True
        if w == "both":
            return ("both", True) if en_ctx else ("bohat", True)
        if w == "bt":
            return ("but", True) if en_ctx else ("baat", True)
        if w == "gay":
            return ("gaye", True) if prev in R.GAYE_PREV else ("ge", True)
        if w == "kya":
            if ne_before and not clause_start and (nxt is None or nxt in R.AFTER_KIYA):
                return "kiya", True                       # main ne graduation kya hai -> kiya
            return "kya", False
        if w == "zada":
            return ("zada", True) if prev in ("haram", "haraam", "shah", "nawab", "sahib", "sahab", "peer")                 else ("zyada", True)
        if w == "kise":
            if nxt is None or nxt in R.KAISE_NEXT:
                return "kaise", True                      # kise ho / kise pata -> kaise
            return "kisi", True                           # kise din / kise ne / kise insaan -> kisi
        if w == "raah":
            return ("raah", True) if nxt in ("e", "mein", "main", "par", "pe", "se", "dikhaye", "dikha") else ("raha", True)
        if w == "ur":
            return ("your", True) if en_ctx else ("aur", True)
        if w in ("see", "too"):
            return (w, True) if en_ctx else ({"see": "se", "too": "to"}[w], True)
        if w == "mare":
            if nxt is None or nxt in R.MARE_NEXT or prev in ("mar", "maar", "ko", "goli", "joote", "jootay", "thappar"):
                return "mare", True                       # headshot mare / mare gaye
            return "mere", True                           # mare bhai / mare pas -> mere
        if w == "ata":
            return ("ata", True) if nxt in ("karein", "karen", "kare", "kre", "kren", "kry", "farmaye", "farmay", "farma",
                                            "kiya", "kia", "ki", "kar", "karta", "karte") else ("aata", True)
        if w in ("kia", "kiya"):
            if ne_before and not clause_start and (nxt is None or nxt in R.AFTER_KIYA):
                return "kiya", True
            if clause_start or nxt in ("?", None) or nxt in ("baat", "aap", "ap", "tum", "hum", "yeh", "ye",
                                                            "woh", "wo", "kabhi", "sach"):
                return "kya", True
            return ("kiya", True) if w == "kiya" else ("kya", True)
        if w == "ta":
            return ("ta", True) if en_ctx else ("tha", True)
        if w == "wha":
            if clause_start or prev in ("are", "arey", "aray", "wah", "wha", "waah") or nxt in ("kya", "wha", "wah", "bhai",
                                                                                     "ji", "yaar", "yar"):
                return "wah", True
            return "wahan", True
        if w == "yea":
            return ("yeah", True) if en_ctx else ("yeh", True)
        if w == "den":
            return ("den", True) if en_ctx else ("dein", True)
        if w == "lie":
            return ("lie", True) if en_ctx else ("liye", True)
        if w == "jan":
            return ("jan", True) if en_ctx else ("jaan", True)
        if w == "ku":
            return ("ko", True) if prev not in (None,) and nxt in ("bhi", "pata", "kya", "kuch") else ("kyun", True)
        if w == "ana":
            return "aana", True
        if w == "sara":
            return ("sara", True) if clause_start else ("saara", True)
        if w == "hotai":
            return "hoti", True
        if w == "mian":
            return "mian", True
        if w == "ty":
            return "ty", True
        return w, True

    # ---------------------------------------------------------------- sentence
    def normalize(self, text):
        """-> list of token dicts {noisy, norm, lang, how, ambiguous}"""
        spans = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
        raw = [text[a:b] for a, b in spans]
        low = [t.lower() for t in raw]
        words = [t for t in low if is_word(t)]
        en_n = sum(w in self.en_only for w in words)
        ur_n = sum((w in self.canon or w in self.curated) and w not in self.en_only for w in words)
        sent_en = en_n >= 3 and en_n > 1.5 * ur_n
        self._sent_en = sent_en
        out = []
        for i, (t, w) in enumerate(zip(raw, low)):
            if not is_word(t):
                out.append({"noisy": t, "norm": collapse_punct(t), "lang": "OTH", "how": "symbol",
                            "ambiguous": False})
                continue
            prev = low[i - 1] if i > 0 and is_word(low[i - 1]) else None
            nxt = low[i + 1] if i + 1 < len(low) and is_word(low[i + 1]) else None
            clause_start = i == 0 or bool(re.match(r"^[.,!?;:()\"\-]+$", low[i - 1]))
            en_ctx = sent_en or ((prev in self.en_only) and (nxt in self.en_only or nxt is None)) or \
                (prev is None and nxt in self.en_only and (i + 2 >= len(low) or low[i + 2] in self.en_only))
            if w == "e" and prev is not None and nxt is not None and not en_ctx:
                out.append({"noisy": t, "norm": "e", "lang": "UR", "how": "izafat", "ambiguous": False})
                continue
            if w in self.CTX and w not in self.curated:
                ne_before = any(x in R.NE_WORDS for x in low[max(0, i - 15):i])
                norm, amb = self._ctx(w, prev, nxt, en_ctx, clause_start,
                                      prev_tok=low[i - 1] if i > 0 else None, ne_before=ne_before)
                lang = "EN" if (norm == w and w in self.english and en_ctx) else "UR"
                out.append({"noisy": t, "norm": norm, "lang": lang, "how": "context", "ambiguous": amb})
                continue
            hit = self.lookup(w)
            if hit is None and re.search(r"(.)\1{2,}", w):
                for cand in (squash(w, 1), squash(w, 2), re.sub(r"(.)\1+", r"\1", w)):
                    h = self.lookup(cand)
                    if h:
                        hit = (h[0], "elongation")
                        break
                if hit is None:
                    hit = (squash(w, 1), "elongation-unknown")
            if hit is None:
                out.append({"noisy": t, "norm": w, "lang": "UR", "how": "unknown", "ambiguous": False})
                continue
            norm, how = hit
            out.append({"noisy": t, "norm": norm, "lang": self.lang_of(norm, how), "how": how, "ambiguous": False})
        for o, sp in zip(out, spans):
            o["span"] = sp
        return out


def render(text, toks):
    """Gold text = the input with each token replaced in place (keeps the writer's spacing around symbols).
    Repeated punctuation is collapsed ('...' -> '.') and followed by a space before a word; spaces before
    , . ! ? are removed; deleted tokens (izafat) take one neighbouring space with them."""
    out, pos = [], 0
    for t in toks:
        a, b = t["span"]
        gap = text[pos:a]
        if gap:
            out.append(" " if gap.strip() == "" else gap)
        norm = t["norm"]
        if norm == "" and out and out[-1] == " ":
            out.pop()
        out.append(norm)
        if t["lang"] == "OTH" and len(t["noisy"]) > 1 and len(norm) == 1 and b < len(text) and                 re.match(r"[^\W\d_]", text[b]):
            out.append(" ")
        pos = b
    s = "".join(out)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


# ================================================================ source 1: templated part of dataset.csv
def read_new_csv(log):
    with open(NEW_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    hdr, rows = rows[0], rows[1:]
    assert hdr == ["noisy_text", "normalized_text"], hdr
    rows = [(html.unescape(a).strip(), html.unescape(b).strip()) for a, b in rows]
    # The file is two blocks: templated rows (gold uses a small vocabulary, names / apps capitalised), then real
    # comments. The boundary is the split point that best separates rows whose gold words all come from the
    # templated vocabulary (learned from the first half of the file) from rows that do not.
    half = len(rows) // 2
    vocab = {w for a, b in rows[:half] for w in re.findall(r"[^\W\d_]+", b.lower())}
    templ_like = [all(w in vocab for w in re.findall(r"[^\W\d_]+", b.lower())) and a != b for a, b in rows]
    best, boundary, score = -1, half, sum(not x for x in templ_like[half:])
    for i in range(half, len(rows)):
        if score > best:
            best, boundary = score, i
        score += 1 if templ_like[i] else -1
    log["dataset_csv"] = {"rows": len(rows), "templated_block": boundary, "social_block": len(rows) - boundary}
    return rows[:boundary], rows[boundary:]


def entity_class(w):
    if w in R.PEOPLE:
        return "P"
    if w in R.CITIES:
        return "C"
    if w in R.APPS:
        return "A"
    if w in R.SUBJECTS:
        return "S"
    if w in R.NUMBERS:
        return "N"
    if w in R.TIME_WORDS:
        return "T"
    if w in R.KIN:
        return "K"
    if w in R.FOODS:
        return "F"
    if w in R.ENGLISH_SYN:
        return "E"
    return None


CURATED = None


def plausible(noisy, gold):
    """is `noisy` a believable spelling of `gold`? (catches generator misalignments, not real variants)"""
    global CURATED
    if CURATED is None:
        CURATED = parse_social_map()
    n = squash(noisy.replace(" ", "").lower(), 2)
    if n == gold or CURATED.get(n) == gold or len(gold) < 4:
        return True
    return sim(n, gold) >= 0.34 or sim("".join(sorted(n)), "".join(sorted(gold))) >= 0.7


def frame_of(gold_words):
    out, i = [], 0
    while i < len(gold_words):
        w = gold_words[i]
        if w in R.TIME_HEADS and i + 1 < len(gold_words) and gold_words[i + 1] in R.TIME_TAILS + R.TIME_WORDS:
            out.append("<T>")
            i += 2
            continue
        if w == "thodi" and gold_words[i + 1:i + 3] == ["der", "baad"]:
            out.append("<T>")
            i += 3
            continue
        c = entity_class(w)
        out.append("<%s>" % c if c else w)
        i += 1
    dedup = [x for k, x in enumerate(out) if k == 0 or x != out[k - 1] or not x.startswith("<")]
    return " ".join(dedup)


def process_templated(pairs, lex, log):
    seen, recs, dropped = set(), [], []
    pair_votes = defaultdict(Counter)
    for k, (noisy, gold) in enumerate(pairs):
        if (noisy, gold) in seen:
            continue
        seen.add((noisy, gold))
        gtok = [collapse_punct(t) for t in tokenize(gold.lower())]
        gtok = [R.CONVENTION.get(t, t) if t in ("ami", "abu") else t for t in gtok]
        ntok = tokenize(noisy)
        groups = align(ntok, gtok)
        toks, bad = [], None
        for src, tgt in groups:
            if not tgt:   # noisy token with no gold counterpart (stray punctuation / duplicated word)
                if any(is_word(x) for x in src):
                    bad = bad or "extra_noisy_word:%s" % " ".join(src)
                continue
            if not src:
                if any(is_word(x) for x in tgt):
                    bad = bad or "missing_noisy_word:%s" % " ".join(tgt)
                for g in tgt:
                    toks.append({"noisy": "", "norm": g})
                continue
            s = " ".join(src)
            if len(tgt) == 1 and is_word(tgt[0]) and not plausible(s, tgt[0]):
                bad = bad or "misaligned:%s->%s" % (s, tgt[0])
            if len(tgt) == 2:   # merged in noisy text: first token carries the noisy form
                toks.append({"noisy": s, "norm": tgt[0], "merged": True})
                toks.append({"noisy": "", "norm": tgt[1]})
            else:
                toks.append({"noisy": s, "norm": tgt[0], "split": len(src) > 1})
        if bad:
            dropped.append({"source": "templated", "row": k, "reason": bad.split(":")[0], "detail": bad,
                            "noisy": noisy, "normalized": gold})
            continue
        for t in toks:
            w = t["norm"]
            c = entity_class(w)
            if c in ("P", "C", "A"):
                t["lang"] = "NE"
            elif c in ("S", "E") and w != "urdu":
                t["lang"] = "EN"
            elif is_word(w):
                t["lang"] = "UR"
            else:
                t["lang"] = "OTH"
            t["urdu"] = lex[w]["urdu"] if t["lang"] == "UR" and w in lex else (URDU_PUNCT.get(w, w)
                                                                                if t["lang"] == "OTH" else "")
            t["changed"] = t["noisy"] != t["norm"]
            if t["noisy"] and " " not in t["noisy"] and not t.get("merged"):
                pair_votes[t["noisy"].lower()][w] += 1
        gw = [t["norm"] for t in toks if is_word(t["norm"])]
        ents = sorted({"%s:%s" % (entity_class(w), w) for w in gw if entity_class(w) in ("P", "C", "A", "S", "F", "E")})
        recs.append({"source": "templated", "row": k, "noise_level": "synthetic", "label": "",
                     "noisy": noisy, "normalized": detok([t["norm"] for t in toks]), "tokens": toks,
                     "frame": frame_of(gw), "entities": ents})
    learned = {}
    for v, c in pair_votes.items():
        tot = sum(c.values())
        w, n = c.most_common(1)[0]
        if tot >= 3 and n / tot >= 0.95 and v != w:
            learned[v] = w
    log["templated"] = {"unique_pairs": len(seen), "kept": len(recs), "dropped": dict(Counter(d["reason"] for d in dropped)),
                        "frames": len({r["frame"] for r in recs}), "learned_variant_pairs": len(learned)}
    return recs, dropped, learned


def split_templated(recs, rng, log):
    """Hold out whole entities, whole sentence frames and a share of the misspellings themselves, so
    test_synth_ood is out of distribution (the templated noise reuses a small pool of misspellings)."""
    pools = defaultdict(set)
    for r in recs:
        for e in r["entities"]:
            pools[e.split(":")[0]].add(e)
    n_hold = {"P": 2, "C": 2, "A": 1, "S": 1, "F": 3, "E": 10}
    held = set()
    for c, k in n_hold.items():
        items = sorted(pools[c])
        rng.shuffle(items)
        held |= set(items[:k])
    frames = sorted({r["frame"] for r in recs})
    rng.shuffle(frames)
    fsplit = {}
    for i, f in enumerate(frames):
        fsplit[f] = "test" if i % 10 == 0 else ("dev" if i % 10 == 1 else "train")
    def misspellings(r):
        return {t["noisy"].lower() for t in r["tokens"] if t["noisy"] and " " not in t["noisy"]
                and not t.get("merged") and t["noisy"].lower() != t["norm"] and is_word(t["norm"])}
    mcount = Counter(m for r in recs for m in misspellings(r))
    cands = sorted(m for m, n in mcount.items() if 2 <= n <= 30)
    rng.shuffle(cands)
    held_sp = set(cands[: int(0.25 * len(cands))])
    out = defaultdict(list)
    for r in recs:
        ent_ood = any(e in held for e in r["entities"])
        sp_ood = bool(misspellings(r) & held_sp)
        fs = fsplit[r["frame"]]
        if fs == "test" or ent_ood or sp_ood:
            r["ood_reason"] = "+".join(x for x, y in (("frame", fs == "test"), ("entity", ent_ood),
                                                      ("spelling", sp_ood)) if y)
            out["test_synth_ood"].append(r)
        elif fs == "dev":
            out["dev"].append(r)
        else:
            out["train"].append(r)
    # in-distribution test: random train-frame rows whose gold sentence is not in train
    tr = out["train"]
    rng.shuffle(tr)
    gold_count = Counter(key_of(r["normalized"]) for r in tr)
    iid, rest = [], []
    for r in tr:
        k = key_of(r["normalized"])
        if len(iid) < 2000 and gold_count[k] == 1:
            iid.append(r)
        else:
            rest.append(r)
    out["train"], out["test_synth_iid"] = rest, iid
    log["templated_split"] = {"held_out_entities": sorted(held), "held_out_misspellings": len(held_sp),
                              "frames": dict(Counter(fsplit.values())),
                              "rows": {k: len(v) for k, v in out.items()},
                              "ood_reason": dict(Counter(r["ood_reason"] for r in out["test_synth_ood"]))}
    return out


# ================================================================ source 2: Dakshina (via v3)
def process_dakshina(log):
    out = defaultdict(list)
    n_conv = Counter()
    for split, name in (("train", "train"), ("dev", "dev"), ("test_real", "test_real")):
        for r in read_jsonl(os.path.join(SRC, "dakshina_" + name + ".jsonl")):
            if r["source"] != "pivot":
                continue
            toks = []
            words = [t for t in r["tokens"] if t["lang"] != "OTH"]
            question = r["normalized"].rstrip().endswith("?") or r["urdu"].rstrip().endswith("؟")
            for i, t in enumerate(r["tokens"]):
                t = {k: t[k] for k in ("noisy", "norm", "urdu", "lang", "ambiguous") if k in t}
                old = t["norm"]
                if t["lang"] == "UR":
                    t["norm"] = R.CONVENTION.get(old, old)
                    if t.get("urdu") == "کیا" and old == "kya":
                        t["norm"] = "kya" if (words and t is words[0]) or i == 0 or question else "kiya"
                        t["ambiguous"] = True
                if t["norm"] != old:
                    n_conv["%s->%s" % (old, t["norm"])] += 1
                t["changed"] = t["noisy"] != t["norm"]
                toks.append(t)
            out[split].append({"id": r["id"], "source": "dakshina", "noise_level": "natural", "label": "",
                               "noisy": r["noisy"], "normalized": detok([t["norm"] for t in toks]),
                               "urdu": r["urdu"], "tokens": toks})
    log["dakshina"] = {"rows": {k: len(v) for k, v in out.items()}, "convention_changes": dict(n_conv.most_common())}
    return out


# ================================================================ source 3: real social part of dataset.csv
def name_list():
    names = set(R.PEOPLE) | set(R.CITIES) | set(R.APPS)
    names |= set("""khan imran nawaz shareef sharif shahbaz shehbaz maryam zardari bilawal bhutto niazi altaf hussain
    muhammad mohammad ahmed ahmad iqbal azam tareen rana malik abdul aziz salman usman ayeza iqra feroze feroz kinza
    sahir murad arsal meerab meerub qudsia qudsiya feriha fariha anaya murtasim ibrahim afia hashim farhad haya waheed
    edhi modi jahangir anwar punjab sindh kashmir london america hindustan bharat dubai saudi iran turkey china
    afghanistan bangladesh pakistan india karachi lahore peshawar quetta multan islamabad rawalpindi fazal rehman
    raheel bajwa musharraf qadri hafeez babar shoaib afridi akmal amir wasim akram sarfaraz rizwan hamza bilal
    ali umar hasan hassan husain fatima ayesha zainab khadija hira sajal mahira humayun saeed fawad danish""".split())
    return names


def process_social_new(pairs, norm, uci_keys, log):
    recs, dropped, seen = [], [], set()
    for k, (noisy, src_gold) in enumerate(pairs):
        if (noisy, src_gold) in seen:
            continue
        seen.add((noisy, src_gold))
        reason = None
        n_qatil = len(re.findall(r"\bqatil\b", noisy.lower()))
        words = [t for t in tokenize(noisy) if is_word(t)]
        if n_qatil:
            reason = "qatil_artifact"
        elif len(words) < 2:
            reason = "too_short"
        elif len(words) > 60:
            reason = "too_long"
        elif re.search(r"https?://|www\.", noisy, re.I):
            reason = "url"
        if reason:
            dropped.append({"source": "social_new", "row": k, "reason": reason, "noisy": noisy,
                            "normalized": src_gold})
            continue
        toks = norm.normalize(noisy)
        unknown = [t["noisy"] for t in toks if t["how"] in ("unknown", "elongation-unknown", "fuzzy")]
        en = sum(t["lang"] == "EN" for t in toks)
        ur = sum(t["lang"] == "UR" for t in toks)
        lang_hint = "english" if en > 2 * ur else ("mixed" if en >= 2 else "urdu")
        flat = []
        for t in toks:
            parts = t["norm"].split()
            if len(parts) > 1:   # merged word split on Urdu boundaries
                flat.append(dict(t, norm=parts[0], merged=True))
                for p in parts[1:]:
                    flat.append({"noisy": "", "norm": p, "lang": "UR", "how": t["how"], "ambiguous": False})
            else:
                flat.append(t)
        for t in flat:
            t["changed"] = t["noisy"] != t["norm"]
        recs.append({"source": "social_new", "row": k, "noise_level": "natural", "label": "",
                     "lang_hint": lang_hint, "noisy": noisy, "normalized": render(noisy, toks),
                     "source_normalized": src_gold, "tokens": flat,
                     "tier": "gold" if not unknown else "silver", "unknown_words": unknown,
                     "n_ambiguous": sum(t["ambiguous"] for t in flat),
                     "in_uci": key_of(noisy) in uci_keys})
    log["social_new"] = {"unique_pairs": len(seen), "kept": len(recs),
                         "dropped": dict(Counter(d["reason"] for d in dropped)),
                         "tier": dict(Counter(r["tier"] for r in recs)),
                         "lang_hint": dict(Counter(r["lang_hint"] for r in recs)),
                         "agrees_with_source_normalized": sum(key_of(r["normalized"]) == key_of(r["source_normalized"])
                                                              for r in recs)}
    return recs, dropped


# ================================================================ source 4: UCI corpus (sentiment) + v3 drafted gold
PUNJABI = set("""nu kolo tusi tussi sanu sannu tenu tainu mainu menu vich kithe kiddan tuhada tuhanu sada sadda
assi ainu onu ohnu ohna jinu kenda kendi kehnda karda kardi gal gall""".split())


def load_uci(norm):
    rows = []
    with open(UCI_CSV, encoding="utf-8", errors="replace") as f:
        for i, rec in enumerate(csv.reader(f)):
            if len(rec) < 2 or not rec[0].strip():
                continue
            text = re.sub(r"\s+", " ", rec[0]).strip()
            label = {"Positive": "pos", "Negative": "neg", "Neutral": "neu", "Neative": "neg"}.get(rec[1].strip(), "")
            if not label:
                continue
            words = [w.lower() for w in tokenize(text) if is_word(w)]
            n = max(1, len(words))
            ur = sum(norm.lookup(w) is not None and w not in norm.en_only for w in words) / n
            en = sum(w in norm.en_only for w in words) / n
            pa = sum(w in PUNJABI for w in words)
            if pa >= 2 or (pa == 1 and n <= 6):
                hint = "punjabi?"
            elif en >= 0.6 and ur < 0.4:
                hint = "english"
            elif ur >= 0.5:
                hint = "urdu" if en < 0.15 else "mixed"
            else:
                hint = "unknown"
            rows.append({"uci_row": i, "text": text, "label": label, "n_words": len(words), "lang_hint": hint,
                         "has_url": bool(re.search(r"https?://|www\.", text, re.I))})
    seen, out = set(), []
    for r in rows:
        k = key_of(r["text"])
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out, len(rows)


def convert_uci_draft(lex):
    out = []
    for r in read_jsonl(os.path.join(SRC, "uci_test_draft.jsonl")):
        toks = []
        for t in r["tokens"]:
            t = dict(t)
            old = t["norm"]
            if t["lang"] == "UR":
                t["norm"] = R.CONVENTION.get(old, old)
                t["urdu"] = lex[t["norm"]]["urdu"] if t["norm"] in lex else ""
            if "noisy" in t:
                t["changed"] = t["noisy"] != t["norm"]
            t.pop("in_lexicon", None)
            toks.append(t)
        out.append({"id": r["id"], "source": "uci_social", "noise_level": "natural", "label": r["label"],
                    "lang_hint": r["lang_hint"], "noisy": r["noisy"],
                    "normalized": detok([t["norm"] for t in toks]), "tokens": toks, "aligned": r["aligned"],
                    "status": r["status"], "annotator": r["annotator"]})
    return out


# ================================================================ social test annotation sheet
SHEET = os.path.join(ANN, "social_test_sheet.tsv")
GOLD = os.path.join(ANN, "social_test_gold.tsv")


def tagged(tokens):
    """tokens -> 'word word/E name/N' (untagged = Urdu, symbols untagged)"""
    out = []
    for t in tokens:
        tag = {"EN": "/E", "NE": "/N"}.get(t["lang"], "")
        out.append(t["norm"] + tag)
    return " ".join(out)


def parse_tagged(text):
    toks = []
    for item in text.split():
        word, tag = item, None
        if "/" in item and item.rsplit("/", 1)[1] in ("E", "N", "O", "U"):
            word, tag = item.rsplit("/", 1)
        lang = {"E": "EN", "N": "NE", "O": "OTH", "U": "UR"}.get(tag) or ("UR" if is_word(word) else "OTH")
        toks.append({"norm": word, "lang": lang})
    return toks


def choose_test_sample(recs, rng, n=300):
    """Random sample (not only easy 'gold' rows) of real comments for the hand-checked test set."""
    pool = [r for r in recs if not r["in_uci"] and 4 <= sum(is_word(t["noisy"]) for t in r["tokens"]) <= 35
            and r["lang_hint"] != "english"]
    pool.sort(key=lambda r: r["row"])
    rng.shuffle(pool)
    seen, sample = set(), []
    for r in pool:
        k = key_of(r["noisy"])
        if k in seen:
            continue
        seen.add(k)
        sample.append(r)
        if len(sample) == n:
            break
    return sample


def write_sheet(sample):
    os.makedirs(ANN, exist_ok=True)
    with open(SHEET, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "status", "noisy", "auto_gold", "source_normalized", "note"])
        for r in sample:
            w.writerow([r["id"], "auto", r["noisy"], tagged(r["tokens"]), r["source_normalized"], ""])


def read_gold_sheet():
    if not os.path.exists(GOLD):
        return {}
    out = {}
    with open(GOLD, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[r["id"]] = r
    return out


# ================================================================ lexicon output
def build_lexicon(lex, splits, norm):
    """canonical -> variants with where they were seen; train-safe version drops eval-only material."""
    seen = defaultdict(lambda: defaultdict(Counter))   # canon -> variant -> split -> count
    lang = {}
    for split, rows in splits.items():
        for r in rows:
            for t in r["tokens"]:
                if t["lang"] == "OTH" or not t.get("noisy") or " " in t["noisy"] or t.get("merged") or                         t.get("how") in ("unknown", "elongation-unknown", "fuzzy"):
                    continue
                c = t["norm"].lower()
                seen[c][t["noisy"].lower()][split] += 1
                lang.setdefault(c, t["lang"])
    rows = []
    for c in sorted(set(seen) | set(lex)):
        e = lex.get(c, {"urdu": "", "variants": set()})
        variants = set(e["variants"]) | set(seen.get(c, {}))
        variants.discard(c)
        ty = {"EN": "english", "NE": "name"}.get(lang.get(c, "UR"), "urdu")
        c_train = sum(seen[c][v].get("train", 0) for v in seen.get(c, {}))
        c_eval = sum(n for v in seen.get(c, {}) for s, n in seen[c][v].items() if s != "train")
        canon_eval_only = c_eval > 0 and c_train == 0
        for v in sorted(variants) or [""]:
            st = seen.get(c, {}).get(v, {})
            src = "lexicon" if v in e["variants"] else ""
            tr, ev = st.get("train", 0), sum(n for s, n in st.items() if s != "train")
            rows.append({"canonical": c, "urdu": e["urdu"], "type": ty, "variant": v,
                         "in_v3_lexicon": int(v in e["variants"]), "train_count": tr, "eval_count": ev,
                         "eval_only": int(ev > 0 and tr == 0),
                         "canonical_only_in_eval": int(canon_eval_only)})
    return rows


# ================================================================ commands
def gather(log, rng):
    lex = load_v3_lexicon()
    templ_pairs, social_pairs = read_new_csv(log)
    templ, templ_dropped, learned = process_templated(templ_pairs, lex, log)
    tsplit = split_templated(templ, rng, log)
    dk = process_dakshina(log)
    core_freq = Counter(t["norm"] for rows in [templ] + list(dk.values()) for r in rows for t in r["tokens"]
                        if t["lang"] == "UR" and is_word(t["norm"]))
    norm = Normalizer(lex, core_freq, name_list())
    uci_rows, n_uci_raw = load_uci(norm)
    uci_draft = convert_uci_draft(lex)
    uci_test_keys = {key_of(r["noisy"]) for r in uci_draft}
    social, social_dropped = process_social_new(social_pairs, norm, {key_of(r["text"]) for r in uci_rows}, log)
    return dict(lex=lex, templ=templ, tsplit=tsplit, templ_dropped=templ_dropped, norm=norm, uci_rows=uci_rows,
                n_uci_raw=n_uci_raw, uci_draft=uci_draft, uci_test_keys=uci_test_keys, social=social,
                social_dropped=social_dropped, dk=dk)


def cmd_review(args):
    """Type-level review lists of what the social normalizer does (for a person to check)."""
    log = {}
    g = gather(log, random.Random(SEED))
    by_type = defaultdict(Counter)
    src_map = defaultdict(Counter)
    for r in g["social"]:
        st = r["source_normalized"].split()
        for t in r["tokens"]:
            if t["noisy"]:
                by_type[t["noisy"].lower()][(t["norm"], t["how"])] += 1
    os.makedirs(ANN, exist_ok=True)
    freq = Counter({w: sum(c.values()) for w, c in by_type.items()})
    with open(os.path.join(ANN, "review_types.tsv"), "w", encoding="utf-8") as f:
        f.write("noisy\tcount\tdecisions\n")
        for w, n in freq.most_common(args.top):
            dec = " | ".join("%s(%s)x%d" % (nm, how, c) for (nm, how), c in by_type[w].most_common())
            f.write("%s\t%d\t%s\n" % (w, n, dec))
    unk = Counter()
    for r in g["social"]:
        for w in r["unknown_words"]:
            unk[w.lower()] += 1
    with open(os.path.join(ANN, "review_unknown.tsv"), "w", encoding="utf-8") as f:
        f.write("word\tcount\n")
        for w, n in unk.most_common():
            f.write("%s\t%d\n" % (w, n))
    print(json.dumps(log, indent=1, ensure_ascii=False)[:3000])
    print("wrote review_types.tsv (top %d) and review_unknown.tsv (%d types)" % (args.top, len(unk)))


def cmd_build(args):
    rng = random.Random(SEED)
    log = {}
    g = gather(log, rng)
    os.makedirs(V4, exist_ok=True)

    # ---- social (new): leakage-safe grouping, hand-checked test sample, dev, train
    social = g["social"]
    for i, r in enumerate(sorted(social, key=lambda r: r["row"])):
        r["id"] = "sn-%05d" % r["row"]
    sample = choose_test_sample(social, random.Random(SEED + 1), n=args.test_size)
    if not os.path.exists(SHEET) or args.force_sheet:
        write_sheet(sample)
    elif args.refresh_sheet:
        with open(SHEET, encoding="utf-8") as f:
            ids = [r["id"] for r in csv.DictReader(f, delimiter="	")]
        by = {r["id"]: r for r in social}
        write_sheet([by[i] for i in ids])
    sheet_ids = []
    with open(SHEET, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sheet_ids.append(r["id"])
    by_id = {r["id"]: r for r in social}
    gold_sheet = read_gold_sheet()
    test_social = []
    for sid in sheet_ids:
        r = by_id[sid]
        a = gold_sheet.get(sid)
        if a and a["status"] == "skip":
            continue
        if a:
            toks = parse_tagged(a["gold"])
            noisy_words = tokenize(r["noisy"])
            groups = align(noisy_words, [t["norm"] for t in toks])
            ntoks, k = [], 0
            for src, tgt in groups:
                for j, g_ in enumerate(tgt):
                    t = toks[k]
                    k += 1
                    t["noisy"] = " ".join(src) if j == 0 else ""
                    if len(tgt) > 1 and j == 0:
                        t["merged"] = True
                    ntoks.append(t)
            for t in ntoks:
                t["urdu"] = g["lex"][t["norm"]]["urdu"] if t["lang"] == "UR" and t["norm"] in g["lex"] else ""
                t["changed"] = t["noisy"] != t["norm"]
            r = dict(r, tokens=ntoks, normalized=detok([t["norm"] for t in ntoks]), status=a["status"],
                     annotator=a.get("annotator", ""), note=a.get("note", ""))
        else:
            r = dict(r, status="auto", annotator="rules")
        test_social.append(r)
    test_keys = set()
    for r in test_social + g["uci_draft"]:
        test_keys |= {key_of(r["noisy"]), key_of(r["normalized"])}
    rest = [r for r in social if r["id"] not in set(sheet_ids)]
    leak = [r for r in rest if key_of(r["noisy"]) in test_keys or key_of(r["normalized"]) in test_keys]
    rest = [r for r in rest if not (key_of(r["noisy"]) in test_keys or key_of(r["normalized"]) in test_keys)]
    # group duplicates (same noisy key or same gold key) so a group never straddles train/dev
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in rest:
        a, b = "n:" + key_of(r["noisy"]), "g:" + key_of(r["normalized"])
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for r in rest:
        groups[find("n:" + key_of(r["noisy"]))].append(r)
    gkeys = sorted(groups, key=stable_hash)
    social_dev, social_train, n_dev = [], [], 0
    for gk in gkeys:
        rows = groups[gk]
        if n_dev < args.dev_size and all(r["tier"] == "gold" for r in rows):
            social_dev += rows
            n_dev += len(rows)
        else:
            social_train += rows
    log["social_new_split"] = {"test_social": len(test_social), "dev": len(social_dev), "train": len(social_train),
                               "train_tier": dict(Counter(r["tier"] for r in social_train)),
                               "removed_test_overlap": len(leak),
                               "test_status": dict(Counter(r["status"] for r in test_social))}

    # ---- UCI sentiment corpus: rows equal to any normalization test row are put in its test split
    uci = g["uci_rows"]
    uci_test_ids = {r["id"] for r in g["uci_draft"]}
    ordered = sorted(uci, key=lambda r: r["uci_row"])
    r2 = random.Random(SEED + 2)
    r2.shuffle(ordered)
    n = len(ordered)
    for i, r in enumerate(ordered):
        r["id"] = "uci-%05d" % r["uci_row"]
        r["source"] = "uci_social"
        r["split"] = "test" if i < int(0.1 * n) else ("dev" if i < int(0.2 * n) else "train")
        if r["id"] in uci_test_ids or key_of(r["text"]) in test_keys:
            r["split"] = "test"
            r["in_normalization_test"] = True
    sentiment = sorted(ordered, key=lambda r: r["uci_row"])
    log["uci"] = {"raw_rows": g["n_uci_raw"], "after_dedup": len(uci),
                  "split": dict(Counter(r["split"] for r in sentiment)),
                  "lang_hint": dict(Counter(r["lang_hint"] for r in sentiment))}

    # ---- assemble splits
    ts, dk = g["tsplit"], g["dk"]
    for name, rows in ts.items():
        for r in rows:
            r["id"] = "tp-%05d" % r["row"]
    splits = {
        "train": dk["train"] + ts["train"] + social_train,
        "dev": dk["dev"] + ts["dev"] + social_dev,
        "test_real": dk["test_real"],
        "test_social": test_social,
        "test_social_uci": g["uci_draft"],
        "test_synth_iid": ts["test_synth_iid"],
        "test_synth_ood": ts["test_synth_ood"],
    }
    # final leakage guard: nothing in an eval set has its noisy or gold sentence in train
    train_keys = {key_of(r["noisy"]) for r in splits["train"]} | {key_of(r["normalized"]) for r in splits["train"]}
    leaks = {}
    for name, rows in splits.items():
        if name == "train":
            continue
        bad = [r for r in rows if key_of(r["noisy"]) in train_keys or key_of(r["normalized"]) in train_keys]
        if bad and name in ("dev", "test_synth_iid", "test_synth_ood", "test_real"):
            ids = {id(r) for r in bad}
            splits[name] = [r for r in rows if id(r) not in ids]
        leaks[name] = len(bad)
    log["eval_rows_overlapping_train_removed"] = leaks
    for name, rows in splits.items():
        for r in rows:
            r["split"] = name
            r.pop("row", None)

    for name, rows in splits.items():
        write_jsonl(os.path.join(V4, name + ".jsonl"), rows)
    write_jsonl(os.path.join(V4, "social_sentiment.jsonl"), sentiment)
    write_jsonl(os.path.join(V4, "dropped.jsonl"), g["templ_dropped"] + g["social_dropped"] + [
        {"source": "social_new", "id": r["id"], "reason": "overlaps_test", "noisy": r["noisy"]} for r in leak])

    # ---- golden pairs (flat CSV, same two columns as the input + provenance)
    with open(os.path.join(V4, "golden_pairs.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noisy_text", "normalized_text", "split", "source", "tier", "id"])
        for name, rows in splits.items():
            for r in rows:
                tier = r.get("tier", "gold")
                if name == "test_social":
                    tier = "claude-reviewed"
                elif name == "test_social_uci":
                    tier = "claude-draft"
                w.writerow([r["noisy"], r["normalized"], name, r["source"], tier, r["id"]])

    # ---- lexicon
    lexrows = build_lexicon(g["lex"], splits, g["norm"])
    cols = ["canonical", "urdu", "type", "variant", "in_v3_lexicon", "train_count", "eval_count", "eval_only",
            "canonical_only_in_eval"]
    for fname, keep in (("lexicon_full.tsv", lambda r: True),
                        ("lexicon_trainsafe.tsv", lambda r: not r["eval_only"] and not r["canonical_only_in_eval"])):
        with open(os.path.join(V4, fname), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for r in lexrows:
                if keep(r):
                    w.writerow(r)
    log["lexicon"] = {"canonicals": len({r["canonical"] for r in lexrows}), "rows": len(lexrows),
                      "eval_only_variants_removed_from_trainsafe": sum(r["eval_only"] for r in lexrows),
                      "eval_only_canonicals_removed": len({r["canonical"] for r in lexrows if r["canonical_only_in_eval"]})}
    log["records"] = {k: len(v) for k, v in splits.items()}
    log["records_by_source"] = {k: dict(Counter(r["source"] for r in v)) for k, v in splits.items()}
    with open(os.path.join(V4, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: log[k] for k in ("dataset_csv", "templated", "templated_split", "social_new",
                                          "social_new_split", "records", "eval_rows_overlapping_train_removed",
                                          "lexicon")}, indent=1, ensure_ascii=False))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--force-sheet", action="store_true", help="re-sample annotation/social_test_sheet.tsv")
    b.add_argument("--refresh-sheet", action="store_true", help="re-run the rules on the existing sample")
    b.add_argument("--test-size", type=int, default=300)
    b.add_argument("--dev-size", type=int, default=1000)
    b.set_defaults(fn=cmd_build)
    v = sub.add_parser("review")
    v.add_argument("--top", type=int, default=2500)
    v.set_defaults(fn=cmd_review)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
