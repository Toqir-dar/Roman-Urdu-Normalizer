# Roman Urdu normalization dataset — v4 (full rebuild with `dataset.csv`)

v4 is built from the input files in `../../external_raw/` and `../../sources/` by `../../build_v4.py` (resources in
`v4_resources.py`, reviewed word lists in `v4_review.py`). Rebuild: `python build_v4.py build && python audit_v4.py`
(standard library only, ~3 min, deterministic). Earlier versions (v2, v3) and the model code were removed.

## What is in `external_raw/dataset.csv` (74,860 pairs) and what v4 does with it

| Block | Rows | What it is | Quality of the given `normalized_text` | v4 treatment |
|---|---|---|---|---|
| rows 1–46,770 | 46,770 (all unique) | templated sentences (≈2,300 frames, 810 gold word types, 14 names, 12 cities, 11 apps) with generated noise | good: gold is the template; noise is varied | kept as gold; lowercased; `ami/abu → ammi/abbu`; token-aligned; split by frame, entity **and misspelling** |
| rows 46,771–74,860 | 28,090 (23,393 unique) | real YouTube/Facebook comments | **unreliable**: a word-by-word lookup (`baar→bazar`, `bad→baad`, `wah→wahan`, `kise→kaise`, `sare→share`, `jaye→jayega`, `deen→dena`), many noisy words left as is (`he`, `me`, `ni`, `b`, `or`), HTML entities, injected token `qatil` in 3,768 rows | original normalization **discarded**; text re-normalized with v4 rules (below); `qatil` rows dropped |

Measured on the 297 hand-checked test comments: the file's own normalization has **WER 27.1** against the reviewed
gold, almost no better than leaving the text unchanged (32.4); v4's rules reach 9.4 before review (that number is
optimistic, because the review started from the rule output).

## Sources

| source | rows | gold | where |
|---|---|---|---|
| `templated` | 46,770 | generated (reliable) | train / dev / test_synth_iid / test_synth_ood |
| `social_new` | 19,284 kept | v4 rules; tier `gold` (all words known) or `silver` (has unknown / fuzzy-matched words); 297 checked sentence by sentence | train / dev / test_social |
| `dakshina` | 3,917 | cleaned Dakshina gold (`sources/dakshina_*.jsonl`), remapped to v4 conventions (Urdu script decides `kya`/`kiya`) | train / dev / test_real |
| `uci_social` | 186 (+19,417 sentiment-only) | earlier draft gold (`sources/uci_test_draft.jsonl`) remapped to v4 conventions | test_social_uci, social_sentiment.jsonl |

## Files

| file | rows | |
|---|---|---|
| `golden_pairs.csv` | 68,166 | **all pairs, flat**: `noisy_text, normalized_text, split, source, tier, id` |
| `train.jsonl` | 48,703 | 27,594 templated + 17,963 social (5,212 gold, 12,751 silver) + 3,146 Dakshina |
| `dev.jsonl` | 4,831 | 3,454 templated (unseen frames) + 1,000 social (gold tier) + 377 Dakshina |
| `test_real.jsonl` | 394 | Dakshina (formal, Wikipedia-style) |
| `test_social.jsonl` | 297 | **real comments, rules + sentence-level review by Claude** (`status` ok / edited) |
| `test_social_uci.jsonl` | 186 | UCI comments, earlier draft gold (Claude, not human-checked) |
| `test_synth_iid.jsonl` | 2,000 | templated, train frames / entities / misspellings |
| `test_synth_ood.jsonl` | 11,755 | templated with an unseen frame, entity and/or misspelling (`ood_reason`) |
| `social_sentiment.jsonl` | 19,417 | UCI corpus with sentiment labels and `split` (normalization-test rows forced into `test`) |
| `lexicon_full.tsv`, `lexicon_trainsafe.tsv` | 30,198 canonicals | variants with train / eval counts; train-safe drops 2,419 eval-only variants and 1,339 eval-only words |
| `dropped.jsonl` | | every removed row with its reason |
| `stats.json`, `generalization_report.json` | | build log and audit |

Record fields: `id, split, source, noisy, normalized, tokens[{noisy, norm, lang, changed, urdu?, how?, ambiguous?,
merged?}]`, plus `frame, entities, ood_reason` (templated), `tier, unknown_words, n_ambiguous, lang_hint,
source_normalized` (social; `source_normalized` is the file's original, kept for comparison only), `urdu` (Dakshina).
`lang`: UR / EN / NE (name) / OTH. `how` says which rule produced a social token.

## Gold conventions (v4)

- Canonical spellings follow the templated data: `nahi, mein` (in) / `main` (I), `hoon, yeh, woh, to, ke, hai,
  hain, bohat, acha, kya` (what) / `kiya` (did), `ammi, abbu`. forms of earlier versions are remapped (`nahin→nahi, toh→to, keh→ke`).
  کہ and کے are both `ke` (Roman Urdu writers do not separate them); the `urdu` field keeps the difference.
- Lowercase everywhere. English words and names are kept as written; chat English is standardized (`plz→please`, `u→you`).
- Merged words are split on Urdu word boundaries (`hogaya → ho gaya`, `maine → main ne`, `sabsy → sab se`).
- Elongation removed (`nahiii → nahi`), repeated punctuation collapsed (`... → .`), laughter and emoji kept.
- A standalone izafat `e` is kept (`raah e haq`); ڑ may be written `r` or `d` (`larki`, `thodi`) — both accepted.
- Context rules decide `main/mein`, `he → hai/hi/he`, `han → hain/haan`, `or → aur/or`, `bad → baad/bad`,
  `chor → chhor/chor`, `kise → kaise/kisi`, `mare → mere/mare`, `ata → ata/aata`, `ur → aur/your`; these tokens are
  flagged `ambiguous`.

## How much each set tests generalization (`python audit_v4.py`)

| | test_real | test_social | test_social_uci | test_synth_iid | test_synth_ood | dev_dakshina | dev_templated | dev_social |
|---|---|---|---|---|---|---|---|---|
| sentences | 394 | 297 | 186 | 2000 | 11755 | 377 | 3454 | 1000 |
| tokens needing a change % | 20.5 | 31.6 | 30.9 | 44.4 | 44.1 | 20.7 | 41.5 | 25.5 |
| gold Urdu word not in train nor train-safe lexicon % | 8.0 | 5.7 | 3.6 | 0.0 | 0.9 | 8.2 | 0.4 | 0.6 |
| noisy spelling unseen in train % | 10.9 | 8.5 | 8.5 | 1.0 | 4.8 | 11.4 | 1.9 | 1.0 |
| corrections whose exact pair is in train % | 67.7 | 72.6 | 60.1 | 97.4 | 87.5 | 67.2 | 95.4 | 98.2 |
| name types unseen % | 16.7 | 21.4 | 89.1 | 0.0 | 0.0 | 23.5 | 0.0 | 0.0 |
| sentence frames unseen % | – | – | – | 3.0 | 54.5 | – | 100.0 | – |
| WER, input unchanged | 22.41 | 32.39 | 35.14 | 45.58 | 45.07 | 22.93 | 42.6 | 25.55 |
| WER, train dictionary + train-safe lexicon | 11.78 | 16.32 | 22.2 | 3.82 | 8.48 | 12.68 | 4.62 | 4.72 |

Read it as: the templated data is easy to memorize (dictionary WER 3.8 in distribution); holding out frames,
entities and misspellings more than doubles the error (8.5). The real sets (`test_real`, `test_social`,
`test_social_uci`) are where generalization shows. Report every model on each set separately; never pool them.

## Annotation status — read before publishing numbers

- `test_social` (297) and every `social_new` row: gold written by rules + word lists that **Claude (AI)** reviewed
  (`v4_review.py`: about 1,800 frequent word types), and the 297 test sentences were corrected one by one by Claude
  (151 edited, 3 skipped as garbled; audit trail in `annotation/social_test_edits.tsv`). **No human has checked
  them yet.** Have a Roman Urdu speaker review `annotation/social_test_gold.tsv` (set `status`), rebuild, and
  report agreement on an overlap of 30–50 sentences.
- `silver` social training rows contain words the rules did not know (kept as written) or guessed with a fuzzy
  chat-spelling rule; expect roughly 10% word error against careful gold. Use them for training, never for evaluation.
- `test_social_uci` is an earlier Claude draft (not human-checked), remapped to v4 conventions.

## Known limits

- The templated block has a small vocabulary and a closed set of names, apps and cities; the names also occur in the
  real data, so "name types unseen" is 0 for `test_synth_ood`.
- `test_social` is small (297 sentences, 3,634 words): report bootstrap confidence intervals.
- Rules cannot always resolve `main/mein`, `he`, `han` and agreement errors made by the writer (`gaya hain`);
  the reviewed test gold fixes spelling, not grammar, except where a form is a clear misspelling.
- `lang_hint` and the EN/NE tags in social rows come from word lists and are approximate.

## Sources and licences

- `external_raw/dataset.csv`: origin and licence not recorded in the file — find out before publishing.
- Google Dakshina dataset (Roark et al., 2020), cleaned copy in `sources/` — check its licence and cite it.
- UCI "Roman Urdu Data Set" (Sharf & Rahman, 2018), CC BY 4.0 per UCI — verify and cite.
