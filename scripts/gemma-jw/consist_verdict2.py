"""Final pass list, comparing translations on their normalised form.

The first pass compared raw strings and scored Yoruba 2/20 while it was
returning the right word every time - its tone marks simply moved. Comparing
NFD-stripped, case-folded text measures whether the model means the same word,
which is the question.

The bar comes from the controls. Languages Gemma demonstrably handles cluster
near the top; anything far below that is not a language it knows, however
fluent the output looks.
"""
import json, statistics, unicodedata
from collections import Counter
from pathlib import Path

HERE = Path("/mnt/volume_d2wey28/projects/shola-translate")
d = json.loads((HERE / "consistency-jw2.json").read_text())
CONTROLS = {"swa", "yor", "hau", "zul", "lin", "sna"}


def norm(t):
    t = unicodedata.normalize("NFD", str(t).casefold())
    return "".join(c for c in t if not unicodedata.combining(c)).strip()


def stability(v):
    samples = v.get("all") or []
    if len(samples) < 2:
        return None
    n = min(len(s) for s in samples)
    return sum(1 for i in range(n)
               if len({norm(s[i]) for s in samples}) == 1), n


scores = {}
for k, v in d.items():
    r = stability(v)
    scores[k] = r

print("controls, normalised:")
ctrl = []
for k in sorted(CONTROLS):
    v, r = d.get(k), scores.get(k)
    if v and r:
        print(f"  {v['name']:12s} {r[0]:>2}/{r[1]}")
        ctrl.append(r[0] / r[1])
if not ctrl:
    raise SystemExit("controls did not parse")

# Two thirds of the weakest control. Generous on purpose: the cost of keeping a
# doubtful language is a speaker correcting bad options, which is the normal
# work; the cost of dropping a real one is a language with nothing in it.
bar_frac = min(ctrl) * 0.66
print(f"\ncontrol fractions {[round(c,2) for c in ctrl]} -> bar "
      f"{bar_frac:.0%} of words stable")

keep, drop = [], []
for k, v in sorted(d.items()):
    if k in CONTROLS:
        continue
    r = scores.get(k)
    if not r or r[1] == 0:
        drop.append((k, v["name"], "did not parse"))
        continue
    frac = r[0] / r[1]
    if frac < bar_frac:
        drop.append((k, v["name"], f"{r[0]}/{r[1]} stable ({frac:.0%})"))
    else:
        keep.append((k, v["name"], r[0], r[1]))

(HERE / "jw-final-langs.tsv").write_text(
    "".join(f"{c}\t{n}\n" for c, n, _, _ in keep), encoding="utf-8")
print(f"\nKEPT {len(keep)}   DROPPED {len(drop)}")
if keep:
    print("median stability of kept: "
          f"{statistics.median([a/b for _,_,a,b in keep]):.0%}")

lines = [f"bar: {bar_frac:.0%} of words stable across 3 samples",
         f"kept {len(keep)}, dropped {len(drop)}", "", "KEPT"]
for c, n, a, b in sorted(keep, key=lambda r: -(r[2]/r[3])):
    lines.append(f"  {c:6s} {n[:32]:34s} {a}/{b}")
lines += ["", "DROPPED"]
for c, n, why in drop:
    lines.append(f"  {c:6s} {n[:32]:34s} {why}")
(HERE / "consistency-jw-summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nweakest kept:")
for c, n, a, b in sorted(keep, key=lambda r: r[2]/r[3])[:8]:
    print(f"  {c:6s} {n[:30]:32s} {a}/{b}")
print("\nsample of dropped:")
for c, n, why in drop[:8]:
    print(f"  {c:6s} {n[:30]:32s} {why}")
