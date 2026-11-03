"""Markdown table of benchmark scores: the best single raw phone vs. the pipeline.

Coverage and tone are compared against the best raw phone *for that metric* (a hard baseline: no single phone is
best at everything). Level wobble is read at the comp stage, before the mix's intentional vocal riding, compression
and added reverb, which a fast-level comparison against the original studio mix would count as error.
"""
import json
import sys
from pathlib import Path

B = Path(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else "bench"
print("| case | phones | coverage: best phone -> output | tone error dB: best phone -> output | "
      "tone wander dB: best phone -> output | level wobble dB: best phone -> comp |")
print("|---|---|---|---|---|---|")
for f in sorted(B.glob(f"*/out/score_{tag}.json")):
    r = json.loads(f.read_text())
    raws = [v for k, v in r.items() if k.startswith("raw_")]
    if "master" not in r or not raws:
        continue
    m, c = r["master"], r.get("comp", {})
    best = lambda key, fn=min: fn(v[key] for v in raws)
    print(f"| {f.parts[-3]} | {len(raws)} | {best('coverage', max):.2f} -> {m['coverage']:.2f} | "
          f"{best('tone_db'):.1f} -> {m['tone_db']:.1f} | {best('tone_wander_db'):.2f} -> {m['tone_wander_db']:.2f} | "
          f"{best('wobble_db'):.2f} -> {c.get('wobble_db', float('nan')):.2f} |")
