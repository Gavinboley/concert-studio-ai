"""Score a benchmark case: pipeline master, its comp (pre-mix), and each raw phone, vs. the clean reference."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from concert_studio.v8.simulate import score  # noqa: E402
case, tag = Path(sys.argv[1]), sys.argv[2]
ref = case / "reference.wav"
rows = {}
m = case / "out" / f"{tag}_master_44k24.wav"
if m.exists():
    rows["master"] = score(ref, m)
cm = case / "work" / "comp1" / "comp_music.wav"
if cm.exists():
    rows["comp"] = score(ref, cm)
for p in sorted((case / "clips").glob("*.m4a")):
    w = case / "out" / f"raw_{p.stem}.wav"
    if not w.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(p), "-ar", "44100", str(w)], check=True)
    rows[f"raw_{p.stem}"] = score(ref, w)
(case / "out" / f"score_{tag}.json").write_text(json.dumps(rows, indent=1))
for k, v in rows.items():
    print(f"{case.name:14s} {k:14s} " + "  ".join(f"{a} {b}" for a, b in v.items()))
