#!/usr/bin/env bash
# Run the V8 pipeline on every simulated case in $BENCH_DIR and score it against the clean reference.
#
#   python -m concert_studio.v8.simulate make --ref song.wav --out $BENCH_DIR/rock_3 --n 3 --seed 1
#   BENCH_DIR=~/cs_bench bash bench/run_bench.sh
#   python bench/summarize.py ~/cs_bench          # markdown table
#
# Env: PY (python), CAP (optional command prefix, e.g. a memory-capped scope), CASES, STAGES, TAG, MIXSET,
#      CS_MODELS (shared model cache). Stages run one at a time on purpose: separation is memory hungry.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
B=${BENCH_DIR:?set BENCH_DIR to a folder of cases made by simulate make}
PY=${PY:-python}
CAP=${CAP:-}
TAG=${TAG:-bench}
MIXSET=${MIXSET:-}
CASES=${CASES:-$(cd "$B" && ls -d */ | tr -d / | tr '\n' ' ')}
STAGES=${STAGES:-"align crowd quality comp stems mix score"}
COMP=${COMP:-w_pq=12,w_ce=2,w_coh=2,w_pump=8,switch_cost=80,min_dwell_s=12,fast_lock=0.8}
export PYTHONUNBUFFERED=1
ST=$B/status.txt
run() { local c=$1 s=$2; shift 2; echo "$(date +%T) $c $s START" >> "$ST"
        "$@" > "$B/$c/log_$s.txt" 2>&1; local rc=$?; echo "$(date +%T) $c $s END rc=$rc" >> "$ST"; return $rc; }
for c in $CASES; do
  [ -d "$B/$c/clips" ] || continue
  W=$B/$c/work; mkdir -p "$W" "$B/$c/out"
  for s in $STAGES; do
    case $s in
      align)   run $c align   $CAP $PY -m concert_studio.v8.run align --src "$B/$c/clips" --work "$W" ;;
      crowd)   run $c crowd   $CAP $PY -m concert_studio.v8.run crowd --work "$W" ;;
      quality) run $c quality $CAP $PY concert_studio/v8/quality.py --work "$W" ;;
      comp)    run $c comp    $CAP $PY -m concert_studio.v8.run comp --work "$W" --name 1 --set "$COMP" ;;
      stems)   run $c stems   $CAP $PY -m concert_studio.v8.run stems --work "$W" --name 1 ;;
      mix)     run $c mix     $CAP $PY -m concert_studio.v8.run mix --work "$W" --out "$B/$c/out" --tag "$TAG" --name 1 \
                                   ${MIXSET:+--set "$MIXSET"} ;;
      score)   run $c score   $CAP $PY bench/score_case.py "$B/$c" "$TAG" ;;
    esac || { echo "$(date +%T) $c FAILED at $s" >> "$ST"; break; }
  done
done
echo "$(date +%T) ALL DONE" >> "$ST"
