"""ConcertStudio V8: best-take comping + neural stem remaster.

V1-V7 fused POVs per STFT bin (median/Wiener consensus). That produces musical-noise
shimmer, comb filtering between POVs at different distances, and smeared transients.
V8 instead:

1. ingest   - decode every clip, record codec profile (HE-AACv2 = fake SBR highs and
              parametric stereo), collapse to mono.
2. align    - onset-novelty xcorr for the coarse offset, windowed GCC-PHAT for the
              sub-sample lag and drift, per-window validity (drops talking / spliced
              footage that matches nothing).
3. quality  - per-window score per POV: crowd-to-music ratio from a crowd-removal
              RoFormer, trusted bandwidth, clipping, level/occlusion, validity.
4. comp     - Viterbi best-take path with a switching cost, loudness + LTAS matching so
              takes are interchangeable, equal-power crossfades. Nothing is stacked.
5. neural   - RoFormer stems (vocals / bass / drums / guitar / piano / other), vocal
              de-reverb.
6. mix      - per-stem studio processing, mono-compatible synthetic stereo, light bus
              glue, true-peak limited master.
"""
