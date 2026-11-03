"""Synthetic 125 BPM electronic instrumental (no vocals, no guitar) for the genre benchmark."""
import sys, numpy as np, soundfile as sf
from scipy import signal
SR = 44100; bpm = 125.0; beat = 60 / bpm; dur = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0
n = int(dur * SR); t = np.arange(n) / SR; rng = np.random.default_rng(7)
x = np.zeros((n, 2))
def add(sig, at, pan=0.0, g=1.0):
    i = int(at * SR); j = min(n, i + len(sig))
    if i >= n: return
    x[i:j, 0] += sig[: j - i] * g * (1 - pan) ; x[i:j, 1] += sig[: j - i] * g * (1 + pan)
kt = np.arange(int(0.35 * SR)) / SR
kick = np.sin(2 * np.pi * (45 * kt + 75 * (1 - np.exp(-kt * 30)) / 30)) * np.exp(-kt * 9)
cl = signal.sosfilt(signal.butter(2, [900, 3000], 'band', fs=SR, output='sos'), rng.standard_normal(int(0.2 * SR))) * np.exp(-np.arange(int(0.2 * SR)) / SR * 25)
hat = signal.sosfilt(signal.butter(4, 7000, 'high', fs=SR, output='sos'), rng.standard_normal(int(0.04 * SR))) * np.exp(-np.arange(int(0.04 * SR)) / SR * 90)
prog = [(45, [57, 60, 64]), (41, [57, 60, 65]), (48, [55, 60, 64]), (43, [55, 59, 62])]   # Am F C G
hz = lambda m: 440 * 2 ** ((m - 69) / 12)
nb = int(dur / beat)
side = np.ones(n)
for b in range(nb):
    at = b * beat; bar = b // 4; root, chord = prog[bar % 4]
    sect = (bar // 8) % 4                                     # build / drop / break / drop
    if sect != 2:
        add(kick, at, g=0.9)
        i = int(at * SR); side[i:i + int(0.18 * SR)] = np.minimum(side[i:i + int(0.18 * SR)], np.linspace(0.25, 1, int(0.18 * SR)))
    if b % 2 == 1 and sect != 2: add(cl, at, g=0.35)
    for h in range(2): add(hat, at + h * beat / 2, pan=0.3, g=0.12 if sect else 0.05)
    # bass: off-beat 8ths
    bt = np.arange(int(beat / 2 * SR)) / SR
    bs = signal.sawtooth(2 * np.pi * hz(root - 12) * bt) * np.exp(-bt * 4)
    add(bs, at + beat / 2, g=0.28 if sect != 2 else 0.12)
    # arp lead 16ths
    for s_ in range(4):
        m = chord[(b * 4 + s_) % 3] + 12
        at2 = np.arange(int(beat / 4 * SR)) / SR
        ld = signal.square(2 * np.pi * hz(m) * at2, 0.3) * np.exp(-at2 * 14)
        add(ld, at + s_ * beat / 4, pan=0.4 * np.sin(b), g=0.06 if sect in (1, 3) else 0.03)
# pads: detuned saws per bar
for bar in range(int(dur / (4 * beat)) + 1):
    root, chord = prog[bar % 4]; at = bar * 4 * beat
    pt = np.arange(int(4 * beat * SR)) / SR
    pad = sum(signal.sawtooth(2 * np.pi * hz(m) * (1 + d) * pt) for m in chord for d in (-0.004, 0.0, 0.005))
    pad *= np.minimum(1, pt / 0.3) * np.minimum(1, (4 * beat - pt) / 0.3)
    pad = signal.sosfilt(signal.butter(2, 2200, 'low', fs=SR, output='sos'), pad)
    add(pad, at, pan=-0.2, g=0.025)
x[:, :] = x * np.where(np.arange(n) >= 0, 1, 1)[:, None]
pad_side = side[:, None]
x = x * (0.6 + 0.4 * pad_side)
x = x / np.max(np.abs(x)) * 0.9
sf.write(sys.argv[1], x.astype(np.float32), SR, subtype='PCM_24')
print('wrote', sys.argv[1], dur)
