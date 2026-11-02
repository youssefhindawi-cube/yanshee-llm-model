"""Light-weight noise suppression for the live Arabic transcriber (numpy + scipy only, no models).

Two parts:
  * Highpass: a stateful 4th-order Butterworth high-pass that removes mains hum and low rumble from the
    continuous stream before voice detection and transcription.
  * SpectralGate: per-phrase spectral subtraction. The noise spectrum is estimated from the phrase's own
    quietest time-frequency cells (a low percentile per frequency bin), so it follows the browser's automatic
    gain instead of assuming a fixed room level. Moderate settings: the aim is a cleaner signal for Whisper,
    not pretty audio.
"""
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

RATE = 16000
N_FFT, HOP = 512, 128                       # 32 ms window, 8 ms hop at 16 kHz
_WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float32)


class Highpass:
    def __init__(self, cutoff_hz=100.0, order=4, sr=RATE):
        self.b, self.a = butter(order, cutoff_hz / (sr / 2.0), btype="high")
        self.zi = lfilter_zi(self.b, self.a) * 0.0

    def __call__(self, frame_int16):
        y, self.zi = lfilter(self.b, self.a, frame_int16.astype(np.float64), zi=self.zi)
        return np.clip(y, -32768, 32767).astype(np.int16)


def _stft(x):
    n = 1 + max(0, (len(x) - N_FFT)) // HOP
    pad = (n - 1) * HOP + N_FFT - len(x)
    x = np.pad(x, (0, max(0, pad)))
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP][:n]
    return np.fft.rfft(frames * _WIN, axis=1).T          # (bins, frames)


def _istft(spec, length):
    frames = np.fft.irfft(spec.T, n=N_FFT, axis=1) * _WIN
    out = np.zeros((frames.shape[0] - 1) * HOP + N_FFT, dtype=np.float64)
    norm = np.zeros_like(out)
    for i, f in enumerate(frames):
        out[i * HOP:i * HOP + N_FFT] += f
        norm[i * HOP:i * HOP + N_FFT] += _WIN ** 2
    out /= np.maximum(norm, 1e-3)
    return out[:length]


class SpectralGate:
    def __init__(self, strength=1.0, floor=0.15, noise_percentile=20):
        self.over = 1.0 * strength          # oversubtraction factor
        self.floor = floor                  # minimum gain (keeps a little background so Whisper is not confused by holes)
        self.pct = noise_percentile

    def __call__(self, audio_int16):
        x = audio_int16.astype(np.float32)
        if len(x) < N_FFT * 4:
            return audio_int16
        spec = _stft(x)
        power = np.abs(spec) ** 2
        noise = np.percentile(power, self.pct, axis=1, keepdims=True) * 2.0   # low percentile under-estimates: lift it
        gain = 1.0 - self.over * noise / np.maximum(power, 1e-9)
        gain = np.sqrt(np.clip(gain, self.floor ** 2, 1.0))
        gain = 0.5 * gain + 0.25 * np.roll(gain, 1, axis=1) + 0.25 * np.roll(gain, -1, axis=1)   # smooth over time
        y = _istft(spec * gain, len(x))
        return np.clip(y, -32768, 32767).astype(np.int16)


def snr_estimate(audio_int16):
    """Rough speech-to-noise ratio in dB: loud frames vs quiet frames, for diagnostics."""
    x = audio_int16.astype(np.float64)
    n = len(x) // HOP
    if n < 10:
        return float("nan")
    e = np.array([np.mean(x[i * HOP:(i + 1) * HOP] ** 2) for i in range(n)]) + 1e-9
    e.sort()
    return 10 * np.log10(np.mean(e[-max(1, n // 10):]) / np.mean(e[:max(1, n // 5)]))
