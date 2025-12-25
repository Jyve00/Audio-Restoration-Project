# Python features for audio restoration

# importing libraries for Python operations

import soundfile as sf
import numpy as np
import librosa
from scipy import signal
import noisereduce as nr
import matplotlib
import pydub


# Step 1: Ingest + metadata + checks
def read_audio(path, target_sr=None, mono=False):
    data, sr = sf.read(path, always_2d=True)   # shape (n_samples, n_channels)
    if mono:
        data = np.mean(data, axis=1, keepdims=True)
    if target_sr and sr != target_sr:
        import librosa
        data = librosa.resample(data.T, sr, target_sr).T
        sr = target_sr
    # convert to float32 in -1..1 if not already
    if data.dtype.kind != 'f':
        data = data.astype('float32') / np.max(np.abs(data))
    return data, sr

audio, sr = read_audio("noisy.wav", mono=False)


# Step 2: Remove Clicks & Pops (transients)
    # Use Median filter on waveform (simple method)
    # Interpolated across detected spikes
    # Spectral median filtering (better for hard and more aggressive clicks)

def remove_clicks_median(x, kernel_len=5):
    # x: 1D numpy array
    from scipy.signal import medfilt
    return medfilt(x, kernel_len)

# channel-wise
clean = np.zeros_like(audio)
for c in range(audio.shape[1]):
    clean[:,c] = remove_clicks_median(audio[:,c], kernel_len=7)


# alternative methos: detect samples with suddwen large differences and replace by linear interpolation around region


# Step 3: Hum/ narrowband tonal interference (notch filters)
    #if a 50/60Hz ground hum is present, use IIR notch (bandstop) filter.

def notch_filter(sig, sr, freq=60.0, Q=30.0):
    # freq: center frequency to remove (Hz), Q: quality factor
    b, a = signal.iirnotch(w0=freq/(sr/2), Q=Q)
    return signal.filtfilt(b, a, sig)

# Apply for fundamental and harmonics
clean_hum = audio.copy()
harmonics = [60, 120, 180]  # change to 50 if required
for h in harmonics:
    for c in range(clean_hum.shape[1]):
        clean_hum[:, c] = notch_filter(clean_hum[:, c], sr, freq=h, Q=30) # the higher the Q the more narrower the notch filter


# Step 4: Broadband noise reduction (spectral gating/subtraction)
    # estimate noise profile from a noise-only segment, then subtract. The noisereduce package exposes a simple API.


# If you have a noise-only segment, use it. Otherwise NR can estimate.
# Example: take first 0.5s as noise estimate (only if it's truly noise)
noise_clip = clean_hum[:int(0.5*sr), :]  # shape (n, channels)

denoised = np.zeros_like(clean_hum)
for c in range(clean_hum.shape[1]):
    denoised[:,c] = nr.reduce_noise(y=clean_hum[:,c], sr=sr, y_noise=noise_clip[:,c])



# Filter 

def highpass(sig, sr, cutoff=80.0, order=4):
    sos = signal.butter(order, cutoff, btype='highpass', fs=sr, output='sos')
    return signal.sosfiltfilt(sos, sig)

eq = denoised.copy()
for c in range(eq.shape[1]):
    eq[:,c] = highpass(eq[:,c], sr, cutoff=80)
