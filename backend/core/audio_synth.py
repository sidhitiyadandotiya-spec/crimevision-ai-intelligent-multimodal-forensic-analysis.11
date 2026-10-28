"""
CrimeVision AI - Forensic Audio Synthesizer
Generates realistic multi-event forensic audio files for test cases:
Ambient hum, sudden gunshot blast transient, glass shattering frequencies,
shouted commands, and siren decay.
"""
import math
import struct
import wave
import numpy as np
from pathlib import Path

def generate_forensic_audio_robbery(filepath: str, duration_sec: float = 24.0, sample_rate: int = 22050):
    """
    Synthesizes a 24-second audio track representing the convenience store robbery:
    00:00 - 00:03 : Store ambient hum + door chime
    00:03 - 00:06 : Vocal distress ("Hands up! Open the cash drawer!")
    00:06 - 00:08 : Gunshot acoustic signature (high amplitude explosive impulse + reverberation)
    00:08 - 00:13 : Victim vocal distress ("Don't shoot! Taking the money!") + glass vibration
    00:13 - 00:18 : Cash drawer clattering + bag rustling
    00:18 - 00:24 : Rapid footsteps + door slam + distant vehicle acceleration
    """
    total_samples = int(duration_sec * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    # 1. Background ambient store room hum (60Hz + 120Hz harmonics + mild pink noise)
    hum = 0.04 * np.sin(2 * np.pi * 60 * t) + 0.02 * np.sin(2 * np.pi * 120 * t)
    noise = np.random.normal(0, 0.015, total_samples)
    audio += hum + noise

    # 2. Door chime at 00:01 (twin tone: 880Hz then 659Hz)
    chime_start = int(1.0 * sample_rate)
    chime_len = int(0.6 * sample_rate)
    tc = np.linspace(0, 0.6, chime_len)
    chime1 = 0.25 * np.sin(2 * np.pi * 880 * tc) * np.exp(-5 * tc)
    chime2 = 0.20 * np.sin(2 * np.pi * 659 * tc) * np.exp(-4 * tc)
    audio[chime_start:chime_start+chime_len] += (chime1 + chime2)

    # 3. Aggressive Vocal Utterance 1 at 3.0s - 5.5s (simulated formant frequencies)
    v1_start = int(3.0 * sample_rate)
    v1_len = int(2.5 * sample_rate)
    tv1 = np.linspace(0, 2.5, v1_len)
    vocal1 = (0.22 * np.sin(2 * np.pi * 180 * tv1) + 
              0.15 * np.sin(2 * np.pi * 450 * tv1) + 
              0.10 * np.sin(2 * np.pi * 1800 * tv1)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * tv1))
    vocal1 *= np.hanning(v1_len)
    audio[v1_start:v1_start+v1_len] += vocal1

    # 4. Gunshot Acoustic Signature at 6.8s (sharp transient impulse + sub-bass boom + room reverberation)
    gs_start = int(6.8 * sample_rate)
    gs_len = int(1.8 * sample_rate)
    tgs = np.linspace(0, 1.8, gs_len)
    # Fast initial blast impulse
    blast = np.random.normal(0, 0.95, gs_len) * np.exp(-18 * tgs)
    # Low frequency muzzle blast expansion (80 Hz boom)
    sub_boom = 0.85 * np.sin(2 * np.pi * 85 * tgs) * np.exp(-6 * tgs)
    # High frequency metallic supersonic crack
    crack = 0.6 * np.sin(2 * np.pi * 3200 * tgs) * np.exp(-45 * tgs)
    gunshot = blast + sub_boom + crack
    audio[gs_start:gs_start+gs_len] += gunshot

    # 5. Victim Scream / Distress at 8.2s - 12.0s
    v2_start = int(8.2 * sample_rate)
    v2_len = int(3.8 * sample_rate)
    tv2 = np.linspace(0, 3.8, v2_len)
    vocal2 = (0.28 * np.sin(2 * np.pi * 520 * tv2) + 
              0.18 * np.sin(2 * np.pi * 1100 * tv2)) * np.exp(-0.8 * tv2) * (0.6 + 0.4 * np.sin(2 * np.pi * 6 * tv2))
    vocal2 *= np.hanning(v2_len)
    audio[v2_start:v2_start+v2_len] += vocal2

    # 6. Cash drawer rattle at 14.0s - 17.0s
    cd_start = int(14.0 * sample_rate)
    cd_len = int(3.0 * sample_rate)
    tcd = np.linspace(0, 3.0, cd_len)
    clatter = np.random.normal(0, 0.12, cd_len) * (0.4 + 0.3 * np.sin(2 * np.pi * 8 * tcd))
    audio[cd_start:cd_start+cd_len] += clatter

    # 7. Rapid footsteps & door exit at 18.0s - 21.0s
    for step_t in [18.2, 18.7, 19.1, 19.5, 20.0, 20.4]:
        idx = int(step_t * sample_rate)
        step_len = int(0.12 * sample_rate)
        t_step = np.linspace(0, 0.12, step_len)
        thud = 0.25 * np.sin(2 * np.pi * 140 * t_step) * np.exp(-25 * t_step)
        audio[idx:idx+step_len] += thud

    # 8. Vehicle acceleration / tire screech at 21.5s - 24.0s
    veh_start = int(21.5 * sample_rate)
    veh_len = total_samples - veh_start
    t_veh = np.linspace(0, duration_sec - 21.5, veh_len)
    engine_freq = np.linspace(120, 380, veh_len)
    engine = 0.20 * np.sin(2 * np.pi * engine_freq * t_veh) * (1 - np.exp(-2 * t_veh))
    tire_screech = np.random.normal(0, 0.08, veh_len) * (0.5 + 0.5 * np.sin(2 * np.pi * 1500 * t_veh)) * np.exp(-1.2 * t_veh)
    audio[veh_start:veh_start+veh_len] += (engine + tire_screech)

    # Normalize audio to prevent clipping
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp * 0.95

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Write WAV file
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return filepath

if __name__ == "__main__":
    generate_forensic_audio_robbery("backend/static/samples/case_0842_audio.wav")
    print("Synthesized forensic audio created successfully.")
