from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TTS_MODEL = os.environ.get("AGENT_VOICE_TTS_MODEL", "k2-fsa/OmniVoice")
ASR_MODEL = os.environ.get("AGENT_VOICE_ASR_MODEL", "openai/whisper-large-v3-turbo")
SAMPLE_RATE = 24000
REQUIRED_FILES = {
    "k2-fsa/OmniVoice": (3.27, ["model.safetensors", "audio_tokenizer/model.safetensors"]),
    "openai/whisper-large-v3-turbo": (1.62, ["model.safetensors"]),
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_downloaded(repo: str) -> bool | None:
    from huggingface_hub import try_to_load_from_cache

    if repo not in REQUIRED_FILES:
        return None
    return all(isinstance(try_to_load_from_cache(repo, f), str) for f in REQUIRED_FILES[repo][1])


def load(words: bool) -> None:
    # Whisper on MPS transcribes nothing when first loaded after an OmniVoice generation.
    if words:
        asr_pipeline()
    tts_model()


@lru_cache(maxsize=1)
def tts_model():
    import torch
    from omnivoice import OmniVoice

    device = best_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    log(f"loading {TTS_MODEL} on {device}")
    return OmniVoice.from_pretrained(TTS_MODEL, device_map=device, dtype=dtype)


@lru_cache(maxsize=1)
def asr_pipeline():
    import torch
    from transformers import pipeline

    device = best_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    log(f"loading {ASR_MODEL} on {device} for word timestamps")
    return pipeline("automatic-speech-recognition", model=ASR_MODEL, device=device, dtype=dtype)


@dataclass(frozen=True)
class Voice:
    design: str | None = None
    prompt_path: str | None = None


def _generate_kwargs(voice: Voice) -> dict:
    if voice.prompt_path:
        from omnivoice import VoiceClonePrompt

        return {"voice_clone_prompt": VoiceClonePrompt.load(voice.prompt_path)}
    if voice.design:
        return {"instruct": voice.design}
    return {}


def synthesize(text: str, out: Path, voice: Voice, speed: float = 1.0, steps: int = 32) -> float:
    import soundfile as sf

    audio = tts_model().generate(text=text, speed=speed, num_step=steps, **_generate_kwargs(voice))[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, SAMPLE_RATE)
    return round(len(audio) / SAMPLE_RATE, 3)


def save_voice(ref_audio: Path, ref_text: str | None, out: Path) -> None:
    if not ref_text:
        ref_text = " ".join(w["word"] for w in align(ref_audio))
    prompt = tts_model().create_voice_clone_prompt(ref_audio=str(ref_audio), ref_text=ref_text)
    out.parent.mkdir(parents=True, exist_ok=True)
    prompt.save(str(out))


def first_sentence(text: str) -> str:
    match = re.match(r"(.+?[.!?])(\s|$)", text)
    return match.group(1) if match else text


def align(audio: Path) -> list[dict]:
    import soundfile as sf

    data, rate = sf.read(str(audio), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    duration = round(len(data) / rate, 3)
    result = asr_pipeline()({"raw": data, "sampling_rate": rate}, return_timestamps="word", generate_kwargs={"task": "transcribe"})
    if not result.get("text", "").strip():
        raise RuntimeError(f"alignment heard no words in {audio}")
    words = []
    for chunk in result.get("chunks", []):
        start, end = chunk["timestamp"]
        word = chunk["text"].strip()
        if word and start is not None:
            words.append({"word": word, "start": round(start, 3), "end": round(end if end is not None else duration, 3)})
    return words
