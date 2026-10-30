---
name: agent-voice
description: Generate natural local speech with measured word timestamps through the agent-voice CLI (OmniVoice voice, Whisper timing). Use when a task needs narration, a voice-over, spoken audio for a video or page, word timings to sync picture to speech, or one consistent voice across many clips, and the user wants it local, offline, or without a paid API. Non-commercial use only.
---

# agent-voice

Every command prints one JSON object on stdout, errors included (`{"ok": false, "error": ...}`).

1. **Check.** `agent-voice doctor`. If the command is missing: `uv tool install git+https://github.com/klaygomes/agent-voice@v0.1.0` (Apple Silicon, Linux or Windows). If `ready` is false, tell the user `download_gb` and ask before the first download.
2. **One clip.** `agent-voice speak "text" --out a.wav --words`.
3. **A narration.** One scene per line in `vo.txt` (several sentences per line are fine), then `agent-voice script vo.txt --out vo/ --voice "<description>"` into a fresh directory. Every line shares one speaker, saved as `vo/voice.pt`.
4. **Choosing the voice.** `--voice "female, middle-aged, low pitch, american accent"` for English or Chinese; no voice option for other languages; `--ref clip.wav` to clone 3 to 10 s of a voice the user may use; `--prompt voice.pt` to reuse an earlier speaker. Use only one.
5. **Timing.** `vo/lines.json` has each scene's `start`, `end`, `dur` and `words`. `vo/words.json` has every word on one timeline, measured on the audio.

Write numbers as words. Keep `--speed 1.0` for a human pace.

**License.** The OmniVoice weights are CC-BY-NC, so generated audio is for non-commercial use only. Say so before the user uses it commercially.

Full reference: https://github.com/klaygomes/agent-voice/blob/main/AGENTS.md
