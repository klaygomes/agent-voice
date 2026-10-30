# agent-voice for agents

Local text to speech with measured word timestamps. Every command prints exactly one JSON object on stdout, including errors (`{"ok": false, "error": "..."}`). Progress goes to stderr. Exit codes: 0 ok, 1 failed, 2 bad arguments, 130 interrupted. Only `--version` prints plain text.

## Before the first use

```bash
agent-voice doctor
```

If `"ready": false`, the model weights are not downloaded yet. Tell the user the size in `download_gb` (4.89 GB the first time) and ask before anything else. Never download models silently. The first `speak`, `script` or `align` call downloads them.

Not installed? `uv tool install git+https://github.com/klaygomes/agent-voice@v0.1.0`. Supported: macOS on Apple Silicon (MPS), Linux and Windows (CUDA or CPU). Intel Macs are not supported, because torch 2.8.0 has no wheel for them.

## Commands

| Command | Use it for | Output |
|---|---|---|
| `agent-voice speak "text" --out a.wav [--words]` | one clip | `path`, `duration`, `sample_rate`, optional `words` |
| `agent-voice script vo.txt --out dir/` | a narration, one line per scene | `lineNN.wav`, `lines.json`, `words.json`, `voice.pt` in a fresh directory |
| `agent-voice align a.wav` | word times of any wav | `words: [{word, start, end}]` |
| `agent-voice doctor` | readiness | `device`, `models`, `ready`, `download_gb` |

Choose the speaker with at most one of:

- `--voice "female, middle-aged, low pitch, american accent"`: design a voice. Attributes: gender, age, pitch, `whisper` style, English accent, Chinese dialect. Stable for English and Chinese only.
- no option: the model picks a voice. Use this for other languages.
- `--ref clip.wav [--ref-text "..."]`: clone a voice from 3 to 10 s of audio. Without `--ref-text`, Whisper transcribes the clip. Only clone voices the user is allowed to use.
- `--prompt voice.pt`: reuse the speaker saved by an earlier `script` run.

`script` freezes the speaker once, from a short reference (the first sentence of line 1, or your `--ref` clip), saves it as `voice.pt`, and uses it for every line, so the whole narration is one person.

## Rules

- One scene per line in the script file. A line may hold several sentences: the model phrases them naturally.
- Keep `--speed 1.0` for narration that should sound human.
- Write numbers as words ("four thousand"), or the model may read them digit by digit.
- `--out` for `script` must be a directory without old `lineNN.wav` files.
- Check `ok` before you report success. Word times in `words.json` are measured on the audio, on one continuous timeline. `lines.json` holds each line's `start`, `end`, `dur` and its own `words`.

## License of the audio

The code is MIT. The OmniVoice model weights are licensed **CC-BY-NC**: generated audio is for non-commercial use only. Its audio tokenizer is under the Boson Higgs Audio 2 Community License. Tell the user this before they use audio in anything commercial. See NOTICE.

`AGENT_VOICE_TTS_MODEL` and `AGENT_VOICE_ASR_MODEL` override the two Hugging Face models; `doctor` then reports `download_gb: null`.
