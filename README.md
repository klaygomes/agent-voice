# agent-voice

Local text to speech for agents, with word timestamps measured on the audio.

- **Voice:** [OmniVoice](https://github.com/k2-fsa/OmniVoice): 600+ languages, designed or cloned voices.
- **Timing:** [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), word-level.
- **Interface:** a CLI that prints one JSON object per command. No server, no API key, nothing leaves your machine.

```bash
uv tool install git+https://github.com/klaygomes/agent-voice@v0.1.0
agent-voice doctor
agent-voice speak "Whistle at four thousand hertz." --out whistle.wav --words
```

```json
{"ok": true, "path": "/tmp/whistle.wav", "duration": 2.1, "sample_rate": 24000,
 "words": [{"word": "Whistle", "start": 0.0, "end": 0.42}]}
```

A narration with one speaker across every scene:

```bash
agent-voice script vo.txt --out vo/ --voice "female, middle-aged, low pitch, american accent"
```

This writes `vo/line01.wav ...`, `vo/lines.json` (scene timeline), `vo/words.json` (word timeline) and `vo/voice.pt` (the speaker, reusable with `--prompt`).

For agents: [AGENTS.md](AGENTS.md). Install the skill with `npx skills add klaygomes/agent-voice`.

## Requirements

- macOS on Apple Silicon, Linux or Windows. Intel Macs are not supported (no torch 2.8.0 wheel).
- Python 3.12, which uv installs for you.
- Packages: about 1 GB on macOS. On Linux with CUDA, torch pulls several GB of NVIDIA libraries.
- Model weights: 4.89 GB, downloaded on the first `speak`, `script` or `align`.

`uv tool install` does not read `uv.lock`. The direct dependencies are pinned exactly, but OmniVoice's own dependencies resolve fresh at install time. For the tested set, clone the repo and run `uv sync --locked`.

## License

The code is MIT. **The OmniVoice model weights are CC-BY-NC, so the audio you generate is for non-commercial use only.** The audio tokenizer is under the Boson Higgs Audio 2 Community License. The required notices are in [NOTICE](NOTICE).

## Development

```bash
uv sync --locked
uv run pytest
```

The tests use a fake engine and download nothing.

## Credits

The agent-first shape (local models, JSON contracts, ask before downloads) follows ideas from [VoiceStudio](https://github.com/debpalash/VoiceStudio). No VoiceStudio code is included.
