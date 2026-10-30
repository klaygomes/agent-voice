"""agent-voice: every command prints one JSON object on stdout; progress goes to stderr."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

VERSION = version("agent-voice")
_stdout = None


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), file=_stdout or sys.stdout, flush=True)


def fail(message: str, code: int = 1) -> int:
    emit({"ok": False, "error": message})
    return code


class JsonParser(argparse.ArgumentParser):
    def error(self, message):
        emit({"ok": False, "error": f"{self.prog}: {message}"})
        sys.exit(2)


def require_files(*paths: str | None) -> None:
    for p in paths:
        if p and not Path(p).is_file():
            raise FileNotFoundError(2, "No such file", p)


def voice_from(args, prompt: Path | None = None):
    from .engine import Voice

    if prompt:
        return Voice(prompt_path=str(prompt))
    return Voice(design=args.voice, prompt_path=str(Path(args.prompt).resolve()) if args.prompt else None)


def cmd_doctor(args) -> int:
    from .engine import ASR_MODEL, REQUIRED_FILES, TTS_MODEL, best_device, is_downloaded

    models = {m: is_downloaded(m) for m in (TTS_MODEL, ASR_MODEL)}
    missing = [m for m, ok in models.items() if not ok]
    unknown = [m for m in missing if m not in REQUIRED_FILES]
    emit({
        "ok": True,
        "version": VERSION,
        "device": best_device(),
        "models": models,
        "ready": not missing,
        "download_gb": None if unknown else round(sum(REQUIRED_FILES[m][0] for m in missing), 2),
    })
    return 0


def cmd_speak(args) -> int:
    from . import engine

    require_files(args.prompt, args.ref)
    out = Path(args.out)
    engine.load(words=args.words or bool(args.ref))
    prompt = None
    if args.ref:
        prompt = out.with_suffix(".voice.pt")
        engine.save_voice(Path(args.ref), args.ref_text, prompt)
    duration = engine.synthesize(args.text, out, voice_from(args, prompt), speed=args.speed, steps=args.steps)
    result = {"ok": True, "path": str(out.resolve()), "duration": duration, "sample_rate": engine.SAMPLE_RATE}
    if args.words:
        result["words"] = engine.align(out)
    emit(result)
    return 0


def cmd_script(args) -> int:
    from . import engine

    require_files(args.file, args.prompt, args.ref)
    lines = [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return fail("the script has no lines")
    out = Path(args.out)
    if any(out.glob("line*.wav")):
        return fail(f"{out} already holds lineNN.wav files; use a fresh --out directory")
    engine.load(words=True)

    if args.prompt:
        prompt = Path(args.prompt).resolve()
    else:
        prompt = out / "voice.pt"
        reference = out / "voice-reference.wav"
        if args.ref:
            reference, ref_text = Path(args.ref), args.ref_text
        else:
            ref_text = engine.first_sentence(lines[0])
            engine.synthesize(ref_text, reference, voice_from(args), speed=args.speed, steps=args.steps)
        engine.save_voice(reference, ref_text, prompt)
    voice = engine.Voice(prompt_path=str(prompt))

    entries, words, clock = [], [], 0.0
    for i, text in enumerate(lines, 1):
        clip = out / f"line{i:02d}.wav"
        duration = engine.synthesize(text, clip, voice, speed=args.speed, steps=args.steps)
        line_words = engine.align(clip)
        words += [{**w, "start": round(w["start"] + clock, 3), "end": round(w["end"] + clock, 3)} for w in line_words]
        entries.append({"i": i, "text": text, "clip": str(clip.resolve()), "dur": duration,
                        "start": round(clock, 3), "end": round(clock + duration, 3), "words": line_words})
        engine.log(f"line {i}/{len(lines)}  {duration:.2f} s  {text[:60]}")
        clock += duration
    (out / "lines.json").write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "words.json").write_text(json.dumps(words, indent=1, ensure_ascii=False), encoding="utf-8")
    emit({"ok": True, "dir": str(out.resolve()), "lines": len(entries), "speech_seconds": round(clock, 2),
          "words_per_minute": round(len(words) / clock * 60) if clock else 0,
          "files": {"lines": str((out / "lines.json").resolve()), "words": str((out / "words.json").resolve()),
                    "voice": str(prompt)}})
    return 0


def cmd_align(args) -> int:
    from . import engine

    require_files(args.audio)
    engine.asr_pipeline()
    emit({"ok": True, "path": str(Path(args.audio).resolve()), "words": engine.align(Path(args.audio))})
    return 0


def add_voice_options(p: argparse.ArgumentParser) -> None:
    who = p.add_mutually_exclusive_group()
    who.add_argument("--voice", help='design a speaker, e.g. "male, middle-aged, low pitch, american accent"')
    who.add_argument("--ref", help="clone a speaker from a 3-10 s reference wav")
    who.add_argument("--prompt", help="reuse a saved voice.pt from an earlier run")
    p.add_argument("--ref-text", help="transcript of --ref; measured with Whisper when omitted")
    p.add_argument("--speed", type=float, default=1.0, help="above 1 is faster")
    p.add_argument("--steps", type=int, default=32, help="diffusion steps; 16 is faster, 32 is cleaner")


def build_parser() -> argparse.ArgumentParser:
    ap = JsonParser(prog="agent-voice", description=__doc__)
    ap.add_argument("--version", action="version", version=VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True, parser_class=JsonParser)

    sub.add_parser("doctor", help="report the device and which models are downloaded").set_defaults(fn=cmd_doctor)

    p = sub.add_parser("speak", help="one text to one wav")
    p.add_argument("text")
    p.add_argument("--out", required=True)
    p.add_argument("--words", action="store_true", help="also return measured word timestamps")
    add_voice_options(p)
    p.set_defaults(fn=cmd_speak)

    p = sub.add_parser("script", help="one wav per line (one scene) of a text file, same speaker for every line")
    p.add_argument("file")
    p.add_argument("--out", required=True, help="a fresh directory")
    add_voice_options(p)
    p.set_defaults(fn=cmd_script)

    p = sub.add_parser("align", help="measured word timestamps for a wav")
    p.add_argument("audio")
    p.set_defaults(fn=cmd_align)
    return ap


def main(argv=None) -> int:
    global _stdout
    _stdout = sys.stdout
    args = build_parser().parse_args(argv)
    if args.cmd in ("speak", "script") and args.ref_text and not args.ref:
        return fail("--ref-text needs --ref", 2)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            return args.fn(args)
    except KeyboardInterrupt:
        return fail("interrupted", 130)
    except FileNotFoundError as err:
        return fail(f"file not found: {err.filename}")
    except Exception as err:
        return fail(f"{type(err).__name__}: {err}")


if __name__ == "__main__":
    sys.exit(main())
