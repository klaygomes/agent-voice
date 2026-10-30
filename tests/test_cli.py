import json

import pytest

from agent_voice import cli, engine


@pytest.fixture
def fake_engine(monkeypatch):
    calls = {"synth": [], "saved": [], "loads": []}

    def synthesize(text, out, voice, speed=1.0, steps=32):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"RIFF")
        calls["synth"].append({"text": text, "out": out.name, "voice": voice})
        return 2.0

    def align(audio):
        return [{"word": "hello", "start": 0.1, "end": 0.5}, {"word": "there", "start": 0.6, "end": 1.2}]

    def save_voice(ref_audio, ref_text, out):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"prompt")
        calls["saved"].append({"ref": str(ref_audio), "text": ref_text})

    monkeypatch.setattr(engine, "load", lambda words: calls["loads"].append(words))
    monkeypatch.setattr(engine, "asr_pipeline", lambda: None)
    monkeypatch.setattr(engine, "synthesize", synthesize)
    monkeypatch.setattr(engine, "align", align)
    monkeypatch.setattr(engine, "save_voice", save_voice)
    return calls


def run(capsys, *argv):
    try:
        code = cli.main(list(argv))
    except SystemExit as exit_:
        code = exit_.code
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1, lines
    return code, json.loads(lines[0])


def write_script(tmp_path, text):
    script = tmp_path / "vo.txt"
    script.write_text(text, encoding="utf-8")
    return str(script)


def test_speak_returns_path_duration_and_words(tmp_path, capsys, fake_engine):
    code, out = run(capsys, "speak", "hello there", "--out", str(tmp_path / "a.wav"), "--words")
    assert code == 0 and out["ok"]
    assert out["duration"] == 2.0 and out["sample_rate"] == 24000
    assert [w["word"] for w in out["words"]] == ["hello", "there"]


def test_speak_without_words_flag_omits_words(tmp_path, capsys, fake_engine):
    _, out = run(capsys, "speak", "hello", "--out", str(tmp_path / "a.wav"))
    assert "words" not in out and fake_engine["loads"] == [False]


def test_script_designs_a_short_reference_then_reuses_it_for_every_line(tmp_path, capsys, fake_engine):
    script = write_script(tmp_path, "First scene. It is long.\n\nSecond scene.\nThird scene.\n")
    code, out = run(capsys, "script", script, "--out", str(tmp_path / "vo"), "--voice", "female, low pitch")
    assert code == 0 and out["lines"] == 3
    reference, *lines = fake_engine["synth"]
    assert reference["text"] == "First scene." and reference["voice"].design == "female, low pitch"
    assert [l["out"] for l in lines] == ["line01.wav", "line02.wav", "line03.wav"]
    assert all(l["voice"].prompt_path.endswith("voice.pt") for l in lines)
    assert fake_engine["saved"] == [{"ref": str(tmp_path / "vo" / "voice-reference.wav"), "text": "First scene."}]


def test_script_with_ref_clones_the_reference_itself(tmp_path, capsys, fake_engine):
    ref = tmp_path / "me.wav"
    ref.write_bytes(b"RIFF")
    run(capsys, "script", write_script(tmp_path, "One.\nTwo.\n"), "--out", str(tmp_path / "vo"), "--ref", str(ref))
    assert fake_engine["saved"] == [{"ref": str(ref), "text": None}]
    assert [s["out"] for s in fake_engine["synth"]] == ["line01.wav", "line02.wav"]


def test_script_with_prompt_saves_nothing_new(tmp_path, capsys, fake_engine):
    prompt = tmp_path / "mine.pt"
    prompt.write_bytes(b"prompt")
    _, out = run(capsys, "script", write_script(tmp_path, "One.\nTwo.\n"), "--out", str(tmp_path / "vo"), "--prompt", str(prompt))
    assert fake_engine["saved"] == [] and out["files"]["voice"] == str(prompt.resolve())
    assert all(s["voice"].prompt_path == str(prompt.resolve()) for s in fake_engine["synth"])


def test_script_offsets_word_times_into_one_timeline(tmp_path, capsys, fake_engine):
    _, out = run(capsys, "script", write_script(tmp_path, "One.\nTwo.\n"), "--out", str(tmp_path / "vo"))
    lines = json.loads((tmp_path / "vo" / "lines.json").read_text())
    words = json.loads((tmp_path / "vo" / "words.json").read_text())
    assert [l["start"] for l in lines] == [0.0, 2.0]
    assert [w["start"] for w in words] == [0.1, 0.6, 2.1, 2.6]
    assert out["speech_seconds"] == 4.0


def test_script_refuses_a_directory_with_old_clips(tmp_path, capsys, fake_engine):
    (tmp_path / "vo").mkdir()
    (tmp_path / "vo" / "line01.wav").write_bytes(b"old")
    code, out = run(capsys, "script", write_script(tmp_path, "One.\n"), "--out", str(tmp_path / "vo"))
    assert code == 1 and "fresh" in out["error"] and fake_engine["loads"] == []


def test_empty_script_fails_with_json(tmp_path, capsys, fake_engine):
    code, out = run(capsys, "script", write_script(tmp_path, "\n\n"), "--out", str(tmp_path / "vo"))
    assert code == 1 and out == {"ok": False, "error": "the script has no lines"}


def test_missing_prompt_fails_before_loading_models(tmp_path, capsys, fake_engine):
    code, out = run(capsys, "speak", "hi", "--out", str(tmp_path / "a.wav"), "--prompt", str(tmp_path / "nope.pt"))
    assert code == 1 and "nope.pt" in out["error"] and fake_engine["loads"] == []


def test_argument_errors_are_json(capsys, fake_engine):
    code, out = run(capsys, "speak", "hi")
    assert code == 2 and not out["ok"] and "--out" in out["error"]


def test_voice_options_are_mutually_exclusive(tmp_path, capsys, fake_engine):
    code, out = run(capsys, "speak", "hi", "--out", "a.wav", "--voice", "male", "--prompt", "p.pt")
    assert code == 2 and "not allowed" in out["error"]


def test_ref_text_needs_ref(tmp_path, capsys, fake_engine):
    code, out = run(capsys, "speak", "hi", "--out", str(tmp_path / "a.wav"), "--ref-text", "hello")
    assert code == 2 and out["error"] == "--ref-text needs --ref"


def test_align_raises_when_whisper_hears_nothing(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf

    sf.write(str(tmp_path / "s.wav"), np.zeros(2400, dtype="float32"), 24000)
    monkeypatch.setattr(engine, "asr_pipeline", lambda: (lambda *a, **k: {"text": " ", "chunks": []}))
    with pytest.raises(RuntimeError, match="heard no words"):
        engine.align(tmp_path / "s.wav")


def test_align_ends_an_open_last_word_at_the_clip_end(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf

    sf.write(str(tmp_path / "a.wav"), np.zeros(24000, dtype="float32"), 24000)
    chunks = [{"text": " Hello", "timestamp": (0.0, 0.4)}, {"text": " there.", "timestamp": (0.4, None)}]
    monkeypatch.setattr(engine, "asr_pipeline", lambda: (lambda *a, **k: {"text": "Hello there.", "chunks": chunks}))
    assert engine.align(tmp_path / "a.wav") == [{"word": "Hello", "start": 0.0, "end": 0.4}, {"word": "there.", "start": 0.4, "end": 1.0}]


def test_first_sentence():
    assert engine.first_sentence("Your line hums. Fix it.") == "Your line hums."
    assert engine.first_sentence("No full stop here") == "No full stop here"
