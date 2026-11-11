#!/usr/bin/env python3
"""Offline Whisper transcription; stdout contains metadata, never transcript text."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time


class UserError(Exception):
    """An actionable configuration error with no meeting content."""


def offline_environment():
    # Override inherited online/telemetry settings for this process only.
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYANNOTE_METRICS_ENABLED": "0",
        "DO_NOT_TRACK": "1",
    })


@dataclass
class Track:
    path: Path
    speaker: str
    offset: float = 0.0


@dataclass
class Utterance:
    start: float
    end: float
    text: str
    speaker: str


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", action="append", type=Path, required=True)
    result.add_argument("--speaker", action="append", default=[])
    result.add_argument("--offset", action="append", type=float, default=[])
    result.add_argument("--output", type=Path)
    result.add_argument("--inspect", action="store_true", help="Metadata only; no inference.")
    result.add_argument("--single-speaker", action="store_true")
    result.add_argument("--unattributed", action="store_true", help="Explicitly omit speaker identification.")
    result.add_argument("--diarization-model", type=Path, help="Local community-1 pipeline directory.")
    result.add_argument("--speaker-map", type=Path, help="Confirmed raw diarization label to name JSON.")
    result.add_argument("--num-speakers", type=int)
    result.add_argument("--model", default="turbo", help="Cached faster-whisper alias or local model directory.")
    result.add_argument("--language", default="auto")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--compute-type", help="Defaults to int8 on CPU, float16 on CUDA.")
    result.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    result.add_argument("--batch-size", type=int, default=8)
    return result


def resolve_tracks(args):
    paths = [p.expanduser().resolve(strict=True) for p in args.input]
    if any(not p.is_file() for p in paths):
        raise UserError("Every input must be a local file.")
    if len(set(paths)) != len(paths):
        raise UserError("The same input file was supplied more than once.")
    if args.offset and len(args.offset) != len(paths):
        raise UserError("Supply one --offset per input, including zeros.")
    offsets = args.offset or [0.0] * len(paths)
    if any(not math.isfinite(x) or x < 0 for x in offsets):
        raise UserError("Offsets must be finite, nonnegative seconds on the meeting timeline.")
    if args.speaker and len(args.speaker) != len(paths):
        raise UserError("Supply one --speaker per input.")
    if any(not name.strip() or any(c in name for c in "\r\n") for name in args.speaker):
        raise UserError("Speaker labels must be nonempty single-line names.")
    labels = args.speaker or ["Unassigned speaker"] * len(paths)
    return [Track(path, label.strip(), offset) for path, label, offset in zip(paths, labels, offsets)]


def validate_mode(args, tracks):
    if args.threads < 1 or args.batch_size < 1:
        raise UserError("Threads and batch size must be positive.")
    if args.num_speakers is not None and args.num_speakers < 1:
        raise UserError("Speaker count must be positive.")
    if len(tracks) > 1:
        if len(args.speaker) != len(tracks):
            raise UserError("Participant tracks require one --speaker per input.")
        if args.single_speaker or args.unattributed or args.diarization_model:
            raise UserError("Participant tracks cannot use a single-recording speaker mode.")
    else:
        if sum(bool(mode) for mode in (args.single_speaker, args.unattributed, args.diarization_model)) != 1:
            raise UserError("Choose --single-speaker, --unattributed, or --diarization-model for one recording.")
        if (args.diarization_model or args.unattributed) and args.speaker:
            raise UserError("Only known single-speaker recordings may use --speaker; diarization uses --speaker-map.")
        if args.single_speaker and not args.speaker:
            tracks[0].speaker = "Speaker 1"
    if (args.speaker_map or args.num_speakers is not None) and not args.diarization_model:
        raise UserError("Speaker mapping and speaker count require diarization.")
    if args.diarization_model and not args.diarization_model.expanduser().is_dir():
        raise UserError("The diarization model must be an existing local directory.")


def validate_output(path, tracks):
    if path is None:
        raise UserError("Supply an explicitly confirmed --output path.")
    path = path.expanduser()
    parent = path.parent.resolve(strict=True)
    output = parent / path.name
    if output.exists() or output.is_symlink():
        raise UserError("Output already exists; choose a new confirmed path.")
    if output in {track.path for track in tracks}:
        raise UserError("Output must not replace a source recording.")
    if output.suffix.lower() != ".md":
        raise UserError("The transcript output must have an .md extension.")
    return output


def inspect_track(track):
    import av

    with av.open(str(track.path), metadata_errors="ignore") as container:
        audio = list(container.streams.audio)
        if not audio:
            raise UserError("An input has no audio stream.")
        return {
            "file": str(track.path),
            "duration_seconds": container.duration / av.time_base if container.duration is not None else None,
            "container_start_seconds": container.start_time / av.time_base if container.start_time is not None else None,
            "channels": audio[0].codec_context.channels,
            "offset_seconds": track.offset,
        }


def segment_words(segment, speaker, offset):
    """Fall back to the segment when alignment omits text; never silently drop it."""
    words = list(segment.words or [])
    aligned = "".join(word.word for word in words)
    if not words or re.sub(r"\s+", "", aligned) != re.sub(r"\s+", "", segment.text):
        return [Utterance(segment.start + offset, segment.end + offset, segment.text.strip(), speaker)]
    return [
        Utterance(word.start + offset, word.end + offset, word.word, speaker)
        for word in words
    ]


def diarize(audio, args):
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(str(args.diarization_model.expanduser().resolve()))
    if args.device == "cuda":
        pipeline.to(torch.device("cuda"))
    options = {"num_speakers": args.num_speakers} if args.num_speakers is not None else {}
    result = pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}, **options)
    # Preserve regular overlap annotations instead of forcing one speaker at a time.
    return [
        (float(turn.start), float(turn.end), str(label))
        for turn, _, label in result.speaker_diarization.itertracks(yield_label=True)
    ]


def speaker_for_word(start, end, intervals, mapping):
    scores = defaultdict(float)
    for left, right, label in intervals:
        overlap = min(end, right) - max(start, left)
        if overlap > 0:
            scores[label] += overlap
    if not scores:
        return "Unassigned speaker"
    if len(scores) > 1:
        return "Uncertain speaker"
    label = next(iter(scores))
    return mapping.get(label, label)


def load_mapping(path):
    if path is None:
        return {}
    mapping = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not v.strip()
        or "\n" in v or "\r" in v for k, v in mapping.items()
    ):
        raise UserError("Speaker mapping must be a JSON object with single-line string names.")
    return {key: value.strip() for key, value in mapping.items()}


def make_turns(words):
    """Join nearby words from one track; keep gaps and speaker changes."""
    turns = []
    for word in words:
        if not word.text.strip():
            continue
        if not all(math.isfinite(v) for v in (word.start, word.end)) or word.start < 0 or word.end < word.start:
            raise UserError("Invalid word timestamps; no transcript was written.")
        if (
            turns and turns[-1].speaker == word.speaker
            and 0 <= word.start - turns[-1].end <= 1.5
            and word.end - turns[-1].start <= 35
        ):
            # Whisper words retain leading whitespace; segment fallbacks may not.
            separator = "" if word.text[:1].isspace() or word.text[:1] in ".,!?;:" else " "
            turns[-1].text += separator + word.text
            turns[-1].end = max(turns[-1].end, word.end)
        else:
            turns.append(Utterance(word.start, word.end, word.text.strip(), word.speaker))
    original = re.sub(r"\s+", "", "".join(word.text for word in words))
    merged = re.sub(r"\s+", "", "".join(turn.text for turn in turns))
    if original != merged:
        raise UserError("Text changed during merging; no transcript was written.")
    return turns


def timestamp(seconds):
    value = max(0, int(seconds))
    return f"{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d}"


def markdown_text(text):
    return re.sub(r"([\\`*_\[\]<>])", r"\\\1", text)


def render(turns, duration, attribution):
    lines = [
        "# Meeting transcript", "",
        "- **Duration:** " + timestamp(math.ceil(duration)),
        "- **Processing:** local Whisper; automatic transcription.",
        "- **Speaker labels:** " + attribution,
        "",
        "Times are approximate and relative to the meeting start. Overlapping contributions are ordered by start time. Recognition errors may remain.",
        "",
    ]
    if not turns:
        lines.extend(["No intelligible speech was detected.", ""])
    for turn in sorted(turns, key=lambda x: (x.start, x.end, x.speaker)):
        label = markdown_text(turn.speaker)
        text = markdown_text(re.sub(r"\s+", " ", turn.text).strip())
        lines.extend([
            f"**[{timestamp(turn.start)}–{timestamp(math.ceil(turn.end))}] {label}:** {text}", ""
        ])
    return "\n".join(lines)


def write_private(path, text):
    """Exclusive creation avoids both ordinary overwrites and symlink replacement."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def transcribe(args, tracks, output):
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from faster_whisper.audio import decode_audio

    mapping = load_mapping(args.speaker_map)
    model = WhisperModel(
        args.model, device=args.device,
        compute_type=args.compute_type or ("int8" if args.device == "cpu" else "float16"),
        cpu_threads=args.threads, local_files_only=True,
    )
    started = time.monotonic()
    turns = []
    duration = 0.0
    speech_seconds = 0.0
    token_count = 0
    for index, track in enumerate(tracks, 1):
        audio = decode_audio(str(track.path))
        intervals = diarize(audio, args) if args.diarization_model else []
        unknown_keys = set(mapping) - {label for _, _, label in intervals}
        if unknown_keys:
            raise UserError("Speaker mapping contains labels not produced by diarization.")
        pipeline = BatchedInferencePipeline(model=model)
        segments, info = pipeline.transcribe(
            audio, language=None if args.language == "auto" else args.language,
            batch_size=args.batch_size, beam_size=5, temperature=0.0,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 450, "speech_pad_ms": 200},
            word_timestamps=True, condition_on_previous_text=False,
        )
        duration = max(duration, track.offset + info.duration)
        speech_seconds += info.duration_after_vad
        words = []
        last_report = time.monotonic()
        for segment in segments:
            current = segment_words(segment, track.speaker, track.offset)
            if args.diarization_model:
                for word in current:
                    word.speaker = speaker_for_word(
                        word.start - track.offset, word.end - track.offset, intervals, mapping
                    )
            words.extend(current)
            if time.monotonic() - last_report >= 20:
                print(json.dumps({"track": index, "processed_seconds": round(segment.end, 1)}), flush=True)
                last_report = time.monotonic()
        token_count += len(words)
        turns.extend(make_turns(words))
        print(json.dumps({"track": index, "complete": True, "duration_seconds": round(info.duration, 1)}), flush=True)
    attribution = (
        "local diarization; uncertain attribution is marked." if args.diarization_model
        else "unattributed; speaker identities were not inferred." if args.unattributed
        else "supplied participant labels; anonymous if no name was given."
    )
    write_private(output, render(turns, duration, attribution))
    return {
        "output": str(output), "tracks": len(tracks), "turns": len(turns),
        "alignment_units": token_count, "speech_seconds": round(speech_seconds, 1),
        "duration_seconds": round(duration, 1), "elapsed_seconds": round(time.monotonic() - started, 1),
        "speech_detected": bool(turns),
    }


def main(argv=None):
    offline_environment()
    logging.getLogger("faster_whisper").setLevel(logging.ERROR)
    args = parser().parse_args(argv)
    try:
        tracks = resolve_tracks(args)
        if args.inspect:
            print(json.dumps({"tracks": [inspect_track(track) for track in tracks]}, indent=2))
            return 0
        validate_mode(args, tracks)
        output = validate_output(args.output, tracks)
        print(json.dumps(transcribe(args, tracks, output)), flush=True)
        return 0
    except UserError as error:
        print("ERROR: " + str(error), file=sys.stderr)
    except FileExistsError:
        print("ERROR: Output already exists; choose a new confirmed path.", file=sys.stderr)
    except (FileNotFoundError, NotADirectoryError):
        print("ERROR: A required local input, model, or destination directory is missing.", file=sys.stderr)
    except Exception as error:
        # Third-party exceptions can contain input or model content; avoid dumping them.
        print(
            "ERROR: Local processing failed (" + type(error).__name__
            + "). Check local dependencies and cached model assets; no transcript content was printed.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
