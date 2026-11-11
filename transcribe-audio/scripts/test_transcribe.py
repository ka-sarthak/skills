"""Behavior checks use synthetic text and mocked models; no downloads or recordings."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location("local_transcribe", Path(__file__).with_name("transcribe.py"))
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def segment(text, start=0.0, end=None):
    end = start + 0.8 if end is None else end
    return SimpleNamespace(
        start=start, end=end, text=text,
        words=[SimpleNamespace(start=start, end=end, word=text)],
    )


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a = self.root / "a.wav"
        self.b = self.root / "b.wav"
        self.a.write_bytes(b"synthetic input a")
        self.b.write_bytes(b"synthetic input b")
        self.output = self.root / "transcript.md"

    def args(self, *extras):
        return module.parser().parse_args(["--input", str(self.a), *extras])

    def runtime(self, results=None, failure=None):
        backend = ModuleType("faster_whisper")
        backend.__path__ = []
        backend.WhisperModel = Mock(return_value=object(), side_effect=failure)
        pipeline = Mock()
        info = SimpleNamespace(duration=3.0, duration_after_vad=1.0)
        pipeline.transcribe.side_effect = [
            (iter(rows), info) for rows in (results or [[segment("Private alpha.")]])
        ]
        backend.BatchedInferencePipeline = Mock(return_value=pipeline)
        decoder = ModuleType("faster_whisper.audio")
        decoder.decode_audio = Mock(return_value=[0.0] * 10)
        return backend, decoder, pipeline

    def run_main(self, args, backend, decoder):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {"faster_whisper": backend, "faster_whisper.audio": decoder}):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = module.main(args)
        return status, out.getvalue(), err.getvalue()

    def test_participant_offsets_names_and_private_output_without_stdout_content(self):
        backend, decoder, pipeline = self.runtime([
            [segment("Private alpha.")], [segment("Secret beta.")],
        ])
        args = [
            "--input", str(self.a), "--speaker", "Alice", "--offset", "10",
            "--input", str(self.b), "--speaker", "Bob", "--offset", "0",
            "--output", str(self.output),
        ]
        status, out, err = self.run_main(args, backend, decoder)
        self.assertEqual((status, err), (0, ""))
        content = self.output.read_text()
        self.assertLess(content.index("Bob"), content.index("Alice"))
        self.assertIn("00:00:10", content)
        self.assertIn("Private alpha.", content)
        self.assertIn("Secret beta.", content)
        self.assertNotIn("Private alpha.", out)
        self.assertNotIn("Secret beta.", out)
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o600)
        self.assertEqual(self.a.read_bytes(), b"synthetic input a")
        self.assertTrue(backend.WhisperModel.call_args.kwargs["local_files_only"])
        self.assertTrue(pipeline.transcribe.call_args.kwargs["vad_filter"])
        self.assertTrue(pipeline.transcribe.call_args.kwargs["word_timestamps"])
        before = self.output.read_bytes()
        status, _, _ = self.run_main(args, backend, decoder)
        self.assertEqual(status, 1)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(backend.WhisperModel.call_count, 1)

    def test_participant_words_are_merged_before_forming_turns(self):
        for index, (bob_start, bob_end, alice_offset, bob_offset, starts) in enumerate([
            (0.9, 1.5, 0, 0, ["00:00:00", "00:00:00", "00:00:01"]),
            (0, 0.6, 2, 3, ["00:00:02", "00:00:03", "00:00:03"]),
            (0.4, 1.2, 0, 0, ["00:00:00", "00:00:00", "00:00:01"]),
        ]):
            with self.subTest(bob_start=bob_start, offsets=(alice_offset, bob_offset)):
                alice = segment("First. Second.", end=2.4)
                alice.words = [
                    segment("First.", end=0.8).words[0],
                    segment(" Second.", 1.6, 2.4).words[0],
                ]
                backend, decoder, _ = self.runtime([
                    [alice], [segment("Reply.", bob_start, bob_end)],
                ])
                output = self.root / f"interleaved-{index}.md"
                status, out, err = self.run_main([
                    "--input", str(self.a), "--speaker", "Alice", "--offset", str(alice_offset),
                    "--input", str(self.b), "--speaker", "Bob", "--offset", str(bob_offset),
                    "--output", str(output),
                ], backend, decoder)
                self.assertEqual((status, err), (0, ""))
                rows = [line for line in output.read_text().splitlines() if line.startswith("**[")]
                self.assertEqual([line.split("] ", 1)[1] for line in rows], [
                    "Alice:** First.", "Bob:** Reply.", "Alice:** Second.",
                ])
                for row, start in zip(rows, starts):
                    self.assertTrue(row.startswith(f"**[{start}–"))
                summary = json.loads(out.splitlines()[-1])
                self.assertEqual((summary["turns"], summary["alignment_units"]), (3, 3))

    def test_english_is_default_and_explicit_languages_are_preserved(self):
        for index, (extras, expected) in enumerate([
            ([], "en"), (["--language", "auto"], None), (["--language", "de"], "de"),
        ]):
            with self.subTest(language=expected):
                backend, decoder, pipeline = self.runtime()
                status, _, err = self.run_main([
                    "--input", str(self.a), "--single-speaker",
                    "--output", str(self.root / f"language-{index}.md"), *extras,
                ], backend, decoder)
                self.assertEqual((status, err), (0, ""))
                self.assertEqual(pipeline.transcribe.call_args.kwargs["language"], expected)

    def test_mixed_recording_with_confirmed_label_mapping(self):
        backend, decoder, _ = self.runtime()
        mapping = self.root / "labels.json"
        mapping.write_text(json.dumps({"SPEAKER_00": "Alice"}))
        with patch.object(module, "diarize", return_value=[(0, 3, "SPEAKER_00")]):
            status, out, err = self.run_main([
                "--input", str(self.a), "--diarization-model", str(self.root),
                "--speaker-map", str(mapping), "--output", str(self.output),
            ], backend, decoder)
        self.assertEqual((status, err), (0, ""))
        self.assertIn("Alice", self.output.read_text())
        self.assertNotIn("Private alpha.", out)

    def test_explicit_unattributed_mode_never_invents_a_named_speaker(self):
        backend, decoder, _ = self.runtime()
        status, out, err = self.run_main([
            "--input", str(self.a), "--unattributed", "--output", str(self.output),
        ], backend, decoder)
        self.assertEqual((status, err), (0, ""))
        content = self.output.read_text()
        self.assertIn("Unassigned speaker", content)
        self.assertIn("unattributed", content)
        self.assertNotIn("Private alpha.", out)

    def test_unattributed_recording_cannot_be_labeled_as_a_known_single_speaker(self):
        args = self.args("--unattributed", "--speaker", "Alice")
        with self.assertRaises(module.UserError):
            module.validate_mode(args, module.resolve_tracks(args))

    def test_single_input_does_not_silently_assume_one_speaker(self):
        args = self.args()
        with self.assertRaises(module.UserError):
            module.validate_mode(args, module.resolve_tracks(args))

    def test_named_tracks_cannot_also_use_diarization(self):
        args = self.args(
            "--speaker", "Alice", "--input", str(self.b), "--speaker", "Bob",
            "--diarization-model", str(self.root),
        )
        with self.assertRaises(module.UserError):
            module.validate_mode(args, module.resolve_tracks(args))

    def test_offsets_must_be_explicit_for_each_track_and_finite(self):
        for extras in [
            ["--input", str(self.b), "--offset", "3"],
            ["--offset", "nan"], ["--offset", "-1"],
        ]:
            with self.subTest(extras=extras), self.assertRaises(module.UserError):
                module.resolve_tracks(self.args(*extras))

    def test_duplicate_inputs_rejected(self):
        with self.assertRaises(module.UserError):
            module.resolve_tracks(self.args("--input", str(self.a)))

    def test_alignment_fallback_preserves_unaligned_words(self):
        data = segment("Keep this missing word.")
        data.words = [SimpleNamespace(start=0, end=1, word="Keep")]
        words = module.segment_words(data, "Alice", 5)
        self.assertEqual(words[0].text, data.text)
        self.assertEqual(words[0].start, 5)
        self.assertEqual(module.make_turns(words)[0].text, data.text)

    def test_turns_preserve_long_pauses_and_changes_of_speaker(self):
        U = module.Utterance
        words = [
            U(0, 1, "One", "Alice"), U(1.1, 2, " two.", "Alice"),
            U(12, 13, "Three.", "Alice"), U(13, 14, "Four.", "Bob"),
        ]
        turns = module.make_turns(words)
        self.assertEqual([t.text for t in turns], ["One two.", "Three.", "Four."])

    def test_aligned_pieces_preserve_original_spacing(self):
        for pieces in [
            ["Open", "AI", "'s", " demo."],
            ["We", "'re", " ready", ",", " Alice", "."],
            ["  Hello ", "world.  "],
            ["Hello", " ", "world."],
            ["你", "好"],
        ]:
            with self.subTest(pieces=pieces):
                data = segment("".join(pieces))
                data.words = [
                    SimpleNamespace(start=i / 10, end=(i + 1) / 10, word=text)
                    for i, text in enumerate(pieces)
                ]
                turns = module.make_turns(module.segment_words(data, "Alice", 0))
                self.assertEqual([turn.text for turn in turns], [data.text.strip()])
                self.assertIn(f"Alice:** {data.text.strip()}", module.render(turns, 1, "supplied"))

    def test_fallback_boundaries_get_only_needed_separators(self):
        for first_fallback, second_fallback, first_text, second_text, expected in [
            (True, True, "First.", "Second.", "First. Second."),
            (True, False, "First.", "Second.", "First. Second."),
            (False, True, "First.", "Second.", "First. Second."),
            (True, True, "Hello", ", world.", "Hello, world."),
            (False, True, "Hello ", "world.", "Hello world."),
            (True, False, "Hello", " world.", "Hello world."),
        ]:
            with self.subTest(fallbacks=(first_fallback, second_fallback), expected=expected):
                first, second = segment(first_text), segment(second_text, 0.8)
                if first_fallback:
                    first.words = []
                if second_fallback:
                    second.words = []
                words = module.segment_words(first, "Alice", 0) + module.segment_words(second, "Alice", 0)
                self.assertEqual([turn.text for turn in module.make_turns(words)], [expected])

    def test_mixed_overlap_and_uncovered_words_are_not_guessed(self):
        intervals = [(0, 2, "SPEAKER_00"), (1, 3, "SPEAKER_01")]
        self.assertEqual(module.speaker_for_word(0, 0.5, intervals, {"SPEAKER_00": "Alice"}), "Alice")
        self.assertEqual(module.speaker_for_word(1, 1.5, intervals, {}), "Uncertain speaker")
        self.assertEqual(module.speaker_for_word(4, 5, intervals, {}), "Unassigned speaker")

    def test_existing_or_symlink_outputs_are_not_overwritten(self):
        target = self.root / "existing.md"
        target.write_text("preserve")
        self.output.symlink_to(target)
        with self.assertRaises(module.UserError):
            module.validate_output(self.output, [module.Track(self.a, "Alice")])
        with self.assertRaises(FileExistsError):
            module.write_private(self.output, "replace")
        self.assertEqual(target.read_text(), "preserve")

    def test_backend_failure_does_not_expose_content_or_write_output(self):
        backend, decoder, _ = self.runtime(failure=RuntimeError("confidential meeting phrase"))
        status, out, err = self.run_main([
            "--input", str(self.a), "--single-speaker", "--output", str(self.output),
        ], backend, decoder)
        self.assertEqual(status, 1)
        self.assertNotIn("confidential meeting phrase", out + err)
        self.assertFalse(self.output.exists())

    def test_offline_flags_override_inherited_online_settings(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "PYANNOTE_METRICS_ENABLED": "1"}):
            module.offline_environment()
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["PYANNOTE_METRICS_ENABLED"], "0")

    def test_no_speech_is_reported_without_inventing_content(self):
        for index, rows in enumerate([[], [segment("   ")]]):
            with self.subTest(rows=rows):
                backend, decoder, _ = self.runtime([rows])
                output = self.root / f"empty-{index}.md"
                status, out, err = self.run_main([
                    "--input", str(self.a), "--single-speaker", "--output", str(output),
                ], backend, decoder)
                self.assertEqual((status, err), (0, ""))
                self.assertFalse(json.loads(out.splitlines()[-1])["speech_detected"])
                self.assertIn("No intelligible speech", output.read_text())


if __name__ == "__main__":
    unittest.main()
