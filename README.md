# Agent skills

Reusable workflows for coding agents. Each skill has a `SKILL.md` with instructions
for the agent, plus any supporting scripts.

## Available skills

- **[transcribe-audio](transcribe-audio/SKILL.md)** — Transcribe local audio with
  Whisper and create a timestamped transcript and concise Markdown meeting notes.
  Supports a single recording or separate participant tracks.

## Use a skill with your agent

Clone or download this repository onto the machine where your agent runs. Use an
agent with local file and shell access, and ask it to read the skill's `SKILL.md`
and follow its instructions. Use the full path to that file if your agent is
working outside this repository.

### transcribe-audio

From the repository root, give your agent a prompt like this, replacing the paths
with your own:

```text
Read transcribe-audio/SKILL.md and use it to transcribe
/recordings/meeting.m4a. Save meeting-transcript.md and meeting-notes.md
to /output/meeting/. Keep audio local. Before processing, ask whether
you may read transcript text in this chat to write notes or whether
all text processing must also stay local.
```

The skill produces:

- `meeting-transcript.md`: a timestamped transcript with speaker labels where
  available.
- `meeting-notes.md`: concise action points and notes covering decisions,
  constraints, and unresolved questions.

For separate participant tracks, provide all file paths, the participant for each
track, and any known start offsets relative to the meeting's beginning. Say whether
multiple files are participant tracks or successive clips.

Audio transcription uses local Whisper models. Reading transcript text in a hosted
agent exposes that text to the hosted assistant; keeping text processing local
requires a suitable local language model. The agent checks the Python environment
and cached models and asks before downloading missing dependencies or model
weights. A mixed recording may also need local speaker diarization models.
