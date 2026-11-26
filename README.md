# Podcast edit pipeline

Turns one raw recording per speaker (each records locally, clap at the start) into a finished Premiere Pro sequence:
cuts (fillers, stutters, restarts, pauses, editorial cuts), split/solo layout, transitions, chapter and short markers,
intro/outro, driven by Claude Code and the premiere-mcp server (separate repo). A local review panel lets the
editor approve or undo every risky cut before anything is placed.

The repo is the engine only. Everything about a show (speakers, Premiere template, track layout, the editor's rules,
episode cut lists) lives in a **show profile** folder outside the repo: see `docs/show-profile.md` and the example in
`examples/show/`. The workflow and cut rules for Claude are in `CLAUDE.md`.

```
tq/          Python: sync helpers, transcripts, cut list (build_edl), detectors, review panel, Premiere plan
premiere/    UXP gotchas (read premiere/_README.md) + a fallback step script
examples/    a two-speaker example show profile
docs/        show profile format
.claude/     /edit-episode command + editorial/runner subagents
```

Runs on Windows and macOS. Premiere Pro 25.6+ (UXP API).

## Setup

After the steps below, `python tq/doctor.py` (Mac: `python3`) checks everything and says how to fix what's missing.

### 1. Prerequisites

| | Windows | macOS |
|---|---|---|
| Python 3.10+ | `winget install Python.Python.3.13` | `brew install python` |
| ffmpeg + ffprobe | `winget install Gyan.FFmpeg` | `brew install ffmpeg` |
| Node.js 22+ (MCP server) | `winget install OpenJS.NodeJS.LTS` | `brew install node` |
| Claude Code | `npm install -g @anthropic-ai/claude-code` | same |

Restart the terminal so the new commands are on PATH.

### 2. Clone both repos side by side

```
git clone <url>/premiere-mcp.git
git clone <url>/<this repo>.git
```

Set up premiere-mcp as its README says: `npm install` in `server/`, the **MCP Bridge** UXP plugin
(install the `.ccx` in `plugin/` or load `plugin/manifest.json` with UXP Developer Tools) and the
**MCP QE Helper** CEP panel (needed for `ppro_audio_transitions` / `ppro_execute_extendscript`):

- Enable unsigned CEP panels once (CSXS.12 for Premiere 25.x; setting the key for several versions is harmless):
  - Windows: `reg add HKCU\Software\Adobe\CSXS.12 /v PlayerDebugMode /t REG_SZ /d 1 /f`
  - macOS: `defaults write com.adobe.CSXS.12 PlayerDebugMode 1`
- Link the panel folder into the CEP extensions folder:
  - Windows (PowerShell): `New-Item -ItemType Junction -Path "$env:APPDATA\Adobe\CEP\extensions\dev.premieremcp.qehelper" -Target "<premiere-mcp>\cep"`
  - macOS: `mkdir -p ~/Library/Application\ Support/Adobe/CEP/extensions && ln -s "<premiere-mcp>/cep" ~/Library/Application\ Support/Adobe/CEP/extensions/dev.premieremcp.qehelper`
- Restart Premiere; Window > Extensions (Legacy) > MCP QE Helper once if it does not start by itself.

### 3. Python packages

**Windows / Linux:**

```
python -m pip install -r requirements.txt           # the editing scripts (numpy)
python -m pip install -r requirements-gpu.txt       # transcription + word alignment (torch, torchaudio, faster-whisper)
```

With an NVIDIA GPU, install the CUDA build of torch first (see the comment in `requirements-gpu.txt`). Without one
everything still runs on the CPU, just slower.

**Mac (Apple Silicon):** one file covers everything:

```
python3 -m pip install -r requirements-mac.txt
```

### Transcription: which engine

The pipeline needs verbatim transcripts (um/uh, repeats and false starts kept: those are what it cuts). Word times
don't need to be perfect: `tq/align.py` fixes them from the sound afterwards.

| machine | engine | command |
|---|---|---|
| any (default) | Whisper large-v3-turbo with a filler prompt: faster-whisper on Windows/Linux, mlx-whisper (GPU) on Apple Silicon | `python tq/asr_whisper.py EP01` |
| NVIDIA GPU, most fillers | NVIDIA Nemotron 3.5 Transcribe via [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) | `python tq/asr_nemotron.py EP01` |
| Mac with 32 GB+, most fillers (untested) | the same Nemotron through NeMo-Speech.cpp's Metal build, IQ4_NL model | `python tq/asr_nemotron.py EP01` with `asr_device` in local.json |

Measured on a real 45-minute, two-mic episode (same audio, then the same forced alignment):

| engine | speech missed (mic 1 / mic 2) | fillers found | time (RTX 5090) |
|---|---|---|---|
| Nemotron 3.5 Transcribe | 3.6 s / 8.0 s | 126 / 44 | ~25 min (with Premiere open) |
| Whisper turbo, faster-whisper's own voice detection | 3.4 s / 6.0 s | 104 / 21 | ~1 min |
| Whisper turbo, `asr_whisper.py` voice detection (what runs on a Mac) | 6.7 s / 9.9 s | 100 / 20 | ~1 min |
| Whisper turbo, no voice detection | 8.5 s / 11.1 s | 86 / 24 | ~1 min |

Whisper misses some fillers; `tq/build_edl.py` also finds fillers from the sound (steady-pitch "uhh" with no word),
which recovers part of the gap. On an Apple Silicon Mac, mlx-whisper turbo is expected to take a few minutes per hour
of audio (published benchmarks; not measured here). CrisperWhisper is not recommended on a Mac (CPU only there) and its
weights are licensed for non-commercial research only. After transcribing, set `transcripts_unaligned` in the episode
config to the new files (`<id>_tr_whisper.json` or `<id>_tr_nemotron.json`).

### 4. Show profile and local settings

Make a show profile (copy `examples/show/` somewhere outside the repo and edit it; `docs/show-profile.md` explains
every field), then:

```
cp local.example.json local.json                   # Windows: copy local.example.json local.json
```

Edit `local.json` (gitignored):

- `show_dir`: your show profile folder (scripts then take a bare episode name, e.g. `EP01`)
- `media_root`: folder holding the raw episode folders (episode configs store paths relative to it)
- `premiere_mcp`: path of the premiere-mcp clone
- `gpu_python`: Python with torch/torchaudio (empty = the Python you run the script with)
- `whisper_python`: Python with faster-whisper for `tq/fill_holes.py` (empty = same)
- `ffmpeg`: ffmpeg executable (`ffmpeg` if on PATH)

Optional: a `CLAUDE.local.md` (gitignored) in the repo root containing `@<show_dir>/SHOW.md` so Claude Code loads the
show's rules automatically. Check the setup with `python tq/tqconfig.py EP01`.
`local.json` is found in the repo root or any parent folder, or at the path in `TQ_LOCAL`.

### 5. MCP server for Claude Code

`.mcp.json` starts the premiere-mcp server from the environment variable `PREMIERE_MCP_DIR`.
Optional: `PREMIERE_MCP_PYTHON` (Python with faster-whisper), `FFMPEG_PATH`, `FFPROBE_PATH`; unset means
premiere-mcp's defaults (`python`/`python3`, `ffmpeg`, `ffprobe` from PATH).

Windows (then open a new terminal):

```
setx PREMIERE_MCP_DIR "C:\path\to\premiere-mcp"
setx PREMIERE_MCP_PYTHON "C:\path\to\python.exe"
```

macOS (add to `~/.zshrc`):

```
export PREMIERE_MCP_DIR="$HOME/path/to/premiere-mcp"
export PREMIERE_MCP_PYTHON="$(which python3)"
```

Start `claude` in this folder, approve the `premiere-pro` server, run `/mcp` to see it connected, and open
Window > UXP Plugins > MCP Bridge in Premiere.

## Per episode

```
python tq/new_episode.py EP01 <raw file per speaker, in show.json order>
```

then in Claude Code `/edit-episode EP01`, or step by step as in `CLAUDE.md` (sync, transcribe, `tq/run_post.py`,
review panel `tq/review.py` on http://127.0.0.1:8790, then the premiere-mcp batch tools).

Episode work folders (`<media_root>/<EP>/_work`: wavs, transcripts, edl/plan/review json) stay outside the repo.

## Publishing your own copy without your private history

If you keep your own fork with private material in its history, `python scripts/publish.py "what changed"` pushes
only the current files as one new commit to a separate remote (default `github`), and refuses if a file is local-only,
looks like a secret, or matches a word in your gitignored `publish-blocklist.txt` (show names, email, paths: one
regular expression per line). `python scripts/publish.py --help` has the one-time setup.

## Notes

- `plan.json["spec"]` carries every show-specific Premiere setting, so the premiere-mcp tools stay generic.
- The analysis code is written for any number of speakers (tested with two); the Premiere layout is split (all speakers) or solo (one speaker).
  Split transforms are per speaker in show.json, so a 3-up split is a matter of settings and a background.
- Mac: `docs/mac-test.md` is a first-run checklist.
- Models in Claude Code: the `editorial` subagent runs on Opus, `runner` on Sonnet.
