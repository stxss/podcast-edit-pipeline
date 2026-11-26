# Podcast edit pipeline (engine)

Turns one raw recording per speaker (each speaker records locally, clap at the start) into a finished Premiere Pro
sequence: cuts, split/solo layout, transitions, markers, intro/outro. This repo is the engine only. Everything about a
particular show lives in its **show profile**, a folder outside the repo:
- `show.json`: speakers, Premiere track layout, template items/nests, transforms, pacing and detector tuning
  (format: `examples/show/show.json`, field list in `docs/show-profile.md`)
- `episodes/<EP>.json`: one config per episode (cuts, chapters, sync, ...)
- `SHOW.md`: the show's own rules (style, cut habits, what the editor wants) - read it before any episode work
- `EDITORIAL.md`: editorial targets for the `editorial` agent (cold open length, chapter count, shorts)
- `learnings.md`: measurements and history for that show; optional `premiere/` fallback step scripts

Per-machine paths live in the gitignored `local.json` (template `local.example.json`): `show_dir` (the default show
profile), `media_root` (raw media; per show under "shows"), premiere-mcp clone, GPU Python, ffmpeg.
`python tq/tqconfig.py <EP>` prints what resolves. A gitignored `CLAUDE.local.md` can import the show's rules
(`@<show_dir>/SHOW.md`) so they load automatically on that machine.
Never put show content (names, episode text, cut lists) or absolute machine paths into this repo.

## Working rules
- Time every automated editing job with the system clock (`Date.now()` inside `ppro_execute_script`) and report it.
- Never delete files (renders, assets, bin items) without asking. Removing clips/markers inside a sequence is fine.
- If an edit would overwrite the editor's manual work on a sequence, ask first or work on a duplicate.

## Tools
- MCP `premiere-pro` (see `.mcp.json`; needs the env var `PREMIERE_MCP_DIR` = the premiere-mcp clone). Premiere must have Window > UXP Plugins > MCP Bridge open.
- Transcription: NVIDIA Nemotron 3.5 Transcribe, verbatim mode (`python tq/asr_nemotron.py <EP>`; NeMo-Speech.cpp in `tools/nemo-speech`, GGUF in the HF cache, `asr_device` in local.json). It won a bake-off (`tq/asr_bakeoff.py`, `tq/asr_score.py`): least missed speech with the most fillers kept. Word times always come from `tq/align.py` (sound), Nemotron's own times for words the aligner can't hear (numbers). Fallback: CrisperWhisper (`transcribe_media`, model `nyrahealth/faster_CrisperWhisper`) + `tq/fill_holes.py`.
- `tq/align.py` / `tq/verify_cuts.py` need torch + torchaudio: run them with `gpu_python` from local.json.
- Scripts in `tq/` take an episode: a config path or a bare name (`EP01`) looked up in `show_dir/episodes/`. Speakers are the show's list; the first speaker's audio is the time base ("reference audio"); per-speaker files in the work dir are named by speaker id (`<id>.wav`, `<id>_tr_*.json`).
- Premiere placement = six premiere-mcp batch tools that take `plan_file` (absolute path of `<work>/plan.json`) (list in `premiere/_README.md`, read it first: UXP gotchas). They execute `plan.json` from `tq/make_plan.py`; all show-specific names come from `plan.json["spec"]`. Change make_plan.py keys together with premiere-mcp `server/src/plan-tools.js`.

## Pipeline (run as `/edit-episode <EP>`)
1. `python tq/new_episode.py <EP> <raw per speaker, in show order>`; import the raw files; create a video-only and an audio-only subclip per speaker named as in the episode config `premiere.items` (`createSubClipAction(name,in,out,false,{takeVideo,takeAudio})`). Intro/outro come from the nests named in show.json.
2. Sync from the clap: audio onset (5 ms RMS jump) and the video frame where the hands meet (check a frame sheet: the frame-diff peak is the swing, contact is 1-2 frames later; seek once and count frames, -ss per clip isn't frame-exact on HEVC). Write `sync.shift_s` (seconds to add to each speaker's clock to reach the reference clock) and `sync.offsets_frames` (`<id>V`, `<id>A`: source frame offsets vs reference audio) to the config.
3. Extract 16 kHz mono `<id>.wav` per speaker, then `python tq/asr_nemotron.py <EP>` (writes `<id>_tr_nemotron.json`; set `transcripts_unaligned`). Set `mic_threshold_db` per speaker from a bleed check (mic level while only the other speakers talk vs own words).
4. `python tq/tokens.py <EP>` (indexed tokens: words, fillers, repeats, silence, noise), then the editorial pass (subagent `editorial`) on `tokens.txt`; it answers manual cuts as token ranges, `tq/tokens.py <EP> --cuts` turns them into seconds. Write show start/end, manual cuts, cold open, chapters and shorts into the config.
5. `python tq/run_post.py <EP>`: align, build_edl, detect, verify_cuts, make_plan, fades, tokens and check_render (the whole edit rendered and re-transcribed, differences tied to joins), independent steps in parallel, every step timed.
6. Review panel: `python tq/review.py <EP>` (run in the background) -> http://127.0.0.1:8790. The editor keeps/undoes EDL cuts, ticks detector cuts/mutes (high pre-ticked), draws extra cut/mute sections (+ cut / + mute or C / M, Del deletes), edits cold open/chapters/shorts, Save -> review.json + config. Then rerun build_edl (applies review.json; `--no-review` ignores it).
7. Premiere (save first, close other long timeline tabs), one call per step with `{sequence, plan_file}`: `ppro_place_from_plan` (must return ok:true), `ppro_set_layout`, `ppro_add_video_transitions`, `ppro_set_markers_from_json`, `ppro_place_intro_outro`; then `ppro_audio_transitions {sequence, plan_file, all_edits: true}` (2f Constant Power on the dialogue tracks; `premiere/09_dialogue_fades.js` only as fallback), then `ppro_audio_transitions {sequence, plan_file}` for the per-cut fades in `plan.json` `audio_fades`. Then save.
8. Verify: export frames (split, solo, dissolve, outro), check colors/transitions from the saved .prproj, check no gate/cut edge lands inside a word.
9. After the editor's own pass: `python tq/compare_editor.py <EP>` against their timeline (timeline dump + `editor_truth` in the config) -> fold the differences into the show's SHOW.md / show.json.

## Cut rules (engine defaults; a show can tune them in show.json "cut"/"detect" and SHOW.md)
- Pacing: leave pauses < 0.6 s alone; when cutting a pause keep ~0.20 s after the last word, ~0.14 s before the next. No micro-trims.
- Never cut inside a kept word; snap edges to the quietest frame nearby. A filler cut with no quiet valley is worse than the filler: skip it.
- Cut: um/uh, stutters, false starts ("And-", "I-", "So", "you know", "I mean" before the real start), restarts (keep the clean take), coughs, breaths that stick out, clicks/lip smacks, background noise.
- "like": cut only when repeated, stuttered or a pure filler; keep it when it belongs to the sentence. Keep most natural "I, I" repeats; don't over-cut.
- Off-screen speaker: their backchannels ("yeah", "right") go unless the dialogue needs them.
- A mic that hears the other speakers (no headphones): gate each mic to its own speaker's speech (build_edl does this).
- Be careful with manual restart cuts: prefer cutting less.

## Layout (from show.json "premiere")
- Split by default; solo when one speaker holds the floor >= `cut.solo_min_s` of kept time. `layout_transition` at every layout switch on all layout tracks.
- Tracks, backgrounds, divider, split transforms, labels, intro/outro nests and the outro overlay window are all show settings; `docs/show-profile.md` lists them.
