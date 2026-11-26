# Show profile

A show profile is a folder outside this repo (keep it in its own private repo if you like):

```
<show>/
  show.json            speakers, layout, Premiere template, tuning (below)
  episodes/EP01.json   one config per episode (tq/new_episode.py makes it)
  SHOW.md              the show's rules for Claude: speakers, recording setup, the editor's preferences, cut habits
  EDITORIAL.md         targets for the editorial agent: cold open length, chapters, shorts, what to keep
  learnings.md         optional: measurements, history, decisions
  premiere/            optional: fallback step scripts for this show (tq/step.py finds them by file name)
```

Point `show_dir` in `local.json` at it. Scripts then accept a bare episode name (`python tq/build_edl.py EP01`).
A gitignored `CLAUDE.local.md` in the repo root with the line `@<show_dir>/SHOW.md` makes Claude Code load the
show's rules automatically. `examples/show/` is a complete two-speaker example.

## show.json

| key | meaning |
|---|---|
| `id`, `name` | short id (used for a per-show `media_root` in local.json `shows.<id>`) and display name |
| `fps` | sequence frame rate (default 30) |
| `speakers` | list, in order; the FIRST speaker's audio is the time base. Each: `id` (file names, config keys), `name` (shown in the panel and used in subclip names), `label` (Premiere label index for that speaker's subclips), `color` (review panel, optional) |
| `cut` | build_edl tuning: `merge_gap`, `tail`, `head` (pause handling, s), `solo_min_s` (kept seconds one speaker must hold the floor for a solo layout), `solo_enter_s`, `solo_absorb_s`, `keep_repeats_extra` (words said twice on purpose that must never be cut as stutters) |
| `detect` | overrides for the detector tunables (`P` in `tq/detect.py`), e.g. `nm_speakers` |
| `work_pattern` | work folder per episode, relative to media_root (default `{episode}/_work`) |
| `episode_defaults` | extra keys copied into every new episode config |
| `premiere` | everything placed in Premiere (below) |

### premiere

Track names are `V<n>` / `A<n>` (1-based, as in Premiere).

| key | meaning |
|---|---|
| `tracks.solo.<id>` / `tracks.split.<id>` | video track of that speaker's full-frame (solo) and split-screen clip |
| `tracks.audio.<id>` | dialogue track of that speaker |
| `background` | split background item: `key` (plan item key), `name` (project item name), `track`, `in` (0, or `"timeline"` for a long nest whose in-point follows the timeline), `label`, `strip_audio` (nested sequence that drops audio on the dialogue track) |
| `solo_background.<id>` | background item shown while that speaker is solo (same fields) |
| `divider` | item laid over split stretches only (`key`, `name`, `track`), or null |
| `cold_open_audio` | speakers whose audio plays in the cold open (default all) |
| `intro_len` | frames from the end of the cold open to the start of the show |
| `outro_window` | `overlay_from_end` / `slide_back_from_end` (frames): an outro overlay (e.g. a discount code) and the speakers sliding out under it, or null |
| `intro` / `outro` | nest sequence placed trimmed: `nest` (name), `track`, `audio_track`, `in`/`out` (frames inside the nest), `anchor` (frame inside the nest that lands on the cold-open end / the outro overlay start), `offset` (frames), `level_db` (volume on the nest's audio clip; nested audio can render louder), optional `ref_cold_open_frames` / `ref_show_end_s` (only for plan diagnostics) |
| `nest_shell_items` / `nest_shell_tracks` | item names removed from those tracks before the nests are placed (old loose intro/outro pieces) |
| `layout_transition` | video transition at every layout switch: `id` (Premiere match name), `frames`, `alignment` |
| `split_transforms.<id>` | effect settings for that speaker's split clip: list of `{comp, param or param_index, value}`; `comp` = `motion` (the clip's Motion) or `t0`, `t1`, ... (Transform effects appended to the clip) |
| `outro_slide` | `frames`, `rest`, `down`, `up_from` (Transform positions) for the speakers sliding out under the outro overlay, or null |
| `marker_colors` | `chapter`, `short`, `cold_open` color indexes |
| `keep_markers` | marker names never removed |
| `park_sequence`, `template_sequences` | an empty sequence the tools bring to front while editing, and template sequences never used for that |
| `item_names` | subclip name patterns for new episodes, e.g. `{episode} {Name} VIDEO` |

`tq/make_plan.py` resolves all of this into `plan.json["spec"]` for the premiere-mcp batch tools.

## Episode config (episodes/EP01.json)

| key | meaning |
|---|---|
| `episode`, `work`, `raw.<id>` | name, work folder and raw recordings (relative to media_root or absolute) |
| `transcripts`, `transcripts_verbatim`, `transcripts_unaligned` | per speaker, file names in the work folder |
| `sync.shift_s.<id>` | seconds to add to that speaker's clock to get the reference clock (0 for the first speaker) |
| `sync.offsets_frames` | `<id>V` / `<id>A`: source frame offsets of each subclip vs reference audio |
| `show.start` / `show.end` | reference seconds where the show starts and ends |
| `mic_threshold_db.<id>` | mic level that counts as that speaker talking |
| `manual_cuts` | `[a, b, why]` in reference seconds (editorial cuts) |
| `cold_open_frames` | `[[a, b], ...]` reference frames, in playing order |
| `chapters`, `shorts` | `[t, title]` / `[a, b, title]` in reference seconds |
| `premiere.sequence`, `premiere.items` | sequence name; `<id>V` / `<id>A` -> project item (subclip) names |
| `editor_truth` | optional, for `tq/compare_editor.py` / `detect.py --score`: `tracks` (timeline dump), `intro_end`, `video_tracks`, `video_offset`, `audio_tracks` |
