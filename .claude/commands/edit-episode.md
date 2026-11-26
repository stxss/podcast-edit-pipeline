---
description: Edit an episode end to end (args: episode name, e.g. EP08)
---
Edit episode $ARGUMENTS following CLAUDE.md and the show profile (`show_dir` in local.json: read SHOW.md first). Start a timer with the system clock (Date.now() in ppro_execute_script) and report the elapsed time at the end.
1. Ask the editor for anything missing (raw file names if not obvious in `<media_root>/$ARGUMENTS/`, whether an existing sequence may be overwritten).
2. Run setup + sync + transcription (runner subagent). Show the measured offsets.
3. Run the editorial subagent on the tokens (cold open, chapters, shorts, manual cuts).
4. `python tq/run_post.py $ARGUMENTS`. Start the review panel (`python tq/review.py $ARGUMENTS`, background) and give the editor the link; wait until they say they saved.
5. build_edl again (applies review.json) -> make_plan -> fades.
6. Premiere (runner subagent): save the project and close other long timeline tabs first (open timelines make UXP edits fail). P = absolute path of <work>/plan.json, one call each (see premiere/_README.md): ppro_place_from_plan {sequence, plan_file: P} (must return ok:true), ppro_set_layout, ppro_add_video_transitions, ppro_set_markers_from_json, ppro_place_intro_outro (all {sequence, plan_file: P}), then ppro_audio_transitions {sequence, plan_file: P, all_edits: true} (2f Constant Power on the dialogue tracks), then ppro_audio_transitions {sequence, plan_file: P} (the per-cut fades set in the panel, plan.audio_fades, one call); save. Report each tool's ms.
7. Verify (frames, colors, transitions, words not clipped) and report: final length, cuts, layout runs, anything left for the editor.
