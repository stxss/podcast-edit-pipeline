Placing an episode = six premiere-mcp tool calls (each does a whole step inside Premiere and reports its own ms).
They execute plan.json from `tq/make_plan.py`; every track, template item, label, transform and nest name comes from
`plan.json["spec"]`, which make_plan fills from the show profile (show.json "premiere"). Nothing show-specific lives in
the tools. Run them on a sequence the editor agreed to overwrite (or a duplicate). P = absolute path of <work>/plan.json
(episode_file = the episode config also works; a plan without spec is refused: rerun tq/make_plan.py).
SAVE FIRST and close other long timeline tabs (see the rules below: open timelines make UXP edits fail; it crashed once).
  1. ppro_place_from_plan       {sequence, plan_file: P}     clear spec.clear_tracks, divider pieces, place every clip,
                                strip nest audio, enable/disable flags, check every clip vs plan (ok:true, mismatches:[])
  2. ppro_set_layout            {sequence, plan_file: P}     spec.split_transforms + spec.outro_slide; safe to rerun
  3. ppro_add_video_transitions {sequence, plan_file: P}     spec.layout_transition at plan.switches on the layout tracks
  4. ppro_set_markers_from_json {sequence, plan_file: P}     replaces markers (keeps spec.keep_markers)
  5. ppro_place_intro_outro     {sequence, plan_file: P}     spec.nests (trimmed from plan.intro_nest / outro_nest, level_db)
  6. ppro_audio_transitions     {sequence, plan_file: P, all_edits: true}  2f Constant Power on every edit of spec.audio_tracks (skips existing)
     ppro_audio_transitions     {sequence, plan_file: P}     per-cut fades from plan.audio_fades, all in one call
Check: ppro_read_timeline {sequence, effects, keyframes, transitions, out_file} reads everything back in one call.
To try a changed tool without restarting Claude: `node test/call.mjs tool <name> '<json>'` in premiere-mcp/server.

The write tools put an empty "park" sequence (spec.park_sequence) in front while they edit, then bring the target back
and close the park tab; results carry parked_on.

`09_dialogue_fades.js` (volume-keyframe fades, fallback for audio transitions) is a step script, sent with
`python tq/step.py <episode> 09_dialogue_fades.js` -> ppro_execute_script (__WORK__ = work dir, __SEQ__ = sequence).
A show profile can keep its own fallback scripts in `<show>/premiere/`; step.py finds them by file name.

Rules learned the hard way (UXP API, Premiere 26.5):
- Open timelines: with several long sequences open as timeline tabs (or a long nest in front), UXP objects go stale
  between reading and committing, even inside one lock, for long stretches. One placement crashed Premiere while
  retrying into that. With the extra tabs closed: 3 placements in a row, 0 retries. So: save, close other long timeline
  tabs (ExtendScript Sequence.close() closes a tab), and commit by POSITION: record which item (index on its track),
  then re-read getTrackItems inside the lock right before the commit (no await in between), retry with a growing wait.
  A tool that still fails changed nothing: just rerun it.
- Placing a nested sequence drops an audio item on the chosen audio track even when the nest has no audio, and it
  OVERWRITES the dialogue there: place nests (backgrounds, divider) before the dialogue, then strip their audio.
- A nest piece with in-point 0 has no material before its start: a centered transition at its head comes out 5f one-sided.
- Use ppro.TickTime.createWithTicks(String(8467200000*frames)), not the T() helper, for overwrite/insert.
- Pass the RAW project item (not ClipProjectItem.cast) to createOverwriteItemAction.
- Every await after you create an action can invalidate objects. Collect everything first, then create all actions inside one executeTransaction.
- An action must be created AND committed inside the same lockedAccess. Created in one, committed in another: silently dropped.
- createOverwriteItemAction uses the project item's in/out at the moment it RUNS: set-in/out + overwrite for several clips in
  one transaction all come in full length. So placing = in/out transaction + overwrite transaction per clip (1111 clips: ~17 s
  when the whole loop runs inside one lockedAccess, 24 s with a lock per transaction).
- Script objects (project items, the editor, track items) go stale without any await: after ~35 placements inside one lock,
  or after a long read loop before a commit -> "The script object is no longer valid". Catch it, refetch, resume.
- Removing an EMPTY TrackItemSelection throws "The script object is no longer valid": skip the remove when nothing matched.
- TrackItemSelection.createEmptySelection(cb) returns a boolean: do the remove inside the callback.
- createMoveAction moves clips but NOT their transitions (they get stranded). Don't move clips that carry transitions; use nests.
- No audio transitions in UXP at all. Use the MCP tool ppro_audio_transitions (CEP 'MCP QE Helper', QE DOM).
  It skips edges that already have a transition (QE addTransition REPLACES them - it ate hand-made Custom Fades in a test).
  Don't combine with 09_dialogue_fades.js (volume keyframes): pick one.
- Per-cut fades: use ppro_audio_transitions with edges / plan_file. The old per-fade call (from=t-0.05, to=t+0.05) picks
  the clip STARTING at t and rewrites its END edge, not the cut at t (at a back-to-back cut the crossfade belongs to the
  previous clip's end). edges matches the clip edge at t itself.
- UXP getTrackItems(TRANSITION) returns nulls (count only) on audio AND video tracks. Read transitions via QE:
  qe track getTransitionAt(k) (gaps come back as unnamed entries). Impact Blur Dissolve reads back as "Blur Dissolve".
- Still images (png) report a source in-point of 108000 frames: compare only their timeline range.
- Point params (Position, Anchor Point) come back from getStartValue() as {value:{0:x,1:y}}, not {x,y}.
- The first ExtendScript call after a big UXP edit can take ~20 s (seen once; 0.3-3 s after).
- ExtendScript Sequence.clone() makes "<name> Copy" next to the original: the safe way to get a test duplicate.
- Intro/outro nests are placed trimmed with in/out on the project item (nothing inside moves).
  Nested audio rendered +6.45 dB louder than the same clips loose in a sequence (clips with their own volume changes are
  boosted, a flat 0 dB clip isn't): compensate with spec.nests[].level_db.
- ExtendScript exportAsMediaDirect fails ("Unknown Error") on 26.5; use UXP EncoderManager.exportSequence(IMMEDIATELY).
- exportSequenceFrame needs a native folder path with a trailing separator ("C:\dir\\" or "/Users/x/dir/"), else it
  returns false. The PNG is written after the call returns: wait for the file before reading it.
- ExtendScript clip.remove() is slow (~1-2 per second on long sequences); the MCP call times out but the script keeps
  running. Check progress through UXP (it stays responsive) instead of resending. UXP createRemoveItemsAction removes
  1600 clips in 0.2 s: always clear through UXP.
