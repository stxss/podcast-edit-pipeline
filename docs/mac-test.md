# Testing on a Mac (Apple Silicon)

Nothing here has run on a Mac yet. This walks through it once; note what fails and send it back.

1. Install the basics (Terminal):
   ```
   brew install python ffmpeg node
   ```
2. Clone both repos next to each other, then in `podcast-edit-pipeline`:
   ```
   python3 -m pip install -r requirements-mac.txt
   cp local.example.json local.json
   ```
   Edit `local.json`: `show_dir` (copy `examples/show` somewhere and point at it, or your real show profile),
   `media_root`, `premiere_mcp`. Leave `gpu_python` and `whisper_python` empty (one Python does everything).
3. Run the checker and fix what it says:
   ```
   python3 tq/doctor.py
   ```
4. Transcription speed and quality (the main unknown on a Mac). Put two 16 kHz mono files `<id>.wav` (one per speaker)
   in an episode work folder with an episode config, then:
   ```
   time python3 tq/asr_whisper.py EP01
   ```
   Write down the time per file. Then `python3 tq/run_post.py EP01 --no-render-check` (alignment + cut list on CPU;
   write down that time too), and `python3 tq/run_post.py EP01` once with the render check.
5. Review panel: `python3 tq/review.py EP01`, open http://127.0.0.1:8790, play a few joins.
6. Premiere: premiere-mcp README (UXP plugin, CEP helper with the macOS commands), `export PREMIERE_MCP_DIR=...` in
   `~/.zshrc`, start `claude` in this folder, `/mcp` should show premiere-pro connected.

Optional, if the Mac has 32 GB+ memory: build NeMo-Speech.cpp with its `metal-asr` preset, download the Nemotron 3.5
Transcribe IQ4_NL + encoder GGUF files, set `nemo_speech`, `nemotron_model` and `asr_device` in local.json and compare
`tq/asr_nemotron.py` against `tq/asr_whisper.py` on the same episode (`tq/asr_score.py` prints the comparison).
