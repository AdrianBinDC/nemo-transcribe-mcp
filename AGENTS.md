# Agents

Build from `docs/SPEC.md`. This repo is the MCP client. It does not load weights or start vLLM.

## This Spark

- Transcribe server: `NEMOTRON_BASE_URL` in `.env`
- Model: `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`
- vLLM 0.32.0, context 65,536, `--max-num-seqs 1`
- Port 8000 is a different model. Do not send audio there.
- Serve script: `/home/abolinger/bin/serve-nemotron-3.5-transcribe.sh`

## Each call

- Normalize to 16 kHz mono 16-bit PCM before upload. One audio file per request.
- `max_tokens = 65536 - ceil(seconds × 12.5) - pad`. This prompt was about 60 tokens. The test used a 512 pad as spare room. Do not send a fixed 16,384.
- `finish_reason: length` means the transcript was cut off.
- Speaker tags are `<spk:0>` … `<spk:7>`. Numbers start over on every request. The model does not write markdown.
- Word times did not return here. The CTC middleware is loaded, but this checkpoint has no `adapter_path`, so a 200 with no `ctc_timestamps` is not word times. Do not write SRT or VTT. For subtitles, use another model.

## Long files

- Cap a piece at about 55 minutes. One `silencedetect` pass, then cut at the longest pause in the last 20% of that window.
- Speaker jobs repeat 10 seconds at each join. Matching those labels across pieces is not done yet.
- A 2 hour 47 minute show took 18 minutes of model time, one request at a time.

## Do not publish

- `tests/audio/` is gitignored. Do not commit audio or transcripts.
- Do not put episode titles, politician names, or partisan show names in GitHub text.

## Not built yet

MCP package, stitching speakers across pieces, jobs, sites other than YouTube, prompt files, init/doctor.
