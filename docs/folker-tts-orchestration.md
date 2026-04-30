# Folker TTS orchestration notes

This document captures the transferred Folker audio workflow for Cursor Cloud agents. The
source playbook was written for a local OpenClaw-style environment; the intended target is a
two-repository setup:

- `chatterbox-pipeline`: this repository; contains `cbpipe` and TTS defaults.
- `tts-pipeline-jobs`: private job orchestration repository; contains job YAMLs, prompt assets,
  speaker registry, production log, and GitHub Actions workflows for the local self-hosted runner.

## Current environment status

- `TTS_JOBS_GITHUB_TOKEN` is expected to be available in Cursor Cloud sessions.
- The token must have at least read/write access to the private `tts-pipeline-jobs` repository.
- A clone test against `MoebiusSt/tts-pipeline-jobs` returned HTTP 403 in this session, so future
  agents must verify the actual owner/repository name and token permissions before cloning.
- Clone the jobs repository outside this repository, for example:
  `git clone https://x-access-token:${TTS_JOBS_GITHUB_TOKEN}@github.com/<owner>/tts-pipeline-jobs.git ../tts-pipeline-jobs`

## Workflow stages

The production workflow remains AP1-AP7:

1. Discover new WordPress posts via WP-CLI.
2. Extract raw plaintext from DIVI content.
3. Transform raw text to TTS-ready `_prep.txt` with local Ollama/Gemma.
4. Generate the `cbpipe` job config.
5. Render TTS with `cbpipe`.
6. Convert WAV to MP3 and upload to WordPress media.
7. Insert the DIVI audio-player row and mark the post as audio-ready.

Gemma must only handle AP2. WordPress operations, DIVI insertion, config generation, and quality
control remain agent-managed tasks.

## VRAM serialization

The local runner must never run Gemma and `cbpipe` at the same time.

- Gemma (`ollama run gemma4:26b`) can consume roughly 14-15 GB VRAM.
- `cbpipe` TTS rendering can consume roughly 8-14 GB VRAM depending on model.
- GitHub Actions should serialize the stages with explicit dependencies:
  `text-transform` runs first, `tts-render` uses `needs: text-transform`.
- If a single workflow contains multiple articles, process Gemma jobs sequentially and then TTS
  jobs sequentially.

## Suggested jobs repository layout

```text
tts-pipeline-jobs/
  assets/
    prompts/
      folker_de.txt
      folker_en.txt
    speakers_registry.yaml
  jobs/
    pending/folker/<article_slug>.yaml
    running/folker/
    done/folker/
  memory/
    YYYY-MM-DD.md
  tts-production-log.yaml
  workflows/
```

Prompt files and `speakers_registry.yaml` should be copied into the private jobs repository so
they are versioned. Keep the old local `folker-tts-workflow` tree during the transition.

## Job YAML contract

Each article job should carry both the AP2 transform request and the final `cbpipe` config fields:

```yaml
text_transform:
  raw_text: folker/<article_slug>_raw.txt
  prep_text: folker/<article_slug>_prep.txt
  article_lang: de
  tts_model: qwen3
  ollama_model: gemma4:26b

parent: folker/defaults/qwen3.yaml
job:
  name: "folker"
  run_label: "<article-slug>"
input:
  text_file: folker/<article_slug>_prep.txt
```

`job.name` must stay `folker`; `job.run_label` is the article slug. Completed output belongs under
`output/folker/erledigt/`.

## AP2 prompt rules

The AP2 prompt is model-dependent:

- `qwen3`: use `<speaker>` tags only for continuous foreign-language passages longer than three
  words.
- `vibevoice`: do not emit `<speaker>` tags.
- `chatterbox_multi`: keep the base prompt speaker-tag rules.

Use Ollama's OpenAI-compatible endpoint from the self-hosted WSL runner:
`http://localhost:11434/v1/chat/completions`.

## AP2 structure checks

After Gemma creates `_prep.txt`, run lightweight checks before TTS:

- Contains `Einleitung.` and `Einleitung ende.`
- Contains `Ein Text von`
- Contains `Hier endet der Text`
- Contains no LaTeX artifacts such as `$\\rightarrow$` or `$\\text{`
- Contains no escaped backtick speaker tags such as `` `<speaker:` ``

Only rerun Gemma when these checks fail.

## WordPress discovery and tracking

Use WP-CLI for discovery and extraction. The production log is authoritative for skip decisions:

- `produced`: posts already completed.
- `disabled_audio`: posts that have an audio row but no current audio tag.

Before producing a candidate, first check `tts-production-log.yaml`, then server-side grep for
`Zeile - Audio Player` in `post_content`.

Raw article text should be extracted directly from DIVI shortcode content. Do not rely on
`apply_filters`; it does not reliably render DIVI modules in WP-CLI context.

## DIVI insertion rules

- Use library item `373353` for German and `373352` for English.
- Insert the audio rows as direct children of the first `[et_pb_section]`.
- Always create the dual-row structure:
  - Paywall row with Steady code modules for F+ posts.
  - Backup/free row with only the real audio player.
- Never set `module_class` on `et_pb_audio`; presets must provide the classes.
- The replacement/teaser player preset must use only `audio-artikel-note`, never
  `audio-artikel-player audio-artikel-note`.
- Always set `theme_builder_area="post_content"`.
- Remove `template_type="row"` from inserted rows.
- Set the column preset to `_module_preset="default"`.
- Build DIVI dynamic-title tokens with raw `»` and `«` bytes, not JSON unicode escapes.
- Add the WordPress `audio` term by slug, not by numeric ID.

## Memory notes

Agents should keep short dated notes under `memory/` in the jobs repository and append structured
entries to `tts-production-log.yaml` after AP7. The notes should include completed AP stages,
post IDs, speaker/model decisions, produced MP3 URL, DIVI/paywall state, and lessons learned.
