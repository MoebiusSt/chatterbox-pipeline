#!/usr/bin/env python3
"""
generate_benchmark_yamls.py

Generates one YAML per speaker for the speaker-benchmark campaign.
Reads the template at _template.yaml and substitutes speaker-specific
placeholders. Writes to ./generated/ alongside the test texts.

Usage:
    python3 generate_benchmark_yamls.py

Output:
    generated/bench_<speaker_id>.yaml    (one per speaker)
    generated/README.md                  (summary + recommended run order)
"""

from pathlib import Path
import re

# All speakers, with the duplicate (stephan_moebius_4) deduplicated.
# Naming rule: anything with "_DE" or "DE_" in the filename is German; everything else is English.
SPEAKERS = [
    "annie_coleman_rothenberg_2_15s.wav",
    "cori_samuel_1_15s.wav",
    "cori_samuel_2_15s.wav",
    "cori_samuel_3_15s.wav",
    "david_barnes_1_15s.wav",
    "david_barnes_2_15s.wav",
    "david-barnes_2_DE-fake_15s.wav",
    "gesine_1_15s.wav",
    "gesine_2_15s.wav",
    "gesine_3_15s.wav",
    "glen_hallstrom_1_15s.wav",
    "glen_hallstrom_2_15s.wav",
    "glen_hallstrom_3_15s.wav",
    "maryjohn_15s.wav",
    "mike_kamp_1_15s.wav",
    "peter_yearsley_1_15s.wav",
    "stephan_moebius_2_15s.wav",
    "stephan_moebius_3_15s.wav",
    "stephan_moebius_4_15s.wav",
    "martina-zimmermann_DE_15s.wav",
    "mike_kamp_2_DE_15s.wav",
    "stefan_backes_DE_15s.wav",
    "stephan_moebius_7_DE_15s.wav",
    "ulrich_joosten_1_DE_15s.wav",
    "ulrich_joosten_2_DE_15s.wav",
]

RUN_DATE = "2026-04-23"  # adjust to your actual run date
OUTPUT_DIR = Path("generated")
TEMPLATE_PATH = Path("config/speaker_bench_template.yaml")


def is_de(filename: str) -> bool:
    """A filename belongs to a German speaker if it contains '_DE' or 'DE_'."""
    return "_DE" in filename or "DE_" in filename


def speaker_id_from_filename(filename: str) -> str:
    """Strip '_15s.wav' suffix to obtain a clean speaker ID for YAML use.

    Examples:
        'peter_yearsley_1_15s.wav'      -> 'peter_yearsley_1'
        'david-barnes_2_DE-fake_15s.wav' -> 'david-barnes_2_DE-fake'
    """
    name = filename
    if name.endswith(".wav"):
        name = name[:-4]
    if name.endswith("_15s"):
        name = name[:-4]
    return name


def safe_run_label_slug(speaker_id: str) -> str:
    """Build a filesystem-safe slug for run_label.

    Keeps hyphens and underscores, strips anything else that might confuse
    downstream tools.
    """
    return re.sub(r"[^A-Za-z0-9_\-]", "_", speaker_id)


def generate_yaml(template: str, speaker_id: str, reference_audio: str,
                  language: str, text_file: str, run_label: str) -> str:
    """Substitute template placeholders. Uses explicit replace rather than
    .format() to avoid issues with braces that may appear elsewhere."""
    out = template
    out = out.replace("__SPEAKER_ID__", speaker_id)
    out = out.replace("__REFERENCE_AUDIO__", reference_audio)
    out = out.replace("__LANGUAGE__", language)
    out = out.replace("__TEXT_FILE__", text_file)
    out = out.replace("__RUN_LABEL__", run_label)
    return out


def main() -> None:
    if not TEMPLATE_PATH.exists():
        raise SystemExit(f"Template not found: {TEMPLATE_PATH}")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    OUTPUT_DIR.mkdir(exist_ok=True)

    generated_files = []
    de_count = 0
    en_count = 0

    for filename in SPEAKERS:
        speaker_id = speaker_id_from_filename(filename)
        language = "de" if is_de(filename) else "en"
        text_file = "test_text_DE.txt" if language == "de" else "test_text_EN.txt"
        run_label = f"{RUN_DATE}_bench_{safe_run_label_slug(speaker_id)}"

        yaml_content = generate_yaml(
            template,
            speaker_id=speaker_id,
            reference_audio=filename,
            language=language,
            text_file=text_file,
            run_label=run_label,
        )

        out_path = OUTPUT_DIR / f"bench_{speaker_id}.yaml"
        out_path.write_text(yaml_content, encoding="utf-8")

        generated_files.append((out_path.name, language, filename))
        if language == "de":
            de_count += 1
        else:
            en_count += 1

    # Write a README with summary and recommended run order
    readme_lines = [
        "# Speaker benchmark campaign",
        "",
        f"Generated {len(generated_files)} per-speaker YAMLs "
        f"({de_count} DE, {en_count} EN).",
        "",
        "## Recommended run order",
        "",
        "Run serially, one task per YAML. Estimated 60-70 minutes per speaker "
        "(~10 candidates × 5-6 chunks × 73 s). Full series ≈ 25-30 hours.",
        "",
        "## Files",
        "",
        "| YAML | Language | Reference audio |",
        "|---|---|---|",
    ]
    for yaml_name, lang, ref_audio in sorted(generated_files):
        readme_lines.append(f"| {yaml_name} | {lang.upper()} | {ref_audio} |")
    readme_lines.extend([
        "",
        "## Shared inputs",
        "",
        "- `test_text_DE.txt` — used by all DE speakers",
        "- `test_text_EN.txt` — used by all EN speakers",
        "",
        "Place both files in the same input directory that cbpipe expects, "
        "then kick off the campaign. After each task finishes, its "
        "`analysis_metrics.json` is ready for inspection.",
    ])
    (OUTPUT_DIR / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Generated {len(generated_files)} YAMLs in {OUTPUT_DIR}/")
    print(f"  DE speakers: {de_count}")
    print(f"  EN speakers: {en_count}")
    print(f"See {OUTPUT_DIR}/README.md for the run plan.")


if __name__ == "__main__":
    main()
