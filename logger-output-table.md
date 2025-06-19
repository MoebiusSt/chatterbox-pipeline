# Logger-Ausgaben Übersicht - TTS Pipeline Enhanced

Diese Tabelle enthält alle Logger-Ausgaben des vollständigen TTS-Pipeline-Programms zur Laufzeit.

## Legende

- **Level**: `S` = Standard (primary), `V` = Verbose-only, `W` = Warning, `E` = Error
- **Empfehlung**: `S` = Standard beibehalten, `V` = In Verbose-Mode, `R` = Entfernen/Reduzieren

## Main Pipeline (src/main.py)

| Logger-Text | Aktueller Level | Empfehlung | Begründung |
|-------------|-----------------|------------|------------|
| "Using device: {device}" | S | S | Wichtige Hardware-Info für User |
| "TTS PIPELINE - TASK-BASED EXECUTION SYSTEM" | S | S | Programm-Header |
| "Execution cancelled by user" | S | S | User-relevante Statusmeldung |
| "Invalid execution plan" | E | E | Kritischer Fehler |
| "Starting batch execution mode" | S | V | Wichtiger Modus-Wechsel |
| "Starting single task execution mode" | S | V | Wichtiger Modus-Wechsel |
| "Detailed report saved: {report_path}" | S | S | Output-Location für User |

## Task Executor (src/pipeline/task_executor.py)

| Logger-Text | Aktueller Level | Empfehlung | Begründung |
|-------------|-----------------|------------|------------|
| "Starting task execution from job: {job_name}" | S (primary) | S | Start-Indikator für User |
| "Task directory: {directory}" | V (verbose) | V | Debug-Info |
| "Current completion stage: {stage}" | S (primary) | S | Wichtiger Status für User |
| "Missing components: {components}" | V (verbose) | V | Debug-Info |
| "🔄 Forcing final audio regeneration" | S (primary) | S | User-relevante Aktion |
| "⚠️ Missing candidates detected - must generate first" | S (primary) | S | Wichtige Warnung |
| "✅ All candidates available - proceeding to assembly only" | S (primary) | S | Wichtiger Status |
| "TTSGenerator initialized with automatic model loading" | S | V | Zu technisch für Standard |
| "Generated {count} text chunks" | S | S | Wichtiger Fortschritt |
| "Loading reference audio from: {path}" | S | V | Zu detailliert |
| "🎙️ Reference audio prepared successfully" | S | S | Wichtiger Erfolg |
| "Starting candidate generation for {count} chunks" | S | S | Wichtiger Fortschritt |
| "Generated candidates for chunk {idx}: {count} files" | V (verbose) | V | Detail-Info |
| "⚠️ Chunk {idx}: Missing {count} candidates, regenerating..." | S | S | Wichtige Warnung |
| "Starting Whisper validation for {count} candidates" | S | S | Wichtiger Fortschritt |
| "Validation Results Summary:" | S | S | Wichtige Zusammenfassung |
| "✅ Starting audio assembly with {count} best candidates" | S | S | Wichtiger Fortschritt |
| "🎵 Final audio saved: {filename}" | S | S | Wichtiges Ergebnis |
| "Task completed successfully in {time} seconds" | S (primary) | S | Wichtige Zusammenfassung |
| "Removed existing final audio file: {file}" | S | V | Debug-Info |
| "⚡ GENERATION PHASE: Processing {total} chunks" | S (primary) | S | Wichtiger Phasen-Header |
| "🎯 CHUNK {num}/{total}" | S (primary) | S | Wichtiger Fortschritt |
| "Text length: {len} characters" | S (primary) | V | Zu detailliert |
| "Preview: \"{preview}\"" | S (primary) | V | Zu detailliert |
| "✓ Candidates already exist for chunk {num}, skipping" | V (verbose) | V | Debug-Info |
| "⚡ Found {existing}/{total} candidates - generating {missing} missing" | S (primary) | S | Wichtiger Status |
| "⚡ Generating candidates..." | S (primary) | S | Wichtiger Fortschritt |
| "❌ Failed to generate missing candidates for chunk {num}" | E | E | Kritischer Fehler |
| "❌ Failed to generate candidates for chunk {num}" | E | E | Kritischer Fehler |
| "❌ Failed to save candidates for chunk {num}" | E | E | Kritischer Fehler |
| "✅ Successfully generated {count} missing candidates" | S (primary) | V | Wichtiger Erfolg |
| "✅ Successfully generated {count} candidates" | S (primary) | S | Wichtiger Erfolg |
| "🚦 VALIDATION PHASE: Processing {count} chunks" | S (primary) | S | Wichtiger Phasen-Header |
| "🎯 CHUNK {num}/{total}" | S (primary) | S | Wichtiger Fortschritt |
| "Candidates to validate: {count}" | S (primary) | V | Zu detailliert |
| "🔍 Validating candidate {num}..." | V (verbose) | V | Debug-Info |
| "✓ Whisper result already exists for candidate {num}" | V (verbose) | V | Debug-Info |
| "✅ Valid - candidate {num} (similarity: {sim}, quality: {qual}, overall: {score})" | V (verbose) | V | Zu detailliert |
| "❌ Invalid - candidate {num} (similarity: {sim}, quality: {qual}, overall: {score})" | V (verbose) | V | Zu detailliert |
| "❌ Whisper validation failed for candidate {num}" | W | W | Wichtige Warnung |
| "✅ Validation complete: {valid}/{total} candidates valid" | S (primary) | S | Wichtiger Status |
| "⚠️ All candidates invalid but maximum retry limit reached" | W | W | Wichtige Warnung |
| "⚠️ All candidates invalid - generating {count} retry candidates" | S (primary) | S | Wichtige Aktion |
| "Highest existing candidate: {highest}, next: {next}, max: {max}" | V (debug) | V | Debug-Info |
| "🔁 Validating {count} retry candidates..." | S (primary) | S | Wichtiger Fortschritt |
| "🔍 Validating retry candidate {num}..." | V (verbose) | V | Debug-Info |
| "Failed to save retry candidates for chunk {num}" | W | W | Wichtige Warnung |
| "✓ Saved {count} retry candidates to disk" | V (verbose) | V | Debug-Info |
| "🎉 Retry success: {new} additional valid candidates found!" | S (primary) | S | Wichtiger Erfolg |
| "😞 Retry complete: Still no valid candidates" | S (primary) | S | Wichtiger Status |
| "Failed to generate retry candidates for chunk {num}" | W | W | Wichtige Warnung |
| "Chunk_{idx}: score {min} to {max}. Best candidate: {best} (score: {score})" | S | V | Zu detailliert |
| "Failed to select best candidate for chunk {idx}: {error}" | W | W | Wichtige Warnung |
| "Assembling audio from {count} selected candidates" | S (primary) | S | Wichtiger Fortschritt |
| "Assembly stage completed successfully" | S | S | Wichtiger Erfolg |
| "Audio cleaning applied" | S | V | Technisches Detail |
| "Custom threshold: {orig} * {factor} = {custom}" | S | V | Technisches Detail |
| "Auto-Editor processing applied" | S | V | Technisches Detail |
| "Auto-Editor not available, skipping" | W | V | Technische Warnung |
| "Auto-Editor processing failed: {error}" | W | V | Technische Warnung |
| "Post-processing applied successfully" | S | V | Technisches Detail |
| "Post-processing failed, using original audio: {error}" | W | W | Wichtige Warnung |

## TTS Generator (src/generation/tts_generator.py)

| Logger-Text | Aktueller Level | Empfehlung | Begründung |
|-------------|-----------------|------------|------------|
| "TTSGenerator initialized on device: {device}" | S (primary) | V | Zu technisch |
| "Conditionals prepared successfully (fresh)" | S | V | Technisches Detail |
| "Conditionals were already prepared (cached)" | V (verbose) | V | Debug-Info |
| "Error preparing conditionals: {error}" | E | E | Kritischer Fehler |
| "Empty text provided for generation" | W | W | Debug-Warnung |
| "Generating audio for text (len={len}): '{text}...'" | V (debug) | V | Debug-Info |
| "Model not loaded - generating mock audio for testing" | W | W | Wichtige Test-Info |
| "Generated audio with shape: {shape}" | V (debug) | V | Debug-Info |
| "Error generating audio for text: {error}" | E | E | Kritischer Fehler |
| "Generating {count} diverse candidates for text" | S | S | Wichtiger Fortschritt |
| "Single candidate mode with conservative enabled" | S | V | Technisches Detail |
| "Candidate {idx} ({type}): exag={exag}, cfg={cfg}, temp={temp}" | S | V | Zu detailliert |
| "Generated candidate {idx}/{total}: duration={dur}s" | S | V | Zu detailliert |

## Whisper Validator (src/validation/whisper_validator.py)

| Logger-Text | Aktueller Level | Empfehlung | Begründung |
|-------------|-----------------|------------|------------|
| "Loading Whisper model '{model}' on device '{device}'..." | S | V | Einmalige Initialisierung |
| "Whisper model loaded successfully" | S | V | Einmalige Initialisierung |
| "Failed to load Whisper model: {error}" | E | E | Kritischer Fehler |
| "Transcription completed in {time}s: '{text}...'" | V (debug) | V | Debug-Info |
| "Transcription failed: {error}" | E | E | Kritischer Fehler |
| "Validation failed: {error}" | E | E | Kritischer Fehler |

## Progress Tracker (src/utils/progress_tracker.py)

| Logger-Text | Aktueller Level | Empfehlung | Begründung |
|-------------|-----------------|------------|------------|
| "{description} Progress: [{bar}] {percent}% ({current}/{total})" | S | S | Wichtiger Fortschritt |
| "Elapsed: {time} \| ETA: {eta}" | S | S | Wichtige Zeitinfo |
| "{description} COMPLETED - {total} items in {time}" | S | S | Wichtige Zusammenfassung |
| "Starting {description}: {total} candidates to validate" | S | V | Nur bei mehreren Items |
| "TEXT COMPARISON" (mit Trennlinien) | S | V | Zu detailliert für Standard |
| "ORIGINAL: {text}" | S | V | Debug-Info |
| "WHISPER RESULT: {text}" | S | V | Debug-Info |
| "Validation failed" | S | S | Wichtiger Fehler |

## Config Manager (src/utils/config_manager.py)

| Logger-Text | Aktueller Level | Empfehlung | Begründung |
|-------------|-----------------|------------|------------|
| "Loading default config: {path}" | S | V | Debug-Info |
| "Loading job config: {path}" | S | V | Debug-Info |
| "Missing required config section: {section}" | E | E | Kritischer Fehler |
| "Missing required job field: {field}" | E | E | Kritischer Fehler |
| "Missing required input field: {field}" | E | E | Kritischer Fehler |
| "Saved task config: {path}" | S | V | Debug-Info |
| "Error reading config {file}: {error}" | W | W | Technische Warnung |
| "Error loading task config {file}: {error}" | W | W | Technische Warnung |

## File Manager (src/utils/file_manager.py)

| Logger-Text | Aktueller Level | Empfehlung | Begründung |
|-------------|-----------------|------------|------------|
| "Loaded input text: {path} ({len} characters)" | S | V | Zu detailliert |
| "Saved {count} chunks to {dir}" | S | V | Debug-Info |
| "Error saving chunks: {error}" | E | E | Kritischer Fehler |
| "Loaded {count} chunks from {dir}" | S | V | Debug-Info |
| "Skipping existing candidate file: {filename}" | V (debug) | R | Debug-Info |
| "Saved new candidate file: {filename}" | V (debug) | V | Debug-Info |
| "Copied candidate file: {filename}" | V (debug) | V | Debug-Info |
| "Saved {count} candidates for chunk {idx} (overwrite mode)" | S | V | Zu detailliert |
| "Saved {count} new candidates (skipped {skipped} existing)" | S | V | Zu detailliert |
| "Error saving candidates for chunk {idx}: {error}" | E | E | Kritischer Fehler |
| "Failed to load audio file {file}: {error}" | W | W | Technische Warnung |
| "Loaded {total} candidates for {count} chunks" | S | V | Debug-Info |
| "Error saving Whisper result: {error}" | E | E | Kritischer Fehler |
| "✓ Synced whisper result to enhanced metrics" | V (debug) | V | Debug-Info |
| "Failed to sync whisper result to enhanced metrics: {error}" | W | W | Technische Warnung |
| "Skipping stale validation data - audio file no longer exists" | V (debug) | V | Debug-Info |
| "Removed stale validation data for chunk {idx}" | V (debug) | V | Debug-Info |
| "Reset chunk {idx} best candidate info due to no valid candidates" | V (debug) | V | Debug-Info |
| "Removed stale selected candidate for chunk {idx}" | V (debug) | V | Debug-Info |
| "🔄 Migrating existing whisper files to enhanced metrics format..." | S | V | Einmalige Migration |
| "No existing whisper files found - no migration needed" | S | V | Einmalige Migration |
| "✅ Migration completed: {count}/{total} whisper files migrated" | S | V | Einmalige Migration |
| "Migration failed: {error}" | E | E | Kritischer Fehler |
| "🧹 Cleaning up individual whisper files after migration..." | S | V | Einmalige Bereinigung |
| "Enhanced metrics not found - skipping cleanup for safety" | W | V | Sicherheitswarnung |
| "✅ Cleanup completed: {count} individual whisper files removed" | S | V | Einmalige Bereinigung |
| "Cleanup failed: {error}" | E | E | Kritischer Fehler |
| "Error saving metrics: {error}" | E | E | Kritischer Fehler |
| "Metrics file not found: {path}" | W | W | Technische Warnung |
| "Error saving final audio: {error}" | E | E | Kritischer Fehler |
| "Error loading final audio {file}: {error}" | E | E | Kritischer Fehler |
| "Error loading audio segment {file}: {error}" | E | E | Kritischer Fehler |
| "Audio file not found: {file}" | W | W | Wichtige Warnung |
| "Loaded {count} audio segments" | S | V | Debug-Info |
| "Chunk {idx}: expected {expected}, found {actual} files" | V (debug) | V | Debug-Info |
| "Missing whisper validation for chunk {idx}, candidate {cand}" | V (debug) | V | Debug-Info |
| "Could not parse candidate index from {file}" | W | W | Technische Warnung |

## Model Cache (src/generation/model_cache.py)

| Logger-Text | Aktueller Level | Empfehlung | Begründung |
|-------------|-----------------|------------|------------|
| "🔄 Loading ChatterboxTTS model for device: {device} (cache miss)" | S | S | Einmalige Initialisierung |
| "♻️ Using cached ChatterboxTTS model for device: {device} (cache hit)" | S | S | Caching-Info |
| "ChatterboxTTS.from_pretrained() does not accept attn_implementation" | V (verbose) | V | Technisches Detail |
| "ChatterboxTTS model loaded successfully for device: {device}" | S (primary) | V | Einmalige Initialisierung |
| "Failed to load ChatterboxTTS model for device {device}: {error}" | E | E | Kritischer Fehler |
| "Returning None - will use mock mode for testing" | W | S | Wichtige Test-Info |
| "🗑️ Clearing ChatterboxTTS model cache" | S | V | Cache-Verwaltung |
| "Preparing conditionals for: {filename}" | S | V | Technisches Detail |
| "Conditionals prepared successfully" | S | V | Technisches Detail |
| "Error preparing conditionals: {error}" | E | E | Kritischer Fehler |
| "Conditionals already prepared for: {filename}" | V (verbose) | V | Caching-Info |
| "Could not calculate hash for {file}: {error}" | W | V | Technische Warnung |
| "🔄 Resetting conditional cache state" | S | V | Cache-Verwaltung |

## Candidate Manager (src/generation/candidate_manager.py)

| Logger-Text | Aktueller Level | Empfehlung | Begründung |
|-------------|-----------------|------------|------------|
| "Candidates will be saved to: {dir}" | S | V | Debug-Info |
| "CandidateManager initialized: max_candidates={max}, max_retries={retries}" | S | S | Initialisierung |
| "🔧 Generating ONLY specific candidates {indices} for chunk {idx}" | S | V | Technisches Detail |
| "✅ Generated specific candidate {idx} for chunk {chunk}" | V (debug) | V | Debug-Info |
| "❌ Failed to prepare candidate {idx} for chunk {chunk}: {error}" | E | E | Kritischer Fehler |
| "🗑️ Deleted old whisper file: {filename}" | V (debug) | V | Debug-Info |
| "✅ Successfully generated {count}/{total} specific candidates" | V (debug) | V | Debug-Info |
| "Starting candidate generation for chunk (length: {len} chars)" | S | V | Zu detailliert |
| "Chunk text preview: '{text}...'" | V (verbose) | S | Debug-Info |
| "Phase 1: Generating {count} normal candidates" | S | V | Technisches Detail |
| "Normal generation failed: {error}" | E | E | Kritischer Fehler |
| "Generated {valid}/{total} valid normal candidates" | S | V | Technisches Detail |
| "Generation completed: {total} candidates ({attempts} attempts), success={success}" | S | V | Zu detailliert |
| "No candidates provided for selection" | W | V | Debug-Warnung |
| "Selected shortest candidate with length: {length}" | V (debug) | V | Debug-Info |
| "Selected first candidate" | V (debug) | V | Debug-Info |
| "Selected random candidate" | V (debug) | V | Debug-Info |
| "Unknown selection strategy: {strategy}, using first" | W | W | Technische Warnung |
| "Processing {count} chunks for candidate generation" | S | V | Debug-Info |
| "Processing chunk {current}/{total}" | S | V | Debug-Info |
| "Failed to generate sufficient candidates for chunk {idx}" | W | W | Wichtige Warnung |
| "Error processing chunk {idx}: {error}" | E | E | Kritischer Fehler |
| "Candidate generation completed: {successful}/{total} chunks successful" | S | S | Wichtige Zusammenfassung |
| "Failed to save candidate {idx} for chunk {chunk}: {error}" | E | E | Kritischer Fehler |
| "Saved candidate to: {filepath}" | V (debug) | V | Debug-Info |
| "Saved candidate to correct structure: {filepath}" | V (debug) | V | Debug-Info |
| "Saved candidate metadata: {path}" | V (debug) | V | Debug-Info |
| "Failed to save candidate metadata: {error}" | E | E | Kritischer Fehler |
| "🔍 Selecting best from {count} validated candidates..." | V (debug) | V | Debug-Info |
| "🎯 Considering {count} {pool} candidates" | V (debug) | V | Debug-Info |
| "✅ Selected candidate with quality={quality}, valid={valid}, duration={dur}" | V (debug) | V | Debug-Info |
