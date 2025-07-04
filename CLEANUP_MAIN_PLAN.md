# CLEANUP PLAN: OOC-phonemizer Files aus main entfernen

## ❌ FILES ZU ENTFERNEN:
```bash
# OOC-phonemizer experimental files that should NOT be in main:
rm src/preprocessor/language_tag_processor.py
rm config/phoneme_mappings_minimal.yaml
rm test_language_tag_integration.py
rm test_phoneme_mapping_comparison.py
rm test_improved_ch_mappings.py
rm phoneme_mapping_comparison_results.csv
```

## 🔄 FILES ZU BEREINIGEN:
- `src/preprocessor/text_preprocessor.py` - Remove Language Tag Processing integration
- `src/pipeline/task_executor/stage_handlers/preprocessing_handler.py` - Remove TextPreprocessor integration
- `config/default_config.yaml` - Remove process_language_tags setting

## ✅ FILES ZU BEHALTEN (Bug-Fixes):
- `src/utils/config_manager.py` - run-label filtering + performance optimization
- `src/pipeline/job_manager/execution_planner.py` - run-label filtering + --mode new fix
- `src/pipeline/job_manager_wrapper.py` - run-label parameter
- `src/pipeline/job_manager/job_manager.py` - run-label parameter

## 🎯 ZIEL:
Main Branch enthält nur:
1. ✅ ExecutionPlanner --mode new Bug-Fix
2. ✅ run-label Filterung für alle Modi  
3. ✅ Performance-Optimierung (filename-based pre-filtering)
4. ❌ KEINE OOC-phonemizer experimental features

## 📋 SCHRITTE:
1. git checkout main
2. Delete OOC-phonemizer files
3. Revert OOC-phonemizer changes in existing files
4. git add -A && git commit -m "Remove experimental OOC-phonemizer features from main"
5. git checkout dev-OOC-phonemizer (continue work there) 