# TTS Pipeline CLI Test Report

## Test Setup
- **Test Text**: `test_short.txt` - "This is a short test sentence for TTS pipeline testing."
- **Fast Config**: `num_candidates: 1`, `max_retries: 0`, `conservative_candidate: enabled: true`
- **Device**: CUDA (automatically detected)

## Test Cases

### Test 1: Default Execution (No Arguments)
**Command**: `python src/main.py`
**Expected**: Run default job with default configuration
**Check**: final audio in  \data\output\default\{doc-name}\final\ produced?
**Status**: 
**Duration**: 
**Issues Found**:

### Test 2: Job Config File from non-default directoriy
**Command**: `python src/main.py data/jobs/job1/job_config.yaml --batch`
**Expected**: Run job1 with job configuration in non-interactive mode
**Check**: final audio produced?
**Status**: 
**Duration**: 
**Issues Found**: 

### Test 3: Multiple Job Config Files from non-default directories
**Command**: `python src/main.py data/jobs/job1/job_config.yaml data/jobs/job2/job_config.yaml --batch`
**Expected**: Batch mode with multiple jobs in non-interactive mode
**Check**: all tasks complete, all final audios produced?
**Status**: 
**Duration**: 
**Issues Found**: 

### Test 4.1: Job Name Search
**Command**: `python src/main.py --job "job1" --mode new`
**Expected**: Find and run job1 and run new task in it
**Check**: final audio produced?
**Status**: 
**Duration**:
**Issues Found**:

### Test 4.2: Job Name Search
**Command**: `python src/main.py --job "job1" --mode latest --add-final`
**Expected**: Find and run job1 latest task with add-final making sure that a new final audio file is produces. Also should NOT have to enquire the user with questions even though --batch mode is not specified.
**Check**: final audio produced?
**Status**: 
**Duration**:
**Issues Found**:

### Test 4.3: Job Name Search
**Command**: `python src/main.py --job "job1" --mode all --add-final`
**Expected**: Find and run all tasks of job1, generating new final audio in each task. Also should NOT have to enquire the user with questions even though --batch mode is not specified.
**Check**: final audio produced in all tasks?
**Status**: 
**Duration**:
**Issues Found**:

### Test 5: Task Config File Direct with --add-final flag
**Command**: `python src/main.py data/output/job1/task1_config.yaml`
**Expected**: Run specific task configuration, --add-finale insures that a new final audio will be produced
**Check**: additional final audio produced?
**Status**: 
**Issues Found**:

### Test 6: Multiple specific Task Config Files with --add-final flag
**Command**: `python src/main.py data/output/job1/task1_config.yaml data/output/job2/task2_config.yaml --add-final`
**Expected**: runs multiple tasks in series 
**Check**: all final audio files added?
**Status**: 
**Issues Found**:

### Test 7: Parallel Mode
**Command**: `python src/main.py --parallel data/jobs/job1/job_config.yaml data/jobs/job2/job_config.yaml --batch --mode new`
**Expected**: runs seversal tasks in parallel without interaction, and "--mode new" makes sure that a new task with new final audio will be produced, Parallel execution of multiple jobs
**Check**: final audio produced for both jobs while not having taken twice as much time?
**Status**: 
**Duration**:
**Issues Found**:

### Test 8: Help Output
**Command**: `python src/main.py --help`
**Expected**: Show help message including all possible flags
**Status**: 
**Issues Found**: 

### Test 9: Specific Config File
**Command**: `python src/main.py config/test_fast_config.yaml`
**Expected**: Run with specific configuration
**Status**: 
**Issues Found**:


### Test 10: Error Handling - Missing File
**Command**: `python src/main.py nonexistent.yaml`
**Expected**: Error message for missing file
**Status**: 
**Error**: 

### Test 11: Error Handling - Missing Job
**Command**: `python src/main.py --job "nonexistent_job"`
**Expected**: Error message for missing job
**Status**: 
**Error**: 

### Test 12: Mixed Job Configs
**Command**: `python src/main.py data/jobs/job1/job_config.yaml config/mixed_job_config.yaml`
**Expected**: Batch mode with mixed job configurations
**Status**: 

### Test 13: Task Configs from Different Job Directories
**Command**: `python src/main.py data/output/job1/task1_config.yaml data/output/job2/task2_config.yaml data/output/job3/task3_config.yaml`
**Expected**: Batch mode with tasks from different jobs
**Status**: 

### Test 14: Parallel Mode with Task Configs
**Command**: `python src/main.py --parallel data/output/job1/task1_config.yaml data/output/job2/task2_config.yaml`
**Expected**: Parallel execution of specific tasks
**Status**: 
**Issues Found**: 

### Test 15: Mixed Task and Job Configs
**Command**: `python src/main.py data/jobs/job1/job_config.yaml data/output/job2/task2_config.yaml --batch`
**Expected**: Mixed configuration handling
**Status**: 
**Issues Found**: 

## Summary of Issues Found







## Recommendations


### High Priority Fixes


### Additional Improvements