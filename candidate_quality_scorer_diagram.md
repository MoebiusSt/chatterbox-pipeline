graph TD
    CONFIG["Config YAML<br/>validation:<br/>  similarity_threshold: 0.8<br/>  min_quality_score: 0.75<br/>  whisper_model: 'small'"]
    
    CONFIG --> TTS_GEN["TTS Generation"]
    TTS_GEN --> CANDIDATES["Audio Candidates<br/>(N candidates per chunk)"]
    
    CANDIDATES --> WHISPER_VAL["WhisperValidator"]
    WHISPER_VAL --> TRANSCRIBE["Whisper Transcription<br/>Audio → Text"]
    TRANSCRIBE --> W_SIMILARITY["Simple Similarity<br/>Token-based comparison"]
    W_SIMILARITY --> W_QUALITY["Whisper Quality Score<br/>similarity(70%) + completeness(30%)<br/>(length_score removed)"]
    
    W_QUALITY --> W_VALID{"Is Valid?<br/>similarity ≥ similarity_threshold<br/>AND quality ≥ min_quality_score"}
    
    W_VALID --> FUZZY_MATCHER["FuzzyMatcher<br/>(if used)"]
    FUZZY_MATCHER --> F_SIMILARITY["Advanced Similarity<br/>ratio|partial|token|levenshtein|sequence"]
    
    F_SIMILARITY --> QUALITY_SCORER["QualityScorer"]
    
    W_QUALITY --> QUALITY_SCORER
    
    QUALITY_SCORER --> CALC_COMPONENTS["Calculate Score Components<br/>"]
    
    CALC_COMPONENTS --> SIMILARITY_SCORE["Similarity Score<br/>FuzzyMatcher OR WhisperValidator<br/>Range: 0.0-1.0"]
    CALC_COMPONENTS --> LENGTH_SCORE["Length Score<br/>Text length comparison<br/>completeness(70%) + word_ratio(30%)"]
    CALC_COMPONENTS --> VALIDATION_SCORE["Validation Score<br/>base_score + bonus(0.1) - time_penalty"]
    CALC_COMPONENTS --> PENALTY_SCORE["Penalty Score<br/>errors(0.3) + slow(0.2) + empty(0.4) + low_sim(0.2)"]
    
    SIMILARITY_SCORE --> COMBINE["Combine Scores<br/>WEIGHTED_AVERAGE Strategy"]
    LENGTH_SCORE --> COMBINE
    VALIDATION_SCORE --> COMBINE
    PENALTY_SCORE --> COMBINE
    
    COMBINE --> FINAL_FORMULA["Final Quality Score<br/>weighted_sum × (1.0 - penalty_score)<br/><br/>weights:<br/>similarity: 50%<br/>length: 40%<br/>validation: 10%"]
    
    FINAL_FORMULA --> FINAL_SCORE["Final Quality Score<br/>Range: 0.0-1.0"]
    
    FINAL_SCORE --> RANKING["Rank Candidates<br/>by quality_score (DESC)<br/>Tie-breaker: shorter audio"]
    RANKING --> BEST_CANDIDATE["Select Best Candidate"]
    
    %% Styling
    classDef configStyle fill:#e1f5fe
    classDef whisperStyle fill:#fff3e0
    classDef fuzzyStyle fill:#f3e5f5
    classDef scorerStyle fill:#f1f8e9
    classDef finalStyle fill:#e8f5e8
    classDef formulaStyle fill:#fff8e1
    
    class CONFIG configStyle
    class WHISPER_VAL,TRANSCRIBE,W_SIMILARITY,W_QUALITY,W_VALID whisperStyle
    class FUZZY_MATCHER,F_SIMILARITY fuzzyStyle
    class QUALITY_SCORER,CALC_COMPONENTS,SIMILARITY_SCORE,LENGTH_SCORE,VALIDATION_SCORE,PENALTY_SCORE,COMBINE scorerStyle
    class FINAL_SCORE,RANKING,BEST_CANDIDATE finalStyle
    class FINAL_FORMULA formulaStyle