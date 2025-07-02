# Candidate Quality Scorer - Architecture Diagram

```mermaid
graph TD
    CONFIG["Config YAML<br/>validation:<br/>  similarity_threshold: 0.8<br/>  min_quality_score: 0.75<br/>  whisper_model: 'small'"]
    
    CONFIG --> TTS_GEN["TTS Generation"]
    TTS_GEN --> CANDIDATES["Audio Candidates<br/>(N candidates per chunk)"]
    
    CANDIDATES --> WHISPER_VAL["WhisperValidator"]
    WHISPER_VAL --> TRANSCRIBE["Whisper Transcription<br/>Audio → Text"]
    TRANSCRIBE --> QUALITY_CALC["QualityCalculator"]
    
    QUALITY_CALC --> W_SIMILARITY["Token-based Similarity<br/>Jaccard similarity<br/>original_tokens ∩ transcribed_tokens<br/>original_tokens ∪ transcribed_tokens"]
    QUALITY_CALC --> W_QUALITY["Whisper Quality Score<br/>similarity(70%) + length_score(30%)<br/>length_score = min(1.0, len_transcription/len_original)"]
    
    W_QUALITY --> W_VALID{"Is Valid?<br/>similarity ≥ similarity_threshold<br/>AND quality ≥ min_quality_score<br/>OR flexible validation logic"}
    
    W_VALID --> QUALITY_SCORER["QualityScorer<br/>(separate from WhisperValidator)"]
    
    W_SIMILARITY --> QUALITY_SCORER
    
    QUALITY_SCORER --> CALC_COMPONENTS["Calculate Score Components<br/>"]
    
    CALC_COMPONENTS --> SIMILARITY_SCORE["Similarity Score<br/>From WhisperValidator OR FuzzyMatcher<br/>Range: 0.0-1.0"]
    CALC_COMPONENTS --> LENGTH_SCORE["Length Score<br/>completeness(70%) + word_ratio(30%)<br/>completeness = min(1.0, len_transcription/len_original)<br/>word_score = 1/(1 + |word_ratio - 1|)"]
    CALC_COMPONENTS --> PENALTY_SCORE["Penalty Score<br/>errors(0.3) + empty(0.4) + low_sim(0.2)"]
    
    SIMILARITY_SCORE --> COMBINE["Combine Scores<br/>WEIGHTED_AVERAGE Strategy"]
    LENGTH_SCORE --> COMBINE
    PENALTY_SCORE --> COMBINE
    
    COMBINE --> FINAL_FORMULA["Final Quality Score<br/>weighted_sum × (1.0 - penalty_score)<br/><br/>weights:<br/>similarity: 65%<br/>length: 35%"]
    
    FINAL_FORMULA --> FINAL_SCORE["Final Quality Score<br/>Range: 0.0-1.0"]
    
    FINAL_SCORE --> RANKING["Rank Candidates<br/>by quality_score (DESC)<br/>Tie-breaker: shorter audio"]
    RANKING --> BEST_CANDIDATE["Select Best Candidate"]
    
    %% Styling
    classDef configStyle fill:#e1f5fe
    classDef whisperStyle fill:#fff3e0
    classDef qualityStyle fill:#f3e5f5
    classDef scorerStyle fill:#f1f8e9
    classDef finalStyle fill:#e8f5e8
    classDef formulaStyle fill:#fff8e1
    
    class CONFIG configStyle
    class WHISPER_VAL,TRANSCRIBE,QUALITY_CALC,W_SIMILARITY,W_QUALITY,W_VALID whisperStyle
    class QUALITY_SCORER,CALC_COMPONENTS,SIMILARITY_SCORE,LENGTH_SCORE,PENALTY_SCORE,COMBINE qualityStyle
    class FINAL_SCORE,RANKING,BEST_CANDIDATE finalStyle
    class FINAL_FORMULA formulaStyle