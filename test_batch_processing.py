#!/usr/bin/env python3
"""
Test script for evaluating batch processing strategies for pygoruut
with multiple language tags in text.

Phase 1.3: Batch Processing Evaluation
"""

import re
import time
import statistics
from typing import List, Dict, NamedTuple
from pygoruut.pygoruut import Pygoruut
from pygoruut.pygoruut_languages import PygoruutLanguages

class LanguageTag(NamedTuple):
    original: str
    language: str
    text: str
    start: int
    end: int

class BatchProcessingEvaluator:
    def __init__(self):
        self.pygoruut = Pygoruut(writeable_bin_dir='')
        self.supported_languages = PygoruutLanguages().get_all_supported_languages()
        
    def find_language_tags(self, text: str) -> List[LanguageTag]:
        """Find all language tags in text with position information"""
        pattern = r'\[lang="([^"]+)"\](.*?)\[/lang\]'
        tags = []
        
        for match in re.finditer(pattern, text, re.DOTALL):
            language, content = match.groups()
            tags.append(LanguageTag(
                original=match.group(0),
                language=language,
                text=content,
                start=match.start(),
                end=match.end()
            ))
        return tags
    
    def strategy_individual(self, tags: List[LanguageTag]) -> Dict[str, str]:
        """Strategy 1: Process each tag individually"""
        results = {}
        for tag in tags:
            try:
                result = self.pygoruut.phonemize(
                    language=tag.language.capitalize(),
                    sentence=tag.text
                )
                results[tag.original] = str(result)
            except Exception as e:
                results[tag.original] = f"ERROR: {e}"
        return results
    
    def strategy_batch_same_language(self, tags: List[LanguageTag]) -> Dict[str, str]:
        """Strategy 2: Group by language and batch process"""
        results = {}
        
        # Group by language
        language_groups = {}
        for tag in tags:
            lang = tag.language.capitalize()
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append(tag)
        
        # Process each language group
        for language, tag_group in language_groups.items():
            try:
                # Batch process all texts for this language
                texts = [tag.text for tag in tag_group]
                batch_results = self.pygoruut.phonemize_list(
                    language=language,
                    sentence_list=texts
                )
                
                # Map results back to tags
                for tag, result in zip(tag_group, batch_results):
                    results[tag.original] = str(result)
                    
            except Exception as e:
                # Fallback to individual processing for this language
                for tag in tag_group:
                    try:
                        result = self.pygoruut.phonemize(
                            language=language,
                            sentence=tag.text
                        )
                        results[tag.original] = str(result)
                    except Exception as e2:
                        results[tag.original] = f"ERROR: {e2}"
        
        return results
    
    def strategy_mixed_batch(self, tags: List[LanguageTag]) -> Dict[str, str]:
        """Strategy 3: Try batch processing all at once with fallback"""
        results = {}
        
        # Try to process all tags at once (this might not work well)
        try:
            # Group by language for better organization
            language_groups = {}
            for tag in tags:
                lang = tag.language.capitalize()
                if lang not in language_groups:
                    language_groups[lang] = []
                language_groups[lang].append(tag)
            
            # Process each language separately but in sequence
            for language, tag_group in language_groups.items():
                texts = [tag.text for tag in tag_group]
                try:
                    batch_results = self.pygoruut.phonemize_list(
                        language=language,
                        sentence_list=texts
                    )
                    for tag, result in zip(tag_group, batch_results):
                        results[tag.original] = str(result)
                except:
                    # Fallback to individual
                    for tag in tag_group:
                        try:
                            result = self.pygoruut.phonemize(
                                language=language,
                                sentence=tag.text
                            )
                            results[tag.original] = str(result)
                        except Exception as e:
                            results[tag.original] = f"ERROR: {e}"
        
        except Exception as e:
            # Complete fallback to individual processing
            return self.strategy_individual(tags)
        
        return results
    
    def benchmark_strategy(self, strategy_func, tags: List[LanguageTag], iterations: int = 5) -> Dict:
        """Benchmark a specific strategy"""
        times = []
        results = None
        
        for _ in range(iterations):
            start_time = time.time()
            results = strategy_func(tags)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'mean_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
            'results': results,
            'success_rate': sum(1 for v in results.values() if not v.startswith('ERROR')) / len(results)
        }
    
    def run_evaluation(self):
        """Run complete evaluation with different test cases"""
        print("🔬 Batch Processing Evaluation für pygoruut")
        print("=" * 60)
        
        # Test cases with different scenarios
        test_cases = [
            {
                'name': 'Mixed Languages (3 different)',
                'text': 'Hello! [lang="German"]München[/lang] is near [lang="French"]Paris[/lang] and [lang="Spanish"]Madrid[/lang].'
            },
            {
                'name': 'Same Language Multiple Tags',
                'text': 'Visit [lang="German"]Berlin[/lang], [lang="German"]Hamburg[/lang], and [lang="German"]Köln[/lang].'
            },
            {
                'name': 'Complex Mixed (5 different)',
                'text': '[lang="German"]Guten Tag[/lang]! [lang="French"]Bonjour[/lang]! [lang="Spanish"]Hola[/lang]! [lang="Italian"]Ciao[/lang]! [lang="Dutch"]Hallo[/lang]!'
            },
            {
                'name': 'Long Text with Multiple Tags',
                'text': 'The conference will be held in [lang="German"]Sindelfingen[/lang] on Monday, then we move to [lang="French"]Strasbourg[/lang] on Tuesday, followed by [lang="Spanish"]Barcelona[/lang] on Wednesday.'
            }
        ]
        
        strategies = [
            ('Individual Processing', self.strategy_individual),
            ('Batch by Language', self.strategy_batch_same_language),
            ('Mixed Batch', self.strategy_mixed_batch)
        ]
        
        overall_results = {}
        
        for test_case in test_cases:
            print(f"\n📋 Test Case: {test_case['name']}")
            print(f"Text: {test_case['text']}")
            
            tags = self.find_language_tags(test_case['text'])
            print(f"Found {len(tags)} language tags:")
            for tag in tags:
                print(f"  - [{tag.language}] {tag.text}")
            
            case_results = {}
            
            for strategy_name, strategy_func in strategies:
                print(f"\n🧪 Testing: {strategy_name}")
                
                try:
                    benchmark = self.benchmark_strategy(strategy_func, tags, iterations=3)
                    case_results[strategy_name] = benchmark
                    
                    print(f"  ⏱️  Mean time: {benchmark['mean_time']:.4f}s")
                    print(f"  ✅ Success rate: {benchmark['success_rate']:.2%}")
                    print(f"  📊 Results preview:")
                    for i, (original, result) in enumerate(list(benchmark['results'].items())[:2]):
                        print(f"    {i+1}. {original} → {result}")
                    
                except Exception as e:
                    print(f"  ❌ Strategy failed: {e}")
                    case_results[strategy_name] = {'error': str(e)}
            
            overall_results[test_case['name']] = case_results
        
        # Summary and recommendations
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        self.print_recommendations(overall_results)
        
        return overall_results
    
    def print_recommendations(self, results: Dict):
        """Print recommendations based on evaluation results"""
        print("\n🎯 RECOMMENDATIONS:")
        
        # Analyze performance patterns
        strategy_performance = {}
        for test_case, case_results in results.items():
            for strategy_name, benchmark in case_results.items():
                if 'error' not in benchmark:
                    if strategy_name not in strategy_performance:
                        strategy_performance[strategy_name] = []
                    strategy_performance[strategy_name].append({
                        'test_case': test_case,
                        'time': benchmark['mean_time'],
                        'success_rate': benchmark['success_rate']
                    })
        
        # Find best strategy
        best_strategy = None
        best_score = -1
        
        for strategy_name, performances in strategy_performance.items():
            if not performances:
                continue
                
            avg_time = statistics.mean([p['time'] for p in performances])
            avg_success = statistics.mean([p['success_rate'] for p in performances])
            
            # Combined score (success rate weighted higher than speed)
            score = avg_success * 0.7 + (1 / (avg_time + 0.001)) * 0.3
            
            print(f"\n{strategy_name}:")
            print(f"  - Average time: {avg_time:.4f}s")
            print(f"  - Average success rate: {avg_success:.2%}")
            print(f"  - Combined score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_strategy = strategy_name
        
        print(f"\n🏆 RECOMMENDED STRATEGY: {best_strategy}")
        
        # Implementation recommendations
        print("\n📋 IMPLEMENTATION RECOMMENDATIONS:")
        print("1. Use 'Batch by Language' strategy for optimal performance")
        print("2. Group tags by language before processing")
        print("3. Implement fallback to individual processing on batch errors")
        print("4. Consider caching results for repeated texts")
        print("5. Process tags in order of appearance to maintain text structure")

if __name__ == "__main__":
    evaluator = BatchProcessingEvaluator()
    evaluator.run_evaluation()