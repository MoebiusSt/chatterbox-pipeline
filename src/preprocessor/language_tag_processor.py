"""
Language Tag Processor for TTS Pipeline.
Handles parsing and phoneme transformation of language-tagged text.
"""

import re
import unicodedata
import logging
from typing import List, Dict, NamedTuple, Optional, Any
from pathlib import Path

# Import dependencies
import yaml
import ipatok
import panphon.distance
import torch
from pygoruut.pygoruut import Pygoruut
from dp.phonemizer import Phonemizer

logger = logging.getLogger(__name__)


class LanguageTag(NamedTuple):
    """Represents a parsed language tag with its content."""
    original: str
    language: str
    text: str


class LanguageTagProcessor:
    """
    Processes language tags in text using pygoruut (G2P) and DeepPhonemizer (P2G).
    
    Transformation flow:
    1. Parse: [lang="de"]München[/lang] → ("de", "München")
    2. G2P: "München" → "mˈyːnçən" (pygoruut)
    3. IPA Fix: "mˈyːnçən" → "mɹoncen" (custom phoneme-mapping.yaml)
    4. P2G: "mɹoncen" → "moncen" (DeepPhonemizer)
    5. Replace: "Visit moncen today!"
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the language tag processor.
        
        Args:
            config: Configuration dictionary with phoneme_mappings_path
        """
        self.config = config
        
        # Initialize pygoruut
        self.pygoruut = Pygoruut(writeable_bin_dir='')
        
        # Initialize DeepPhonemizer with torch security fix
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
        self.deepphoneme = Phonemizer.from_checkpoint("model_step_140k.pt")
        torch.load = original_load
        
        # Load phoneme mappings
        self.load_phoneme_mappings(config.get('phoneme_mappings_path', 'config/phoneme_mappings.yaml'))
        
        # DeepPhonemizer phone inventory (predefined by model)
        self.MODEL_PHONES = [
            'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 's', 't', 'u', 'v', 'w', 'z', 
            'æ', 'ð', 'ŋ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɝ', 'ɡ', 'ɪ', 'ɫ', 'ɹ', 'ʃ', 'ʊ', 'ʒ', 'θ'
        ]
        
        # Initialize panphon distance calculator
        self.DST = panphon.distance.Distance()
        
        logger.info(f"✅ LanguageTagProcessor initialized with {len(self.MODEL_PHONES)} model phones")

    def find_language_tags(self, text: str) -> List[LanguageTag]:
        """
        Find all language tags in text.
        
        Args:
            text: Input text potentially containing language tags
            
        Returns:
            List of LanguageTag objects
        """
        pattern = r'\[lang="([^"]+)"\](.*?)\[/lang\]'
        tags = []
        
        for match in re.finditer(pattern, text, re.DOTALL):
            language, content = match.groups()
            tags.append(LanguageTag(
                original=match.group(0),
                language=language,
                text=content.strip()
            ))
        
        if tags:
            logger.info(f"Found {len(tags)} language tags: {[f'{tag.language}' for tag in tags]}")
        
        return tags

    def process_text(self, text: str) -> str:
        """
        Main processing function using optimized batch-by-language strategy.
        
        Args:
            text: Input text with language tags
            
        Returns:
            Processed text with language tags replaced by phoneme transformations
        """
        # 1. Find language tags
        tags = self.find_language_tags(text)
        
        if not tags:
            return text
        
        # 2. Group tags by language (optimization from evaluation)
        language_groups: Dict[str, List[LanguageTag]] = {}
        for tag in tags:
            # Normalize language code using mappings
            normalized_lang = self.language_mappings.get(tag.language, tag.language.capitalize())
            if normalized_lang not in language_groups:
                language_groups[normalized_lang] = []
            language_groups[normalized_lang].append(tag)
        
        # 3. Process each language group using segmentation-based processing
        transformations = {}
        for language, tag_group in language_groups.items():
            try:
                logger.info(f"Processing {len(tag_group)} texts for language: {language}")
                
                # Process each tag individually with segmentation
                for tag in tag_group:
                    transformed_text = self.transform_language_tag(tag)
                    transformations[tag.original] = transformed_text
                    logger.debug(f"Transformed: [{tag.language}]{tag.text}[/lang] → {transformed_text}")
                    
            except Exception as e:
                logger.warning(f"Processing failed for language {language}: {e}")
                # Fallback: keep original text for this language group
                for tag in tag_group:
                    transformations[tag.original] = tag.text
        
        # 4. Replace tags in original text (maintain order)
        processed_text = text
        logger.debug(f"Original text: '{text}'")
        logger.debug(f"Transformations: {transformations}")
        
        for tag in tags:
            logger.debug(f"Processing tag: original='{tag.original}', replacement='{transformations.get(tag.original, 'NOT_FOUND')}'")
            if tag.original in transformations:
                old_text = processed_text
                processed_text = processed_text.replace(tag.original, transformations[tag.original])
                logger.info(f"Transformed: [{tag.language}]{tag.text}[/lang] → {transformations[tag.original]}")
                logger.debug(f"Replacement: '{old_text}' → '{processed_text}'")
            else:
                logger.warning(f"Tag '{tag.original}' not found in transformations!")
        
        return processed_text

    def transform_language_tag(self, tag: LanguageTag) -> str:
        """
        Transform individual language tag: Text → IPA → English.
        Uses word-based processing with punctuation tracking to preserve context and punctuation.
        
        Args:
            tag: LanguageTag to transform
            
        Returns:
            Transformed text with preserved punctuation and spacing
        """
        try:
            # Step 1: Parse text into words and punctuation with position tracking
            tokens = self.tokenize_with_punctuation(tag.text)
            logger.debug(f"Tokenized: {tokens}")
            
            # Step 2: Get IPA for the entire text (maintaining full context)
            normalized_lang = self.language_mappings.get(tag.language, tag.language.capitalize())
            ipa_result = self.pygoruut.phonemize(
                language=normalized_lang,
                sentence=tag.text  # Full context!
            )
            ipa_text = str(ipa_result)
            logger.debug(f"IPA from pygoruut: '{ipa_text}'")
            
            # Step 3: Extract words from IPA (remove punctuation for processing)
            ipa_words = self.extract_words_from_ipa(ipa_text)
            logger.debug(f"IPA words: {ipa_words}")
            
            # Step 4: Transform each IPA word to English
            english_words = []
            for ipa_word in ipa_words:
                if ipa_word.strip():  # Skip empty words
                    english_word = self.transform_ipa_word(ipa_word, tag.language)
                    english_words.append(english_word)
                else:
                    english_words.append("")
            
            logger.debug(f"English words: {english_words}")
            
            # Step 5: Reconstruct text with original punctuation and spacing
            result = self.reconstruct_with_punctuation(tokens, english_words)
            logger.debug(f"Reconstructed: '{result}'")
            
            return result
            
        except Exception as e:
            # Fallback: return original text on error
            logger.warning(f"Failed to process [{tag.language}]{tag.text}[/lang]: {e}")
            return tag.text

    def tokenize_with_punctuation(self, text: str) -> List[Dict[str, str]]:
        """
        Tokenize text into words and punctuation, preserving order and spacing.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens with type and text
        """
        import re
        
        # Pattern to match words (including contractions and compound words)
        word_pattern = r"[a-zA-ZäöüÄÖÜßàáâäèéêëìíîïòóôöùúûüÿñç]+(?:[-'][a-zA-ZäöüÄÖÜßàáâäèéêëìíîïòóôöùúûüÿñç]+)*"
        
        tokens = []
        last_end = 0
        
        for match in re.finditer(word_pattern, text):
            # Add non-word content before this word
            if match.start() > last_end:
                non_word_text = text[last_end:match.start()]
                if non_word_text:
                    tokens.append({
                        'type': 'non_word',
                        'text': non_word_text
                    })
            
            # Add word
            tokens.append({
                'type': 'word',
                'text': match.group()
            })
            
            last_end = match.end()
        
        # Add remaining non-word content
        if last_end < len(text):
            non_word_text = text[last_end:]
            if non_word_text:
                tokens.append({
                    'type': 'non_word',
                    'text': non_word_text
                })
        
        return tokens

    def extract_words_from_ipa(self, ipa_text: str) -> List[str]:
        """
        Extract word-level IPA from full IPA text by removing punctuation.
        
        Args:
            ipa_text: Full IPA text with punctuation
            
        Returns:
            List of IPA words
        """
        import re
        
        # Simplified approach: split on common punctuation and whitespace
        # This preserves word boundaries while removing punctuation
        words = re.findall(r'[^\s\.,!?;:()"\'-]+', ipa_text)
        return [word for word in words if word.strip()]

    def transform_ipa_word(self, ipa_word: str, language: str) -> str:
        """
        Transform a single IPA word to English approximation.
        
        Args:
            ipa_word: Single IPA word
            language: Language code
            
        Returns:
            English approximation
        """
        try:
            # Apply custom mappings
            normalized_ipa = self.apply_phoneme_mappings(ipa_word, language)
            
            # Map to DeepPhonemizer phone inventory
            mapped_phones = self.map_phones(normalized_ipa)
            
            # DeepPhonemizer P2G
            if mapped_phones:
                result = self.deepphoneme.phonemise_list([mapped_phones], lang="eng")
                return result.phonemes[0]
            else:
                return ipa_word
                
        except Exception as e:
            logger.warning(f"Failed to transform IPA word '{ipa_word}': {e}")
            return ipa_word

    def reconstruct_with_punctuation(self, tokens: List[Dict[str, str]], english_words: List[str]) -> str:
        """
        Reconstruct text using original punctuation and spacing with transformed words.
        
        Args:
            tokens: Original tokens with punctuation
            english_words: Transformed English words
            
        Returns:
            Reconstructed text
        """
        result = []
        word_index = 0
        
        for token in tokens:
            if token['type'] == 'word':
                # Replace word with English transformation
                if word_index < len(english_words):
                    result.append(english_words[word_index])
                    word_index += 1
                else:
                    result.append(token['text'])  # Fallback
            else:
                # Keep punctuation and spacing as-is
                result.append(token['text'])
        
        return ''.join(result)

    def apply_phoneme_mappings(self, ipa_text: str, language: str) -> str:
        """
        Apply custom phoneme mappings from YAML configuration.
        
        Args:
            ipa_text: IPA text to transform
            language: Language code for language-specific mappings
            
        Returns:
            Transformed IPA text
        """
        # Unicode normalization
        normalized = unicodedata.normalize("NFC", ipa_text)
        
        # Apply global mappings (including R-Sound fixes)
        if hasattr(self, 'global_mappings'):
            for old_phone, new_phone in self.global_mappings.items():
                normalized = normalized.replace(old_phone, new_phone)
        
        # Apply language-specific mappings from YAML
        if hasattr(self, 'custom_mappings') and language in self.custom_mappings:
            for old_phone, new_phone in self.custom_mappings[language].items():
                normalized = normalized.replace(old_phone, new_phone)
        
        return normalized

    def map_phones(self, ipa_text: str) -> str:
        """
        Map IPA phones to DeepPhonemizer phone inventory using panphon distance.
        
        Args:
            ipa_text: IPA text to map
            
        Returns:
            Mapped phone sequence
        """
        phones = ipatok.tokenise(ipa_text)
        mapped = []
        
        for phone in phones:
            # Force r-sounds to English r (critical for quality)
            if phone in ['r', 'ɾ', 'ʁ', 'ʀ', 'ɹ']:
                mapped.append('ɹ')
            else:
                # Find closest phone in DeepPhonemizer inventory
                distances = [
                    self.DST.hamming_feature_edit_distance(phone, model_phone) 
                    for model_phone in self.MODEL_PHONES
                ]
                best_phone = self.MODEL_PHONES[distances.index(min(distances))]
                mapped.append(best_phone)
        
        return "".join(mapped)

    def load_phoneme_mappings(self, mappings_path: str):
        """
        Load phoneme mappings and language mappings from YAML file.
        
        Args:
            mappings_path: Path to YAML file with mappings
        """
        try:
            with open(mappings_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
                self.language_mappings = data.get('language_mappings', {})
                self.global_mappings = data.get('global_mappings', {})
                self.custom_mappings = data.get('custom_mappings', {})
                
                logger.info(f"Loaded phoneme mappings from {mappings_path}")
                logger.info(f"Language mappings: {len(self.language_mappings)} entries")
                logger.info(f"Global mappings: {len(self.global_mappings)} entries")
                logger.info(f"Custom mappings: {len(self.custom_mappings)} languages")
                
        except Exception as e:
            logger.warning(f"Could not load phoneme mappings from {mappings_path}: {e}")
            logger.info("Using fallback mappings")
            
            # Fallback mappings
            self.language_mappings = {
                'de': 'German', 'en': 'English', 'fr': 'French', 'es': 'Spanish',
                'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'ru': 'Russian',
                'ja': 'Japanese', 'zh': 'Chinese', 'ga': 'Irish', 'da': 'Danish'
            }
            self.global_mappings = {
                'r': 'ɹ', 'ɾ': 'ɹ', 'ʁ': 'ɹ', 'ʀ': 'ɹ', 
                'l': 'ɫ', 'g': 'ɡ', 'ʌ': 'ə'
            }
            self.custom_mappings = {}

    def validate_language_support(self, language_code: str) -> bool:
        """
        Validate if a language is supported by pygoruut.
        
        Args:
            language_code: Language code to validate
            
        Returns:
            True if supported, False otherwise
        """
        try:
            # Check if language code is in our mappings
            normalized_lang = self.language_mappings.get(language_code, language_code.capitalize())
            
            # Try to get supported languages from pygoruut
            # This is a simple validation - in practice, pygoruut will fail gracefully
            return normalized_lang in self.language_mappings.values()
            
        except Exception:
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'supported_languages': len(self.language_mappings),
            'global_mappings': len(self.global_mappings),
            'custom_language_mappings': len(self.custom_mappings),
            'model_phones': len(self.MODEL_PHONES)
        } 