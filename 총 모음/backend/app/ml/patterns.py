"""
Rule-based pattern matching for euphemism detection
"""
import re
from typing import List, Dict


class PatternMatcher:
    """Regular expression based pattern matcher for Korean euphemisms"""

    def __init__(self):
        """Initialize pattern matchers with comprehensive patterns"""
        self.patterns = {
            #    ()
            'company_anonymized': r'[A-Z-]\s*\s*(||||)',
            'initial_company': r'[A-Z]\s*(||||||||||||)',
            'company_abbreviated': r'[-]{1,2}\s*(||||||||)',

            #    ()
            'person_initial': r'[A-Z-]\s*|[-]{1,2}',
            'person_anonymized': r'[-]{1,2}\s*\s*|\s*[-]{1,2}\s*',
            'person_title': r'(|)\s*[-]+\s*|[-]+\s*|[-]+\s*',

            # /  ()
            'country_reference': r'\s*|\s*|\s*|\s*|\s*',
            'country_initial': r'[A-Z]\s*|[-]\s*',
            'location_vague': r'\s*|\s*|\s*|\s*',
            'neighbor_country': r'(|||)\s*(|)|\s*',

            # / 
            'industry_reference': r'\s*|\s*|\s*|\s*',
            'industry_anonymous': r'[-]{2,4}\s*|[A-Z]\s*',

            # /  (  - "", ""    )
            'government_reference': r'\s*||\s*|\s*',
            'government_anonymous': r'(||||||||||||||||||||||||||||||||)\s*(|)',

            #   ()
            'initial_pattern': r'[A-Z]{2,5}(?=[^A-Z]|$)',  # ABC, ABCD 
            'mixed_initial': r'[A-Z]+[0-9]+|[0-9]+[A-Z]+',  # A1, 1A 

            #   
            'symbol_anonymize': r'[]{2,}|[*]{2,}|[X]{2,}|[?]{2,}',
            'blank_pattern': r'[_]{2,}|[-]{3,}',

            #  
            'pronoun_reference': r'\s*(||||)||\s*(||)',

            #  
            'number_anonymize': r'[0-9]+\s*(||)',

            #   ( )
            'wiki_redaction': r'\[\]|\[\]|\[\]',
            'honorific_avoid': r'\s*[-]{2,3}|\s*[-]{2,3}\s*',

            # / 
            'slang_reference': r'[-]{1,2}\s*\s*|[A-Z]\s*\s*',

            #  / 
            'korean_abbreviation': r'[--]{2,6}',  # , , ,  
            'laugh_expression': r'[]{2,}|[]{2,}|[]{2,}',  # , , 
            'yamin_pattern': r'(|||||||.|.|.)',  #  
            'internet_slang_common': r'(||||||||||||)',  #  
            'profanity_disguised': r'(|||||||[.@#]|[.@#])',  #  
            'neologism_common': r'(||||||||[-]+|[-]+)',  # 
            'meme_expression': r'(|+|+|+|+)',  #  

            #  
            'media_anonymous': r'\s*(|||)',
            'broadcast_initial': r'[A-Z]{3,4}\s*(TV|)',

            # /  (Irony/Sarcasm)
            'sarcasm_praise': r'(|||)\s*(|||).{0,10}(?|?|?|)',
            'sarcasm_with_laugh': r'.{2,15}(?|?|)\s*[]{2,}',
            'sarcasm_exaggeration': r'(|||)',
            'sarcasm_fake_thanks': r'(|).{0,5}(?|)\s*[]{2,}',
            'sarcasm_rhetorical': r'(\s*|\s*.{2,6}(|)||)',
            #    (Ironic curse expressions)
            'ironic_blessing_curse': r'(||||||)\s*(?||||||)\s*(|||)',
            'family_curse': r'(?||||).{0,5}(|||)',
            'sarcasm_blessing': r'(|||).{0,3}(|||)',
        }

        # Compile patterns for efficiency
        self.compiled_patterns = {
            name: re.compile(pattern)
            for name, pattern in self.patterns.items()
        }

        #   (    )
        self.pattern_priority = [
            #   (  -   )
            'ironic_blessing_curse',  # "  "  
            'family_curse',
            'sarcasm_blessing',
            #   (   )
            'yamin_pattern',
            'internet_slang_common',
            'profanity_disguised',
            'neologism_common',
            'meme_expression',
            'laugh_expression',
            'korean_abbreviation',
            # /  (+  !)
            'company_anonymized',  # "S ", "S "  
            'initial_company',     # "S", "L"
            'company_abbreviated', # " " ()
            #  
            'person_anonymized',
            'person_initial',
            'person_title',        # " " 
            # / 
            'country_initial',
            'country_reference',
            #  
            'symbol_anonymize',
            'initial_pattern',
            'mixed_initial',
            # ... 
        ]
    
    def detect_patterns(self, text: str) -> List[Dict]:
        """
        Detect euphemism patterns in text with priority-based matching

        Args:
            text: Input text to analyze

        Returns:
            List of detected patterns with metadata
        """
        detected = []
        covered_spans = set()  #     

        #    
        for pattern_name in self.pattern_priority:
            if pattern_name not in self.compiled_patterns:
                continue

            regex = self.compiled_patterns[pattern_name]
            matches = regex.finditer(text)

            for match in matches:
                start, end = match.start(), match.end()

                #     
                if any(start < e and end > s for s, e in covered_spans):
                    continue  #  

                detected.append({
                    'type': pattern_name,
                    'text': match.group(),
                    'start': start,
                    'end': end,
                    'confidence': self._get_confidence(pattern_name),
                    'method': 'pattern_matching'
                })

                covered_spans.add((start, end))

        #    (  )
        remaining_patterns = set(self.compiled_patterns.keys()) - set(self.pattern_priority)

        for pattern_name in remaining_patterns:
            regex = self.compiled_patterns[pattern_name]
            matches = regex.finditer(text)

            for match in matches:
                start, end = match.start(), match.end()

                if any(start < e and end > s for s, e in covered_spans):
                    continue

                detected.append({
                    'type': pattern_name,
                    'text': match.group(),
                    'start': start,
                    'end': end,
                    'confidence': self._get_confidence(pattern_name),
                    'method': 'pattern_matching'
                })

                covered_spans.add((start, end))

        #   
        detected.sort(key=lambda x: x['start'])

        return detected

    def _get_confidence(self, pattern_type: str) -> float:
        """
        Get confidence score based on pattern type

        Args:
            pattern_type: Type of pattern

        Returns:
            Confidence score (0.0 - 1.0)
        """
        #   
        high_confidence = {
            'company_abbreviated', 'initial_company', 'person_anonymized',
            'symbol_anonymize', 'wiki_redaction',
            #   ( )
            'yamin_pattern', 'internet_slang_common', 'profanity_disguised',
            'neologism_common'
        }

        medium_confidence = {
            'company_anonymized', 'person_initial', 'country_reference',
            'industry_reference', 'government_reference',
            #   ( )
            'meme_expression', 'laugh_expression', 'korean_abbreviation'
        }

        #      (Gemini fallback )
        low_confidence_irony = {
            'sarcasm_praise', 'sarcasm_with_laugh', 'sarcasm_exaggeration',
            'sarcasm_fake_thanks', 'sarcasm_rhetorical',
            'ironic_blessing_curse', 'family_curse', 'sarcasm_blessing'
        }

        if pattern_type in low_confidence_irony:
            return 0.55  # Gemini fallback threshold(0.7)  

        if pattern_type in high_confidence:
            return 0.95
        elif pattern_type in medium_confidence:
            return 0.85
        else:
            return 0.75
    
    def get_pattern_category(self, pattern_type: str) -> str:
        """
        Get category for pattern type
        
        Args:
            pattern_type: Type of detected pattern
            
        Returns:
            Category name
        """
        categories = {
            'company_anonymized': 'organization',
            'person_initial': 'person',
            'country_reference': 'location',
            'initial_company': 'organization',
            'industry_reference': 'organization',
            'government_reference': 'organization',
            'location_vague': 'location',
            #   
            'korean_abbreviation': 'internet_slang',
            'laugh_expression': 'internet_slang',
            'yamin_pattern': 'internet_slang',
            'internet_slang_common': 'internet_slang',
            'profanity_disguised': 'internet_slang',
            'neologism_common': 'internet_slang',
            'meme_expression': 'internet_slang',
            #  
            'sarcasm_praise': 'irony',
            'sarcasm_with_laugh': 'irony',
            'sarcasm_exaggeration': 'irony',
            'sarcasm_fake_thanks': 'irony',
            'sarcasm_rhetorical': 'irony',
            #   
            'ironic_blessing_curse': 'irony',
            'family_curse': 'irony',
            'sarcasm_blessing': 'irony',
        }
        return categories.get(pattern_type, 'unknown')
    
    def validate_pattern(self, text: str, pattern_type: str) -> bool:
        """
        Validate if text matches specific pattern
        
        Args:
            text: Text to validate
            pattern_type: Pattern type to check
            
        Returns:
            True if valid pattern
        """
        if pattern_type not in self.compiled_patterns:
            return False
        
        return bool(self.compiled_patterns[pattern_type].search(text))
    
    def extract_initial(self, text: str) -> str:
        """
        Extract initial letter from euphemism
        
        Args:
            text: Euphemism text (e.g., "S ")
            
        Returns:
            Initial letter (e.g., "S")
        """
        match = re.search(r'^([A-Z-])', text)
        if match:
            return match.group(1)
        return ""
