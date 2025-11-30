"""
Korean NLP utilities
"""
import re
from typing import List, Dict, Tuple


class KoreanTextProcessor:
    """Korean text processing utilities"""
    
    def __init__(self):
        """Initialize Korean text processor"""
        self.jamo_initial = ''
        self.jamo_medial = ''
        self.jamo_final = ' '
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Korean text
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except Korean, English, numbers
        text = re.sub(r'[^\w\s-]', '', text)
        
        return text.strip()
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split on Korean sentence endings
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract potential keywords from text
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction based on word frequency
        words = text.split()
        
        # Filter out short words and duplicates
        keywords = []
        seen = set()
        
        for word in words:
            if len(word) >= 2 and word not in seen:
                if self.is_korean(word) or self.is_english(word):
                    keywords.append(word)
                    seen.add(word)
        
        return keywords
    
    def is_korean(self, text: str) -> bool:
        """
        Check if text contains Korean characters
        
        Args:
            text: Input text
            
        Returns:
            True if contains Korean
        """
        return bool(re.search(r'[-]', text))
    
    def is_english(self, text: str) -> bool:
        """
        Check if text contains English characters
        
        Args:
            text: Input text
            
        Returns:
            True if contains English
        """
        return bool(re.search(r'[a-zA-Z]', text))
    
    def decompose_hangul(self, char: str) -> Tuple[str, str, str]:
        """
        Decompose Hangul character into jamo
        
        Args:
            char: Single Hangul character
            
        Returns:
            Tuple of (initial, medial, final) jamo
        """
        if not self.is_korean(char) or len(char) != 1:
            return ('', '', '')
        
        code = ord(char) - 0xAC00
        
        if code < 0 or code > 11171:
            return ('', '', '')
        
        initial = self.jamo_initial[code // 588]
        medial = self.jamo_medial[(code % 588) // 28]
        final = self.jamo_final[code % 28]
        
        return (initial, medial, final.strip())
    
    def get_initial_consonant(self, text: str) -> str:
        """
        Get initial consonant of Korean text
        
        Args:
            text: Korean text
            
        Returns:
            Initial consonant
        """
        if not text or not self.is_korean(text[0]):
            return ''
        
        initial, _, _ = self.decompose_hangul(text[0])
        return initial
    
    def count_words(self, text: str) -> int:
        """
        Count words in text
        
        Args:
            text: Input text
            
        Returns:
            Word count
        """
        # Split by whitespace
        words = text.split()
        return len([w for w in words if w.strip()])
    
    def count_characters(self, text: str, include_spaces: bool = False) -> int:
        """
        Count characters in text
        
        Args:
            text: Input text
            include_spaces: Whether to include spaces
            
        Returns:
            Character count
        """
        if include_spaces:
            return len(text)
        return len(text.replace(' ', ''))
    
    def highlight_text(
        self,
        text: str,
        start: int,
        end: int,
        highlight_char: str = '*'
    ) -> str:
        """
        Highlight portion of text
        
        Args:
            text: Input text
            start: Start position
            end: End position
            highlight_char: Character for highlighting
            
        Returns:
            Text with highlighted portion
        """
        before = text[:start]
        highlighted = text[start:end]
        after = text[end:]
        
        return f"{before}{highlight_char}{highlighted}{highlight_char}{after}"


# Global processor instance
korean_processor = KoreanTextProcessor()


# Convenience functions for backward compatibility
def normalize_text(text: str) -> str:
    """
    Normalize Korean text

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    return korean_processor.normalize_text(text)


def tokenize_korean(text: str) -> List[str]:
    """
    Tokenize Korean text into sentences

    Args:
        text: Input text

    Returns:
        List of sentences (tokens)
    """
    return korean_processor.tokenize_sentences(text)
