"""
Main euphemism detection engine coordinating all components

Enhanced 3-Stage Pipeline with Gemini Integration:
1. Pattern matching (rule-based) - Fast, reliable detection
2. Context-aware analysis - Deeper understanding with context keywords
3. Entity resolution + Gemini explanation (when API key available)
"""
import time
import os
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from app.ml.patterns import PatternMatcher
from app.ml.kobert_model import KoELECTRADetector
from app.ml.inference import EntityResolver
from app.core.cache import cached

# Gemini integration (optional)
try:
    from app.ml.gemini_detector import GeminiEuphemismDetector
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)


# Context keywords for entity resolution
CONTEXT_KEYWORDS = {
    "company": {
        "": ["", "", "", "", "OLED", "", "DS"],
        "SK": ["", "", "", "", "5G", ""],
        "LG": ["", "TV", "", "", "", ""],
        "": ["", "", "", "", "", ""],
        "": ["", "SUV", "", "", ""],
        "": ["", "", "AI", "", "", ""],
        "": ["", "", "", "", "", ""],
        "": ["", "", "", ""],
        "": ["", "", "", "", ""],
        "": ["", "", "", "", ""],
    },
    "country": {
        "": ["", "", "", "", "", "", "", ""],
        "": ["", "", "", "", "", ""],
        "": ["", "", "IRA", "", "", ""],
        "": ["", "", "", "", "", ""],
        "": ["", "", "", "", "", ""],
    },
    "person": {
        "": ["", "", "", "", ""],
        "": ["", "", "", "", "", ""],
        "": ["", "", "", "", "", ""],
    }
}


class EuphemismDetector:
    """
    Enhanced detection engine with context-aware entity resolution

    3-Stage Pipeline:
    1. Pattern matching (rule-based) - Fast, reliable detection
    2. Context-aware analysis - Keyword-based entity inference
    3. Entity resolution + Gemini explanation (optional)
    """

    def __init__(self, model_path: str = None, gemini_api_key: str = None):
        """
        Initialize all detection components

        Args:
            model_path: Path to trained KoELECTRA model (optional)
            gemini_api_key: Google Gemini API key (optional)
        """
        self.pattern_matcher = PatternMatcher()
        self.ml_detector = KoELECTRADetector(model_path=model_path)
        self.entity_resolver = EntityResolver()
        self.processing_time = 0.0
        self._is_loaded = True

        # Gemini integration (optional - for AI explanation)
        self.gemini_detector = None
        gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if GEMINI_AVAILABLE and gemini_key:
            try:
                self.gemini_detector = GeminiEuphemismDetector(api_key=gemini_key)
                logger.info("Gemini detector initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini detector: {e}")

        logger.info(f"EuphemismDetector initialized (Gemini: {'enabled' if self.gemini_detector else 'disabled'})")
    
    @cached(ttl=3600)
    def detect_and_resolve(self, text: str, include_explanation: bool = False) -> Dict:
        """
        Complete detection and resolution pipeline

        Args:
            text: Input text to analyze
            include_explanation: Whether to include AI explanation (requires Gemini)

        Returns:
            Detection results with resolved entities
        """
        start_time = time.time()

        # Stage 1: Pattern matching (fast, reliable)
        pattern_detections = self.detect_patterns(text)

        # Stage 2: Context-aware entity inference
        context_enhanced = self._enhance_with_context(pattern_detections, text)

        # Stage 3: Entity resolution with knowledge base
        resolved_entities = self.resolve_entities(context_enhanced, text)

        # Optional: Add AI explanation (if Gemini available)
        if include_explanation and self.gemini_detector:
            resolved_entities = self._add_ai_explanations(resolved_entities, text)

        # Calculate processing time
        self.processing_time = time.time() - start_time

        return {
            'text': text,
            'detections': resolved_entities,
            'total_detected': len(resolved_entities),
            'processing_time': self.processing_time,
            'gemini_enabled': self.gemini_detector is not None,
            'stages': {
                'pattern_matching': len(pattern_detections),
                'context_enhanced': len(context_enhanced),
                'resolved': len(resolved_entities)
            }
        }

    def _enhance_with_context(self, detections: List[Dict], text: str) -> List[Dict]:
        """
        Stage 2: Enhance detections with context keyword analysis

        Args:
            detections: Initial pattern detections
            text: Original text

        Returns:
            Enhanced detections with context-based suggestions
        """
        enhanced = []
        text_lower = text.lower()

        for detection in detections:
            # Get pattern type category
            pattern_type = detection.get('type', '')
            euphemism_text = detection.get('text', '')

            # Determine entity category
            category = self._get_entity_category(pattern_type)

            # Extract initial letter for company/person inference
            initial = self._extract_initial(euphemism_text)

            # Find context-based candidates
            context_candidates = []
            context_keywords_found = []

            if category in CONTEXT_KEYWORDS:
                for entity, keywords in CONTEXT_KEYWORDS[category].items():
                    # Check initial match
                    initial_match = False
                    if initial:
                        # S  → , SK 
                        if initial.upper() == entity[0].upper():
                            initial_match = True
                        #    ( → )
                        elif initial in entity:
                            initial_match = True

                    # Check keyword match in context
                    keyword_matches = [kw for kw in keywords if kw in text_lower]

                    if keyword_matches:
                        score = len(keyword_matches) * 0.2
                        if initial_match:
                            score += 0.5
                        context_candidates.append({
                            'entity': entity,
                            'score': min(score, 0.95),
                            'keywords': keyword_matches,
                            'initial_match': initial_match
                        })
                        context_keywords_found.extend(keyword_matches)

            # Sort by score
            context_candidates.sort(key=lambda x: x['score'], reverse=True)

            # Add to detection
            enhanced_detection = {
                **detection,
                'category': category,
                'initial': initial,
                'context_candidates': context_candidates[:3],  # Top 3
                'context_keywords': list(set(context_keywords_found))[:5]
            }

            enhanced.append(enhanced_detection)

        return enhanced

    def _get_entity_category(self, pattern_type: str) -> str:
        """Map pattern type to entity category"""
        company_patterns = ['company_anonymized', 'initial_company', 'company_abbreviated',
                           'industry_reference', 'industry_anonymous']
        country_patterns = ['country_reference', 'country_initial', 'neighbor_country',
                           'location_vague']
        person_patterns = ['person_initial', 'person_anonymized', 'person_title']

        if pattern_type in company_patterns:
            return 'company'
        elif pattern_type in country_patterns:
            return 'country'
        elif pattern_type in person_patterns:
            return 'person'
        return 'unknown'

    def _extract_initial(self, text: str) -> str:
        """Extract initial letter from euphemism text"""
        import re
        #      
        match = re.search(r'^([A-Z-])', text)
        if match:
            return match.group(1)
        return ""

    def _add_ai_explanations(self, detections: List[Dict], text: str) -> List[Dict]:
        """Add AI-generated explanations using Gemini"""
        if not self.gemini_detector:
            return detections

        try:
            for detection in detections:
                explanation = self.gemini_detector.explain(
                    euphemism=detection['text'],
                    context=text,
                    resolved_entity=detection.get('entity', '')
                )
                detection['ai_explanation'] = explanation
        except Exception as e:
            logger.warning(f"Failed to generate AI explanation: {e}")

        return detections

    def explain(self, text: str) -> Dict:
        """
        Get detailed AI explanation for euphemisms in text

        Args:
            text: Input text

        Returns:
            Detailed explanation including AI analysis
        """
        # Basic detection first
        result = self.detect_and_resolve(text, include_explanation=True)

        # If Gemini available, add comprehensive explanation
        if self.gemini_detector:
            try:
                gemini_result = self.gemini_detector.detect(text)
                result['ai_analysis'] = {
                    'summary': gemini_result.get('summary', ''),
                    'detailed_analysis': gemini_result.get('analysis', ''),
                    'confidence': gemini_result.get('confidence', 0.0)
                }
            except Exception as e:
                logger.warning(f"Gemini analysis failed: {e}")
                result['ai_analysis'] = None

        return result
    
    def detect_patterns(self, text: str) -> List[Dict]:
        """
        Stage 1: Rule-based pattern detection
        
        Args:
            text: Input text
            
        Returns:
            List of pattern-based detections
        """
        return self.pattern_matcher.detect_patterns(text)
    
    def detect_with_ml(self, text: str) -> List[Dict]:
        """
        Stage 2: ML-based detection using KoELECTRA
        
        Args:
            text: Input text
            
        Returns:
            List of ML-based detections
        """
        return self.ml_detector.predict(text)
    
    def resolve_entities(self, detections: List[Dict], text: str) -> List[Dict]:
        """
        Stage 3: Resolve detected euphemisms to actual entities
        With Gemini fallback for UNKNOWN entities

        Args:
            detections: List of detected euphemisms
            text: Original text for context

        Returns:
            List of detections with resolved entities
        """
        resolved = []
        unresolved_for_gemini = []  # Gemini fallback 

        for detection in detections:
            # Extract context window around detection
            context = self._extract_context(text, detection)

            # Resolve entity
            resolution = self.entity_resolver.resolve_entity(
                detection['text'],
                context
            )

            # Merge detection with resolution
            resolved_detection = {
                **detection,
                'entity': resolution['entity'],
                'entity_confidence': resolution['confidence'],
                'alternatives': resolution.get('alternatives', []),
                'resolution_method': 'knowledge_base'
            }

            # Check if entity is UNKNOWN or low confidence - mark for Gemini fallback
            # UNKNOWN  fallback,  confidence(< 0.7) fallback
            if resolution['entity'] == 'UNKNOWN' or resolution['confidence'] < 0.7:
                unresolved_for_gemini.append((len(resolved), resolved_detection))

            resolved.append(resolved_detection)

        # Gemini fallback for unresolved entities
        if unresolved_for_gemini and self.gemini_detector:
            resolved = self._resolve_with_gemini(resolved, unresolved_for_gemini, text)

        return resolved

    def _resolve_with_gemini(
        self,
        resolved: List[Dict],
        unresolved: List[tuple],
        text: str
    ) -> List[Dict]:
        """
        Use Gemini to resolve entities that ML/KB couldn't handle

        Args:
            resolved: List of all resolved detections
            unresolved: List of (index, detection) tuples for unresolved items
            text: Original text for context

        Returns:
            Updated resolved list with Gemini results
        """
        try:
            logger.info(f"Using Gemini fallback for {len(unresolved)} unresolved entities")

            # Build focused prompt for unresolved entities
            euphemisms_to_resolve = [det['text'] for _, det in unresolved]

            #   
            irony_patterns = ['sarcasm_praise', 'sarcasm_with_laugh', 'sarcasm_exaggeration',
                             'sarcasm_fake_thanks', 'sarcasm_rhetorical']
            has_irony = any(det.get('type', '') in irony_patterns for _, det in unresolved)

            if has_irony:
                #    
                prompt = f"""    .  /  .

: "{text}"

 : {', '.join(euphemisms_to_resolve)}

   JSON  :
```json
[
  {{
    "expression": "",
    "entity": " ",
    "confidence": 0.0-1.0,
    "explanation": "",
    "is_irony": true/false,
    "surface_meaning": " ",
    "intended_meaning": "  ",
    "irony_type": "sarcasm/understatement/exaggeration/none"
  }}
]
```

  :
1. " ", "" +  →   
2.   "", "" →    
3.      
4.  (, )    
5.     vs  
6.  JSON  """
            else:
                #   (  )
                prompt = f"""       .

: "{text}"

 : {', '.join(euphemisms_to_resolve)}

   JSON  :
```json
[
  {{"expression": "", "entity": " ", "confidence": 0.0-1.0, "explanation": ""}}
]
```

:
1.       
2.   confidence  
3. 'S '        
4. ' '       
5.  JSON  """

            # Call Gemini
            response = self.gemini_detector.model.generate_content(prompt)

            # Parse response
            import re
            import json

            json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.text.strip()

            gemini_results = json.loads(json_str)

            # Create lookup map
            gemini_map = {}
            for result in gemini_results:
                if isinstance(result, dict) and 'expression' in result:
                    gemini_map[result['expression']] = result

            # Update resolved list with Gemini results
            for idx, detection in unresolved:
                expr = detection['text']
                if expr in gemini_map:
                    gemini_result = gemini_map[expr]
                    resolved[idx]['entity'] = gemini_result.get('entity', detection['entity'])
                    resolved[idx]['entity_confidence'] = gemini_result.get('confidence', 0.7)
                    resolved[idx]['resolution_method'] = 'gemini_fallback'
                    resolved[idx]['gemini_explanation'] = gemini_result.get('explanation', '')

                    #   
                    if gemini_result.get('is_irony'):
                        resolved[idx]['is_irony'] = True
                        resolved[idx]['surface_meaning'] = gemini_result.get('surface_meaning', '')
                        resolved[idx]['intended_meaning'] = gemini_result.get('intended_meaning', '')
                        resolved[idx]['irony_type'] = gemini_result.get('irony_type', 'sarcasm')
                        logger.info(f"Irony detected: '{expr}' → surface: '{resolved[idx]['surface_meaning']}', intended: '{resolved[idx]['intended_meaning']}'")
                    else:
                        logger.info(f"Gemini resolved: '{expr}' → '{resolved[idx]['entity']}'")

        except Exception as e:
            logger.warning(f"Gemini fallback failed: {e}")
            # Keep original UNKNOWN results

        return resolved
    
    def _merge_detections(
        self,
        pattern_detections: List[Dict],
        ml_detections: List[Dict]
    ) -> List[Dict]:
        """
        Merge detections from different sources, removing duplicates
        
        Args:
            pattern_detections: Rule-based detections
            ml_detections: ML-based detections
            
        Returns:
            Merged and deduplicated detections
        """
        all_detections = pattern_detections + ml_detections
        
        # Remove duplicates based on text position
        unique_detections = []
        seen_positions = set()
        
        for detection in all_detections:
            position = (detection['start'], detection['end'])
            if position not in seen_positions:
                unique_detections.append(detection)
                seen_positions.add(position)
        
        return unique_detections
    
    def _extract_context(
        self,
        text: str,
        detection: Dict,
        window_size: int = 100
    ) -> str:
        """
        Extract context window around detection
        
        Args:
            text: Full text
            detection: Detection dictionary with start/end positions
            window_size: Context window size in characters
            
        Returns:
            Context text
        """
        start = detection.get('start', 0)
        end = detection.get('end', len(text))
        
        # Calculate context boundaries
        context_start = max(0, start - window_size)
        context_end = min(len(text), end + window_size)
        
        return text[context_start:context_end]
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        Analyze multiple texts in parallel
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of analysis results
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self.detect_and_resolve, texts))
        return results
    
    async def stream_analyze(self, text: str) -> Dict:
        """
        Async streaming analysis for real-time processing
        
        Args:
            text: Input text
            
        Returns:
            Analysis result
        """
        # For now, just wrap synchronous call
        # Can be enhanced with actual streaming in future
        return self.detect_and_resolve(text)
    
    def get_processing_time(self) -> float:
        """Get last processing time"""
        return self.processing_time
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return self._is_loaded
    
    def get_statistics(self) -> Dict:
        """
        Get detector statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'models_loaded': self._is_loaded,
            'pattern_count': len(self.pattern_matcher.patterns),
            'ml_model': 'KoELECTRA',
            'last_processing_time': self.processing_time
        }
