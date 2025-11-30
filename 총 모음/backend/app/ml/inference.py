"""
Entity resolution and inference engine - Enhanced with Korean initial mapping
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
import logging
import re

from app.core.config import settings

logger = logging.getLogger(__name__)


# /    (S → , SK )
INITIAL_MAPPING = {
    #   →   
    'S': ['', '', '', '', 'SK', 'SK', 'SK', 'SPC', '', ''],
    'L': ['LG', 'LG', 'LG', '', 'LS'],
    'H': ['', '', '', '', ''],
    'K': ['', '', 'KT'],
    'N': ['', '', '', '', ''],
    'C': ['CJ', '', ''],
    'G': ['GS', ''],
    'A': ['', '', ''],
    'D': [''],
    'P': [''],
    'T': ['', 'TSMC', ''],
    'M': ['', ''],
    'Y': ['YG'],
    'J': ['JYP'],
    #  
    '': ['', '', '', ''],
    '': ['LG', 'LG', 'LG'],
    '': ['LG', 'LG', 'LG'],
    '': ['', ''],
    '': ['SK', 'SK', 'SK'],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
}

#   
COUNTRY_INITIAL_MAPPING = {
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    '': [''],
    'C': [''],  # China
    'J': [''],  # Japan
    'U': [''],  # USA
    'R': [''],  # Russia
    'N': [''],  # North Korea
    'G': [''],  # Germany
}

#    ( )
PERSON_INITIAL_MAPPING = {
    #  () →  
    '': ['', '', ''],
    '': ['', '', ''],
    '': [''],
    '': ['', ''],
    '': ['', ''],
    '': [''],
    '': [''],
    '': [''],
    #  
    'L': ['', ''],  # Lee
    'C': ['', ''],  # Chung/Choi
}

# " "      ( →  )
COUNTRY_CONTEXT_KEYWORDS = {
    '': ['', '', '', '', '', '', '', '', '', '', '',
             '', '', '', '', '', '', '', '', ''],
    '': ['', '', '', '', '', '', '',
             '', '', '', '', '', '', ''],
    '': ['IRA', '', '', '', '', '', '',
             '', '', '', '', '', '', ''],
    '': ['', '', '', '', '', '', 'NLL', '', '', '', ''],
    '': ['', '', '', '', '', '', ''],
    '': ['', 'TSMC', '', ''],
    '': ['', '', '', '', '', ''],
}


class EntityResolver:
    """Resolve euphemisms to actual entities using context and knowledge base"""

    def __init__(self, knowledge_base_path: str = None):
        """
        Initialize sentence transformer and knowledge base

        Args:
            knowledge_base_path: Path to knowledge base JSON file
        """
        self.encoder = None
        self._encoder_available = False

        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(settings.SENTENCE_TRANSFORMER)
            self._encoder_available = True
            logger.info("SentenceTransformer loaded successfully")
        except Exception as e:
            logger.warning(f"SentenceTransformer initialization failed: {e}. Using fallback scoring.")

        self.knowledge_base_path = knowledge_base_path or os.path.join(
            os.path.dirname(__file__), 'knowledge_base.json'
        )
        self.knowledge_base = self._load_knowledge_base()
        #      
        self.threshold = 0.65  #  0.85 

        # Create reverse index for aliases
        self._build_alias_index()

    def _load_knowledge_base(self) -> Dict:
        """
        Load knowledge base from JSON file

        Returns:
            Dictionary with entity categories and metadata
        """
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Knowledge base not found at {self.knowledge_base_path}, using empty KB")
            return {
                'companies': {},
                'countries': {},
                'industries': {},
                'media': {},
                'government': {}
            }

    def _build_alias_index(self):
        """Build reverse index mapping aliases to entity names"""
        self.alias_to_entity = {}

        for category_name, category_data in self.knowledge_base.items():
            # Skip if category_data is not a dict
            if not isinstance(category_data, dict):
                continue

            for entity_name, entity_info in category_data.items():
                # Handle case where entity_info is a string instead of dict
                if isinstance(entity_info, str):
                    # Simple string description - just index the entity name
                    self.alias_to_entity[entity_name] = entity_name
                    continue

                # Normal dict case - index aliases
                if isinstance(entity_info, dict):
                    for alias in entity_info.get('aliases', []):
                        self.alias_to_entity[alias] = entity_name

                    # Index the entity name itself
                    self.alias_to_entity[entity_name] = entity_name

    def _check_alias_exact_match(self, euphemism: str) -> Optional[str]:
        """
        Check if euphemism exactly matches any known alias in knowledge base

        Args:
            euphemism: Euphemism text to check

        Returns:
            Entity name if exact alias match found, None otherwise
        """
        # Check direct alias index first
        if euphemism in self.alias_to_entity:
            return self.alias_to_entity[euphemism]

        # Check all categories for alias matches (more thorough)
        for category_name, category_data in self.knowledge_base.items():
            if not isinstance(category_data, dict):
                continue

            for entity_name, entity_info in category_data.items():
                if not isinstance(entity_info, dict):
                    continue

                aliases = entity_info.get('aliases', [])

                # Exact match check
                if euphemism in aliases:
                    logger.debug(f"Alias exact match: '{euphemism}' in aliases of '{entity_name}'")
                    return entity_name

                # Partial match for patterns like " " matching ""
                # Check if euphemism contains title suffix patterns
                title_patterns = ['', '', '', '', '', '', '', '', '']
                for title in title_patterns:
                    if title in euphemism:
                        # Extract the name part before title
                        name_part = euphemism.replace(title, '').strip()
                        # Check if this matches first character of any alias
                        for alias in aliases:
                            if name_part and alias.startswith(name_part):
                                logger.debug(f"Title pattern match: '{euphemism}' -> '{entity_name}' via alias '{alias}'")
                                return entity_name

        return None

    def resolve_entity(self, euphemism: str, context: str) -> Dict:
        """
        Resolve euphemism to actual entity

        Args:
            euphemism: Detected euphemism text
            context: Surrounding context

        Returns:
            Resolution result with entity and confidence
        """
        # Check internet_slang first for direct lookup (, , )
        if 'internet_slang' in self.knowledge_base:
            slang_data = self.knowledge_base['internet_slang']
            # Direct match
            if euphemism in slang_data:
                slang_info = slang_data[euphemism]
                if isinstance(slang_info, dict):
                    meaning = slang_info.get('meaning', euphemism)
                    category = slang_info.get('category', 'internet_slang')
                    return {
                        'entity': meaning,
                        'confidence': 0.95,
                        'alternatives': [],
                        'category': category,
                        'description': slang_info.get('description', '')
                    }
            # Check aliases
            for slang_name, slang_info in slang_data.items():
                if isinstance(slang_info, dict):
                    aliases = slang_info.get('aliases', [])
                    if euphemism in aliases:
                        meaning = slang_info.get('meaning', slang_name)
                        category = slang_info.get('category', 'internet_slang')
                        return {
                            'entity': meaning,
                            'confidence': 0.95,
                            'alternatives': [],
                            'category': category,
                            'description': slang_info.get('description', '')
                        }

        # Check irony (/) for direct lookup
        if 'irony' in self.knowledge_base:
            irony_data = self.knowledge_base['irony']
            # Direct match
            if euphemism in irony_data:
                irony_info = irony_data[euphemism]
                if isinstance(irony_info, dict):
                    meaning = irony_info.get('meaning', euphemism)
                    category = irony_info.get('category', 'irony')
                    return {
                        'entity': meaning,
                        'confidence': 0.90,
                        'alternatives': [],
                        'category': category,
                        'description': irony_info.get('description', '')
                    }
            # Check aliases and partial matches for irony patterns
            for irony_name, irony_info in irony_data.items():
                if isinstance(irony_info, dict):
                    aliases = irony_info.get('aliases', [])
                    # Exact alias match
                    if euphemism in aliases:
                        meaning = irony_info.get('meaning', irony_name)
                        category = irony_info.get('category', 'irony')
                        return {
                            'entity': meaning,
                            'confidence': 0.90,
                            'alternatives': [],
                            'category': category,
                            'description': irony_info.get('description', '')
                        }
                    # Partial match: check if euphemism contains the irony pattern
                    if irony_name in euphemism or euphemism in irony_name:
                        meaning = irony_info.get('meaning', irony_name)
                        category = irony_info.get('category', 'irony')
                        return {
                            'entity': meaning,
                            'confidence': 0.85,
                            'alternatives': [],
                            'category': category,
                            'description': irony_info.get('description', '')
                        }
                    # Check aliases for partial match
                    for alias in aliases:
                        if alias in euphemism or euphemism in alias:
                            meaning = irony_info.get('meaning', irony_name)
                            category = irony_info.get('category', 'irony')
                            return {
                                'entity': meaning,
                                'confidence': 0.85,
                                'alternatives': [],
                                'category': category,
                                'description': irony_info.get('description', '')
                            }

        #  NEW: Check for exact alias match FIRST (highest priority)
        alias_match = self._check_alias_exact_match(euphemism)
        if alias_match:
            logger.info(f"Exact alias match found: '{euphemism}' -> '{alias_match}'")
            return {
                'entity': alias_match,
                'confidence': 0.95,  # High confidence for exact alias match
                'alternatives': [],
                'match_type': 'alias_exact'
            }

        # Generate candidate entities
        candidates = self._generate_candidates(euphemism, context)

        if not candidates:
            return {
                'entity': 'UNKNOWN',
                'confidence': 0.0,
                'alternatives': []
            }

        # Calculate similarity scores with alias bonus
        scores = self._calculate_similarity(context, candidates, euphemism)

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Log top candidates for debugging
        logger.debug(f"Top candidates for '{euphemism}': {scores[:3]}")

        # Check if best match meets threshold
        if scores[0][1] >= self.threshold:
            return {
                'entity': scores[0][0],
                'confidence': float(scores[0][1]),
                'alternatives': [
                    {'entity': entity, 'confidence': float(score)}
                    for entity, score in scores[1:3]
                ]
            }

        return {
            'entity': 'UNKNOWN',
            'confidence': float(scores[0][1]) if scores else 0.0,
            'alternatives': [
                {'entity': entity, 'confidence': float(score)}
                for entity, score in scores[:3]
            ]
        }
    
    def _generate_candidates(self, euphemism: str, context: str) -> List[str]:
        """
        Generate candidate entities based on euphemism and context using KB

        Args:
            euphemism: Euphemism text
            context: Context text

        Returns:
            List of candidate entity names
        """
        candidates = set()

        # Extract initial if present
        initial = self._extract_initial(euphemism)

        # 0. INITIAL_MAPPING    ()
        if initial:
            # /     
            if initial in INITIAL_MAPPING:
                candidates.update(INITIAL_MAPPING[initial])
            #    
            if initial in COUNTRY_INITIAL_MAPPING:
                candidates.update(COUNTRY_INITIAL_MAPPING[initial])

        # 1. Pattern-based candidate generation by category
        # Companies
        if any(word in euphemism for word in [' ', ' ', ' ', '', '', '']):
            company_candidates = self._get_category_candidates('companies', initial, context)
            candidates.update(company_candidates)
            #   
            if initial and initial in INITIAL_MAPPING:
                candidates.update(INITIAL_MAPPING[initial])

        # Countries - " "   
        if any(word in euphemism for word in [' ', ' ', ' ', ' ', ' ', '']):
            # " " :     
            if ' ' in euphemism or ' ' in euphemism:
                context_based_countries = self._resolve_country_from_context(context)
                candidates.update(context_based_countries)
            else:
                country_candidates = self._get_category_candidates('countries', initial, context)
                candidates.update(country_candidates)
            #    
            if initial and initial in COUNTRY_INITIAL_MAPPING:
                candidates.update(COUNTRY_INITIAL_MAPPING[initial])

        # Industries
        if '' in euphemism or '' in euphemism:
            industry_candidates = self._get_category_candidates('industries', initial, context)
            candidates.update(industry_candidates)

        # Media
        if any(word in euphemism for word in ['', '', '', '']):
            media_candidates = self._get_category_candidates('media', initial, context)
            candidates.update(media_candidates)

        # Government
        if any(word in euphemism for word in ['', '', '', '', '', '']):
            gov_candidates = self._get_category_candidates('government', initial, context)
            candidates.update(gov_candidates)

        # People - " ", " ", ""   
        if any(word in euphemism for word in ['', '', '', '', '', '', '', '', '', '', '']):
            people_candidates = self._get_category_candidates('people', initial, context)
            candidates.update(people_candidates)
            #    
            if initial and initial in PERSON_INITIAL_MAPPING:
                candidates.update(PERSON_INITIAL_MAPPING[initial])

        # Internet slang /  /  - Direct lookup
        if 'internet_slang' in self.knowledge_base:
            slang_data = self.knowledge_base['internet_slang']
            # Direct match check
            if euphemism in slang_data:
                candidates.add(euphemism)
            # Check aliases
            for slang_name, slang_info in slang_data.items():
                if isinstance(slang_info, dict):
                    aliases = slang_info.get('aliases', [])
                    if euphemism in aliases or euphemism == slang_name:
                        candidates.add(slang_name)

        # 2. Keyword-based matching from context
        context_lower = context.lower()
        for category_name, category_data in self.knowledge_base.items():
            # Skip if category_data is not a dict
            if not isinstance(category_data, dict):
                continue

            for entity_name, entity_info in category_data.items():
                # Handle case where entity_info is a string instead of dict
                if isinstance(entity_info, str):
                    # Simple string description - no keywords to check
                    continue

                # Normal dict case - check keywords
                if isinstance(entity_info, dict):
                    keywords = entity_info.get('keywords', [])
                    if any(keyword.lower() in context_lower for keyword in keywords):
                        candidates.add(entity_name)

        return list(candidates)

    def _get_category_candidates(
        self,
        category: str,
        initial: str,
        context: str
    ) -> List[str]:
        """
        Get candidate entities from specific category

        Args:
            category: Category name (companies, countries, etc.)
            initial: Initial letter from euphemism
            context: Context text

        Returns:
            List of candidate entity names
        """
        candidates = []
        category_data = self.knowledge_base.get(category, {})

        # Filter by initial if present -    
        if initial:
            #      
            if category == 'companies' and initial in INITIAL_MAPPING:
                for mapped_entity in INITIAL_MAPPING[initial]:
                    # KB   
                    if mapped_entity in category_data:
                        candidates.append(mapped_entity)
                    #    KB  alias 
                    for entity_name, entity_info in category_data.items():
                        if isinstance(entity_info, dict):
                            aliases = entity_info.get('aliases', [])
                            if mapped_entity in aliases or mapped_entity == entity_name:
                                if entity_name not in candidates:
                                    candidates.append(entity_name)

            if category == 'countries' and initial in COUNTRY_INITIAL_MAPPING:
                for mapped_entity in COUNTRY_INITIAL_MAPPING[initial]:
                    if mapped_entity in category_data:
                        candidates.append(mapped_entity)
                    for entity_name, entity_info in category_data.items():
                        if isinstance(entity_info, dict):
                            aliases = entity_info.get('aliases', [])
                            if mapped_entity in aliases or mapped_entity == entity_name:
                                if entity_name not in candidates:
                                    candidates.append(entity_name)

            #  KB    
            for entity_name, entity_info in category_data.items():
                # Check if initial matches entity name or any alias
                if entity_name.startswith(initial):
                    if entity_name not in candidates:
                        candidates.append(entity_name)
                    continue

                # Handle case where entity_info is a string instead of dict
                if isinstance(entity_info, str):
                    # Simple string description - no aliases
                    continue

                # Check aliases for dict entity_info
                if isinstance(entity_info, dict):
                    aliases = entity_info.get('aliases', [])
                    if any(alias.startswith(initial) for alias in aliases):
                        if entity_name not in candidates:
                            candidates.append(entity_name)

        # If no initial or no matches, use keyword matching
        if not candidates:
            context_lower = context.lower()
            for entity_name, entity_info in category_data.items():
                # Handle case where entity_info is a string instead of dict
                if isinstance(entity_info, str):
                    # Simple string description - no keywords to check
                    continue

                # Check keywords for dict entity_info
                if isinstance(entity_info, dict):
                    keywords = entity_info.get('keywords', [])
                    # Check if any keyword appears in context
                    if any(keyword.lower() in context_lower for keyword in keywords):
                        candidates.append(entity_name)

        # If still no candidates, return all from category (fallback)
        if not candidates:
            candidates = list(category_data.keys())

        return candidates
    
    def _extract_initial(self, text: str) -> str:
        """
        Extract initial letter from euphemism

        Args:
            text: Euphemism text

        Returns:
            Initial letter or empty string
        """
        import re
        match = re.search(r'^([A-Z-])', text)
        return match.group(1) if match else ""

    def _resolve_country_from_context(self, context: str) -> List[str]:
        """
        " "       

        Args:
            context:  

        Returns:
               ( )
        """
        context_lower = context.lower()
        country_scores = {}

        for country, keywords in COUNTRY_CONTEXT_KEYWORDS.items():
            score = 0
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in context_lower:
                    score += 1
                    matched_keywords.append(keyword)

            if score > 0:
                country_scores[country] = (score, matched_keywords)

        #   
        sorted_countries = sorted(
            country_scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )

        #    ( 3)
        result = [country for country, _ in sorted_countries[:3]]

        #         
        if not result:
            result = ['', '', '', '']

        logger.debug(f"Country context resolution: {context[:50]}... -> {result}")
        return result
    
    def _calculate_similarity(
        self,
        context: str,
        candidates: List[str],
        euphemism: str = ""
    ) -> List[Tuple[str, float]]:
        """
        Calculate semantic similarity between context and candidates
        Enhanced with alias bonus scoring

        Args:
            context: Context text
            candidates: List of candidate entities
            euphemism: Original euphemism text for alias matching

        Returns:
            List of (entity, score) tuples
        """
        # If encoder is not available, use keyword-based fallback scoring
        if not self._encoder_available:
            return self._fallback_scoring(context, candidates, euphemism)

        # Encode context
        context_embedding = self.encoder.encode(context)

        scores = []
        for candidate in candidates:
            # Get candidate description from knowledge base
            description = self._get_entity_description(candidate)

            # Encode description
            candidate_embedding = self.encoder.encode(description)

            # Calculate cosine similarity
            similarity = np.dot(context_embedding, candidate_embedding) / (
                np.linalg.norm(context_embedding) * np.linalg.norm(candidate_embedding)
            )

            #  NEW: Add alias bonus for matching
            alias_bonus = self._get_alias_bonus(candidate, euphemism)
            final_score = min(similarity + alias_bonus, 0.98)

            scores.append((candidate, final_score))

        return scores

    def _get_alias_bonus(self, candidate: str, euphemism: str) -> float:
        """
        Calculate bonus score for alias matches

        Args:
            candidate: Candidate entity name
            euphemism: Original euphemism text

        Returns:
            Bonus score (0.0 - 0.4)
        """
        if not euphemism:
            return 0.0

        # Search for candidate in knowledge base
        for category_name, category_data in self.knowledge_base.items():
            if not isinstance(category_data, dict):
                continue

            if candidate in category_data:
                entity_info = category_data[candidate]
                if not isinstance(entity_info, dict):
                    continue

                aliases = entity_info.get('aliases', [])

                # Exact alias match - high bonus
                if euphemism in aliases:
                    logger.debug(f"Exact alias match bonus: {euphemism} -> {candidate}")
                    return 0.4

                # Check if euphemism is substring of any alias
                for alias in aliases:
                    # e.g., " " in "S "
                    if euphemism in alias:
                        logger.debug(f"Substring alias match bonus: {euphemism} in {alias} -> {candidate}")
                        return 0.35

                    # e.g., "S " ends with " "
                    if alias.endswith(euphemism):
                        logger.debug(f"Suffix alias match bonus: {alias} ends with {euphemism} -> {candidate}")
                        return 0.35

                    # Check for title pattern matches (e.g., " " -> "")
                    title_patterns = ['', '', '', '', '']
                    for title in title_patterns:
                        if title in euphemism:
                            name_part = euphemism.replace(title, '').strip()
                            if name_part and candidate.startswith(name_part):
                                logger.debug(f"Title pattern bonus: {euphemism} -> {candidate}")
                                return 0.3

        return 0.0

    def _fallback_scoring(
        self,
        context: str,
        candidates: List[str],
        euphemism: str = ""
    ) -> List[Tuple[str, float]]:
        """
        Fallback keyword-based scoring when encoder is not available
         :   + alias   

        Args:
            context: Context text
            candidates: List of candidate entities
            euphemism: Original euphemism for alias matching

        Returns:
            List of (entity, score) tuples
        """
        context_lower = context.lower()
        scores = []

        for candidate in candidates:
            score = 0.55  # Base score for being a candidate (slightly higher)

            #  NEW: Add alias bonus FIRST (highest priority)
            alias_bonus = self._get_alias_bonus(candidate, euphemism)
            if alias_bonus >= 0.35:
                # Alias match found - give high base score
                score = 0.60 + alias_bonus  # 0.95+ for exact alias match

            # Get entity info from knowledge base
            for category_name, category_data in self.knowledge_base.items():
                if not isinstance(category_data, dict):
                    continue

                if candidate in category_data:
                    entity_info = category_data[candidate]

                    if isinstance(entity_info, dict):
                        keywords = entity_info.get('keywords', [])
                        # Count keyword matches
                        keyword_matches = sum(
                            1 for kw in keywords if kw.lower() in context_lower
                        )

                        #       
                        if keyword_matches >= 3:
                            score = max(score, 0.90)  # 3     
                        elif keyword_matches >= 2:
                            score = max(score, 0.80)  # 2    
                        elif keyword_matches >= 1:
                            score = max(score, 0.70)  # 1   threshold 

            # COUNTRY_CONTEXT_KEYWORDS    (  )
            if candidate in COUNTRY_CONTEXT_KEYWORDS:
                country_keywords = COUNTRY_CONTEXT_KEYWORDS[candidate]
                country_keyword_matches = sum(
                    1 for kw in country_keywords if kw.lower() in context_lower
                )
                if country_keyword_matches >= 2:
                    score = max(score, 0.85)  #   2  
                elif country_keyword_matches >= 1:
                    score = max(score, 0.75)  #   1 

            scores.append((candidate, min(score, 0.98)))

        return scores

    def _get_entity_description(self, entity_name: str) -> str:
        """
        Get comprehensive description for entity from knowledge base

        Args:
            entity_name: Entity name to look up

        Returns:
            Combined description with keywords and aliases
        """
        # Search through all categories
        for category_name, category_data in self.knowledge_base.items():
            if entity_name in category_data:
                entity_info = category_data[entity_name]

                # Handle case where entity_info is a string instead of dict
                if isinstance(entity_info, str):
                    # Simple string description - return as is
                    return entity_info

                # Normal dict case - combine description, keywords, and aliases
                if isinstance(entity_info, dict):
                    description = entity_info.get('description', '')
                    keywords = ', '.join(entity_info.get('keywords', []))
                    aliases = ', '.join(entity_info.get('aliases', []))
                    meaning = entity_info.get('meaning', '')  # For internet_slang

                    # Create comprehensive description for better matching
                    if meaning:
                        full_description = f"{description}. : {meaning}. : {keywords}"
                    else:
                        full_description = f"{description}. : {keywords}. : {aliases}"
                    return full_description

        # Fallback to entity name if not found in KB
        return entity_name
    
    def batch_resolve(
        self,
        euphemisms: List[str],
        contexts: List[str]
    ) -> List[Dict]:
        """
        Batch resolve multiple euphemisms
        
        Args:
            euphemisms: List of euphemism texts
            contexts: List of context texts
            
        Returns:
            List of resolution results
        """
        results = []
        for euphemism, context in zip(euphemisms, contexts):
            results.append(self.resolve_entity(euphemism, context))
        return results
