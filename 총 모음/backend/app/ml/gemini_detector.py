"""
Gemini API   /  

 : 15 requests/min, 1500 requests/day
"""
import os
import json
import re
from typing import List, Dict, Optional
import time
from functools import lru_cache

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[WARNING] google-generativeai   .")
    print(": pip install google-generativeai")

class GeminiEuphemismDetector:
    """
    Gemini API   /    

    :
    -  (euphemism) 
    -  (neologism) 
    -  (slang)  
    -  (profanity)  
    -  ,    

    :
    - "" → "  "
    - "" → "" → ""
    - "S " →   ""  "SK"
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        

        Args:
            api_key: Gemini API  (  GEMINI_API_KEY )
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai  .\n"
                ": pip install google-generativeai"
            )

        # API  
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API  .\n"
                " GEMINI_API_KEY  api_key  .\n"
                "API  : https://makersuite.google.com/app/apikey"
            )

        genai.configure(api_key=self.api_key)

        #   (gemini-2.0-flash )
        self.model = genai.GenerativeModel('gemini-2.0-flash')

        # Rate limiting ( : 15 req/min)
        self.last_request_time = 0
        self.min_request_interval = 4.0  # 15 req/min = 1 req per 4 seconds

        #  (   )
        self.cache = {}
        self.cache_ttl = 3600  # 1

    def _wait_for_rate_limit(self):
        """Rate limit   """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    @lru_cache(maxsize=1000)
    def _get_cache_key(self, text: str) -> str:
        """  """
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def detect(self, text: str, context: str = "") -> List[Dict]:
        """
         /    

        Args:
            text:  
            context:    ()

        Returns:
              ,    :
            - text:  
            - type:  (euphemism, slang, neologism, profanity)
            - meaning:  
            - explanation:  
            - confidence:  (0-1)
            - alternatives:   ( )
        """
        #  
        cache_key = self._get_cache_key(text + context)
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_result

        # Rate limit 
        self._wait_for_rate_limit()

        #  
        prompt = self._build_prompt(text, context)

        try:
            # Gemini API 
            response = self.model.generate_content(prompt)

            #  
            detections = self._parse_response(response.text)

            # 
            self.cache[cache_key] = (time.time(), detections)

            return detections

        except Exception as e:
            print(f"[ERROR] Gemini API : {e}")
            return []

    def _build_prompt(self, text: str, context: str = "") -> str:
        """
        Gemini API  

         :
        - Few-shot learning  
        - JSON   
        -    
        """
        prompt = f"""  , ,  ,  .
    :

1.  (euphemism): "S ", " ", "K" 
2.  (neologism):    
3.  (slang):    
4.   (profanity): /   

**:**

: " "
:
```json
[
  {{
    "text": "",
    "type": "political_slang",
    "meaning": "  ",
    "explanation": "      ",
    "confidence": 0.95,
    "alternatives": []
  }}
]
```

: "  ?"
:
```json
[
  {{
    "text": "",
    "type": "profanity_disguise",
    "meaning": " ()",
    "explanation": "     ",
    "confidence": 0.85,
    "alternatives": ["     "]
  }}
]
```

: "S   "
:
```json
[
  {{
    "text": "S ",
    "type": "company_anonymization",
    "meaning": "",
    "explanation": "       ",
    "confidence": 0.90,
    "alternatives": ["SK", "SPC"]
  }}
]
```

** :**
1.     (: ""  )
2.   
3.   confidence  
4.  JSON   
5.      [] 

{"** :** " + context if context else ""}

** :**
"{text}"

** (JSON only):**
"""
        return prompt

    def _parse_response(self, response_text: str) -> List[Dict]:
        """
        Gemini  

        Args:
            response_text: Gemini API 

        Returns:
               
        """
        try:
            # JSON   
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # JSON      JSON 
                json_str = response_text.strip()

            # JSON 
            detections = json.loads(json_str)

            # 
            if not isinstance(detections, list):
                print(f"[WARNING]   : {type(detections)}")
                return []

            #     
            normalized = []
            for detection in detections:
                if not isinstance(detection, dict):
                    continue

                #   
                if 'text' not in detection or 'meaning' not in detection:
                    continue

                #  
                normalized_detection = {
                    'text': detection.get('text', ''),
                    'type': detection.get('type', 'unknown'),
                    'meaning': detection.get('meaning', ''),
                    'explanation': detection.get('explanation', ''),
                    'confidence': float(detection.get('confidence', 0.7)),
                    'alternatives': detection.get('alternatives', [])
                }

                normalized.append(normalized_detection)

            return normalized

        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON  : {e}")
            print(f" : {response_text}")
            return []
        except Exception as e:
            print(f"[ERROR]   : {e}")
            return []

    def batch_detect(self, texts: List[str], context: str = "") -> List[List[Dict]]:
        """
           

        Args:
            texts:   
            context:  

        Returns:
                
        """
        results = []
        for text in texts:
            detections = self.detect(text, context)
            results.append(detections)
        return results

    def explain(self, term: str) -> Optional[Dict]:
        """
            

        Args:
            term:  

        Returns:
              
        """
        prompt = f"""  //   :

: "{term}"

  JSON  :
- origin:   
- meaning:  
- usage:    
- connotation: //
- popularity:   (rare/common/viral)
- category:  (political/internet/youth/profanity )

JSON  .
"""

        self._wait_for_rate_limit()

        try:
            response = self.model.generate_content(prompt)

            # JSON   
            json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.text.strip()

            explanation = json.loads(json_str)
            explanation['term'] = term

            return explanation

        except Exception as e:
            print(f"[ERROR]   : {e}")
            return None

    def get_stats(self) -> Dict:
        """
         

        Returns:
            API  
        """
        return {
            'cache_size': len(self.cache),
            'cache_ttl': self.cache_ttl,
            'rate_limit': f'{60/self.min_request_interval:.0f} requests/minute',
            'last_request': time.time() - self.last_request_time if self.last_request_time > 0 else None
        }


#  
if __name__ == "__main__":
    import sys

    # API  
    if not os.getenv("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY  ")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        print("   API  : https://makersuite.google.com/app/apikey")
        sys.exit(1)

    #  
    detector = GeminiEuphemismDetector()

    #  
    test_cases = [
        "   ?",
        " ",
        "S    ",
        " ",
        " ?",
    ]

    print("=== Gemini      ===\n")

    for text in test_cases:
        print(f"[INPUT] {text}")
        detections = detector.detect(text)

        if detections:
            for det in detections:
                print(f"  [DETECT] '{det['text']}' -> {det['meaning']}")
                print(f"     : {det['type']}")
                print(f"     : {det['explanation']}")
                print(f"     : {det['confidence']:.2%}")
                if det['alternatives']:
                    print(f"     : {', '.join(det['alternatives'])}")
        else:
            print("  [OK] No slang/profanity detected")

        print()

    # 
    stats = detector.get_stats()
    print("===   ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
