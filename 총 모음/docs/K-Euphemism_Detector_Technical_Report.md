#      (K-Euphemism Detector)
#   

: 2025 11 30
: v12-irony-entity

---

## 

1.  
2.    
3.     
4. ML/NLP  
5. 3  
6. LLM  (Gemini API)
7.   (Entity Resolution) 
8.   (Knowledge Base) 
9.    
10.     

---

## 1.  

### 1.1 

    "S ", " ", " "   (Euphemism)  ,       AI .

### 1.2  

-   :   30   
- ML  NER: KoELECTRA    
-  :       
- LLM Fallback: Google Gemini 2.0 Flash    
- / :      

### 1.3   URL

-  API: https://k-euphemism-api-245053314944.asia-northeast3.run.app/
- : https://k-euphemism-frontend-245053314944.asia-northeast3.run.app/
- API : https://k-euphemism-api-245053314944.asia-northeast3.run.app/docs

---

## 2.    

### 2.1 Google Cloud Platform 

####  
-  ID: inspiring-list-465300-p6
- : asia-northeast3 ()

#### Cloud Run 

|  |  |  |
|---------|-------|------|
| k-euphemism-api | 8Gi RAM, 4 vCPU |  API  |
| k-euphemism-frontend | 2Gi RAM, 2 vCPU | Next.js  |

#### Cloud Storage 

```
gs://k-euphemism-data/
  - training/           #   (train.jsonl, validation.jsonl, test.jsonl)
  - statistics/         #   
  - exports/            #  
  - job_results/        #  
```

### 2.2   

```
backend/
  app/
    main.py              # FastAPI 
    api/
      routes/
        analyze.py       # /api/analyze 
        websocket.py     #  WebSocket
        labeling.py      #  API
    core/
      config.py          #  
      cache.py           #  
    ml/
      detector.py        #   
      patterns.py        #  
      kobert_model.py    # KoELECTRA NER 
      inference.py       #   
      gemini_detector.py # Gemini API 
      knowledge_base.json #  
```

### 2.3 Docker  

```dockerfile
FROM python:3.9-slim
WORKDIR /app

#   
RUN apt-get update && apt-get install -y gcc g++ git

# Python  
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# KoELECTRA   
RUN python -c "from transformers import ElectraTokenizer; \
    ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')"

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.4   

|  |  |
|-------|------|
| GEMINI_API_KEY | Google Gemini API   |
| MODEL_GCS_BUCKET |   GCS  |
| MODEL_GCS_PATH |  GCS  |
| AUTO_DOWNLOAD_MODEL | GCS    |

---

## 3.     

### 3.1   

####   (scripts/namuwiki_crawler.py)

 "", "", ""       .

  :
- : " ", "S", "L" 
- : "K", " ", "" 
- : " ", " " 

#### AI Hub  

AI Hub     NER     .

### 3.2   (scripts/data_preprocessing.py)

#### 3.2.1  

```python
def clean_text(text):
    # HTML  
    text = re.sub(r'<[^>]+>', '', text)
    #   
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    #   
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

#### 3.2.2  

        .

#### 3.2.3  

 :
-  : 10 , 500 
-  : UNKNOWN   
-   : 2 
-   
-   

#### 3.2.4  

1.  : "S " -> "S", "S  "
2.   :
   - "{euphemism}() ."
   - "{euphemism}() {entity}() ."

#### 3.2.5   

       .  :  500 .

### 3.3 BIO  

|  |  |  |
|-----|------|-----|
| O | Outside (-) | "", "" |
| B-EUPHEMISM | Begin (  ) | "S" |
| I-EUPHEMISM | Inside (  ) | "" |

### 3.4    (JSONL)

```json
{
  "text": "S    .",
  "tokens": ["S", "", "", "", "", "", "", "", "", "."],
  "ner_tags": ["B-EUPHEMISM", "I-EUPHEMISM", "O", "O", "O", "O", "O", "O", "O", "O"],
  "patterns": [{"text": "S ", "start": 0, "end": 5, "type": "company_anonymized"}]
}
```

---

## 4. ML/NLP  

### 4.1 KoELECTRA  NER 

#### 4.1.1  

-  : monologg/koelectra-base-v3-discriminator
- : Token Classification (NER)
-   : 3 (O, B-EUPHEMISM, I-EUPHEMISM)

#### 4.1.2   (kobert_model.py)

```python
class KoELECTRADetector:
    TAG_O = 0   # Outside
    TAG_B = 1   # Begin-EUPHEMISM
    TAG_I = 2   # Inside-EUPHEMISM

    def __init__(self, model_path=None):
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.model = ElectraForTokenClassification.from_pretrained(model_path, num_labels=3)
        self.model.eval()
```

#### 4.1.3   

1.      
2. GCS    (AUTO_DOWNLOAD_MODEL=true )
3.     (Fallback)

### 4.2   (training/train_koelectra_ner.py)

#### 4.2.1 

|  |  |  |
|---------|---|------|
| batch_size | 16 |   |
| num_epochs | 10 |   |
| learning_rate | 2e-5 |  |
| warmup_steps | 500 | Warmup  |
| weight_decay | 0.01 |   |
| max_grad_norm | 1.0 |   |
| max_length | 128 |    |

#### 4.2.2  

```python
# AdamW with different weight decay for bias and LayerNorm
optimizer_grouped_parameters = [
    {'params': [...], 'weight_decay': 0.01},  #  
    {'params': [...], 'weight_decay': 0.0}    # bias, LayerNorm
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

# Linear warmup scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)
```

#### 4.2.3 Early Stopping

 F1-Score 3       .

#### 4.2.4  

- Precision:       
- Recall:      
- F1-Score: Precision Recall  

---

## 5. 3  

### 5.1   (detector.py)

```
 
     |
     v
[1:  ] ----->    
     |
     v
[2:  ] ----->    
     |
     v
[3:  ] ----->  + Gemini Fallback
     |
     v
 
```

### 5.2 1:     (patterns.py)

#### 5.2.1  

|  |   |  |
|---------|----------|------|
| company_anonymized | `[A-Z-]\s*\s*(\|\|)` | S , L  |
| initial_company | `[A-Z]\s*(\|\|)` | S, L |
| person_initial | `[A-Z-]\s*` | K,  |
| person_anonymized | `[-]{1,2}\s*\s*` | ,  |
| country_reference | `\s*\|\s*` |  ,   |
| internet_slang | `[--]{2,6}` | , ,  |
| sarcasm_praise | `(\|\|).*.*` |   |

#### 5.2.2  

    ,   , , ,   .

```python
pattern_priority = [
    'ironic_blessing_curse',    #  
    'yamin_pattern',            # 
    'company_anonymized',       # S 
    'initial_company',          # S
    'person_anonymized',        # 
    ...
]
```

#### 5.2.3  

   (covered_spans)       .

### 5.3 2:    (detector.py)

#### 5.3.1   

```python
CONTEXT_KEYWORDS = {
    "company": {
        "": ["", "", "", ""],
        "SK": ["", "", "", ""],
        "LG": ["", "TV", "", ""],
        ...
    },
    "country": {
        "": ["", "", "", "", ""],
        "": ["", "", "", ""],
        ...
    }
}
```

#### 5.3.2  

1.      (company/country/person)
2.     (S, L, ,  )
3.    
4.   +    
5.  3  

### 5.4 3:   (inference.py)

#### 5.4.1   

```python
INITIAL_MAPPING = {
    'S': ['', '', 'SK', 'SK'],
    'L': ['LG', 'LG', ''],
    'H': ['', '', ''],
    ...
}

COUNTRY_INITIAL_MAPPING = {
    '': [''],
    '': [''],
    '': [''],
    ...
}
```

#### 5.4.2  

1.  /  
2.  alias  
3.    ( +  )
4.    (SentenceTransformer)
5. (0.65)    
6.   UNKNOWN  (Gemini Fallback )

---

## 6. LLM  (Gemini API)

### 6.1 Gemini Fallback  (gemini_detector.py)

#### 6.1.1  

-    UNKNOWN 
- (confidence) 0.7  

#### 6.1.2  

```python
model = genai.GenerativeModel('gemini-2.0-flash')
```

#### 6.1.3 Rate Limiting

-    4  
- LRU     

### 6.2  

#### 6.2.1    

```
       .

: "{text}"
 : {euphemisms}

   JSON  :
[
  {"expression": "", "entity": " ", "confidence": 0.0-1.0, "explanation": ""}
]

:
1.       
2.   confidence  
3. 'S '        
```

#### 6.2.2   

```
    .  /  .

 :
1. " ", "" +  ->   
2.   "", "" ->    
3.  (, )    

 :
[
  {
    "expression": "",
    "is_irony": true/false,
    "surface_meaning": " ",
    "intended_meaning": "  ",
    "irony_type": "sarcasm/understatement/exaggeration/none"
  }
]
```

---

## 7.   (Entity Resolution) 

### 7.1 EntityResolver  (inference.py)

#### 7.1.1 

```python
class EntityResolver:
    def __init__(self):
        # SentenceTransformer  ( )
        self.encoder = SentenceTransformer('jhgan/ko-sbert-sts')

        #  
        self.knowledge_base = self._load_knowledge_base()

        #  
        self.threshold = 0.65

        #   
        self._build_alias_index()
```

#### 7.1.2  

1. **   **: knowledge_base['internet_slang']  
2. **  **: knowledge_base['irony']  
3. **  **:   aliases  
4. ** **:   +  
5. ** **: SentenceTransformer  Fallback 
6. ** **:   ,  UNKNOWN

### 7.2 " "  

#### 7.2.1   

```python
COUNTRY_CONTEXT_KEYWORDS = {
    '': ['', '', '', '', '', ''],
    '': ['', '', '', '', ''],
    '': ['IRA', '', '', '', ''],
    '': ['', '', '', '', ''],
}
```

#### 7.2.2  

1.        
2.       
3.      : ['', '', '', '']

---

## 8.   (Knowledge Base) 

### 8.1  

```
backend/app/ml/knowledge_base.json
```

### 8.2  ( 965)

```json
{
  "companies": {
    "": {
      "description": "  ,    ",
      "aliases": ["", "S", "S", "", "S "],
      "keywords": ["", "", "", "OLED", ""]
    },
    ...
  },
  "countries": {
    "": {
      "description": " ,    ",
      "aliases": ["", "", " ", "C"],
      "keywords": ["", "", "", "", ""]
    },
    ...
  },
  "people": {
    "": {
      "description": " ",
      "aliases": [" ", " ", ""],
      "keywords": ["", "", "", ""]
    },
    ...
  },
  "internet_slang": {
    "": {
      "meaning": " ",
      "category": "laugh",
      "aliases": ["", ""],
      "description": "   "
    },
    "": {
      "meaning": "/",
      "category": "abbreviation",
      "aliases": ["", ""],
      "description": "  "
    },
    ...
  },
  "irony": {
    " ": {
      "meaning": "/ ",
      "category": "sarcasm",
      "aliases": ["", " "],
      "description": "    "
    },
    ...
  }
}
```

### 8.3   

|  |   |  |
|---------|--------|------|
| companies |  57 |    |
| countries |  18 |   |
| people |  30 | , ,  |
| organizations |  20 |  ,  |
| industries |  15 |   |
| internet_slang |  50 |  , ,  |
| irony |  10 | ,   |

---

## 9.    

### 9.1 API  

```
POST /api/analyze
{
  "text": "S    .      ."
}
```

### 9.2  

####  1:  
```python
patterns = detector.detect_patterns(text)
# : [
#   {"text": "S ", "type": "company_anonymized", "confidence": 0.85},
#   {"text": " ", "type": "country_reference", "confidence": 0.85}
# ]
```

####  2:  
```python
enhanced = detector._enhance_with_context(patterns, text)
# : [
#   {"text": "S ", "initial": "S", "context_candidates": [
#     {"entity": "", "score": 0.9, "keywords": ["", ""]}
#   ]},
#   {"text": " ", "context_candidates": [
#     {"entity": "", "score": 0.85, "keywords": ["", ""]}
#   ]}
# ]
```

####  3:  
```python
resolved = detector.resolve_entities(enhanced, text)
# : [
#   {"text": "S ", "entity": "", "entity_confidence": 0.92},
#   {"text": " ", "entity": "", "entity_confidence": 0.88}
# ]
```

####  4: Gemini Fallback ()
```python
if resolution['entity'] == 'UNKNOWN' or resolution['confidence'] < 0.7:
    gemini_result = gemini_detector.detect(text)
    # Gemini  
```

### 9.3   

```json
{
  "text": "S    .      .",
  "detections": [
    {
      "type": "company_anonymized",
      "text": "S ",
      "start": 0,
      "end": 5,
      "confidence": 0.85,
      "entity": "",
      "entity_confidence": 0.92,
      "resolution_method": "knowledge_base",
      "alternatives": [
        {"entity": "SK", "confidence": 0.75}
      ]
    },
    {
      "type": "country_reference",
      "text": " ",
      "start": 21,
      "end": 25,
      "confidence": 0.85,
      "entity": "",
      "entity_confidence": 0.88,
      "resolution_method": "knowledge_base"
    }
  ],
  "total_detected": 2,
  "processing_time": 0.145,
  "gemini_enabled": true,
  "stages": {
    "pattern_matching": 2,
    "context_enhanced": 2,
    "resolved": 2
  }
}
```

---

## 10.     

### 10.1  

|  |  |  |
|-----|------|-----|
|  API |   | v12-irony-entity |
|  |   | latest |
| Gemini  |  | gemini-2.0-flash |
|   |    | KoELECTRA base |

### 10.2   

|  |   |  |
|-----|--------|-----|
|   | 6 | S , S,   |
|   | 4 | K,  ,  |
|   | 4 |  ,  ,   |
| / | 2 |  ,   |
|   | 7 | , ,  |
| / | 9 |  ,  |

### 10.3  

|  |  |   |
|-----|------|---------|
|   | 200ms  |  100-200ms |
|    | 90%  |  92% ( ) |
|    | 85%  |  85-90% (KB + Gemini) |
|   | 100  |  (Cloud Run  ) |

### 10.4   

1. ** KoELECTRA  **: Vertex AI     
2. ** **:      
3. **  **:      
4. ** **: Redis     
5. ** **:      

---

## :   

|  |  |  |
|-----|------|------|
|  API | backend/app/main.py | FastAPI  |
|   | backend/app/ml/detector.py | 3  |
|   | backend/app/ml/patterns.py |   |
| KoELECTRA | backend/app/ml/kobert_model.py | NER  |
|   | backend/app/ml/inference.py | Entity Resolution |
| Gemini  | backend/app/ml/gemini_detector.py | LLM Fallback |
|  | backend/app/ml/knowledge_base.json |   DB |
|  | scripts/data_preprocessing.py |   |
|   | backend/training/train_koelectra_ner.py |   |

---

   .
