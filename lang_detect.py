# lang_detect.py
from transformers import AutoTokenizer
# Try direct import instead of Auto class
try:
    from transformers import XLMRobertaForSequenceClassification
    use_direct_import = True
except ImportError:
    from transformers import AutoModelForSequenceClassification
    use_direct_import = False

import torch
import os
import shutil

# --- FIX: Moved LANG_MODEL definition here so it's available for cache clearing ---
LANG_MODEL = "papluca/xlm-roberta-base-language-detection"

# --- Clear Hugging Face cache for this model (optional, but good for debugging) ---
# This will force a fresh download of the model and its configuration.
# You might want to comment this out after the first successful run.
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
model_cache_path = os.path.join(cache_dir, f"models--{LANG_MODEL.replace('/', '--')}")

if os.path.exists(model_cache_path):
    print(f"Clearing cache for {LANG_MODEL}...")
    shutil.rmtree(model_cache_path)
    print("Cache cleared. Model will be re-downloaded.")

# --- End cache clearing ---
tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL)

# --- Load model with appropriate class ---
if use_direct_import:
    print("Using direct XLMRobertaForSequenceClassification import")
    model = XLMRobertaForSequenceClassification.from_pretrained(LANG_MODEL, trust_remote_code=True)
else:
    print("Using AutoModelForSequenceClassification import")
    model = AutoModelForSequenceClassification.from_pretrained(LANG_MODEL, trust_remote_code=True)

def detect_language(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted = torch.argmax(outputs.logits, dim=1)
    return model.config.id2label[predicted.item()]

if __name__ == "__main__":
    test_text = "Hello world! This is a test."
    detected_lang = detect_language(test_text)
    print(f"Detected language for '{test_text}': {detected_lang}")
    
    test_text_hi = "नमस्ते दुनिया! यह एक परीक्षण है।"
    detected_lang_hi = detect_language(test_text_hi)
    print(f"Detected language for '{test_text_hi}': {detected_lang_hi}")