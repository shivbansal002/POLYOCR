# process_pipeline.py

# --- FIX: Changed relative imports to absolute imports ---
# Assuming lang_detect.py, correct_text.py, and translate_text.py are in the same directory
from lang_detect import detect_language
from correct_text import correct_ocr_text
from translate_text import translate_text
import string # Added for spellchecker integration
from spellchecker import SpellChecker # Added for spellchecker integration

# Initialize spell checker once
spell = SpellChecker() # Default to English

lang_code_map = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml"
}

def process_ocr_output(text):
    print(f"📥 Input OCR Text: {text}")
    
    # --- Step 1: Language Detection ---
    lang = detect_language(text)
    print(f"🌍 Detected Language: {lang}")
    
    # --- Step 2: Spell Correction (using pyspellchecker logic) ---
    # Tokenize the text into words
    words = text.split()
    # Remove punctuation from words
    translator = str.maketrans('', '', string.punctuation)
    clean_words = [word.translate(translator).strip() for word in words]
    # Filter out empty strings
    clean_words = [word for word in clean_words if word]
    # Run spell correction
    corrected_spell = [spell.correction(word) if spell.correction(word) is not None else word for word in clean_words]
    # Reconstruct the sentence after spell checking
    corrected_spell_text = " ".join(corrected_spell)

    # --- Step 3: Grammar/Contextual Correction (using correct_ocr_text) ---
    # This function uses a Hugging Face model for more advanced correction.
    corrected_grammar = correct_ocr_text(corrected_spell_text)
    print(f"🛠️ Corrected Text: {corrected_grammar}")
    
    # --- Step 4: Translation ---
    translated = corrected_grammar # Default to corrected grammar if no translation needed
    if lang != "en":
        source_lang = lang_code_map.get(lang, "eng_Latn")
        # Only translate if source_lang is in the map and different from target (eng_Latn)
        if source_lang != "eng_Latn":
            translated = translate_text(corrected_grammar, source_lang, target_lang="eng_Latn")
        else: # If detected language is already English (or mapped to English)
            print("🌐 No translation needed (already English).")

    print(f"🌐 Final Processed Text (after translation if applicable): {translated}")

    # --- FIX: Return only the final string, not a dictionary ---
    return translated

if __name__ == "__main__":
    # Example usage for testing this module directly
    sample_ocr_text = "Ths is a smple txt extrcted frm img. It has sum errrs."
    print(f"Original OCR Text:\n{sample_ocr_text}")
    
    processed_text = process_ocr_output(sample_ocr_text)
    print(f"\nFinal Processed Output:\n{processed_text}")

    print("\n--- Testing with Hindi ---")
    sample_ocr_text_hindi = "यह एक सधारण पाठ है।" # Example Hindi text
    processed_hindi_text = process_ocr_output(sample_ocr_text_hindi)
    print(f"\nFinal Processed Hindi Output:\n{processed_hindi_text}")
