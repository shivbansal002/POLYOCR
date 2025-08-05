from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

TRANSLATE_MODEL = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL)

def translate_text(text, source_lang="hin_Deva", target_lang="eng_Latn"):
    """
    Translates text from source_lang to target_lang using the NLLB model.
    """
    print(f"🌐 Translating from {source_lang} to {target_lang}...")

    # NLLB models use specific tokens for languages, e.g., "__eng_Latn__".
    # We need to get the token ID for the target language.
    target_lang_token = f"__{target_lang}__"
    
    forced_bos_token_id = None
    if target_lang_token in tokenizer.vocab:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang_token)
    else:
        print(f"Warning: Target language token '{target_lang_token}' not found in tokenizer vocabulary. "
              "Translation might not be in the desired target language.")
        # As a last resort, if the specific language token isn't found,
        # we can try to use the default BOS token, though translation quality might suffer.
        forced_bos_token_id = tokenizer.bos_token_id 

    if forced_bos_token_id is None:
        print("Error: Could not determine forced_bos_token_id. Translation may fail.")
        return text # Return original text if translation setup fails


    # Encode the input text. NLLB tokenizer handles source language implicitly
    # or expects it as a special token in the input text if explicitly needed.
    # For general NLLB usage with AutoTokenizer, just the text is usually sufficient.
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generate translation
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=128 # Limit output length to prevent very long generations
    )
    
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🌐 Translated Text: {translated_text}")
    return translated_text

if __name__ == "__main__":
    # Example usage for testing this module directly
    test_text_hi = "नमस्ते दुनिया! यह एक परीक्षण है।"
    translated_hi = translate_text(test_text_hi, source_lang="hin_Deva", target_lang="eng_Latn")
    print(f"Hindi to English: {translated_hi}")

    test_text_en = "Hello world! This is a test."
    translated_en = translate_text(test_text_en, source_lang="eng_Latn", target_lang="hin_Deva")
    print(f"English to Hindi: {translated_en}")
