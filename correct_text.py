from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

CORRECT_MODEL = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(CORRECT_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(CORRECT_MODEL)

def correct_ocr_text(text):
    prompt = f"Correct the sentence: {text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_input = "Ths is a smple txt extrcted frm img."
    corrected = correct_ocr_text(test_input)
    print("Corrected Text:", corrected)
