import easyocr
import os
from google.colab import drive

# 1. Mount Google Drive
# This step connects your Colab environment to your Google Drive.
drive.mount('/content/drive')

# ✅ Paths
# Update the input_folder to point to your dataset on Google Drive.
input_folder = r"/content/drive/MyDrive/OCR_datasets/mixed_dataset/"
# You can keep the output folder in Colab's temporary storage or
# change it to a Google Drive path if you want the results saved there.
output_folder = r"/content/drive/MyDrive/OCR_recognized_text/" # Changed to save in Drive
os.makedirs(output_folder, exist_ok=True)

# ✅ Language suffix → EasyOCR language list
lang_map = {
    'en': ['en'],
    'hi': ['hi', 'en'],
    'ch': ['ch_sim', 'en'],
    'ja': ['ja', 'en'],
    'ko': ['ko', 'en'],
    'te': ['te', 'en'],
    'fr': ['fr', 'en'],
    # Add any other languages you might have in your dataset if needed
}

# ✅ Supported image extensions
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# ✅ Loop through each image in the folder
for filename in os.listdir(input_folder):
    if not any(filename.lower().endswith(ext) for ext in valid_extensions):
        continue

    print(f"\n🔍 Processing: {filename}")
    image_path = os.path.join(input_folder, filename)

    # ✅ Extract language from filename: expects _lang.jpg (e.g. img_fr.jpg)
    try:
        lang_key = filename.split('_')[-1].split('.')[0].lower()
        # Handle cases where filename might not have a language suffix,
        # or if the suffix doesn't match a defined language.
        if lang_key not in lang_map:
            print(f"⚠️ Skipping '{filename}': Language code '{lang_key}' not found in lang_map. Defaulting to 'en'.")
            lang_list = ['en'] # Default to English if language not found
        else:
            lang_list = lang_map[lang_key]
    except IndexError:
        print(f"⚠️ Skipping '{filename}': Filename does not contain expected language suffix (e.g., img_en.jpg). Defaulting to 'en'.")
        lang_list = ['en'] # Default to English if format is unexpected
    except KeyError:
        print(f"⚠️ Skipping '{filename}': Language key '{lang_key}' not defined in lang_map. Defaulting to 'en'.")
        lang_list = ['en'] # Should be caught by the above if, but as a fallback

    # ✅ Initialize EasyOCR reader
    try:
        # EasyOCR can utilize GPU in Colab, so set gpu=True for faster processing
        reader = easyocr.Reader(lang_list=lang_list, gpu=True)
    except Exception as e:
        print(f"❌ Error initializing EasyOCR for {filename}: {e}")
        continue

    # ✅ Perform OCR
    try:
        results = reader.readtext(image_path)
    except Exception as e:
        print(f"❌ OCR failed on {filename}: {e}")
        continue

    # ✅ Print and save recognized text
    recognized_lines = []
    if results: # Check if results is not empty
        for (_, text, conf) in results:
            print(f"📝 {text} (Confidence: {conf:.2f})") # Added confidence for better insight
            recognized_lines.append(text)
    else:
        print(f"No text recognized for {filename}")


    out_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
    with open(out_file, 'w', encoding='utf-8') as f:
        if recognized_lines: # Only write if there's content
            for line in recognized_lines:
                f.write(line + "\n")
        else:
            f.write("No text recognized.\n") # Write a note if no text was found

print("\n🎉 OCR processing complete!")