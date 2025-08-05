import sys, glob
from pathlib import Path
import easyocr

# ------------------------- Config
USE_GPU   = False
PARAGRAPH = True
EXTS      = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}

# ------------------------- Language‑specific Readers
LANG_GROUPS = {
    "hi": ['en', 'hi'],
    "ta": ['en', 'ta'],
    "te": ['en', 'te'],
    "ja": ['en', 'ja'],
    "ko": ['en', 'ko'],
    "ch": ['en', 'ch_sim'],
    "en": ['en'],
}

READERS = {}

def load_readers():
    for key, langs in LANG_GROUPS.items():
        READERS[key] = easyocr.Reader(
            langs, gpu=USE_GPU, download_enabled=True, verbose=False
        )

def gather(arg: str):
    """Expand wildcards or recurse into folders and yield image files."""
    p = Path(arg)
    for q in map(Path, glob.glob(str(p), recursive=True)):
        if q.is_file() and q.suffix.lower() in EXTS:
            yield q
        elif q.is_dir():
            yield from (f for f in q.rglob('*') if f.suffix.lower() in EXTS)

def run_reader(reader, img):
    """Run OCR on one image with a given reader and pretty‑print."""
    results = reader.readtext(str(img), detail=1, paragraph=PARAGRAPH)

    if not results:                      # nothing found
        print(f"\n🖼  {img.name} — no text found.\n" + "-"*40)
        return

    print(f"\n🖼  {img.name}\n" + "-"*40)
    for res in results:
        # EasyOCR sometimes returns (bbox, text) and sometimes (bbox, text, conf)
        if len(res) == 3:
            _, txt, conf = res
            print(f"[{conf:5.2f}] {txt}")
        elif len(res) == 2:
            _, txt = res
            print(f"[  N/A ] {txt}")
    print("-"*40)

# ------------------------- Main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python multilang_easyocr.py <images | folders>")
        sys.exit(1)

    files = {Path(f).resolve()
             for arg in sys.argv[1:]
             for f in gather(arg)}

    if not files:
        print("❌  No image files matched your input.")
        sys.exit(1)

    print(f"🔍  Processing {len(files)} image(s)…")
    load_readers()

    for img in sorted(files):
        name = img.name.lower()
        if   'hindi'    in name: run_reader(READERS['hi'], img)
        elif 'tamil'    in name: run_reader(READERS['ta'], img)
        elif 'telugu'   in name: run_reader(READERS['te'], img)
        elif 'chinese'  in name: run_reader(READERS['ch'], img)
        elif 'japanese' in name: run_reader(READERS['ja'], img)
        elif 'korean'   in name: run_reader(READERS['ko'], img)
        else:                    run_reader(READERS['en'], img)

