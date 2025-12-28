import cv2
import re
from paddleocr import PaddleOCR

# ======================================================
# 1. SCADA TAG REGEX + CLASSIFIER
# ======================================================

TAG_REGEX = re.compile(
    r"""
    ^(
        [A-Z]{2,4}                 # PIT, TT, FT, FAN
        |
        [A-Z]{1,4}\d{2,6}          # PT94071, FAN94040
        |
        [A-Z]{1,4}-\d+[A-Z]*       # P-61A, CN-60PT
    )$
    """,
    re.VERBOSE
)

CONFIDENCE_THRESHOLD = 0.75


def classify_ocr(texts, scores):
    """
    Separate valid SCADA tags from noise
    """
    tags, noise = [], []

    for text, score in zip(texts, scores):
        text = text.strip()

        if not text or score < CONFIDENCE_THRESHOLD:
            noise.append((text, score))
            continue

        if TAG_REGEX.match(text):
            tags.append((text, score))
        else:
            noise.append((text, score))

    return tags, noise


# ======================================================
# 2. IMAGE PREPROCESSING (OCR-FRIENDLY)
# ======================================================

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"{image_path} not found")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Remove background noise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    # Scale up for small text
    scaled = cv2.resize(
        denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )

    # PaddleOCR requires 3-channel input
    final_img = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

    return final_img


# ======================================================
# 3. OCR INITIALIZATION
# (DBNet is used internally by PaddleOCR)
# ======================================================

ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True
)


# ======================================================
# 4. OCR EXECUTION
# ======================================================

def run_ocr(image):
    """
    PaddleOCR internally performs:
    - DBNet-based text detection
    - Text recognition
    - Confidence scoring
    """
    result = ocr.predict(image)
    return result[0]


# ======================================================
# 5. MAIN PIPELINE
# ======================================================

if __name__ == "__main__":

    print("\n🔍 Running SCADA OCR Pipeline...\n")

    # Step 1: Preprocess image
    image = preprocess_image("SCADA.jpeg")

    # Step 2: Run OCR
    ocr_output = run_ocr(image)

    # Step 3: Extract OCR results
    texts = ocr_output.get("rec_texts", [])
    scores = ocr_output.get("rec_scores", [])
    boxes = ocr_output.get("rec_boxes", [])

    # Step 4: Filter using rules
    tags, noise = classify_ocr(texts, scores)

    # Step 5: Print results
    print("✅ VALID SCADA TAGS:\n")
    for t, s in tags:
        print(f"{t}  (confidence: {s:.2f})")

    print("\n❌ NOISE / NON-TAGS:\n")
    for t, s in noise:
        print(f"{t}  (confidence: {s:.2f})")

    # Step 6: Structured output (ready for LayoutLM)
    structured_output = [
        {
            "text": t,
            "confidence": s
        }
        for t, s in tags
    ]

    print("\n📦 STRUCTURED OUTPUT (for VLM / LayoutLM):\n")
    print(structured_output)
    print("\n🔍 SCADA OCR Pipeline Complete.\n")