import cv2
import numpy as np
from paddleocr import PaddleOCR

# ======================================================
# 1. IMAGE PREPROCESSING
# ======================================================

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"{image_path} not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    scaled = cv2.resize(
        denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )

    final_img = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    return final_img


# ======================================================
# 2. ROBUST BBOX HANDLING (FIXED)
# ======================================================

def polygon_to_bbox(box):
    """
    Handles ALL PaddleOCR bbox formats safely
    Returns (x1, y1, x2, y2)
    """

    box = np.array(box).astype(float)

    # Case 1: [x1, y1, x2, y2]
    if box.ndim == 1 and box.shape[0] == 4:
        x1, y1, x2, y2 = box
        return x1, y1, x2, y2

    # Case 2: [x1,y1,x2,y2,x3,y3,x4,y4]
    if box.ndim == 1 and box.shape[0] == 8:
        xs = box[0::2]
        ys = box[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    # Case 3: [[x,y],[x,y],[x,y],[x,y]]
    if box.ndim == 2 and box.shape == (4, 2):
        xs = box[:, 0]
        ys = box[:, 1]
        return min(xs), min(ys), max(xs), max(ys)

    raise ValueError(f"Unsupported bbox format: {box}")


def normalize_bbox(box, img_width, img_height):
    x1, y1, x2, y2 = box

    x1 = int(1000 * x1 / img_width)
    y1 = int(1000 * y1 / img_height)
    x2 = int(1000 * x2 / img_width)
    y2 = int(1000 * y2 / img_height)

    return [
        max(0, min(1000, x1)),
        max(0, min(1000, y1)),
        max(0, min(1000, x2)),
        max(0, min(1000, y2)),
    ]


# ======================================================
# 3. INITIALIZE OCR
# ======================================================

ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True
)


# ======================================================
# 4. RUN OCR
# ======================================================

def run_ocr(image):
    result = ocr.predict(image)
    return result[0]


# ======================================================
# 5. MAIN PIPELINE
# ======================================================

if __name__ == "__main__":

    print("\n🔍 Running SCADA OCR → LayoutLM Pipeline...\n")

    IMAGE_PATH = "SCADA1.jpeg"

    image = preprocess_image(IMAGE_PATH)
    img_height, img_width, _ = image.shape

    ocr_output = run_ocr(image)

    texts = ocr_output.get("rec_texts", [])
    scores = ocr_output.get("rec_scores", [])
    boxes = ocr_output.get("rec_boxes", [])

    layoutlm_input = []

    for text, score, box in zip(texts, scores, boxes):

        if not text.strip() or score < 0.5:
            continue

        rect = polygon_to_bbox(box)
        norm_bbox = normalize_bbox(rect, img_width, img_height)

        layoutlm_input.append({
            "text": text,
            "bbox": norm_bbox
        })

    print("📦 LAYOUTLM-READY OUTPUT (sample):\n")
    for item in layoutlm_input:
        print(item)

    print(f"\n✅ Total tokens prepared for LayoutLM: {len(layoutlm_input)}")
    print("\n🔍 SCADA OCR Pipeline Complete.\n")
