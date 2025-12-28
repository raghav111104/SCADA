import cv2
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

    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)


# ======================================================
# 2. SAFE BBOX HANDLING (CRITICAL FIX)
# ======================================================

import numpy as np

def safe_polygon_to_bbox(polygon):
    """
    Handles ALL PaddleOCR v5 formats:
    - numpy array of shape (8,)
    - [[x,y], [x,y], [x,y], [x,y]]
    - [x1, y1, x2, y2]
    """

    # Case 1: numpy array → flatten
    if isinstance(polygon, np.ndarray):
        polygon = polygon.tolist()

    # Case 2: flat list [x1,y1,x2,y2,x3,y3,x4,y4]
    if isinstance(polygon, (list, tuple)) and len(polygon) == 8:
        xs = polygon[0::2]
        ys = polygon[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    # Case 3: list of points [[x,y],...]
    if isinstance(polygon, (list, tuple)) and isinstance(polygon[0], (list, tuple)):
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return min(xs), min(ys), max(xs), max(ys)

    # Case 4: already rectangle
    if isinstance(polygon, (list, tuple)) and len(polygon) == 4:
        return polygon

    return None



def normalize_bbox(box, img_width, img_height):
    x1, y1, x2, y2 = box

    x1 = int(1000 * x1 / img_width)
    y1 = int(1000 * y1 / img_height)
    x2 = int(1000 * x2 / img_width)
    y2 = int(1000 * y2 / img_height)

    # Clamp
    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))

    return [x1, y1, x2, y2]


# ======================================================
# 3. OCR INITIALIZATION (PaddleOCR v5)
# ======================================================

ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True
)


# ======================================================
# 4. MAIN PIPELINE
# ======================================================

if __name__ == "__main__":

    print("\n🔍 Running SCADA OCR → LayoutLM Pipeline...\n")

    IMAGE_PATH = "SCADA1.jpeg"

    image = preprocess_image(IMAGE_PATH)
    img_h, img_w, _ = image.shape

    result = ocr.predict(image)[0]

    texts  = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])
    boxes  = (
        result.get("dt_polys")
        or result.get("rec_polys")
        or []
    )

    print(f"OCR texts detected : {len(texts)}")
    print(f"OCR boxes detected : {len(boxes)}")

    layoutlm_tokens = []

    line_id = 0
    token_id = 0
    prev_y = None
    LINE_GAP = 15  # vertical gap to detect new line

    for text, score, polygon in zip(texts, scores, boxes):

        if not text.strip():
            continue

        rect = safe_polygon_to_bbox(polygon)
        if rect is None:
            continue

        # line grouping
        if prev_y is None or abs(rect[1] - prev_y) > LINE_GAP:
            line_id += 1
            token_id = 0

        token_id += 1
        prev_y = rect[1]

        layoutlm_tokens.append({
            "text": text,
            "bbox": normalize_bbox(rect, img_w, img_h),
            "confidence": round(float(score), 4),
            "line_id": line_id,
            "token_id": token_id
        })

    # ==================================================
    # OUTPUT
    # ==================================================

    print(f"\n✅ Tokens prepared for LayoutLM: {len(layoutlm_tokens)}\n")

    print("📦 SAMPLE OUTPUT:\n")
    for t in layoutlm_tokens:
        print(t)

    print("\n🔍 SCADA OCR Pipeline Complete.\n")
