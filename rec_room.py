import os
import re
import numpy as np
from PIL import Image, ImageDraw
import easyocr
from difflib import SequenceMatcher

def _normalize_room(s: str) -> str:
    """
    Normalize for room codes:
    - lowercase
    - remove spaces/punct
    - map common OCR confusions: O->0 (optional), I->1 (optional)
    """
    s = s.strip().lower()
    s = s.replace("־", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w\-]", "", s)

    # common OCR confusions for room codes
    s = s.replace("o", "0")   # "C32O" -> "C320"
    # s = s.replace("i", "1") # uncomment if needed

    return s

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _make_reader(languages, gpu=False):
    try:
        return easyocr.Reader(list(languages), gpu=gpu)
    except ValueError as e:
        print(f"[WARN] EasyOCR language error: {e}. Falling back to ['en'] only.")
        return easyocr.Reader(["en"], gpu=gpu)

def _scale_image(img_np: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img_np
    pil = Image.fromarray(img_np)
    w, h = pil.size
    pil = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
    return np.array(pil)

def add_arrow_to_match(
    image_path: str,
    query_room_name: str,
    out_path: str = None,
    languages=("en", "iw"),      # Hebrew = iw
    min_confidence: float = 0.15, # lowered for small text
    gpu: bool = False,
    scale: float = 2.0,          # upscale helps a lot
    debug: bool = False,
    fuzzy_threshold: float = 0.86 # similarity threshold
):
    """
    Finds a specific room/class name in image using OCR.
    If found, outputs new image with arrow to the matched text.
    - Uses allowlist to improve detection of room codes.
    - Uses normalization and fuzzy fallback (handles 0/O).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)

    # Upscale for better OCR on small signage
    img_np_big = _scale_image(img_np, scale=scale)

    reader = _make_reader(languages, gpu=gpu)

    # Restrict OCR to likely characters for room codes to reduce noise
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"

    results = reader.readtext(
        img_np_big,
        allowlist=allowlist,
        paragraph=False
    )  # (bbox, text, conf)

    query_n = _normalize_room(query_room_name)

    # 1) exact/contains match pass
    exact_matches = []
    for bbox, text, conf in results:
        if conf < min_confidence:
            continue
        text_n = _normalize_room(text)
        if query_n in text_n or text_n in query_n:
            exact_matches.append((bbox, text, conf, text_n))

    best = None
    if exact_matches:
        # choose highest confidence exact match
        best = max(exact_matches, key=lambda x: x[2])
    else:
        # 2) fuzzy fallback: pick the most similar OCR token to the query
        best_fuzzy = None
        best_score = -1.0
        for bbox, text, conf in results:
            if conf < min_confidence:
                continue
            text_n = _normalize_room(text)
            score = _similarity(text_n, query_n)
            # boost if it's very close length-wise
            if abs(len(text_n) - len(query_n)) <= 1:
                score += 0.02
            if score > best_score:
                best_score = score
                best_fuzzy = (bbox, text, conf, text_n)

        if best_fuzzy and best_score >= fuzzy_threshold:
            best = best_fuzzy
        else:
            if debug:
                print("---- OCR OUTPUT (filtered by confidence) ----")
                for bbox, text, conf in results:
                    if conf >= min_confidence:
                        print(f"text='{text}' conf={conf:.2f} norm='{_normalize_room(text)}'")
                print(f"[DEBUG] Best fuzzy score={best_score:.3f} for query='{query_room_name}' norm='{query_n}'")
            return None

    bbox, text, conf, text_n = best

    # bbox is 4 points in the *upscaled* image coordinates
    pts = np.array(bbox, dtype=np.float32)
    x_min, y_min = float(pts[:, 0].min()), float(pts[:, 1].min())
    x_max, y_max = float(pts[:, 0].max()), float(pts[:, 1].max())

    # Convert back to original image coordinates
    x_min /= scale; y_min /= scale; x_max /= scale; y_max /= scale

    target_x = (x_min + x_max) / 2.0
    target_y = y_min
    box_h = max(10.0, (y_max - y_min))
    start_x = target_x
    start_y = max(0.0, target_y - 1.2 * box_h)

    draw = ImageDraw.Draw(pil_img)

    pad = 6
    draw.rectangle([x_min - pad, y_min - pad, x_max + pad, y_max + pad],
                   outline=(255, 0, 0), width=4)

    draw.line([start_x, start_y, target_x, target_y], fill=(255, 0, 0), width=6)

    head_len = max(12.0, 0.35 * box_h)
    head_w = head_len * 0.8
    p1 = (target_x, target_y)
    p2 = (target_x - head_w / 2.0, target_y - head_len)
    p3 = (target_x + head_w / 2.0, target_y - head_len)
    draw.polygon([p1, p2, p3], fill=(255, 0, 0))

    if out_path is None:
        base, ext = os.path.splitext(image_path)
        out_path = f"{base}_arrow{ext}"

    pil_img.save(out_path)

    if debug:
        print(f"[FOUND] query='{query_room_name}' norm='{query_n}'")
        print(f"[MATCH] ocr_text='{text}' norm='{text_n}' conf={conf:.2f}")
        print(f"[SAVED] {out_path}")

    return {
        "out_path": out_path,
        "matched_text": text,
        "matched_norm": text_n,
        "confidence": float(conf),
        "bbox": bbox
    }

if __name__ == "__main__":
    image_path = "input2.JPG"
    room_name = "C320"

    res = add_arrow_to_match(
        image_path=image_path,
        query_room_name=room_name,
        languages=("en", "iw"),
        gpu=False,
        debug=True
    )

    if res is None:
        print(f"'{room_name}' was NOT found in the image. No output image created.")
    else:
        print(f"Found '{room_name}'! Saved annotated image to: {res['out_path']}")
