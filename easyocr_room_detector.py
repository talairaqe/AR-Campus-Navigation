import os
import re
import numpy as np
from PIL import Image, ImageDraw
import easyocr
from difflib import SequenceMatcher

# -------------------------
# Helpers (your original)
# -------------------------
def _normalize_room(s: str) -> str:
    """
    Normalize for room/cafeteria codes:
    - lowercase
    - remove spaces/punct (keeps & and -)
    - map common OCR confusions: O->0
    """
    s = s.strip().lower()
    s = s.replace("־", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w\-&]", "", s)   # keep & for BITES&BEATS

    # common OCR confusion
    s = s.replace("o", "0")  # "C32O" -> "C320"
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

# -------------------------
# Your original function (unchanged except allowlist includes &)
# -------------------------
def add_arrow_to_match(
    image_path: str,
    query_room_name: str,
    out_path: str = None,
    languages=("en", "iw"),       # Hebrew = iw (EasyOCR sometimes doesn't support 'he')
    min_confidence: float = 0.15,
    gpu: bool = False,
    scale: float = 2.0,
    debug: bool = False,
    fuzzy_threshold: float = 0.86
):
    """
    Finds a specific room/cafe name in image using OCR.
    If found, outputs new image with arrow to the matched text.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)

    # Upscale for better OCR on small signage
    img_np_big = _scale_image(img_np, scale=scale)

    reader = _make_reader(languages, gpu=gpu)

    # Allow & for BITES&BEATS
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-&"

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

# -------------------------
# NEW: detect where you are (C320 vs FABiOLA vs BITES&BEATS)
# -------------------------
def detect_one_of(
    reader,
    pil_img: Image.Image,
    options: dict,
    allowlist: str,
    min_confidence: float = 0.15,
    scale: float = 2.0,
    fuzzy_threshold: float = 0.86,
    debug: bool = False
):
    """
    options: dict {canonical_name: [aliases...]}
    Returns: (best_canonical_name, best_match_dict) or (None, None)

    Scans OCR results ONCE, then tries aliases. Picks the best scored match.
    Score favors exact/contains matches; otherwise confidence+similarity.
    """
    img_np = np.array(pil_img)
    img_np_big = _scale_image(img_np, scale=scale)

    results = reader.readtext(img_np_big, allowlist=allowlist, paragraph=False)

    best_name, best_res = None, None
    best_score = -1.0

    for name, aliases in options.items():
        for alias in aliases:
            query_n = _normalize_room(alias)

            # exact/contains
            exact_candidates = []
            for bbox, text, conf in results:
                if conf < min_confidence:
                    continue
                text_n = _normalize_room(text)
                if query_n in text_n or text_n in query_n:
                    exact_candidates.append((bbox, text, conf, text_n))

            if exact_candidates:
                bbox, text, conf, text_n = max(exact_candidates, key=lambda x: x[2])
                score = float(conf) + 10.0  # big boost for exact-ish
            else:
                # fuzzy
                best_fuzzy = None
                best_sim = -1.0
                for bbox, text, conf in results:
                    if conf < min_confidence:
                        continue
                    text_n = _normalize_room(text)
                    sim = _similarity(text_n, query_n)
                    if abs(len(text_n) - len(query_n)) <= 1:
                        sim += 0.02
                    if sim > best_sim:
                        best_sim = sim
                        best_fuzzy = (bbox, text, conf, text_n)

                if best_fuzzy and best_sim >= fuzzy_threshold:
                    bbox, text, conf, text_n = best_fuzzy
                    score = float(conf) + best_sim
                else:
                    continue

            if score > best_score:
                best_score = score
                best_name = name
                best_res = {"bbox": bbox, "matched_text": text, "confidence": float(conf), "alias_used": alias}

    if debug:
        print(f"[DEBUG] detect_one_of best={best_name} score={best_score:.3f}")

    return best_name, best_res

# -------------------------
# MVP FLOW:
# 1) start image: detect where you are (C320 / FABiOLA / BITES&BEATS) + arrow
# 2) destination: user chooses
# 3) route text: fixed "go straight 44m"
# 4) end image: detect destination + arrow
# -------------------------
def mvp_two_photos_route(
    start_image_path: str,
    end_image_path: str,
    destination: str,  # "C320" or "FABiOLA" or "BITES&BEATS"
    start_out: str = "start_annotated.jpg",
    end_out: str = "end_annotated.jpg",
    languages=("en", "iw"),
    gpu=False,
    scale=2.0,
    min_confidence=0.15,
    fuzzy_threshold=0.86,
    debug=True
):
    if not os.path.exists(start_image_path):
        raise FileNotFoundError(start_image_path)
    if not os.path.exists(end_image_path):
        raise FileNotFoundError(end_image_path)

    reader = _make_reader(languages, gpu=gpu)
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-&"

    # Known places (only what you need now)
    options = {
        "C320": ["C320", "C32O"],
        "FABIOLA": ["FABiOLA", "FABIOLA"],
        "BITES&BEATS": ["BITES&BEATS", "BITES & BEATS", "BITES& BEATS"]
    }

    # 1) Detect start location + annotate
    pil_start = Image.open(start_image_path).convert("RGB")
    where_am_i, _ = detect_one_of(
        reader, pil_start, options,
        allowlist=allowlist,
        min_confidence=min_confidence,
        scale=scale,
        fuzzy_threshold=fuzzy_threshold,
        debug=debug
    )

    if where_am_i is None:
        print("[FAIL] Could not detect start location from the first image.")
        return None

    # Draw arrow on start location using your function
    add_arrow_to_match(
        image_path=start_image_path,
        query_room_name=where_am_i,
        out_path=start_out,
        languages=languages,
        min_confidence=min_confidence,
        gpu=gpu,
        scale=scale,
        debug=False,
        fuzzy_threshold=fuzzy_threshold
    )

    # 2) Destination
    dest = destination.strip().upper()
    if dest not in options:
        raise ValueError("destination must be one of: C320, FABiOLA, BITES&BEATS")

    # 3) Route (fixed for now)
    route_text = "לכי ישר 44 מטרים"
    if debug:
        print(f"[START] {where_am_i}")
        print(f"[DEST]  {dest}")
        print(f"[ROUTE] {route_text}")

    # 4) Detect destination in final image + annotate
    pil_end = Image.open(end_image_path).convert("RGB")
    found_dest, _ = detect_one_of(
        reader, pil_end, {dest: options[dest]},
        allowlist=allowlist,
        min_confidence=min_confidence,
        scale=scale,
        fuzzy_threshold=fuzzy_threshold,
        debug=debug
    )

    if found_dest is None:
        print("[FAIL] Destination not found in the final image. No end annotation created.")
        return {
            "start_location": where_am_i,
            "destination": dest,
            "route": route_text,
            "start_annotated": start_out,
            "end_annotated": None
        }

    add_arrow_to_match(
        image_path=end_image_path,
        query_room_name=dest,
        out_path=end_out,
        languages=languages,
        min_confidence=min_confidence,
        gpu=gpu,
        scale=scale,
        debug=False,
        fuzzy_threshold=fuzzy_threshold
    )

    return {
        "start_location": where_am_i,
        "destination": dest,
        "route": route_text,
        "start_annotated": start_out,
        "end_annotated": end_out
    }

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Put your images here:
    start_image = "start.JPG"   # image near C320 or cafeteria sign
    end_image   = "end.JPG"     # image near the destination sign

    # Choose destination for this run:
    destination = "FABiOLA"  # or "FABiOLA" or "C320"

    result = mvp_two_photos_route(
        start_image_path=start_image,
        end_image_path=end_image,
        destination=destination,
        debug=True
    )

    print("\nRESULT:")
    print(result)
