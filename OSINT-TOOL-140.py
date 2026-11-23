
import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from PIL import Image, ExifTags


VERSION = "2.2.0"


# ==========================================
# 1. IMAGE METADATA
# ==========================================

def rational_to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def get_decimal_from_dms(dms, ref):
    degrees = rational_to_float(dms[0])
    minutes = rational_to_float(dms[1])
    seconds = rational_to_float(dms[2])

    decimal = degrees + minutes / 60 + seconds / 3600

    if str(ref).upper() in ("S", "W"):
        decimal = -decimal

    return round(decimal, 6)


def extract_exif_metadata(image_path):

    metadata = {
        "Camera": "Not available",
        "Date": "Not available",
        "GPS": "Not available",
        "Coordinates": None
    }

    try:
        with Image.open(image_path) as img:

            exif = img.getexif()

            if not exif:
                return metadata

            make = str(exif.get(271, "")).strip()
            model = str(exif.get(272, "")).strip()

            if make or model:
                metadata["Camera"] = f"{make} {model}".strip()

            date_taken = exif.get(36867) or exif.get(306)

            if date_taken:
                metadata["Date"] = str(date_taken)

            try:
                gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo)
            except (AttributeError, KeyError, TypeError):
                gps_info = {}

            if gps_info:

                gps = {}

                for key, value in gps_info.items():
                    tag = ExifTags.GPSTAGS.get(key, key)
                    gps[tag] = value

                lat = gps.get("GPSLatitude")
                lat_ref = gps.get("GPSLatitudeRef")
                lon = gps.get("GPSLongitude")
                lon_ref = gps.get("GPSLongitudeRef")

                if lat and lat_ref and lon and lon_ref:

                    lat_dec = get_decimal_from_dms(lat, lat_ref)
                    lon_dec = get_decimal_from_dms(lon, lon_ref)

                    metadata["GPS"] = f"{lat_dec}, {lon_dec}"
                    metadata["Coordinates"] = (lat_dec, lon_dec)

    except (OSError, ValueError, TypeError) as error:
        print(f"[!] EXIF warning: {error}")

    return metadata


# ==========================================
# 2. IMAGE LOADING
# ==========================================

def load_image(image_path):

    image_data = np.fromfile(
        str(image_path),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_data,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Unable to load image.")

    return image


# ==========================================
# 3. OCR LANGUAGE SUPPORT
# ==========================================

def get_tesseract_languages():

    try:
        return set(
            pytesseract.get_languages(config="")
        )

    except (
        pytesseract.TesseractNotFoundError,
        pytesseract.TesseractError
    ):
        return set()


# ==========================================
# 4. IMPROVED IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    # Enlarge small images
    if max(height, width) < 1800:

        scale = 3 if max(height, width) < 900 else 2

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # Improve contrast
    contrast = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    # Remove noise
    denoised = cv2.GaussianBlur(
        contrast,
        (3, 3),
        0
    )

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return [contrast, adaptive]


# ==========================================
# 5. IMPROVED OCR EXTRACTION
# ==========================================

def extract_text(image_path):

    image = load_image(image_path)

    height, width = image.shape[:2]

    installed_languages = get_tesseract_languages()

    if "eng" not in installed_languages:
        raise RuntimeError(
            "English OCR data is missing. "
            "Install eng.traineddata."
        )

    languages = ["eng"]

    if "ara" in installed_languages:
        languages.append("ara")
    else:
        print(
            "[!] Arabic OCR data not found. "
            "Using English only."
        )

    language_config = "+".join(languages)

    # Full image and smaller regions
    regions = [
        image,
        image[:height // 2, :],
        image[height // 2:, :],
        image[:, :width // 2],
        image[:, width // 2:]
    ]

    detected_words = []

    for region in regions:

        prepared_images = preprocess_image(region)

        for prepared in prepared_images:

            for psm in (6, 11):

                try:

                    data = pytesseract.image_to_data(
                        prepared,
                        lang=language_config,
                        config=f"--oem 3 --psm {psm}",
                        output_type=pytesseract.Output.DICT
                    )

                    for word, confidence in zip(
                        data["text"],
                        data["conf"]
                    ):

                        word = word.strip()

                        try:
                            confidence = float(confidence)
                        except (TypeError, ValueError):
                            continue

                        # Filter weak OCR results
                        if confidence >= 50 and len(word) >= 2:

                            detected_words.append(
                                (word, confidence)
                            )

                except pytesseract.TesseractError as error:

                    print(
                        f"[!] OCR error: {error}"
                    )

    if not detected_words:
        return [], ""

    # Keep the highest-confidence occurrence of each word
    best_words = {}

    for word, confidence in detected_words:

        key = word.casefold()

        if (
            key not in best_words
            or confidence > best_words[key][1]
        ):
            best_words[key] = (word, confidence)

    final_words = [
        item[0]
        for item in best_words.values()
    ]

    raw_text = " ".join(final_words)

    return final_words, raw_text


# ==========================================
# 6. LANGUAGE DETECTION
# ==========================================

def detect_languages(text):

    languages = []

    if re.search(r"[A-Za-z]", text):
        languages.append("English")

    if re.search(r"[\u0600-\u06FF]", text):
        languages.append("Arabic")

    if not languages:
        return "Unknown (no readable text extracted)"

    return " / ".join(languages)


# ==========================================
# 7. OSINT CLUE ANALYSIS
# ==========================================

def analyze_clues(lines, raw_text):

    clues = {
        "Websites": [],
        "Emails": [],
        "Phones": [],
        "Streets": [],
        "Businesses": []
    }

    # Website detection
    url_pattern = re.compile(
        r"\b(?:https?://|www\.)"
        r"[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        re.IGNORECASE
    )

    email_pattern = re.compile(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    # Phone numbers
    phone_pattern = re.compile(
        r"(?<!\w)\+?\d[\d\s()-]{6,18}\d(?!\w)"
    )

    # Use full words to avoid false matches like "st"
    street_pattern = re.compile(
        r"\b(street|road|avenue|boulevard|"
        r"lane|drive|highway)\b",
        re.IGNORECASE
    )

    business_pattern = re.compile(
        r"\b(pharmacy|shop|store|market|"
        r"supermarket|cafe|coffee|restaurant|"
        r"hotel|bank|clinic|hospital|center|"
        r"centre|company|bakery|salon|mall)\b",
        re.IGNORECASE
    )

    clues["Websites"] = sorted(
        set(url_pattern.findall(raw_text))
    )

    clues["Emails"] = sorted(
        set(email_pattern.findall(raw_text))
    )

    phones = set()

    for phone in phone_pattern.findall(raw_text):

        digits = re.sub(r"\D", "", phone)

        if 7 <= len(digits) <= 15:
            phones.add(phone.strip())

    clues["Phones"] = sorted(phones)

    for line in lines:

        if street_pattern.search(line):
            clues["Streets"].append(line)

        if business_pattern.search(line):
            clues["Businesses"].append(line)

    clues["Streets"] = sorted(
        set(clues["Streets"])
    )

    clues["Businesses"] = sorted(
        set(clues["Businesses"])
    )

    return clues


# ==========================================
# 8. REPORT GENERATION
# ==========================================

def generate_report(
    image_path,
    lines,
    languages,
    metadata,
    clues
):

    report = []

    report.append("=" * 60)
    report.append("                 PHOTOCLUE OSINT")
    report.append("          IMAGE ANALYSIS REPORT")
    report.append(f"                 Version {VERSION}")
    report.append("=" * 60)

    report.append(
        f"\nImage: {Path(image_path).name}"
    )

    # DETECTED TEXT
    report.append("\n[1] DETECTED TEXT")
    report.append("-" * 60)

    if lines:

        for line in lines:
            report.append(f"  {line}")

    else:
        report.append("  No readable text detected.")

    # LANGUAGE
    report.append("\n[2] LANGUAGE ANALYSIS")
    report.append("-" * 60)

    report.append(
        f"  Detected language: {languages}"
    )

    # METADATA
    report.append("\n[3] IMAGE METADATA")
    report.append("-" * 60)

    report.append(
        f"  Camera: {metadata['Camera']}"
    )

    report.append(
        f"  Date: {metadata['Date']}"
    )

    report.append(
        f"  GPS: {metadata['GPS']}"
    )

    if metadata["Coordinates"]:

        lat, lon = metadata["Coordinates"]

        map_link = (
            "https://www.openstreetmap.org/"
            f"?mlat={lat}&mlon={lon}"
            f"#map=16/{lat}/{lon}"
        )

        report.append(
            f"  Map: {map_link}"
        )

    # OSINT CLUES
    report.append("\n[4] POTENTIAL OSINT CLUES")
    report.append("-" * 60)

    categories = [
        ("Businesses", "Business"),
        ("Streets", "Street"),
        ("Websites", "Website"),
        ("Emails", "Email"),
        ("Phones", "Phone")
    ]

    found = False

    for category, label in categories:

        for item in clues[category]:

            found = True

            report.append(
                f"  - {label}: {item}"
            )

    if not found:
        report.append(
            "  No structured clues identified."
        )

    # DISCLAIMER
    report.append("\n[5] DISCLAIMER")
    report.append("-" * 60)

    report.append(
        "  Results are automatically extracted clues."
    )

    report.append(
        "  OCR results may contain errors or false positives."
    )

    report.append(
        "  GPS metadata may be missing, inaccurate, or outdated."
    )

    report.append(
        "  Text and business names are not verified facts."
    )

    report.append(
        "  Visual clues do not prove an exact location."
    )

    report.append(
        "  Findings do not independently establish a person's identity."
    )

    report.append("=" * 60)

    return "\n".join(report)


# ==========================================
# 9. MAIN PROGRAM
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="PhotoClue OSINT Image Analysis Tool"
    )

    parser.add_argument(
        "image",
        help="Image file path"
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Save report to a text file"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"PhotoClue OSINT {VERSION}"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():

        print(
            f"[ERROR] Image not found: {image_path}"
        )

        sys.exit(1)

    print("\n[*] PhotoClue OSINT starting...")

    print("[*] Extracting image metadata...")

    available_languages = get_tesseract_languages()

    print(
        "[*] Tesseract languages available: "
        + (
            ", ".join(sorted(available_languages))
            if available_languages
            else "None"
        )
    )

    try:

        metadata = extract_exif_metadata(image_path)

        print(
            "[*] Processing image and extracting text..."
        )

        lines, raw_text = extract_text(image_path)

        print("[*] Detecting languages...")

        languages = detect_languages(raw_text)

        print(
            "[*] Analyzing potential OSINT clues..."
        )

        clues = analyze_clues(lines, raw_text)

        print("[*] Generating report...")

        report = generate_report(
            image_path,
            lines,
            languages,
            metadata,
            clues
        )

        print("\n" + report)

        if args.output:

            Path(args.output).write_text(
                report,
                encoding="utf-8"
            )

            print(
                f"\n[+] Report saved: {args.output}"
            )

        print(
            "\n[+] Analysis completed successfully."
        )

    except Exception as error:

        print(
            f"\n[ERROR] {error}",
            file=sys.stderr
        )

        sys.exit(1)


if __name__ == "__main__":
    main()

