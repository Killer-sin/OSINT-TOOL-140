
import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from PIL import Image, ExifTags


VERSION = "2.0.0"


def rational_to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def get_decimal_from_dms(dms, ref):
    degrees = rational_to_float(dms[0])
    minutes = rational_to_float(dms[1])
    seconds = rational_to_float(dms[2])

    decimal = degrees + (minutes / 60) + (seconds / 3600)

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
                metadata["Date"] = str(date_taken).strip()

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
        print(f"[!] EXIF warning: {error}", file=sys.stderr)

    return metadata


def preprocess_image(image_path):
    try:
        image_data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Unable to decode image.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape

        if width < 2000:
            gray = cv2.resize(
                gray,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )

        denoised = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return thresh

    except Exception as error:
        raise ValueError(f"Image preprocessing failed: {error}") from error


def extract_text(image_path):
    processed_img = preprocess_image(image_path)

    try:
        raw_text = pytesseract.image_to_string(
            processed_img,
            lang="eng+ara"
        )

    except pytesseract.TesseractError:
        print(
            "[!] Arabic OCR unavailable. Falling back to English.",
            file=sys.stderr
        )

        raw_text = pytesseract.image_to_string(
            processed_img,
            lang="eng"
        )

    lines = [
        line.strip()
        for line in raw_text.splitlines()
        if len(line.strip()) > 1
    ]

    return lines, raw_text


def detect_languages(text):
    languages = []

    if re.search(r"[A-Za-z]", text):
        languages.append("English")

    if re.search(r"[\u0600-\u06FF]", text):
        languages.append("Arabic")

    return " / ".join(languages) if languages else "Unknown"


def analyze_clues(lines, raw_text):
    clues = {
        "Websites": [],
        "Emails": [],
        "Phones": [],
        "Streets": [],
        "Businesses": []
    }

    url_pattern = re.compile(
        r"\b(?:https?://|www\.)"
        r"[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        r"(?:/[^\s]*)?",
        re.IGNORECASE
    )

    domain_pattern = re.compile(
        r"\b[a-zA-Z0-9-]+\."
        r"(?:com|net|org|io|co|gov|edu|bh|ae|sa|uk|us)\b",
        re.IGNORECASE
    )

    email_pattern = re.compile(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    phone_pattern = re.compile(
        r"(?<!\w)(?:\+?\d{1,3}[\s-]?)?"
        r"(?:\(?\d{2,4}\)?[\s-]?)?"
        r"\d{3,4}[\s-]?\d{4}(?!\w)"
    )

    street_keywords = (
        r"\b(?:street|st|road|rd|avenue|ave|"
        r"boulevard|blvd|lane|ln|drive|dr|"
        r"way|highway|hwy|شارع|طريق|مجمع)\b"
    )

    business_keywords = (
        r"\b(?:pharmacy|shop|store|market|supermarket|"
        r"cafe|coffee|restaurant|hotel|bank|clinic|"
        r"hospital|center|centre|ltd|llc|inc|co|"
        r"corp|company|bakery|salon|mall|"
        r"صيدلية|مطعم|مقهى|محل|شركة|مركز|بنك)\b"
    )

    urls = set(url_pattern.findall(raw_text))
    domains = set(domain_pattern.findall(raw_text))

    clues["Websites"] = sorted(urls.union(domains))
    clues["Emails"] = sorted(set(email_pattern.findall(raw_text)))

    phones = set()

    for phone in phone_pattern.findall(raw_text):
        digits = re.sub(r"\D", "", phone)

        if 7 <= len(digits) <= 15:
            phones.add(phone.strip())

    clues["Phones"] = sorted(phones)

    for line in lines:
        if re.search(street_keywords, line, re.IGNORECASE):
            clues["Streets"].append(line)

        if re.search(business_keywords, line, re.IGNORECASE):
            clues["Businesses"].append(line)

    clues["Streets"] = sorted(set(clues["Streets"]))
    clues["Businesses"] = sorted(set(clues["Businesses"]))

    return clues


def generate_report(image_path, lines, languages, metadata, clues):
    filename = Path(image_path).name

    report = []

    report.append("=" * 60)
    report.append("                 PHOTOCLUE OSINT")
    report.append("          IMAGE ANALYSIS REPORT")
    report.append(f"                 Version {VERSION}")
    report.append("=" * 60)

    report.append(f"\nImage: {filename}")

    report.append("\n[1] DETECTED TEXT")
    report.append("-" * 60)

    if lines:
        for line in lines:
            report.append(f'  "{line}"')
    else:
        report.append("  No readable text detected.")

    report.append("\n[2] LANGUAGE ANALYSIS")
    report.append("-" * 60)
    report.append(f"  Detected language: {languages}")

    report.append("\n[3] IMAGE METADATA")
    report.append("-" * 60)

    report.append(f"  Camera: {metadata['Camera']}")
    report.append(f"  Date:   {metadata['Date']}")
    report.append(f"  GPS:    {metadata['GPS']}")

    if metadata["Coordinates"]:
        lat, lon = metadata["Coordinates"]

        map_link = (
            f"https://www.openstreetmap.org/"
            f"?mlat={lat}&mlon={lon}#map=16/{lat}/{lon}"
        )

        report.append(f"  Map Link: {map_link}")

    report.append("\n[4] POTENTIAL OSINT CLUES")
    report.append("-" * 60)

    categories = [
        ("Businesses", "Business"),
        ("Streets", "Street"),
        ("Websites", "Website / Domain"),
        ("Emails", "Email"),
        ("Phones", "Phone")
    ]

    has_clues = False

    for category, label in categories:
        if clues[category]:
            has_clues = True

            for item in clues[category]:
                report.append(f"  - {label}: {item}")

    if not has_clues:
        report.append("  No structured clues identified.")

    report.append("\n[5] DISCLAIMER")
    report.append("-" * 60)
    report.append("  Results are automatically extracted clues.")
    report.append("  OCR results may contain errors or false positives.")
    report.append("  GPS metadata may be missing, inaccurate, or outdated.")
    report.append("  Findings do not independently establish a person's")
    report.append("  identity or the exact location of an image.")

    report.append("=" * 60)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="PhotoClue OSINT - Image Metadata and Text Analysis Tool",
        epilog="Example: python photoclue.py photo.jpg -o report.txt"
    )

    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("-o", "--output", help="Optional report output path")

    parser.add_argument(
        "--version",
        action="version",
        version=f"PhotoClue OSINT {VERSION}"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    print("\n[*] PhotoClue OSINT starting...")
    print("[*] Extracting image metadata...")

    try:
        metadata = extract_exif_metadata(image_path)

        print("[*] Processing image and extracting text...")

        lines, raw_text = extract_text(image_path)

        print("[*] Detecting languages...")

        languages = detect_languages(raw_text)

        print("[*] Analyzing potential OSINT clues...")

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
            output_path = Path(args.output)

            output_path.write_text(
                report,
                encoding="utf-8"
            )

            print(f"\n[+] Report saved: {output_path}")

        print("\n[+] Analysis completed successfully.")

    except Exception as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
