import os
import re
import sys
import argparse
import cv2
import pytesseract
from PIL import Image, ExifTags


def get_decimal_from_dms(dms, ref):
    degrees = float(dms[0])
    minutes = float(dms[1]) / 60.0
    seconds = float(dms[2]) / 3600.0
    decimal = degrees + minutes + seconds
    if ref in ['S', 'W']:
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
            exif_raw = img._getexif()
            if not exif_raw:
                return metadata

            exif_data = {}
            for tag_id, value in exif_raw.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[tag] = value

            make = str(exif_data.get("Make", "")).strip()
            model = str(exif_data.get("Model", "")).strip()
            if make or model:
                metadata["Camera"] = f"{make} {model}".strip()

            date_taken = exif_data.get("DateTimeOriginal") or exif_data.get("DateTime")
            if date_taken:
                metadata["Date"] = str(date_taken).strip()

            gps_info = exif_data.get("GPSInfo")
            if gps_info:
                gps_parsed = {}
                for key in gps_info.keys():
                    decode = ExifTags.GPSTAGS.get(key, key)
                    gps_parsed[decode] = gps_info[key]

                lat = gps_parsed.get("GPSLatitude")
                lat_ref = gps_parsed.get("GPSLatitudeRef")
                lon = gps_parsed.get("GPSLongitude")
                lon_ref = gps_parsed.get("GPSLongitudeRef")

                if lat and lat_ref and lon and lon_ref:
                    lat_dec = get_decimal_from_dms(lat, lat_ref)
                    lon_dec = get_decimal_from_dms(lon, lon_ref)
                    metadata["GPS"] = f"{lat_dec}, {lon_dec}"
                    metadata["Coordinates"] = (lat_dec, lon_dec)
    except Exception:
        pass

    return metadata


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_text(image_path):
    processed_img = preprocess_image(image_path)
    try:
        raw_text = pytesseract.image_to_string(processed_img, lang="eng+ara")
    except pytesseract.TesseractError:
        raw_text = pytesseract.image_to_string(processed_img, lang="eng")

    lines = [line.strip() for line in raw_text.splitlines() if len(line.strip()) > 1]
    return lines, raw_text


def detect_languages(text):
    languages = []
    if re.search(r'[A-Za-z]', text):
        languages.append("English")
    if re.search(r'[\u0600-\u06FF]', text):
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

    url_pattern = re.compile(r'\b(?:https?://|www\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\b', re.IGNORECASE)
    domain_pattern = re.compile(r'\b[a-zA-Z0-9-]+\.(?:com|net|org|io|co|gov|edu|bh|ae|sa|uk|us)\b', re.IGNORECASE)
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    phone_pattern = re.compile(r'(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}')

    street_keywords = r'\b(?:street|st|road|rd|avenue|ave|boulevard|blvd|lane|ln|drive|dr|way|highway|hwy|شارع|طريق|مجمع)\b'
    business_keywords = r'\b(?:pharmacy|shop|store|market|supermarket|cafe|coffee|restaurant|hotel|bank|clinic|hospital|center|centre|ltd|llc|inc|co|corp|company|bakery|salon|mall|صيدلية|مطعم|مقهى|محل|شركة|مركز|بنك)\b'

    urls = set(url_pattern.findall(raw_text))
    domains = set(domain_pattern.findall(raw_text))
    clues["Websites"] = sorted(list(urls.union(domains)))
    clues["Emails"] = sorted(list(set(email_pattern.findall(raw_text))))
    clues["Phones"] = sorted(list(set(p.strip() for p in phone_pattern.findall(raw_text) if len(re.sub(r'\D', '', p)) >= 7)))

    for line in lines:
        if re.search(street_keywords, line, re.IGNORECASE):
            clues["Streets"].append(line)
        if re.search(business_keywords, line, re.IGNORECASE):
            clues["Businesses"].append(line)

    clues["Streets"] = sorted(list(set(clues["Streets"])))
    clues["Businesses"] = sorted(list(set(clues["Businesses"])))

    return clues


def generate_report(image_path, lines, languages, metadata, clues):
    filename = os.path.basename(image_path)
    report = []
    report.append("=" * 50)
    report.append("PHOTOCLUE OSINT REPORT")
    report.append("=" * 50)
    report.append(f"Image: {filename}\n")

    report.append("Detected Text:")
    if lines:
        for line in lines:
            report.append(f'  "{line}"')
    else:
        report.append("  No readable text detected.")
    report.append("")

    report.append("Possible Language:")
    report.append(f"  {languages}\n")

    report.append("Image Metadata:")
    report.append(f"  Camera: {metadata['Camera']}")
    report.append(f"  Date: {metadata['Date']}")
    report.append(f"  GPS: {metadata['GPS']}")
    if metadata["Coordinates"]:
        lat, lon = metadata["Coordinates"]
        report.append(f"  Map Link: https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=16/{lat}/{lon}")
    report.append("")

    report.append("Potential OSINT Clues:")
    has_clues = False
    if clues["Businesses"]:
        has_clues = True
        for b in clues["Businesses"]:
            report.append(f"  - Business name: {b}")
    if clues["Streets"]:
        has_clues = True
        for s in clues["Streets"]:
            report.append(f"  - Street name: {s}")
    if clues["Websites"]:
        has_clues = True
        for w in clues["Websites"]:
            report.append(f"  - Website/domain: {w}")
    if clues["Emails"]:
        has_clues = True
        for e in clues["Emails"]:
            report.append(f"  - Email address: {e}")
    if clues["Phones"]:
        has_clues = True
        for p in clues["Phones"]:
            report.append(f"  - Phone number: {p}")
    if not has_clues:
        report.append("  - No specific structured clues identified from text.")
    report.append("")

    report.append("Note:")
    report.append("  The results are clues only and do not automatically confirm")
    report.append("  the exact location or identity of a person.")
    report.append("=" * 50)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="PhotoClue OSINT - Image Reconnaissance Tool")
    parser.add_argument("image", help="Path to the input image file (JPG, PNG, etc.)")
    parser.add_argument("-o", "--output", help="Optional path to save the report as a text file")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: File '{args.image}' not found.")
        sys.exit(1)

    try:
        metadata = extract_exif_metadata(args.image)
        lines, raw_text = extract_text(args.image)
        languages = detect_languages(raw_text)
        clues = analyze_clues(lines, raw_text)
        report = generate_report(args.image, lines, languages, metadata, clues)

        print(report)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\nReport saved to {args.output}")

    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
