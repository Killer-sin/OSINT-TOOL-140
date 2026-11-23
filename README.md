PhotoClue OSINT

PhotoClue OSINT is a Python-based image analysis tool designed to extract metadata, recognize text, and identify potential information clues from images.

Features
EXIF metadata extraction
Camera manufacturer and model detection
Image timestamp extraction
GPS coordinate extraction
OpenStreetMap coordinate links
OCR using Tesseract
English and Arabic text recognition
Website and domain detection
Email address extraction
Phone number detection
Street and business keyword detection
Automatic text report generation
Requirements
Python 3.9+
Tesseract OCR
English Tesseract language data
Arabic Tesseract language data (optional)
Installation

Clone the repository:

bash
git clone https://github.com/YOUR-USERNAME/PhotoClue-OSINT.git
cd PhotoClue-OSINT


Install dependencies:

bash
pip install -r requirements.txt


Install Tesseract OCR separately and ensure it is available in your system PATH.

For Arabic OCR, install the Arabic language data (ara).

Usage

Basic analysis:

bash
python photoclue.py image.jpg


Save the report:

bash
python photoclue.py image.jpg -o report.txt


Display version:

bash
python photoclue.py --version


Display help:

bash
python photoclue.py --help

Disclaimer

This project is intended for educational purposes and authorized image analysis.

Extracted information may be incomplete or inaccurate. Always respect privacy, applicable laws, and data protection requirements.

License

MIT License.
