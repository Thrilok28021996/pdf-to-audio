import re

import PyPDF2

from extract import extract_book

# Putting it all together:
pdf_path = "/home/dev/Downloads/houn.pdf"

# Extract and clean the text
cleaned_text = extract_book(pdf_path)

print("Completed ...........!")
