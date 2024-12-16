import utils.number_utils
import os
import fitz
import pandas as pd

# WARNING: This might need some post-processing, as the text extraction is not perfect

# Volume 1 - 27 - 1
# Volume 2 - 13,14 - 51
# Volume 3 - 11,12 - 151

volume = 3
list_of_plates = [11,12] # list of plates to extract the index from, this is the actual page number in the PDF
sites_start_at = 151 # the page number where the sites start


pdf_path = f'./in/raw/birdsEurope{utils.number_utils.roman_number(volume)}Goul.pdf'

if not os.path.exists(pdf_path):
    print(f'PDF file {pdf_path} does not exist')
    exit(1)
    
pdf_document = fitz.open(pdf_path)

text = []
for list_of_plate in list_of_plates:
    page = pdf_document[list_of_plate - 1]
    words = page.get_text().split("\n")
    
    text.extend(words)
    
print(text)
    
# remove empty lines and lines that only contain punctuation
text = [word for word in text if any([char.isalnum() for char in word])]
# remove leading and trailing punctuation
text = [word.strip(".,;:!?—- ") for word in text]
# remove words that are only one character long
text = [word for word in text if len(word) > 1 or word.isdigit()]
# find all relevant triples
triples = [(text[i], text[i+1], text[i+2]) for i in range(len(text) - 2) if text[i+2].isdigit() and text[i+1].isdigit() == False and text[i].isdigit() == False]

# remove the author name from scientific names
triples = [(triple[0], triple[1].split(",")[0], triple[2]) for triple in triples]

# if the second entry contains only one word find the last triple that contains two words in the second entry, and use the first word of that plus the word from the current triple
for i in range(len(triples)):
    last_double = None
    if len(triples[i][1].split()) == 1:
        for j in range(i-1, -1, -1):
            if len(triples[j][1].split()) == 2:
                last_double = triples[j]
                break
        if last_double is not None:
            triples[i] = (triples[i][0], last_double[1].split()[0] + " " + triples[i][1], triples[i][2])
            
# convert the last entry to an integer
triples = [(triple[0], triple[1], int(triple[2])) for triple in triples]

df = pd.DataFrame(triples, columns=["en_name", "sci_name", "page"])

# find missing numbers in the page column
missing_pages = set(range(sites_start_at, df["page"].max() + 1)) - set(df["page"])
# add the missing pages with empty strings
missing_df = pd.DataFrame([("", "", page) for page in missing_pages], columns=["en_name", "sci_name", "page"])
df = pd.concat([df, missing_df], ignore_index=True)
# sort by page number
df = df.sort_values("page")

# save as csv in in/raw
df.to_csv(f'./out/birdsEurope{utils.number_utils.roman_number(volume)}Goul.csv', index=False)



