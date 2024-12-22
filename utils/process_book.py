import utils.image_utils
import utils.number_utils
import pandas as pd
import cv2
import pymupdf
import os
import numpy as np

volume = 1
selected_pages = [31]
limit = 500
preview = True
show_original = True

if selected_pages is None or len(selected_pages) > 3:
    preview = False

img_root = f'./in/raw/birdsEurope{utils.number_utils.roman_number(volume)}Goul_jp2'
pdf_path = f'./in/raw/birdsEurope{utils.number_utils.roman_number(volume)}Goul.pdf'
index_path = f'./in/raw/birdsEurope{utils.number_utils.roman_number(volume)}Goul.csv'

index = pd.read_csv(index_path)
pdf = pymupdf.open(pdf_path)

page_map = {}

for page_number in range(len(pdf)):
    page = pdf[page_number]
    physical_number = page_number + 1
    label = page.get_label()
    if label is not None:
        page_map[label] = physical_number


for i, row in index.iterrows():
    if i == limit:
        break
    
    if selected_pages and row['page'] not in selected_pages:
        continue
    
    # the page is the labelled page number, not the actual pdf page
    physical_page = page_map[str(row['page'])]
    
    # get the orientation of text on the page to check if it is horizontal or vertical
    page = pdf[physical_page - 1]
    blocks = page.get_text("blocks")
        
    weighted_count_horizontal = 0
    weighted_count_vertical = 0
    for block in blocks:
        x0, y0, x1, y1, text = block[:5]
        width = x1 - x0
        height = y1 - y0

        if width > height:
            weighted_count_horizontal += width * text.count('\n')
        else:
            weighted_count_vertical += height * text.count('\n')

    # open the image
    img_path = f'{img_root}/birdsEurope{utils.number_utils.roman_number(volume)}Goul_{physical_page:04d}.jp2'
    
    # check if the image exists
    if not os.path.exists(img_path):
        print(f'Image file {img_path} does not exist')
        continue
    
    img = cv2.imread(img_path)
    
    if weighted_count_vertical > weighted_count_horizontal:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        
    avg_bg_color = utils.image_utils.detect_background_color_kmeans(img)
    corrected_image = utils.image_utils.correct_color_balance(img, avg_bg_color, strength=1.0)
        
    
    if preview:
        cv2.imshow(f"Corrected: {row['en_name']}", utils.image_utils.resize_preview(corrected_image, 600))
        
    corrected_image, mask = utils.image_utils.reduce_yellow(corrected_image, img, preview=preview, name=row["en_name"], tolerance=30)
    corrected_image = utils.image_utils.remove_text_remains(corrected_image, preview=preview, name=row["en_name"])
    
    corrected_image = cv2.bilateralFilter(corrected_image, 9, 75, 75)

    # apply a brightness and contrast adjustment only on the non-white pixels
    non_white_mask = np.all(corrected_image != [255, 255, 255], axis=-1)
        
    brightness = -40
    contrast = 30
    corrected_image = np.int16(corrected_image)
    corrected_image[non_white_mask] = (
        corrected_image[non_white_mask] * (contrast / 127 + 1) - contrast + brightness
    )
    corrected_image = np.clip(corrected_image, 0, 255)
    corrected_image = np.uint8(corrected_image)
    
    padding = 50
    corrected_image = utils.image_utils.crop_image_to_subject(corrected_image, mask, padding=0, preview=preview, name=row["en_name"])
    corrected_image = cv2.copyMakeBorder(corrected_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    if show_original and preview:
        cv2.imshow(f"Original: {row["en_name"]}", utils.image_utils.resize_preview(img, 600))
    if preview:    
        cv2.imshow(row["en_name"], utils.image_utils.resize_preview(corrected_image, 600))
    
    if not os.path.exists(f'./out/processed/{utils.number_utils.roman_number(volume)}'):
        os.makedirs(f'./out/processed/{utils.number_utils.roman_number(volume)}')
    
    cv2.imwrite(f'./out/processed/{utils.number_utils.roman_number(volume)}/{row["page"]}.jpg', corrected_image)
    print(f'Processed {row["en_name"]} - {row["page"]}')
    
cv2.waitKey(0)
cv2.destroyAllWindows()