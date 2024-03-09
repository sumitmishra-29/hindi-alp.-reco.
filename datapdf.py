import fitz  
from PIL import Image

def extract_images_from_pdf(pdf_path, block_size=(1, 1), grid_size=(7, 12)):
    pdf_document = fitz.open(pdf_path)

    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)

       
        page_width_px = int(page.rect.width)
        page_height_px = int(page.rect.height)

        
        block_width_px = int(page_width_px / grid_size[0])
        block_height_px = int(page_height_px / grid_size[1])

       
        for row in range(grid_size[1]):
            for col in range(grid_size[0]):
               
                left = col * block_width_px
                upper = row * block_height_px
                right = left + block_width_px
                lower = upper + block_height_px

               
                block_pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), clip=(left, upper, right, lower))

               
                block_image = Image.frombytes("RGB", [block_pixmap.width, block_pixmap.height], block_pixmap.samples)

                
                block_image.save(f"page_{page_number + 1}_block_{row + 1}_{col + 1}.png")

if __name__ == "__main__":
    
    input_pdf_path = "crop.pdf"

    block_size_inches = (1, 1)

    grid_dimensions = (7, 12)

    extract_images_from_pdf(input_pdf_path, block_size_inches, grid_dimensions)
