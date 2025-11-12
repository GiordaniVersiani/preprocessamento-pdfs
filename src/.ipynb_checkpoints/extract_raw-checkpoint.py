import pdfplumber
import re

def extract_raw(pdf_path: str, header_height_ratio: float = 0.1, footer_height_ratio: float = 0.1) -> str:
    """
    Extrai texto bruto de um PDF, removendo cabeçalhos, rodapés e numeração de páginas
    com base na análise de layout usando pdfplumber.

    Args:
        pdf_path (str): O caminho para o arquivo PDF.
        header_height_ratio (float): Proporção da altura da página para considerar como região de cabeçalho.
        footer_height_ratio (float): Proporção da altura da página para considerar como região de rodapé.

    Returns:
        str: O texto extraído e limpo do PDF.
    """
    all_cleaned_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_height = page.height
            page_width = page.width

            header_top = 0
            header_bottom = page_height * header_height_ratio
            footer_top = page_height * (1 - footer_height_ratio)
            footer_bottom = page_height

            page_text_elements = page.extract_words(x_tolerance=1, y_tolerance=1, keep_blank_chars=False, use_text_flow=True, horizontal_ltr=True, vertical_ttb=True)
            cleaned_lines = []
            current_line_text = ""
            current_line_y = None

            page_text_elements.sort(key=lambda x: (x["top"], x["x0"]))

            for word in page_text_elements:
                x0, y0, x1, y1 = word["x0"], word["top"], word["x1"], word["bottom"]

                is_header = y0 < header_bottom
                is_footer = y1 > footer_top

                if not is_header and not is_footer:
                    word_text = word["text"].strip()
                    if re.fullmatch(r"\d+", word_text) or \
                       re.fullmatch(r"\(\d+\)", word_text) or \
                       re.fullmatch(r"\[\d+\]", word_text):
                        continue

                    if current_line_y is None or abs(y0 - current_line_y) < (word["height"] / 2):
                        current_line_text += (" " if current_line_text else "") + word_text
                        current_line_y = y0
                    else:
                        cleaned_lines.append(current_line_text.strip())
                        current_line_text = word_text
                        current_line_y = y0
            if current_line_text:
                cleaned_lines.append(current_line_text.strip())

            page_content = "\n".join(line for line in cleaned_lines if line)
            if page_content:
                all_cleaned_text.append(page_content)

    return "\n\n".join(all_cleaned_text).strip()
