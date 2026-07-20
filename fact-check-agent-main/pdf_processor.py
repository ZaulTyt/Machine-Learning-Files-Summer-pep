import fitz  # PyMuPDF
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text(pdf_file) -> str:
    """
    Extracts text from a PDF file stream or a file path.
    
    Args:
        pdf_file: A file-like object (e.g., BytesIO, Streamlit UploadedFile) or a path string.
        
    Returns:
        str: Extracted plain text content.
    """
    text_content = []
    
    try:
        # Open PDF from stream or path
        if isinstance(pdf_file, (str, bytes)):
            doc = fitz.open(pdf_file)
        else:
            # Handle Streamlit UploadedFile or BytesIO
            pdf_bytes = pdf_file.read()
            # Reset seek position just in case
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
        logger.info(f"Opened PDF with {len(doc)} pages.")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            if page_text.strip():
                text_content.append(page_text)
                
        doc.close()
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError(f"Failed to process PDF: {str(e)}")
        
    extracted_text = "\n\n".join(text_content).strip()
    if not extracted_text:
        raise ValueError("The PDF contains no extractable text. It might be scanned or empty.")
        
    return extracted_text
