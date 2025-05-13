import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
import io
import base64
import pandas as pd

def convert_to_base64(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    base64_bytes = base64.b64encode(file_bytes)
    return base64_bytes.decode("utf-8")

def get_words(page, line):
    result = []
    for word in page.words:
        if in_span(word, line.spans):
            result.append(word)
    return result

def in_span(word, spans):
    for span in spans:
        if word.span.offset >= span.offset and (
            word.span.offset + word.span.length
        ) <= (span.offset + span.length):
            return True
    return False

def analyze_layout_from_uploaded_file(uploaded_file, endpoint, key):
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=endpoint, 
        credential=AzureKeyCredential(key)
    )
    
    base64_string = convert_to_base64(uploaded_file)
    
    # Determine content type
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        content_type = "application/pdf"
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        content_type = "image/jpeg"
    elif filename.endswith(".png"):
        content_type = "image/png"
    else:
        raise ValueError("Unsupported file type.")

    poller = document_intelligence_client.begin_analyze_document(
        model_id="prebuilt-layout", 
        analyze_request={"base64Source": base64_string}
    )
    print("Azure Document intelligence called......")
    result = poller.result()
    output = []
    
    # Text extraction
    for page in result.pages:
        if page.lines:
            for line in page.lines:
                words = get_words(page, line)
                word_texts = [word.content for word in words]
                output.append(' '.join(word_texts))

    # Table extraction
    test_df = pd.DataFrame()
    if result.tables:
        for table in result.tables:
            max_row = max(cell.row_index for cell in table.cells) + 1
            max_col = max(cell.column_index for cell in table.cells) + 1
            
            data = [['' for _ in range(max_col)] for _ in range(max_row)]

            for cell in table.cells:
                data[cell.row_index][cell.column_index] = cell.content

            test_df = pd.DataFrame(data)
            test_df.columns = test_df.iloc[0]
            test_df = test_df[1:].reset_index(drop=True)
            text_df = test_df.to_string()
            
    return text_df, output
