import logging
import os
from io import BytesIO
import json
import PyPDF2
from langchain.text_splitter import TokenTextSplitter
from docx import Document as Docx
from message_clients import MessageData


class Document:
    def __init__(self, filename: str, file_data: bytes):
        self.filename = filename
        self.file_data = file_data

    @property
    def file_type(self):
        _, file_extension = os.path.splitext(self.filename)
        return file_extension.lower()

    def textualize_file(self):
        if ".pdf" == self.file_type:
            with BytesIO(self.file_data) as stream:
                pdf_reader = PyPDF2.PdfReader(stream)
                text = ""
                for page in pdf_reader.pages:
                    text += "\n"
                    text += page.extract_text()
            return text
        elif ".docx" == self.file_type:
            with BytesIO(self.file_data) as stream:
                doc = Docx(stream)
                text = [para.text for para in doc.paragraphs]
                # Extracting text from tables
                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            for para in cell.paragraphs:
                                text.append(para.text)
                text = '\n'.join(text)
            return text
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

    def chunk_text(self, text, context_length):
        text_splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=context_length, chunk_overlap=1000)
        chunked_text = text_splitter.split_text(text)
        return chunked_text

    def get_file_content(self) -> str:
        file_text = self.textualize_file()

        return file_text

    def get_chunked_text(self, context_length):
        merged_text = self.get_file_content()
        chunked_text = self.chunk_text(merged_text, context_length)

        return chunked_text


class DocumentLoader:
    def __init__(self):
        pass

    def load_document(self, message: MessageData) -> Document:
        return None


class LocalDocumentLoader(DocumentLoader):
    def __init__(self, base_directory: str):
        super().__init__()
        self.base_directory = base_directory

    def load_document(self, message: MessageData) -> Document:
        with open(os.path.join(self.base_directory, message.context), 'rb') as f:
            doc = f.read()
        return Document(message.context, doc)
