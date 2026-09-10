import logging
import os
from io import BytesIO
from typing import List, Tuple
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document as Docx


# Tokens of overlap between consecutive chunks, so a sentence cut at a boundary still
# appears whole in one of them.
CHUNK_OVERLAP_TOKENS = 200

# Boundaries the splitter tries, in order, before falling back to a hard cut. Blank line
# first (a paragraph), then a line break, then a sentence end, then a word. The previous
# TokenTextSplitter had no notion of any of these: it counted tokens and cut wherever it
# landed, which is why retrieved chunks used to start mid-sentence ("discretos.",
# "máx P .") and why a chunk could carry the tail of one topic and the head of another.
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Prepended to every chunk. It costs ~15 tokens and buys two things: the retriever gets
# the document title as a signal (the course files are named by topic, e.g. "Amplificador
# diferencial.pdf"), and an answer becomes traceable to the source it came from.
SOURCE_HEADER = "[Fuente: {source}]\n"


class Document:
    def __init__(self, filename: List[str], file_data: List[bytes]):
        self.filename = filename
        self.file_data = file_data

    def file_type(self, filename: str):
        _, file_extension = os.path.splitext(filename)
        return file_extension.lower()

    def extract_text(self, filename: str, data: bytes) -> str:
        """Extract the plain text of a single source document.

        Args:
            filename (str): Name of the document, used to pick the reader and to report
                an unsupported extension.
            data (bytes): Raw file contents.

        Raises:
            ValueError: The extension is neither .pdf nor .docx.

        Returns:
            str: The document's text.
        """
        extension = self.file_type(filename)
        if extension == ".pdf":
            with BytesIO(data) as stream:
                pdf_reader = PyPDF2.PdfReader(stream)
                return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        if extension == ".docx":
            with BytesIO(data) as stream:
                doc = Docx(stream)
                paragraphs = [para.text for para in doc.paragraphs]
                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            paragraphs.extend(para.text for para in cell.paragraphs)
            return "\n".join(paragraphs)
        raise ValueError(f"Unsupported file type: {extension} ({filename})")

    def textualize_file(self) -> List[Tuple[str, str]]:
        """Extract every source document, keeping each one separate.

        Returns:
            List[Tuple[str, str]]: One ``(filename, text)`` pair per document, in the
            order the loader read them.
        """
        return [(self.filename[index], self.extract_text(self.filename[index], data))
                for index, data in enumerate(self.file_data)]

    def chunk_text(self, text: str, context_length: int) -> List[str]:
        """Split one document's text into chunks of at most ``context_length`` tokens.

        Args:
            text (str): The document text to split.
            context_length (int): Chunk budget in cl100k_base TOKENS, not characters.
                ``from_tiktoken_encoder`` is what makes the recursive splitter measure in
                tokens; without it ``chunk_size`` would mean characters and the budget
                computed in ``rag/main.py`` would be wrong by a factor of ~3.

        Returns:
            List[str]: The chunks, cut at the most structural boundary available.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=context_length,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            separators=CHUNK_SEPARATORS,
        )
        return text_splitter.split_text(text)

    def get_file_content(self) -> str:
        """Return every document's text concatenated.

        Kept for callers that want the raw corpus. Chunking does NOT go through here:
        see ``get_chunked_text``.

        Returns:
            str: All documents joined with a blank line between them.
        """
        return "\n\n".join(text for _, text in self.textualize_file())

    def get_chunked_text(self, context_length: int) -> List[str]:
        """Chunk the corpus one document at a time, tagging each chunk with its source.

        Chunking per document rather than over one concatenated string is what stops a
        chunk from straddling two unrelated files -- ending in a page about amplifier
        stages and beginning in one about noise. A straddling chunk is worse than useless
        to the retriever: it matches on vocabulary from one topic and delivers the other.

        Args:
            context_length (int): Chunk budget in cl100k_base tokens.

        Returns:
            List[str]: Every chunk of every document, each prefixed with SOURCE_HEADER.
        """
        chunks = []
        for filename, text in self.textualize_file():
            header = SOURCE_HEADER.format(source=filename)
            document_chunks = self.chunk_text(text, context_length)
            chunks.extend(f"{header}{chunk}" for chunk in document_chunks)
            logging.debug(f"{filename}: {len(document_chunks)} chunks")
        logging.info(f"Chunked {len(self.filename)} documents into {len(chunks)} chunks")
        return chunks


class DocumentLoader:
    def __init__(self):
        pass

    def load_document(self) -> Document:
        pass


class LocalDocumentLoader(DocumentLoader):
    def __init__(self, base_directory: str):
        super().__init__()
        self.base_directory = base_directory

    def load_document(self) -> Document:
        documents = []
        files = []
        for filename in sorted(os.listdir(self.base_directory)):
            with open(os.path.join(self.base_directory, filename), 'rb') as f:
                doc = f.read()
            files.append(filename)
            documents.append(doc)
        return Document(files, documents)
