import logging
import os
import re
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

# A line is prose if it contains at least one plausible Spanish word: a run of four or
# more letters holding a vowel. Deliberately crude, because the alternative -- a
# dictionary -- would reject the course's own vocabulary ("transconductancia",
# "Darlington") while accepting extraction debris that happens to spell something.
PROSE_WORD_PATTERN = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñü]{4,}")
PROSE_VOWELS = frozenset("aeiouáéíóúü")

# How many consecutive non-prose lines make a block extraction debris rather than a
# formula. Six, and the number is measured rather than picked. Across the 15 course
# documents, 39% of content lines hold no plausible word -- but run length separates
# the two populations sharply, and mean line length inside a run confirms it:
#
#     run length   1     2    3-4   5-6   7-9  10-19  20+
#     chars/line  20.9  12.1   8.5   8.1   6.9   6.6   5.6
#
# Runs of one are where the intact equations live ("Tn = 290 K ( F - l). (233)",
# "θ1 = θ2 e −(to/D − to) / τ ... (A7)") -- readable, numbered, and exactly the content
# the course is about. By five the lines are symbol fragments, though a measured case in
# `disipa.pdf` ("T a + P.RT ja ≤ Tj máx") is still usable at that length, which is what
# sets the threshold above it rather than at it. Every sampled run of six or more is
# unrecoverable: loose circuit-diagram labels, or a numbered equation shattered into
# two-character pieces.
#
# The block issue #30 reports reaching a student's inbox is a run of nine, so a higher
# threshold would not have caught the bug that prompted this. Measured over the real
# corpus, six removes 4.0% of characters (464286 -> 445880) and takes the chunk count
# from 163 to 148 -- a large share of *lines*, a small share of text, because the debris
# is made of very short ones. The seven `Amplificación - *.pdf` files come through byte
# for byte identical; the damage is concentrated in the older scanned documents, peaking
# at 8.9% in `Amplificador diferencial.pdf`, which is where issue #30's block came from.
#
# What this cannot do is recover a formula. Extraction already destroyed these; the
# filter only stops the wreckage from being quoted to a student as if it were an answer.
# Recovering them needs a different extractor, which is a separate problem.
MIN_NOISE_RUN_LINES = 6


def is_prose_line(line: str) -> bool:
    """Report whether a line carries readable prose.

    Args:
        line (str): A single line of extracted text.

    Returns:
        bool: True when the line holds at least one four-letter-or-longer word with a
        vowel. Blank lines are not prose -- but they do not break a run either, see
        :func:`strip_extraction_noise`.
    """
    return any(PROSE_VOWELS & set(word.lower())
               for word in PROSE_WORD_PATTERN.findall(line))


def strip_extraction_noise(text: str) -> str:
    """Drop runs of MIN_NOISE_RUN_LINES or more consecutive non-prose lines.

    Filtering by run rather than by line is the whole point: the same per-line test
    flags both a shattered formula and an intact one, and only the length of the run
    they sit in tells them apart. See MIN_NOISE_RUN_LINES for the measurements.

    Blank lines neither start nor break a run. pypdf emits them inside mangled formula
    blocks, so counting a blank as prose would split one long run into two short ones
    and let both through.

    Args:
        text (str): One document's extracted text.

    Returns:
        str: The text with debris blocks removed and everything else, prose included,
        byte for byte as it arrived.
    """
    kept: List[str] = []
    pending: List[str] = []

    def flush():
        # Fewer than the threshold: not debris, keep it. The count ignores the blank
        # lines carried along inside the run, matching how the threshold was measured.
        if sum(1 for line in pending if line.strip()) < MIN_NOISE_RUN_LINES:
            kept.extend(pending)
        pending.clear()

    for line in text.split("\n"):
        if not line.strip():
            # Transparent: joins whatever run is open, or trails the prose above it.
            (pending if pending else kept).append(line)
        elif is_prose_line(line):
            flush()
            kept.append(line)
        else:
            pending.append(line)
    flush()
    return "\n".join(kept)


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

        Extraction debris is stripped here rather than at chunking time, so that no
        caller can reach the unfiltered text by accident -- ``get_file_content`` and
        ``get_chunked_text`` both read through this method.

        Returns:
            List[Tuple[str, str]]: One ``(filename, text)`` pair per document, in the
            order the loader read them, with :func:`strip_extraction_noise` applied.
        """
        return [(self.filename[index],
                 strip_extraction_noise(self.extract_text(self.filename[index], data)))
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
