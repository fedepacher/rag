"""Tests for the PDF extraction-noise filter (issue #30).

All pure text in, text out: no PDF reading, no models. The fixtures are verbatim
extractions from the course corpus, so a change in behaviour shows up against real
material rather than against invented strings.
"""
import pytest

from rag.document_loader import (
    MIN_NOISE_RUN_LINES,
    is_prose_line,
    strip_extraction_noise,
)

# The exact block issue #30 reports reaching a student's inbox, from Run D question R4.
# Nine consecutive unreadable lines out of `Amplificador diferencial.pdf`.
R4_NOISE = """Fm
FΠcmcm
iC oC iD oDS 
vC vDS rg
rβrRgRg
vvvv
AA≈
+≈ ==
22FRs  """

# Readable numbered equations that survive extraction intact. These are the course
# content and must never be filtered: each sits alone between prose lines.
GOOD_EQUATIONS = [
    "θ1   =   θ2 e −(to/D − to) / τ   =   θ2 e −to / Dτ e to / τ. (A7)",
    "Tn   =   290 K ( F  −  l). (233)",
]


# --- is_prose_line ---------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "El transistor de efecto de campo presenta una elevada impedancia de entrada.",
    "AMPLIFICADORES DE POTENCIA",
    "La puerta no maneja corriente, salvo alguna corriente de fuga.",
])
def test_prose_is_recognised(line):
    assert is_prose_line(line) is True


@pytest.mark.parametrize("line", R4_NOISE.split("\n"))
def test_every_line_of_the_reported_block_is_not_prose(line):
    assert is_prose_line(line) is False


@pytest.mark.parametrize("line", ["", "   ", "\t"])
def test_blank_lines_are_not_prose(line):
    assert is_prose_line(line) is False


# --- strip_extraction_noise ------------------------------------------------------

def test_the_reported_block_is_removed():
    text = f"El factor de rechazo de modo común se define así:\n{R4_NOISE}\nPor lo tanto conviene maximizarlo."
    cleaned = strip_extraction_noise(text)
    assert "22FRs" not in cleaned
    assert "rβrRgRg" not in cleaned
    assert "El factor de rechazo de modo común se define así:" in cleaned
    assert "Por lo tanto conviene maximizarlo." in cleaned


def test_prose_only_text_is_returned_unchanged():
    """The seven `Amplificación - *.pdf` files measure 0% noise; they must round-trip
    byte for byte, or the filter is rewriting clean documents."""
    text = ("Un amplificador diferencial amplifica la diferencia entre dos señales.\n"
            "Su ganancia en modo común debe ser lo más baja posible.\n"
            "\n"
            "El CMRR relaciona ambas ganancias.")
    assert strip_extraction_noise(text) == text


@pytest.mark.parametrize("equation", GOOD_EQUATIONS)
def test_a_readable_equation_between_prose_survives(equation):
    """Single non-prose lines average 20.9 characters and are where the intact
    equations live. Filtering by line rather than by run would delete them."""
    text = f"La temperatura equivalente de ruido es:\n{equation}\ndonde F es el factor de ruido."
    assert equation in strip_extraction_noise(text)


def test_a_run_just_below_the_threshold_survives():
    """Measured boundary case from `disipa.pdf`: a run of five lines carrying the
    thermal-resistance inequality, which is still readable and still course content."""
    run = ("RT ja .\n"
           "T a   +   P.RT ja    ≤   Tj máx\n"
           "RT jc  RT ca\n"
           "P a j c\n"
           "Ta")
    text = f"La resistencia térmica se calcula así:\n{run}\nCon ese valor se elige el disipador."
    cleaned = strip_extraction_noise(text)
    assert "T a   +   P.RT ja    ≤   Tj máx" in cleaned


def test_a_run_at_the_threshold_is_removed():
    run = "\n".join(f"v{index}D" for index in range(MIN_NOISE_RUN_LINES))
    text = f"Antes del bloque.\n{run}\nDespués del bloque."
    cleaned = strip_extraction_noise(text)
    assert "v0D" not in cleaned
    assert "Antes del bloque." in cleaned
    assert "Después del bloque." in cleaned


def test_blank_lines_do_not_break_a_noise_run():
    """pypdf emits empty lines inside mangled formula blocks — measured in `ruido-t.pdf`.

    Three debris lines, a gap, three more: six non-blank lines in one run, which is the
    threshold. If a blank ended the run this would be two runs of three and both would
    survive, so the assertion is precisely that blanks are transparent.
    """
    run = "()( ) g j\ndtgd n\nnn\n\n\nF F ω=\n21\n. (66)"
    text = f"El espectro resulta:\n{run}\nQue es la expresión buscada."
    cleaned = strip_extraction_noise(text)
    assert "dtgd n" not in cleaned
    assert ". (66)" not in cleaned
    assert "El espectro resulta:" in cleaned
    assert "Que es la expresión buscada." in cleaned


def test_a_gap_does_not_merge_two_short_runs_into_a_removable_one():
    """The mirror of the test above: blanks being transparent must not make the filter
    greedier than the measurement. Two runs of three stay two runs of three."""
    text = ("Primera frase del párrafo.\n"
            "v1D\nv2D\nv3D\n"
            "Una frase intermedia que sí es prosa.\n"
            "v4D\nv5D\nv6D\n"
            "Frase final del párrafo.")
    cleaned = strip_extraction_noise(text)
    assert "v1D" in cleaned
    assert "v6D" in cleaned


def test_a_noise_run_at_the_end_of_a_document_is_removed():
    """The last run has no prose line after it to trigger the flush."""
    text = "Conclusiones del capítulo.\n" + "\n".join(f"R{i}" for i in range(MIN_NOISE_RUN_LINES + 2))
    cleaned = strip_extraction_noise(text)
    assert "R0" not in cleaned
    assert "Conclusiones del capítulo." in cleaned


def test_a_noise_run_at_the_start_of_a_document_is_removed():
    text = "\n".join(f"C{i}" for i in range(MIN_NOISE_RUN_LINES)) + "\nPrimer párrafo real del documento."
    cleaned = strip_extraction_noise(text)
    assert "C0" not in cleaned
    assert cleaned.strip().startswith("Primer párrafo real")


def test_empty_input_is_handled():
    assert strip_extraction_noise("") == ""
