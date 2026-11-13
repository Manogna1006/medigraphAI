from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.clinical.pdf_record_ingestor import chunk_text, extract_text_from_pdf, ingest_pdf_record
from app.clinical.vitals_analyzer import VitalStatus, analyze_vitals, vitals_to_evidence_text


# ---------- vitals_analyzer ----------

def test_normal_vital_flagged_as_normal():
    flags = analyze_vitals({"heart_rate": 75})
    assert flags[0].status == VitalStatus.NORMAL


def test_high_vital_flagged_with_correct_label():
    flags = analyze_vitals({"heart_rate": 140})
    assert flags[0].status == VitalStatus.HIGH
    assert "tachycardia" in flags[0].note


def test_low_vital_flagged_with_correct_label():
    flags = analyze_vitals({"heart_rate": 40})
    assert flags[0].status == VitalStatus.LOW
    assert "bradycardia" in flags[0].note


def test_unknown_vital_is_not_silently_dropped():
    flags = analyze_vitals({"some_unrecognized_metric": 42})
    assert len(flags) == 1
    assert flags[0].status == VitalStatus.UNKNOWN_VITAL


def test_empty_vitals_returns_empty_list():
    assert analyze_vitals({}) == []
    assert analyze_vitals(None) == []


def test_vitals_to_evidence_text_produces_readable_lines():
    text = vitals_to_evidence_text({"heart_rate": 140, "spo2": 99})
    assert "ABOVE" in text
    assert "heart_rate" in text
    assert "spo2" in text


def test_vitals_evidence_line_format_for_high_and_normal():
    flags = analyze_vitals({"heart_rate": 140, "spo2": 98})
    lines = {f.name: f.as_evidence_line() for f in flags}
    assert "ABOVE" in lines["heart_rate"]
    assert "within normal range" in lines["spo2"]


# ---------- pdf_record_ingestor: chunk_text ----------

def test_chunk_text_returns_single_chunk_for_short_text():
    chunks = chunk_text("short clinical note", max_words=200)
    assert len(chunks) == 1


def test_chunk_text_splits_long_text_with_overlap():
    long_text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(long_text, max_words=200, overlap_words=30)
    assert len(chunks) > 1
    # verify overlap: last words of chunk 1 should reappear at start of chunk 2
    assert chunks[0].split()[-1] in chunks[1]


def test_chunk_text_handles_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


# ---------- pdf_record_ingestor: extract_text_from_pdf ----------

def test_extract_text_returns_empty_for_missing_file():
    assert extract_text_from_pdf("/nonexistent/path/record.pdf") == ""


def test_extract_text_handles_reader_exception_gracefully():
    with patch("app.clinical.pdf_record_ingestor.Path.exists", return_value=True), \
         patch("app.clinical.pdf_record_ingestor.PdfReader", side_effect=Exception("corrupted file")):
        assert extract_text_from_pdf("fake.pdf") == ""


def test_extract_text_handles_encrypted_pdf():
    mock_reader = MagicMock()
    mock_reader.is_encrypted = True
    with patch("app.clinical.pdf_record_ingestor.Path.exists", return_value=True), \
         patch("app.clinical.pdf_record_ingestor.PdfReader", return_value=mock_reader):
        assert extract_text_from_pdf("encrypted.pdf") == ""


# ---------- pdf_record_ingestor: ingest_pdf_record ----------

def test_ingest_pdf_record_returns_zero_when_extraction_fails():
    store = MagicMock()
    with patch("app.clinical.pdf_record_ingestor.extract_text_from_pdf", return_value=""):
        written = ingest_pdf_record("missing.pdf", patient_id="patient-1", vector_store=store)
    assert written == 0
    store.add_documents.assert_not_called()


def test_ingest_pdf_record_writes_chunks_tagged_with_patient_id():
    store = MagicMock()
    store.add_documents.return_value = 2
    with patch("app.clinical.pdf_record_ingestor.extract_text_from_pdf", return_value="a" * 1000):
        written = ingest_pdf_record("record.pdf", patient_id="patient-42", vector_store=store)
    assert written == 2
    call_kwargs = store.add_documents.call_args.kwargs
    assert all(src == "patient_record:patient-42" for src in call_kwargs["sources"])
