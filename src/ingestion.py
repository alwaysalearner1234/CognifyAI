"""Multimodal ingestion: PDF textbooks, PPTX slides, video/audio transcripts, plain text.
Every output unit: {unit_id, doc, doc_type, loc, text, topic, subtopic, concepts, has_figure}
loc examples: 'p.12', 'slide 7', 'video 04:32'.
"""
from __future__ import annotations
import re, uuid, json
from pathlib import Path
from typing import List, Dict

from . import config


def chunk_text(text: str, words: int = None, overlap: int = None) -> List[str]:
    words = words or config.CHUNK_WORDS
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    toks = text.split()
    if not toks:
        return []
    out, i = [], 0
    while i < len(toks):
        out.append(" ".join(toks[i:i + words]))
        i += max(1, words - overlap)
    return out


def _mk(doc: str, doc_type: str, loc: str, text: str, has_figure=False) -> Dict:
    return {
        "unit_id": uuid.uuid4().hex[:10],
        "doc": doc, "doc_type": doc_type, "loc": loc,
        "text": text.strip(), "topic": "", "subtopic": "",
        "concepts": [], "has_figure": has_figure,
    }


def ingest_pdf(path: Path) -> List[Dict]:
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    units: List[Dict] = []
    for pno, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        # figure detection: images on page
        has_fig = len(page.get_images(full=True)) > 0
        # save figure thumbnails for UI (best-effort)
        if has_fig:
            try:
                for idx, img in enumerate(page.get_images(full=True)[:2]):
                    xref = img[0]
                    pix = doc.extract_image(xref)
                    ext = pix.get("ext", "png")
                    out = config.UPLOAD_DIR / f"{path.stem}_p{pno}_fig{idx}.{ext}"
                    out.write_bytes(pix["image"])
            except Exception:
                pass
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 40:
            if has_fig:
                units.append(_mk(path.name, "pdf", f"p.{pno}",
                                 f"[Figure/diagram on page {pno} of {path.name} — visual content extracted as image. Refer to saved figure.]",
                                 has_figure=True))
            continue
        for ch in chunk_text(text):
            u = _mk(path.name, "pdf", f"p.{pno}", ch, has_figure=has_fig)
            if has_fig:
                u["text"] += f" [Figure on p.{pno} — see extracted image.]"
            units.append(u)
    return units


def ingest_pptx(path: Path) -> List[Dict]:
    from pptx import Presentation
    prs = Presentation(str(path))
    units: List[Dict] = []
    for sno, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                t = shape.text.strip()
                if t:
                    texts.append(t)
            # image extraction
            try:
                if shape.shape_type == 13:  # picture
                    img = shape.image
                    ext = img.ext or "png"
                    out = config.UPLOAD_DIR / f"{path.stem}_slide{sno}_fig.{ext}"
                    out.write_bytes(img.blob)
            except Exception:
                pass
        has_fig = any(getattr(s, "shape_type", None) == 13 for s in slide.shapes)
        body = re.sub(r"\s+", " ", " ".join(texts)).strip()
        if len(body) < 20:
            body = f"[Slide {sno} is primarily visual — diagram/figure. See extracted image.]"
            has_fig = True
        for ch in chunk_text(body, words=150, overlap=20):
            units.append(_mk(path.name, "pptx", f"slide {sno}", ch, has_figure=has_fig))
    return units


def _parse_timestamp_line(line: str):
    m = re.match(r"\[?(\d{1,2}):(\d{2})(?::(\d{2}))?\]?\s*(.*)", line.strip())
    if m:
        h = 0
        if m.group(3):
            h, mi, s, txt = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        else:
            h, mi, txt = 0, int(m.group(1)), m.group(2)
            s = 0
        secs = h * 3600 + mi * 60 + s
        return secs, txt
    return None, line


def ingest_transcript(path: Path, doc_name: str = None) -> List[Dict]:
    """Handles .srt/.vtt/.txt transcripts and Whisper-produced text. Preserves timestamps."""
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    # strip SRT indices/timestamps into mm:ss lines
    lines = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or re.match(r"^\d+$", ln):
            continue
        m = re.match(r"(\d{2}):(\d{2}):(\d{2})[,\.](\d+)\s*-->", ln)
        if m:
            lines.append(f"[{m.group(1)}:{m.group(2)}:{m.group(3)}]")
            continue
        m2 = re.match(r"(\d{2}):(\d{2})[,\.](\d+)\s*-->", ln)
        if m2:
            lines.append(f"[{m2.group(1)}:{m2.group(2)}]")
            continue
        if re.match(r"WEBVTT|NOTE", ln):
            continue
        lines.append(ln)
    # group into ~90s windows
    units, buf, start = [], [], None
    for ln in lines:
        secs, txt = _parse_timestamp_line(ln)
        if secs is not None and txt:
            if start is None:
                start = secs
            buf.append(txt)
            if secs - start >= 90 and buf:
                mm, ss = divmod(start, 60)
                units.append(_mk(doc_name or path.name, "video", f"{mm:02d}:{ss:02d}",
                                 " ".join(buf)))
                buf, start = [], None
        elif txt:
            buf.append(txt)
            if len(buf) > 120:
                loc = f"{(start or 0)//60:02d}:{(start or 0)%60:02d}"
                units.append(_mk(doc_name or path.name, "video", loc, " ".join(buf)))
                buf, start = [], None
    if buf:
        loc = f"{(start or 0)//60:02d}:{(start or 0)%60:02d}"
        units.append(_mk(doc_name or path.name, "video", loc, " ".join(buf)))
    return units


def ingest_media(path: Path) -> List[Dict]:
    """mp4/mp3/wav: try faster-whisper if installed, else ask for transcript sidecar."""
    try:
        from faster_whisper import WhisperModel  # optional
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(path))
        units, buf, start = [], [], None
        for seg in segments:
            if start is None:
                start = int(seg.start)
            buf.append(seg.text.strip())
            if seg.end - (start or 0) >= 90:
                mm, ss = divmod(int(start or 0), 60)
                units.append(_mk(path.name, "video", f"{mm:02d}:{ss:02d}", " ".join(buf)))
                buf, start = [], None
        if buf:
            mm, ss = divmod(int(start or 0), 60)
            units.append(_mk(path.name, "video", f"{mm:02d}:{ss:02d}", " ".join(buf)))
        return units
    except Exception:
        # Fallback: look for same-stem .srt/.vtt/.txt
        for ext in (".srt", ".vtt", ".txt"):
            side = path.with_suffix(ext)
            if side.exists():
                return ingest_transcript(side, doc_name=path.name)
        return [_mk(path.name, "video", "00:00",
                     f"[Audio/video '{path.name}' uploaded but no transcription engine found. "
                     f"Upload a same-name .srt/.vtt/.txt transcript alongside, or `pip install faster-whisper`.]")]


def ingest_file(path: Path) -> List[Dict]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return ingest_pdf(path)
    if suf in (".pptx", ".ppt"):
        return ingest_pptx(path)
    if suf in (".srt", ".vtt", ".txt", ".md"):
        dtype = "video" if suf in (".srt", ".vtt") else "notes"
        if dtype == "video":
            return ingest_transcript(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        units = []
        for i, para in enumerate(paras or [text]):
            clean = re.sub(r"\s+", " ", para)
            for ch in chunk_text(clean, words=180, overlap=20):
                units.append(_mk(path.name, dtype, f"sec {i+1}", ch))
        return units or [_mk(path.name, dtype, "sec 1", re.sub(r"\s+", " ", text)[:1500])]
    if suf in (".mp4", ".mp3", ".wav", ".m4a"):
        return ingest_media(path)
    raise ValueError(f"Unsupported file type: {suf}")
