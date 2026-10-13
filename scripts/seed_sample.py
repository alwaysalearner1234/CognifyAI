"""Seed KB with bundled sample course (no manual preprocessing needed)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import ingestion, kb

base = Path(__file__).resolve().parent.parent / "data" / "sample_physics"
units = []
for f in base.glob("*.*"):
    units += ingestion.ingest_file(f)
fresh = kb.add_units(units)
print(f"Seeded {len(fresh)} units ({len(kb.load_units())} total). Topics: {sorted({u['topic'] for u in kb.load_units()})}")
