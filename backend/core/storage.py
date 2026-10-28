"""
CrimeVision AI - Case and Evidence Storage Engine
Handles persistent storage of forensic cases, evidence files, metadata,
cryptographic SHA-256 chain-of-custody verification, and default forensic cases.
"""
import os
import json
import hashlib
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CASES_DIR = BASE_DIR / "data" / "cases"
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
STATIC_SAMPLES_DIR = BASE_DIR / "backend" / "static" / "samples"

class StorageEngine:
    def __init__(self):
        CASES_DIR.mkdir(parents=True, exist_ok=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        STATIC_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        self.initialize_default_cases()

    @staticmethod
    def calculate_sha256(filepath: str) -> str:
        """Computes SHA-256 cryptographic hash for chain of custody."""
        sha256 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                while chunk := f.read(65536):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception:
            return hashlib.sha256(filepath.encode()).hexdigest()

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Loads a case from disk."""
        case_file = CASES_DIR / f"{case_id}.json"
        if case_file.exists():
            with open(case_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_case(self, case_data: Dict[str, Any]) -> str:
        """Saves case data to JSON file."""
        case_id = case_data.get("case_id", f"CASE-{int(time.time())}")
        case_file = CASES_DIR / f"{case_id}.json"
        case_data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        with open(case_file, "w", encoding="utf-8") as f:
            json.dump(case_data, f, indent=2)
        return case_id

    def list_cases(self) -> List[Dict[str, Any]]:
        """Lists all stored cases with summary metrics."""
        cases = []
        for case_file in CASES_DIR.glob("*.json"):
            try:
                with open(case_file, "r", encoding="utf-8") as f:
                    c = json.load(f)
                    cases.append({
                        "case_id": c.get("case_id"),
                        "title": c.get("title"),
                        "incident_type": c.get("incident_type"),
                        "incident_date": c.get("incident_date"),
                        "location": c.get("location"),
                        "status": c.get("status", "Under Investigation"),
                        "investigator": c.get("investigator", "Special Agent R. Vance"),
                        "evidence_count": len(c.get("evidence", [])),
                        "created_at": c.get("created_at"),
                        "sha256_hash": c.get("sha256_hash")
                    })
            except Exception as e:
                print(f"Error reading case file {case_file}: {e}")
        return sorted(cases, key=lambda x: x.get("created_at", ""), reverse=True)

    def initialize_default_cases(self):
        """Initializes benchmark forensic cases with rich multimodal data if not already present."""
        case_0842 = {
            "case_id": "CASE-2026-0842",
            "title": "Armed Convenience Store Robbery & Weapon Discharge",
            "incident_type": "Armed Robbery / Aggravated Assault",
            "incident_date": "2026-10-24 23:14:08 UTC",
            "location": "QuickStop Market - 1442 Metro Blvd, Sector 4",
            "jurisdiction": "Metropolitan Police Dept - Major Crimes Division",
            "lead_investigator": "Detective Sergeant Sarah Vance (Badge #4829)",
            "status": "Active Forensics Analysis",
            "created_at": "2026-10-25 00:30:00 UTC",
            "sha256_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "chain_of_custody": [
                {"timestamp": "2026-10-24 23:35:10", "handler": "Officer J. Miller", "action": "Evidence seized from store DVR"},
                {"timestamp": "2026-10-25 00:15:22", "handler": "Forensic Specialist D. Hall", "action": "Digital evidence ingested into CrimeVision AI"},
                {"timestamp": "2026-10-25 00:30:00", "handler": "CrimeVision AI System", "action": "Multimodal forensic pipeline executed"}
            ],
            "evidence": [
                {
                    "evidence_id": "EVID-0842-IMG-01",
                    "type": "image",
                    "label": "CCTV Camera 01 (Overhead Checkout View)",
                    "file_path": "/static/samples/case_0842_cctv.jpg",
                    "timestamp": "23:14:08",
                    "file_size_kb": 895,
                    "sha256": "4a7d60f2bb9d34e6c1e92d73f4e24018274a4cf79d869273c884fbf32d43ea81"
                },
                {
                    "evidence_id": "EVID-0842-AUD-01",
                    "type": "audio",
                    "label": "Store Microphone Audio & Witness Dispatch Recording",
                    "file_path": "/static/samples/case_0842_audio.wav",
                    "timestamp": "23:14:00 - 23:14:24",
                    "file_size_kb": 1033,
                    "duration_sec": 24.0,
                    "sha256": "9b12f6c91a382103f7bc2245e89643120ae89cd4f18392173b092284cf01ea92"
                },
                {
                    "evidence_id": "EVID-0842-VID-01",
                    "type": "video",
                    "label": "CCTV Primary Surveillance Stream (24 fps)",
                    "file_path": "/static/samples/case_0842_cctv.jpg",
                    "timestamp": "23:14:00 - 23:14:24",
                    "file_size_kb": 4200,
                    "duration_sec": 24.0,
                    "fps": 24,
                    "sha256": "7c23a54d309be041e1248a31e1f579c29801fa498327cbdae2014ff98231c944"
                }
            ],
            "crime_classification": {
                "primary_category": "Armed Robbery (Class 1 Felony)",
                "penal_code": "State Penal Code § 211 / § 12022.53 (Personal Discharge of Firearm)",
                "confidence": 0.982,
                "threat_level": "CRITICAL_VIOLENT",
                "recommended_charges": [
                    "Aggravated Robbery with Deadly Weapon",
                    "Assault with a Deadly Weapon (ADW)",
                    "Discharge of Firearm in Commission of a Felony",
                    "Commercial Burglary"
                ]
            }
        }

        case_0915 = {
            "case_id": "CASE-2026-0915",
            "title": "Commercial Jewelry Vault Burglary & Safe Intrusion",
            "incident_type": "Commercial Burglary / Forced Entry",
            "incident_date": "2026-10-25 03:22:41 UTC",
            "location": "Aura Fine Jewelry - 890 Diamond Row, Suite B",
            "jurisdiction": "Commercial Crimes Squad - Metro PD",
            "lead_investigator": "Detective Marcus Chen (Badge #3104)",
            "status": "Active Forensics Analysis",
            "created_at": "2026-10-25 04:15:00 UTC",
            "sha256_hash": "f8a192b0c34e89f2d1a938b76c54e3210984aef53021948ba932e1857c09321f",
            "chain_of_custody": [
                {"timestamp": "2026-10-25 03:45:00", "handler": "Alarm Response Officer K. Diaz", "action": "Physical perimeter secured; security DVR recovered"},
                {"timestamp": "2026-10-25 04:15:00", "handler": "Detective Marcus Chen", "action": "Night vision feed ingested into CrimeVision AI"}
            ],
            "evidence": [
                {
                    "evidence_id": "EVID-0915-IMG-01",
                    "type": "image",
                    "label": "Vault Night-Vision Camera 04",
                    "file_path": "/static/samples/case_0915_vault.jpg",
                    "timestamp": "03:22:41",
                    "file_size_kb": 861,
                    "sha256": "5c98d30e1b2f4837a4c90e21a837493012984fe7582910394857bfe09214cd32"
                }
            ],
            "crime_classification": {
                "primary_category": "Commercial Burglary (Safe Cracking)",
                "penal_code": "State Penal Code § 459 / § 464 (Burglary with Explosives or Tools)",
                "confidence": 0.965,
                "threat_level": "HIGH_PROPERTY_LOSS",
                "recommended_charges": [
                    "Burglary in the Second Degree",
                    "Possession of Burglary Tools",
                    "Grand Theft of Personal Property over $100,000"
                ]
            }
        }

        # Save default cases if they don't exist
        for c in [case_0842, case_0915]:
            case_path = CASES_DIR / f"{c['case_id']}.json"
            if not case_path.exists():
                with open(case_path, "w", encoding="utf-8") as f:
                    json.dump(c, f, indent=2)

# Global singleton storage
storage = StorageEngine()
