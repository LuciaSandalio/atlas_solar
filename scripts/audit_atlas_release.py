#!/usr/bin/env python3
"""Read-only definitive-release preflight; exit 1 if certification is blocked.

This audit does not rewrite a contract or turn file presence into approval.
--output saves a diagnostic JSON, not a release certificate.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import pyarrow.parquet as pq
from models.atlas_contract import validate_models, sha256


def audit(root=ROOT):
    root = Path(root)
    blockers, evidence = [], {}
    contract_path = root/'metadata/fase4_inference_contract.json'
    try:
        contract = json.loads(contract_path.read_text())
        evidence['contract_sha256'] = sha256(contract_path)
        evidence['declared_status'] = contract.get('audit_status')
        model_path = root/contract['models_file']
        evidence['model_sha256'] = sha256(model_path)
        model = joblib.load(model_path)
        evidence['model_type'] = type(model).__name__
        validate_models(model)
    except Exception as exc:
        blockers.append(f'Model/contract incompatibility: {exc}')
    # These legacy entry points are known NOT to implement the requested chain.
    blockers.append('Native annual inference/export replacement is not yet validated; legacy stages are blocked.')
    blockers.append('No reproducible end-to-end release run with independent scientific QA has been recorded.')
    annual_dir = root/'data/ready/atlas_definitivo_2km'
    days_evidence = {}
    for year in range(2020, 2026):
        expected = {t.strftime('%Y%m%d') for t in pd.date_range(f'{year}-01-01', f'{year}-12-31')}
        paths = sorted((root/'data/processed_2km').glob(f'reflectancia_{year}*.parquet'))
        present = {p.stem.removeprefix('reflectancia_') for p in paths}
        missing = sorted(expected-present)
        traced = 0
        for path in paths:
            try:
                meta = pq.read_metadata(path).metadata or {}
                if b'atlas_extraction' in meta:
                    traced += 1
            except Exception as exc:
                blockers.append(f'Unreadable daily file {path.name}: {exc}')
        days_evidence[str(year)] = {'expected_days': len(expected), 'present_days': len(expected & present),
                                     'missing_days': missing, 'files_with_extraction_metadata': traced}
        if missing:
            blockers.append(f'{year}: {len(missing)} missing daily files.')
        if traced != len(paths):
            blockers.append(f'{year}: {len(paths)-traced} daily files without extraction lineage.')
        annual_path = annual_dir/f'atlas_anual_{year}.parquet'
        if not annual_path.exists():
            blockers.append(f'{year}: no canonical annual product.')
            continue
        try:
            schema = pq.read_schema(annual_path)
            required = {'cell_id', 'n_valid_hours', 'n_expected_hours', 'coverage_pct', 'run_id'}
            absent = sorted(required - set(schema.names))
            if absent:
                blockers.append(f'{year}: annual product lacks {absent}.')
        except Exception as exc:
            blockers.append(f'{year}: unreadable annual product: {exc}')
    evidence['daily_files'] = days_evidence
    return {'status': 'BLOCKED' if blockers else 'UNVERIFIED',
            'audited_at': datetime.now(timezone.utc).isoformat(),
            'requested_period': [2020, 2025], 'blockers': blockers, 'evidence': evidence,
            'scope': 'Preflight only; not physical validation or an approval certificate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'])
    for reason in report['blockers']:
        print(f' - {reason}')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
