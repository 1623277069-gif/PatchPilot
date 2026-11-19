"""Shared deterministic I/O. No model outputs or network clients here."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
MODEL = 'gemma-4-31b-it-qat-w4a16-ct'
HF_MODEL = 'google/gemma-4-31B-it-qat-w4a16-ct'


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else
                          json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def journal(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(),
                            **value}, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def model_identity(path):
    """Validate supplied LOCAL official QAT checkpoint, never download/replace it."""
    path = Path(path).resolve()
    cfg = read_json(path / 'config.json')
    quant = cfg.get('quantization_config', {})
    text = cfg.get('text_config', cfg)
    if (cfg.get('model_type') != 'gemma4' or text.get('num_hidden_layers') != 60
            or quant.get('quant_method') != 'compressed-tensors'):
        raise ValueError('Requires the specified 31B Gemma4 QAT compressed-tensors checkpoint')
    groups = quant.get('config_groups', {}).values()
    if not any(g.get('weights', {}).get('num_bits') == 4 for g in groups):
        raise ValueError('Expected QAT INT4 weights')
    if not list(path.glob('*.safetensors')):
        raise ValueError('No local model weights')
    # Filename checks alone are insufficient; capture all weight byte hashes.
    files = {}
    for file in sorted(path.glob('*')):
        if file.is_file() and file.suffix in {'.json', '.jinja', '.safetensors'}:
            h = hashlib.sha256()
            with file.open('rb') as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(block)
            files[file.name] = h.hexdigest()
    return {'model': MODEL, 'hf_model': HF_MODEL, 'path': str(path),
            'files': files, 'fingerprint': digest(files)}
