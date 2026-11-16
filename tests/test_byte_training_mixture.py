import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_mixture_samples_document_windows_and_records_equal_sequences(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(Path(__file__).resolve().parents[1] / "experiments")
    from byte_training_mixture import TrainingMixture

    lengths = b"short\t2\nfirst\t4\nsecond\t5\n"
    body = b"!!aaaaBBBBB"
    enwik8 = b"wikipedia"
    (tmp_path / "c4.bin").write_bytes(body)
    (tmp_path / "c4.spans").write_bytes(lengths)
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(json.dumps({
        "format": 1, "name": "test", "context_bytes": 2, "sampling_weights": {"enwik8": 2, "c4": 1},
        "corpora": {
            "enwik8": {"bytes": len(enwik8), "documents": 1, "sha256": hashlib.sha256(enwik8).hexdigest()},
            "c4": {"file": "c4.bin", "spans_file": "c4.spans", "bytes": len(body), "documents": 3,
                   "sha256": hashlib.sha256(body).hexdigest(), "spans_sha256": hashlib.sha256(lengths).hexdigest()},
        },
    }))
    results = []
    for _ in range(2):
        mix = TrainingMixture(catalogue, np.frombuffer(enwik8, dtype=np.uint8), 2)
        generator = torch.Generator().manual_seed(9631)
        for _ in range(20):
            inputs, labels = mix.batch(3, generator)
            assert inputs[2].tolist() + [labels[2, -1].item()] in [[97] * 3, [66] * 3]
            torch.testing.assert_close(inputs[:, 1:], labels[:, :-1])
        results.append(mix.receipt())
    assert results[0] == results[1]
    assert results[0]["source_targets"] == {"enwik8": 80, "c4": 40}

    (tmp_path / "c4.spans").write_bytes(lengths.replace(b"first\t4", b"first\t3"))
    with pytest.raises(ValueError, match="file checksum differs"):
        TrainingMixture(catalogue, np.frombuffer(enwik8, dtype=np.uint8), 2)
