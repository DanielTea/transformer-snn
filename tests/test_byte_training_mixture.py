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


def test_domain_screen_drops_shared_documents_and_checks_code_hashes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(Path(__file__).resolve().parents[1] / "experiments")
    from prepare_byte_training_mixture import (
        fetch_code, matched_owners, overlap_matchers, owner_matchers, select_documents,
    )

    spec = {"exact_bytes": 16, "exact_stride_bytes": 16, "normalized_words": 4, "normalized_stride_words": 2}
    shared = b"the quick brown fox jumps over the lazy dog again and again"
    candidates = [("a", b"0123456789abcdef" * 4), ("b", shared), ("c", b"zyxwvutsrqponmlk" * 4)]
    owners = owner_matchers(candidates, spec)
    # Candidate a shares exact bytes with the earlier text and candidate b shares its words.
    earlier = [b"header 0123456789abcdef trailer", b"The QUICK brown   fox jumps!"]
    assert matched_owners(owners, earlier) == {0, 1}

    guards = [overlap_matchers([b"protected benchmark sentence one two three"], spec),
              overlap_matchers([candidates[2][1]], spec)]
    documents = [("benchmark", b"x protected benchmark sentence one two three y"), ("heldout", b"..zyxwvutsrqponmlk.." * 2),
                 ("fresh", b"an unrelated training document with plenty of new words")]
    selected, stats = select_documents(iter(documents), 8, 4, guards, set())
    assert [name for name, _ in selected] == ["fresh"]
    assert stats["excluded"] == {"exact_overlap": 2}

    body = b"print('hello')\n"
    blob = hashlib.sha1(body).hexdigest()
    (tmp_path / blob[:2]).mkdir()
    (tmp_path / blob[:2] / blob).write_bytes(body)
    assert fetch_code(tmp_path, "unused://{blob_id}", blob) == body
    (tmp_path / blob[:2] / blob).write_bytes(body + b"#")
    with pytest.raises(ValueError, match="SHA-1"):
        fetch_code(tmp_path, "unused://{blob_id}", blob)
