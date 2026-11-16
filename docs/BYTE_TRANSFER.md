# Byte-model transfer benchmarks

The suite evaluates released checkpoints on pinned text corpora. It
uses identical byte windows and targets for every model. The four
corpus scores are raw bits per byte. LAMBADA reports bits per byte on
the answer word and greedy exact match under two prompt conditions.
enwik8 and PG-19 retain the release leaderboard's established
full-corpus protocols.

| Dataset | Pinned evaluation |
|---|---|
| WikiText-103 | The complete Salesforce `wikitext-103-raw-v1` test split. Empty line records contribute a line break. The reconstructed raw file contains 1,289,123 bytes. |
| C4 English | 256 validation documents. Select the 32 lowest SHA-256 hashes of complete document text in each of the eight cleaned English validation shards. Score up to 2,048 original UTF-8 bytes per document. |
| LAMBADA | The original plain-text test distribution. Score the final word in both the whole passage and its last sentence on the same 5,078 examples. |
| enwik9 | The 1,048,576-byte slice beginning at zero-based offset 100,000,000. Preparation verifies the published full-file SHA-1 and the canonical enwik8 prefix. |
| text8 | The final 1,048,576 bytes of the canonical lowercased letters-and-spaces corpus. Preparation verifies the publisher's MD5 and audits shared training text. |

[WikiText paper](https://arxiv.org/abs/1609.07843) ·
[C4 corpus study](https://aclanthology.org/2021.emnlp-main.98/) ·
[Original LAMBADA dataset](https://zenodo.org/records/2630551) ·
[enwik9 and text8 checksums](https://mattmahoney.net/dc/textdata.html)

## Scoring

Each text window starts from fresh state and consumes up to 256
preceding bytes and 256 scored bytes. The first byte of each document
serves as context. Every subsequent byte is scored once. C4 documents
retain independent boundaries. Both models receive the same input
bytes, and padded positions contribute no score. The score uses each
model's existing vocabulary readout at temperature one.

LAMBADA uses the final whitespace-delimited field of each source
passage as its target. The original source uses tokenized punctuation.
The last-sentence prompt starts at the final span from the pinned
English PySBD segmenter. Seventy-five passages exceed the shared
512-byte context and are excluded from both conditions, leaving
5,078 whole passages. Both conditions score exactly the same
target words. The byte log-loss excludes the preceding separator.

Greedy exact match requires the correct argmax at every target byte.
Those predictions equal an autoregressive greedy continuation up to
the end of the word. The raw report also includes boundary-qualified
exact match, which checks that the next prediction is whitespace or
one of the recorded punctuation bytes. This checks whether the
decoder would stop the word at the correct place.

## Wikipedia overlap

The overlap screen searches the exact **90,000,000 training bytes**
from canonical enwik8. For every WikiText-103, enwik9 and text8
evaluation window, it checks aligned 64-byte strings and 12-word
normalized strings every six words. Word normalization decodes
entities, removes tag syntax and reference bodies, replaces
internal links with their display text, and retains lowercase
ASCII words. The check includes the scored bytes and their supplied
history. The raw results give loss on all targets and on windows
with zero matches under this screen. Sampled exact and normalized
checks provide evidence about overlap at their stated granularity.

| Dataset | Windows | Any flagged | Exact-byte matches | Normalized-word matches |
|---|---:|---:|---:|---:|
| WikiText-103 | 5,036 | 4 | 0 | 4 |
| enwik9 | 4,096 | 470 | 408 | 151 |
| text8 | 4,096 | 46 | 9 | 43 |

The enwik9 byte range starts after the enwik8 source prefix. Shared
phrases and repeated markup can still match elsewhere in the dump.
text8 measures transfer to cleaned Wikipedia from the same snapshot.
The text8 and enwik9 slices contain different text, so their score
difference combines cleanup and content.
[Prepared file and overlap catalogue](../results/event_lm/byte_transfer/prepared_v1.json) ·
[Pinned source files and protocol](../results/event_lm/byte_transfer/sources_v1.json)

## Reproduce

Use the project Python environment. The optional requirements install
the parquet reader, the overlap scanner and the sentence segmenter.
The preparation command downloads only pinned sources, checks their
hashes, and produces local files under `data/transfer-v1`.

```bash
python -m pip install -r requirements-benchmarks.txt
python experiments/prepare_byte_benchmarks.py --enwik8 data/enwik8 \
  --sources data/transfer-v1/sources --output data/transfer-v1/prepared
cmp results/event_lm/byte_transfer/prepared_v1.json data/transfer-v1/prepared/catalogue.json
```

Use `uv pip install -r requirements-benchmarks.txt` with an active
uv environment. For each iteration, supply the checkpoint paths in
comparison order, with full attention first. The evaluator records
the source and prepared catalogue hashes, the checkpoint digests,
training exposure and raw scores.

```bash
study=checkpoints/trace-arrival-study
python experiments/eval_byte_transfer.py --device mps --prepared data/transfer-v1/prepared \
  --checkpoint "$study"/{transformer,control,selected,unmatched}_s9349.pt \
  --json "$study/results/byte_transfer.json"
```

[Preparation](../experiments/prepare_byte_benchmarks.py) ·
[Evaluator](../experiments/eval_byte_transfer.py)
