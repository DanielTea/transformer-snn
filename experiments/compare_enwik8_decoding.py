"""Compare short and longer leads with two decoders and identical seeds."""

import json

import torch

from generate_best_enwik8_samples import ROOT, STUDY, Decoder, display_text, generate


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    results = STUDY / "results"
    previous = json.loads((results / "best_enwik8_generation_samples.json").read_text())
    models = {name: Decoder(ROOT / details["checkpoint"])
              for name, details in previous["models"].items()}
    leads = [
        (1, "== History ==\nThe town developed around a river crossing. The crossing linked farms in the surrounding valley to a market on the opposite bank. Before a permanent bridge was built, travellers and goods crossed the river by ferry. "),
        (2, "== Prime number ==\nThe positive integers are the numbers 1, 2, 3, and so on. A divisor of an integer divides it without a remainder. For example, 2 and 3 are divisors of 6, while 4 does not divide 6. A number such as 6 is called composite. "),
        (3, "== Pacific Ocean ==\nThe Pacific lies between Asia and Australia on the west and the Americas on the east. It meets the Arctic Ocean to the north through the Bering Strait. The continents on its shores enclose numerous seas and gulfs. "),
    ]
    device = torch.device("cpu")
    new_bytes = 192
    output = {"new_bytes": new_bytes, "models": previous["models"], "cases": []}
    for prior_index, lead in leads:
        prior = previous["samples"][prior_index]
        short = prior["prompt"]
        seed = prior["seed"]
        case = {"seed": seed, "short": {"prompt": short}, "long": {"prompt": lead + short}}
        for condition in ("short", "long"):
            row = case[condition]
            prefix = row["prompt"].encode("utf-8")
            assert len(prefix) + new_bytes <= 512
            row.update(prompt_bytes=len(prefix), continuations={})
            for name, model in models.items():
                continuations = {}
                for label, temperature in (("greedy", 0), ("temperature_0.8", 0.8)):
                    suffix = generate(model, prefix, new_bytes, temperature, seed, device)[len(prefix):]
                    assert len(suffix) == new_bytes
                    if condition == "short" and temperature:
                        expected = bytes.fromhex(prior["continuations"][name]["raw_hex"])
                        assert suffix.startswith(expected)
                    grams = {suffix[index:index + 4] for index in range(len(suffix) - 3)}
                    continuations[label] = {
                        "text": display_text(suffix), "raw_hex": suffix.hex(),
                        "repeated_4gram_fraction": 1 - len(grams) / (len(suffix) - 3),
                    }
                    print(f"{condition} {len(prefix)} {name} {label}: {display_text(suffix)!r}", flush=True)
                row["continuations"][name] = continuations
        output["cases"].append(case)
    (results / "enwik8_long_prompt_decoding.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
