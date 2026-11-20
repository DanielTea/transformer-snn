"""Continue a saved training run as a new arm under other settings.

The trainer continues a run only under the settings that it started with. This
writes a copy of a run's saved state (``model.pt.resume``) in which the
recorded settings are changed as asked, under a tag of its own. The trainer
then continues the copy as a new arm when it is called with the new settings:
all that the run has learned stays, and the schedule goes on where it stood.

The long-window arm of the stream-form study left its parent this way at
24,000 updates, for windows of 1,024 bytes in batches of four and reads that
reach 1,024 cells back:

    python experiments/stack/fork_state.py checkpoints/stream_form/windows/model.pt.resume \\
        checkpoints/stream_form/long-windows/model.pt.resume --tag long_windows \\
        --set ctx=1024 bs=4 read_reach=1024

A setting is changed here only in the record. Whether the saved weights and
the optimizer's state fit the new settings is checked by the trainer when it
continues the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def fork(state, tag, settings):
    """The saved state of a run as the state of a new arm with this tag and these settings."""
    unknown = sorted(set(settings) - set(state["args"]))
    if unknown:
        raise ValueError(f"the run has no setting {', '.join(unknown)}")
    # A whole number for a setting that holds a fraction is that fraction, as the trainer reads it from its flag.
    settings = {key: float(value) if isinstance(state["args"][key], float) and isinstance(value, int)
                and not isinstance(value, bool) else value for key, value in settings.items()}
    return state | {"args": state["args"] | settings | {"tag": tag}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state", type=Path, help="the saved state of the parent run")
    parser.add_argument("forked", type=Path, help="where the state of the new arm is written")
    parser.add_argument("--tag", required=True, help="the tag of the new arm, which the trainer is then given")
    parser.add_argument("--set", nargs="+", default=[], metavar="SETTING=VALUE",
                        help="settings of the trainer by the names of their flags, with the values as JSON")
    args = parser.parse_args()
    if args.forked.exists():
        parser.error(f"{args.forked} already exists")
    settings = {}
    for pair in args.set:
        name, found, value = pair.partition("=")
        if not found:
            parser.error(f"{pair}: a setting is given as SETTING=VALUE")
        try:
            settings[name.lstrip("-").replace("-", "_")] = json.loads(value)
        except json.JSONDecodeError:
            settings[name.lstrip("-").replace("-", "_")] = value
    state = torch.load(args.state, map_location="cpu", weights_only=True)
    try:
        forked = fork(state, args.tag, settings)
    except ValueError as error:
        parser.error(str(error))
    args.forked.parent.mkdir(parents=True, exist_ok=True)
    torch.save(forked, args.forked)
    print(f"update {state['step']}: {state['args']['tag']} -> {args.tag} with {settings}")


if __name__ == "__main__":
    main()
