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

A fork that gives a stack trained fast output synapses (``plastic_levels``)
also gives it their gains, at zero: the arm goes on as the same stack, and the
gains learn from the fork on, with no moments of the optimizer yet. Those
settings may be given to a run whose trainer did not have them yet.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

# Settings of trainers newer than a saved run, which a fork may give it all the same.
LATER_SETTINGS = {"plastic_levels", "plastic_rule", "plastic_chunk", "plastic_decay_shift", "plastic_threshold",
                  "plastic_shuffle", "read_credit", "cell_credit"}


def add_plastic_gains(state, count):
    """The state with the gains of trained fast synapses for ``count`` levels added at zero, with no moments of the
    optimizer yet. The gains are the stack's only parameter of its own, so they come first among its parameters:
    first in the running average and first in the optimizer's group of weights without decay at the plain rate. A
    run without such a group gets one after all others, where the trainer's optimizer sorts it
    (common.make_optimizer)."""
    if "plastic_gains" in state["state_dict"]:
        raise ValueError("the run has trained fast synapses already")
    groups = copy.deepcopy(state["optimizer"]["param_groups"])
    index = 1 + max(index for group in groups for index in group["params"])
    free = [group for group in groups if group["weight_decay"] == 0.0 and group.get("lr_scale", 1.0) == 1.0]
    if free:
        free[0]["params"].insert(0, index)
    elif any(group["weight_decay"] == 0.0 or group.get("lr_scale", 1.0) != 1.0 for group in groups):
        raise ValueError("the optimizer's groups leave no place that the trainer would give the gains")
    else:
        groups.append(copy.deepcopy(groups[-1]) | {"params": [index], "weight_decay": 0.0})
    gains = torch.zeros(count)
    forked = state | {"state_dict": state["state_dict"] | {"plastic_gains": gains},
                      "optimizer": state["optimizer"] | {"param_groups": groups}}
    if state.get("average") is not None:
        forked["average"] = [gains.clone(), *state["average"]]
    return forked


def fork(state, tag, settings):
    """The saved state of a run as the state of a new arm with this tag and these settings."""
    unknown = sorted(set(settings) - set(state["args"]) - LATER_SETTINGS)
    if unknown:
        raise ValueError(f"the run has no setting {', '.join(unknown)}")
    # A whole number for a setting that holds a fraction is that fraction, as the trainer reads it from its flag.
    settings = {key: float(value) if isinstance(state["args"].get(key), float) and isinstance(value, int)
                and not isinstance(value, bool) else value for key, value in settings.items()}
    if settings.get("plastic_levels") is not None:
        # The trainer reads the levels as the text of its flag.
        settings["plastic_levels"] = str(settings["plastic_levels"])
    forked = state | {"args": state["args"] | settings | {"tag": tag}}
    if settings.get("plastic_levels") is not None:
        forked = add_plastic_gains(forked, len(settings["plastic_levels"].split(",")))
    return forked


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
