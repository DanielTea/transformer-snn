"""A centrifugal pump on an induction motor, simulated second by second, with faults whose truth is kept.

Eight channels a second: speed (rpm), flow (m3/h), suction and discharge pressure (bar), motor current (A), vibration
(mm/s RMS), bearing and winding temperature (C). The physics is plain and stated here so that every number can be
checked:
  * The pump's head at speed n: H = H0 (n/n0)^2 - k Q^2; the pipes need H_static + R Q^2, R rising as a valve closes,
    so the flow is Q = sqrt((H0 (n/n0)^2 - H_static) / (k + R)), and the head is the system's at that flow.
  * Suction pressure follows the tank level; discharge = suction + rho g H.
  * Shaft power rho g Q H / eta(Q), with the efficiency peaking at the best-efficiency flow; current = power over
    (sqrt(3) V cos(phi) eta_motor).
  * Vibration: a floor, plus imbalance growing with n^2, plus bearing wear, plus the broadband noise of cavitation.
  * Bearing and winding temperatures follow their heat (friction, and I^2) with lags of minutes.
  * Cavitation starts when the net positive suction head available falls below the pump's need, which grows with
    flow: it costs head and flow and shakes pressure, current and vibration.
Operation: a daily demand cycle on the valve, setpoint changes of the speed, stops and starts. Faults, each with a
start, a severity and a ramp, kept as truth: cavitation from a falling tank, a clogging strainer, bearing wear,
impeller imbalance, misalignment, a valve that sticks shut, a seal leak, a drifting sensor and a frozen sensor.

Every channel is written as one byte a second over the channel's fixed range (encode), so a second of a pump is eight
bytes of a stream, the channels always in the same order.

usage: python experiments/pump_sim.py corpus --out data/onemodel/telemetry/pumps-v1.bin --pumps 150 --hours 24
       python experiments/pump_sim.py tests --out data/onemodel/telemetry/tests-v1.bin --per-fault 4 --healthy 9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CHANNELS = ("speed_rpm", "flow_m3h", "suction_bar", "discharge_bar", "current_a", "vibration_mms", "bearing_c",
            "winding_c")
# The fixed range of every channel that its byte spans, lowest to highest.
RANGES = {"speed_rpm": (0.0, 3600.0), "flow_m3h": (0.0, 400.0), "suction_bar": (-0.5, 4.5),
          "discharge_bar": (0.0, 16.0), "current_a": (0.0, 200.0), "vibration_mms": (0.0, 20.0),
          "bearing_c": (0.0, 120.0), "winding_c": (0.0, 160.0)}
FAULTS = ("cavitation", "clogging", "bearing_wear", "imbalance", "misalignment", "valve_stuck", "seal_leak",
          "sensor_drift", "sensor_freeze")
RHO_G = 1000.0 * 9.81
VAPOUR_BAR = 0.03


@dataclass
class Pump:
    """One pump's build and its site, drawn per pump so that every pump has a normal of its own."""
    n0: float = 2950.0          # rated speed, rpm
    h0: float = 70.0            # shut-off head at rated speed, m
    k: float = 4.0e-4           # head lost to flow, m per (m3/h)^2
    h_static: float = 20.0      # static head of the site, m
    r_open: float = 6.0e-4      # pipe resistance with the valve open, m per (m3/h)^2
    q_best: float = 180.0       # best-efficiency flow, m3/h
    eta_best: float = 0.78      # best efficiency
    npsh_r0: float = 3.0        # NPSH the pump needs at its best flow, m
    tank_m: float = 12.0        # normal tank level above the pump, m
    volts: float = 400.0
    vib_floor: float = 1.2      # mm/s
    ambient_c: float = 22.0

    @classmethod
    def draw(cls, rng: np.random.Generator) -> "Pump":
        scale = rng.uniform(0.75, 1.25)
        return cls(n0=float(rng.choice([1475.0, 2950.0])), h0=70.0 * rng.uniform(0.85, 1.15),
                   k=4.0e-4 / scale ** 2 * rng.uniform(0.9, 1.1), h_static=rng.uniform(12.0, 30.0),
                   r_open=6.0e-4 / scale ** 2 * rng.uniform(0.8, 1.2), q_best=180.0 * scale,
                   eta_best=rng.uniform(0.70, 0.84), npsh_r0=rng.uniform(2.0, 4.5), tank_m=rng.uniform(6.0, 18.0),
                   vib_floor=rng.uniform(0.8, 2.0), ambient_c=rng.uniform(5.0, 35.0))


@dataclass
class Fault:
    kind: str
    start: int                  # second it begins
    ramp: int                   # seconds to full severity
    severity: float             # 0 to 1
    channel: str | None = None  # the sensor of a sensor fault

    def level(self, t: np.ndarray) -> np.ndarray:
        return np.clip((t - self.start) / max(self.ramp, 1), 0.0, 1.0) * self.severity * (t >= self.start)


@dataclass
class Run:
    pump: Pump
    seconds: int
    faults: list[Fault] = field(default_factory=list)
    seed: int = 0


def simulate(run: Run) -> np.ndarray:
    """The eight channels of a run, as [seconds, 8] floats in their own units."""
    p, rng, t = run.pump, np.random.default_rng(run.seed), np.arange(run.seconds, dtype=np.float64)
    fault = {f.kind: f.level(t) for f in run.faults if f.channel is None}
    zero = np.zeros(run.seconds)
    # Operation: speed setpoints every few hours, stops, a daily cycle of demand on the valve.
    speed = np.full(run.seconds, p.n0)
    for change in range(0, run.seconds, int(rng.integers(3600, 4 * 3600))):
        speed[change:] = p.n0 * rng.uniform(0.75, 1.0)
    running = np.ones(run.seconds)
    for _ in range(int(rng.integers(0, 3))):
        stop = int(rng.integers(0, max(run.seconds - 1800, 1)))
        running[stop:stop + int(rng.integers(300, 1800))] = 0.0
    # The drive ramps over half a minute; the edges hold their values.
    speed = np.convolve(np.pad(speed * running, (15, 14), mode="edge"), np.ones(30) / 30, mode="valid")
    valve = 1.0 + 0.6 * (1 + np.sin(2 * math.pi * (t / 86400.0 + rng.uniform()))) / 2
    r = p.r_open * valve * (1 + 8.0 * fault.get("valve_stuck", zero))
    # Suction: the tank level falls under cavitation's cause; a strainer clogs; flow pulls the suction down.
    level = p.tank_m * (1 - 0.85 * fault.get("cavitation", zero))
    head_shut = p.h0 * (speed / p.n0) ** 2
    q = np.sqrt(np.clip(head_shut - p.h_static, 0, None) / (p.k + r))
    q = q * (1 - 0.12 * fault.get("seal_leak", zero))
    strainer = 1.0 + 40.0 * fault.get("clogging", zero)
    suction = (level * RHO_G - strainer * 0.02 * q ** 2 * RHO_G / 1000.0) / 1e5
    npsh_a = (suction - VAPOUR_BAR) * 1e5 / RHO_G + 0.5
    npsh_r = p.npsh_r0 * (q / p.q_best).clip(0.3, 2.0) ** 2
    cavitating = np.clip((npsh_r - npsh_a) / np.maximum(npsh_r, 1e-6), 0.0, 1.0) * (speed > 100)
    churn = rng.normal(0, 1, run.seconds)
    q = q * (1 - 0.35 * cavitating * (1 + 0.3 * churn))
    head = (p.h_static + r * q ** 2) * (1 - 0.3 * cavitating)
    discharge = suction + RHO_G * head / 1e5 + 0.02 * rng.normal(0, 1, run.seconds)
    eta = p.eta_best * (1 - 0.6 * ((q - p.q_best) / p.q_best) ** 2).clip(0.2, 1.0)
    eta = eta * (1 - 0.15 * fault.get("seal_leak", zero) - 0.1 * fault.get("misalignment", zero))
    power = RHO_G * (q / 3600.0) * head / np.maximum(eta, 0.05)
    current = power / (math.sqrt(3) * p.volts * 0.85 * 0.93) * (speed > 100) + 8.0 * (speed > 100)
    current = current * (1 + 0.04 * cavitating * churn) + 0.3 * rng.normal(0, 1, run.seconds)
    wear = fault.get("bearing_wear", zero)
    vibration = (p.vib_floor + 3.0 * fault.get("imbalance", zero) * (speed / p.n0) ** 2 + 6.0 * wear ** 2
                 + 2.5 * fault.get("misalignment", zero) + 4.0 * cavitating * np.abs(churn))
    vibration = vibration * (speed > 100) * (1 + 0.05 * rng.normal(0, 1, run.seconds)).clip(0.5, 2.0)
    # Temperatures: first-order lags towards their heat.
    bearing = lag(p.ambient_c + 25.0 * (speed / p.n0) + 30.0 * wear + 12.0 * fault.get("misalignment", zero), 900)
    winding = lag(p.ambient_c + 70.0 * (current / 150.0) ** 2, 1200)
    values = np.stack([speed, q, suction, discharge, current, vibration, bearing, winding], 1)
    # Sensor faults change what is read, not the pump.
    for f in run.faults:
        if f.channel is None:
            continue
        column = CHANNELS.index(f.channel)
        on = t >= f.start
        lo, hi = RANGES[f.channel]
        if f.kind == "sensor_drift":
            values[:, column] += f.level(t) * 0.25 * (hi - lo)
        elif f.kind == "sensor_freeze" and on.any():
            values[on, column] = values[np.argmax(on), column]
    return values


def lag(target: np.ndarray, seconds: float, block: int = 2000) -> np.ndarray:
    """A first-order lag of ``target`` with a time constant of ``seconds``: y[i] = y[i-1] + (target[i] - y[i-1]) /
    seconds, from y[-1] = target[0], computed in closed form a block at a time."""
    a = 1.0 / seconds
    c, out, value = 1.0 - a, np.empty_like(target, dtype=np.float64), float(target[0])
    for start in range(0, len(target), block):
        x = target[start:start + block].astype(np.float64)
        i = np.arange(len(x))
        grow = c ** -i
        out[start:start + len(x)] = c ** (i + 1) * value + a * c ** i * np.cumsum(grow * x)
        value = float(out[start + len(x) - 1])
    return out


def encode(values: np.ndarray) -> np.ndarray:
    """One byte a channel a second over the channel's fixed range, as [seconds, 8] uint8."""
    lo = np.array([RANGES[c][0] for c in CHANNELS])
    hi = np.array([RANGES[c][1] for c in CHANNELS])
    return np.clip(np.round((values - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)


def decode(codes: np.ndarray) -> np.ndarray:
    lo = np.array([RANGES[c][0] for c in CHANNELS])
    hi = np.array([RANGES[c][1] for c in CHANNELS])
    return lo + codes.astype(np.float64) / 255 * (hi - lo)


def draw_run(rng: np.random.Generator, seconds: int, fault_share: float = 0.5) -> Run:
    """A pump for ``seconds``, with one fault in ``fault_share`` of runs, starting in the run's second half."""
    run = Run(Pump.draw(rng), seconds, seed=int(rng.integers(2 ** 31)))
    if rng.uniform() < fault_share:
        kind = str(rng.choice(FAULTS))
        start = int(rng.integers(seconds // 2, seconds - seconds // 8))
        ramp = int(rng.choice([60, 600, 3600])) if kind not in ("sensor_freeze",) else 1
        channel = str(rng.choice(CHANNELS[1:])) if kind.startswith("sensor") else None
        run.faults.append(Fault(kind, start, ramp, float(rng.uniform(0.5, 1.0)), channel))
    return run


def write_corpus(out: Path, pumps: int, seconds: int, seed: int, shares=(0.9, 0.05, 0.05)):
    """Runs of ``pumps`` pumps, each ``seconds`` long, as one byte stream of 8-byte seconds, split by pumps into train,
    validation and test; the faults' truth goes to ``out.truth.json``."""
    rng = np.random.default_rng(seed)
    runs = [draw_run(rng, seconds) for _ in range(pumps)]
    parts, truth, sizes = [], [], {}
    bounds = np.cumsum([0] + [round(share * pumps) for share in shares[:-1]] + [pumps])
    bounds[-1] = pumps
    offset = 0
    for split, (a, b) in zip(("train", "validation", "test"), zip(bounds[:-1], bounds[1:])):
        chunk = []
        for index in range(a, b):
            codes = encode(simulate(runs[index])).reshape(-1)
            truth.append({"split": split, "pump": index, "offset": offset, "bytes": int(codes.size),
                          "build": asdict(runs[index].pump), "seed": runs[index].seed,
                          "faults": [asdict(f) for f in runs[index].faults]})
            chunk.append(codes)
            offset += codes.size
        data = np.concatenate(chunk) if chunk else np.zeros(0, np.uint8)
        parts.append(data)
        sizes[split] = int(data.size)
    raw = np.concatenate(parts).tobytes()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    sidecar = {"format": "raw-bytes-1", "task": "pump telemetry", "channels": list(CHANNELS), "ranges": RANGES,
               "bytes_per_second": len(CHANNELS), "seconds_per_pump": seconds, "pumps": pumps, "seed": seed,
               "sha256": hashlib.sha256(raw).hexdigest(), **sizes}
    Path(f"{out}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    Path(f"{out}.truth.json").write_text(json.dumps(truth) + "\n")
    return sidecar


def write_tests(out: Path, per_fault: int, healthy: int, seconds: int, seed: int):
    """A test set with every fault in ``per_fault`` pumps and ``healthy`` pumps without one: each pump a stream of
    ``seconds``, its fault starting in its second half; bytes to ``out``, truth to ``out.truth.json``."""
    rng = np.random.default_rng(seed)
    runs = []
    for kind in FAULTS:
        for _ in range(per_fault):
            run = Run(Pump.draw(rng), seconds, seed=int(rng.integers(2 ** 31)))
            channel = str(rng.choice(CHANNELS[1:])) if kind.startswith("sensor") else None
            ramp = 1 if kind == "sensor_freeze" else int(rng.choice([60, 600, 1800]))
            start = int(rng.integers(seconds // 2, seconds - seconds // 4))
            run.faults.append(Fault(kind, start, ramp, float(rng.uniform(0.5, 1.0)), channel))
            runs.append(run)
    runs += [Run(Pump.draw(rng), seconds, seed=int(rng.integers(2 ** 31))) for _ in range(healthy)]
    codes = [encode(simulate(run)).reshape(-1) for run in runs]
    raw = np.concatenate(codes).tobytes()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    truth = {"channels": list(CHANNELS), "bytes_per_second": len(CHANNELS), "seconds_per_pump": seconds, "seed": seed,
             "sha256": hashlib.sha256(raw).hexdigest(),
             "pumps": [{"build": asdict(run.pump), "seed": run.seed, "faults": [asdict(f) for f in run.faults]}
                       for run in runs]}
    Path(f"{out}.truth.json").write_text(json.dumps(truth) + "\n")
    return {"pumps": len(runs), "bytes": len(raw), "sha256": truth["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="what", required=True)
    tests = sub.add_parser("tests")
    tests.add_argument("--out", type=Path, required=True)
    tests.add_argument("--per-fault", type=int, default=4)
    tests.add_argument("--healthy", type=int, default=9)
    tests.add_argument("--hours", type=float, default=6.0)
    tests.add_argument("--seed", type=int, default=2)
    corpus = sub.add_parser("corpus")
    corpus.add_argument("--out", type=Path, required=True)
    corpus.add_argument("--pumps", type=int, default=400)
    corpus.add_argument("--hours", type=float, default=24.0)
    corpus.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.what == "tests":
        print(json.dumps(write_tests(args.out, args.per_fault, args.healthy, int(args.hours * 3600), args.seed)))
    else:
        print(json.dumps(write_corpus(args.out, args.pumps, int(args.hours * 3600), args.seed)))


if __name__ == "__main__":
    main()
