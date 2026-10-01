#!/usr/bin/env python3

from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
FILES = [
    "stage08_common.conf",
    "00_minimize.conf",
    "01_heat_100K.conf",
    "02_heat_200K.conf",
    "03_heat_310K.conf",
    "04_equil_npt_500ps.conf",
    "05_acceptance_npt_1ns.conf",
]


def directives(path):
    out = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        out.append((parts[0].lower(), parts[1:] if len(parts) > 1 else [], lineno))
    return out


def vals(ds, key):
    return [(args, line) for k, args, line in ds if k == key.lower()]


def require(cond, msg):
    if not cond:
        raise SystemExit(f"ERROR: {msg}")


def one_value(ds, key, expected=None):
    found = vals(ds, key)
    require(len(found) == 1, f"expected exactly one '{key}' directive, found {len(found)}")
    args, line = found[0]
    require(args, f"'{key}' has no value at line {line}")
    if expected is not None:
        require(args[0].lower() == expected.lower(), f"'{key}' is {args[0]!r}, expected {expected!r} at line {line}")
    return args[0]


def main():
    data = {}
    for name in FILES:
        p = HERE / name
        require(p.is_file(), f"missing configuration file: {name}")
        data[name] = directives(p)
        require(not vals(data[name], "nonbondedFrequency"), f"unsupported legacy spelling 'nonbondedFrequency' remains in {name}")


    for name in FILES[1:]:
        one_value(data[name], "ambercoor", "$stage07_inpcrd")

    common = data["stage08_common.conf"]
    require(not vals(common, "langevin"), "stage08_common.conf must not define 'langevin'; each stage owns that switch")
    one_value(common, "nonbondedFreq", "1")
    one_value(common, "fullElectFrequency", "2")

    mini = data["00_minimize.conf"]
    one_value(mini, "langevin", "off")
    one_value(mini, "temperature")
    require(not vals(mini, "langevinTemp"), "minimization must not define langevinTemp while Langevin is off")

    h100 = data["01_heat_100K.conf"]
    one_value(h100, "langevin", "on")
    one_value(h100, "temperature", "100.0")
    one_value(h100, "langevinTemp", "100.0")
    require(not vals(h100, "binvelocities"), "100 K stage should initialize velocities rather than read prior dynamics velocities")

    for name, prev, target in [
        ("02_heat_200K.conf", "01_heat_100K/heat100.vel", "200.0"),
        ("03_heat_310K.conf", "02_heat_200K/heat200.vel", "310.15"),
        ("04_equil_npt_500ps.conf", "03_heat_310K/heat310.vel", "310.15"),
        ("05_acceptance_npt_1ns.conf", "04_equil_npt_500ps/equil500ps.vel", "310.15"),
    ]:
        ds = data[name]
        one_value(ds, "langevin", "on")
        one_value(ds, "langevinTemp", target)
        one_value(ds, "binvelocities", prev)
        require(not vals(ds, "temperature"), f"{name} must continue velocities and must not reinitialize them with 'temperature'")

    for name in ("01_heat_100K.conf", "02_heat_200K.conf", "03_heat_310K.conf"):
        one_value(data[name], "langevinPiston", "off")
    for name in ("04_equil_npt_500ps.conf", "05_acceptance_npt_1ns.conf"):
        one_value(data[name], "langevinPiston", "on")
        one_value(data[name], "langevinPistonTarget", "1.01325")
        one_value(data[name], "langevinPistonTemp", "310.15")

    print("Stage 08 static configuration validation: PASS")
    print("- Langevin switch defined exactly once per stage")
    print("- NAMD3 nonbondedFreq spelling is used explicitly")
    print("- ambercoor is declared in every AMBER-mode stage, including restart continuations")
    print("- velocities initialized once at 100 K and continued through later stages")
    print("- NVT/NPT thermostat and piston switches are stage-local")


if __name__ == "__main__":
    main()
