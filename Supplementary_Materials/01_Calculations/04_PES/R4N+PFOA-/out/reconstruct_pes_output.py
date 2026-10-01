#!/usr/bin/env python3

import argparse
import hashlib
import os
import tempfile
from pathlib import Path


OUTPUT_NAME = "R4N+PFOA-_0.15M.out.gz"
EXPECTED_OUTPUT_SIZE = 147_853_593
EXPECTED_OUTPUT_SHA256 = "bce958f14737123fa4d1a67b0098b6951878ffdcf8aa79e5e9337fdd6e1f4a6f"
PARTS = (
    (
        "R4N+PFOA-_0.15M.out.gz.part_aa",
        51_380_224,
        "1eaa38c01a53732d09893ca0f1e9abf445af57d169fa064d499f798c8e1eacea",
    ),
    (
        "R4N+PFOA-_0.15M.out.gz.part_ab",
        51_380_224,
        "decdb5c7df41e7cee2b7a025043d20e12e13c9a0efffc663189e575e02dd27a0",
    ),
    (
        "R4N+PFOA-_0.15M.out.gz.part_ac",
        45_093_145,
        "2edc35dea3ca6b83c27a2a7c4a9d6c4248d5805a0d52fbbac7ae63854cecb426",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_parts(root: Path) -> None:
    for name, expected_size, expected_hash in PARTS:
        path = root / name
        if not path.is_file():
            raise SystemExit(f"Missing part: {path}")
        if path.stat().st_size != expected_size:
            raise SystemExit(f"Unexpected size for {name}")
        if file_sha256(path) != expected_hash:
            raise SystemExit(f"SHA-256 mismatch for {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct and verify the compressed ORCA PES output."
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output = root / OUTPUT_NAME
    verify_parts(root)
    if args.check_only:
        print("All three parts passed size and SHA-256 validation.")
        return

    if output.exists() and not args.force:
        if (
            output.stat().st_size == EXPECTED_OUTPUT_SIZE
            and file_sha256(output) == EXPECTED_OUTPUT_SHA256
        ):
            print(f"Existing output already verified: {output}")
            return
        raise SystemExit(f"Refusing to replace unverified output: {output}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=root, prefix=f".{OUTPUT_NAME}.", suffix=".tmp", delete=False
        ) as destination:
            temporary = Path(destination.name)
            digest = hashlib.sha256()
            total_size = 0
            for name, _, _ in PARTS:
                with (root / name).open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
                        digest.update(block)
                        total_size += len(block)

        if total_size != EXPECTED_OUTPUT_SIZE:
            raise SystemExit("Reconstructed output size mismatch")
        if digest.hexdigest() != EXPECTED_OUTPUT_SHA256:
            raise SystemExit("Reconstructed output SHA-256 mismatch")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    print(f"Reconstructed and verified: {output}")
    print(f"SHA-256: {EXPECTED_OUTPUT_SHA256}")


if __name__ == "__main__":
    main()
