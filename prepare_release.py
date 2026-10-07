"""Make a deterministic review archive from an explicit standalone allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
REQUIRED = ["README.md", "ALGORITHM.md", "ADAPTING.md", "PROVENANCE.md", "HANDOFF.md",
            "RELEASE.md", "VERIFICATION.md", "pyproject.toml", "requirements-tested.txt",
            ".gitignore", "prepare_release.py"]


def prepare(output):
    output = Path(output).resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if output.suffix.lower() != ".zip":
        raise ValueError("output must end in .zip")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("archive or neighboring manifest already exists")
    paths = [ROOT / name for name in REQUIRED]
    for directory in ("advg_reference", "examples", "tests"):
        paths += sorted((ROOT / directory).glob("*.py"))
    paths += sorted((ROOT / "configs").glob("*.json"))
    paths += sorted((ROOT / "verification").glob("*.json"))
    paths += sorted((ROOT / "verification").glob("*.npz"))
    paths += [p for name in ("LICENSE", "CITATION.cff") if (p := ROOT / name).is_file()]
    files = {}
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"missing or nonlocal release file: {path.name}")
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    if not any(name.startswith("verification/") for name in files):
        raise ValueError("compact verification evidence is missing")
    if not any(name.startswith("tests/") for name in files):
        raise ValueError("tests are missing")
    for name in ("algorithm", "lq", "pendulum"):
        if f"configs/{name}.json" not in files:
            raise ValueError(f"shared configuration is missing: {name}")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    license_status = "author-supplied LICENSE included" if "LICENSE" in files else "review draft; authors have not supplied a license"
    content_manifest = {"version": "0.4.0", "license_status": license_status, "sha256": hashes}
    files["SOURCE_MANIFEST.json"] = (json.dumps(content_manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo("advg-reference/" + name, date_time=(2026, 10, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    result = {"archive": output.name, "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "file_count": len(files), **content_manifest}
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("archive", "archive_sha256", "file_count", "license_status")}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args().output)


if __name__ == "__main__":
    main()
