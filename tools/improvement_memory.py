"""Export/check compact cross-chat memory. Standard library only; no application imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / "memory" / "improvement"
DOCS = ROOT / "docs" / "improvement"
LIMIT = 199
CORE = [ROOT / name for name in ("AGENTS.md", "CLAUDE.md", "DESIGN.md")]
CORE += [MEMORY / name for name in ("STATE.md", "DECISIONS.md", "HANDOFF.md")]
GENERATED = [MEMORY / "CHAT_CONTEXT.md", MEMORY / "NEXT_PROMPT.md"]
ALLOWED = {"PENDING", "ACTIVE", "DONE", "EVALUATED_NO_GAIN", "BLOCKED", "DEFERRED_DATA", "NEEDS_FIX"}
ARCHIVES = {
    "DESIGN.md": "18ABE14400B614708802AD69CA26D7BC67DB2EBC85DC008958C490541DFE25B6",
    "CLAUDE.original.md": "E3C0B6737D02916078D9C6D07FEA54D8CCE0ACD63753312614733D8A67300E58",
}

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").rstrip()

def manifest() -> dict:
    return json.loads(read(DOCS / "manifest.json"))

def next_step() -> str:
    match = re.search(r"^Next step:\s*(\d{2}|none)\s*$", read(MEMORY / "STATE.md"), re.M)
    if not match:
        raise ValueError("STATE.md needs one 'Next step: NN' or 'Next step: none' line.")
    return match.group(1)

def readiness_errors(data: dict, state: str) -> list[str]:
    errors = []
    steps = {s["id"]: s for s in data["steps"]}
    statuses = dict(re.findall(r"^\|\s*(\d{2})\s*\|[^|]*\|\s*([A-Z_]+)\s*\|", state, re.M))
    finished = {"DONE", "EVALUATED_NO_GAIN"}
    disposed = finished | {"DEFERRED_DATA"}
    queued = re.search(r"^Next step:\s*(\d{2}|none)\s*$", state, re.M)
    active = re.search(r"^Active step:\s*(\d{2}|none)\s*$", state, re.M)
    if not queued or not active:
        return ["STATE needs valid Next step and Active step fields."]
    target, owner = queued.group(1), active.group(1)
    active_rows = [sid for sid, value in statuses.items() if value == "ACTIVE"]
    if owner == "none" and active_rows:
        errors.append("ACTIVE stage exists but Active step is none.")
    if owner != "none":
        if owner != target or statuses.get(owner) not in {"ACTIVE", "NEEDS_FIX"}:
            errors.append("Active step must be the queued ACTIVE/NEEDS_FIX stage.")
        if any(sid != owner for sid in active_rows):
            errors.append("Multiple active writers/stages are not supported in shared memory.")
    def dependencies(sid: str, allow_deferred: bool = False) -> list[str]:
        spec = steps[sid]
        allowed = set(spec.get("allow_deferred", [])) if allow_deferred else set()
        return [dep for dep in spec["depends"]
                if statuses.get(dep) not in finished and not
                (dep in allowed and statuses.get(dep) == "DEFERRED_DATA")]
    for sid, status in statuses.items():
        if sid not in steps:
            continue
        if status in finished and dependencies(sid):
            errors.append(f"Completed stage {sid} has unfinished prerequisites: {dependencies(sid)}")
        if status in finished or sid == target:
            missing = [v for v in steps[sid].get("review_of", []) if statuses.get(v) not in disposed]
            if missing:
                errors.append(f"Stage {sid} needs optional-stage dispositions: {missing}")
        if status == "DEFERRED_DATA":
            note = MEMORY / "stages" / f"{sid}.md"
            if not note.exists():
                errors.append(f"Deferred stage {sid} needs evidence and a resume condition.")
            else:
                content = read(note)
                if not re.search(r"^Candidate enabled:\s*no\s*$", content, re.M | re.I):
                    errors.append(f"Deferred stage {sid} must record 'Candidate enabled: no'.")
                if not re.search(r"^Resume condition:\s*\S.+$", content, re.M):
                    errors.append(f"Deferred stage {sid} needs a concrete Resume condition.")
    if target != "none":
        if target not in steps:
            errors.append(f"Unknown queued stage {target}.")
        else:
            if statuses.get(target) in finished:
                errors.append(f"Queued stage {target} is already complete; choose the next ready stage.")
            missing = dependencies(target, allow_deferred=True)
            if missing:
                errors.append(f"Queued stage {target} is not ready; prerequisites: {missing}")
    elif owner != "none":
        errors.append("No next step can be queued while a stage is active.")
    return errors

def generated_texts() -> dict[Path, str]:
    step = next_step()
    steps = {s["id"]: s for s in manifest()["steps"]}
    if step != "none" and step not in steps:
        raise ValueError(f"Unknown next step {step}.")
    design = read(ROOT / "DESIGN.md")
    architecture = design.split("## Architecture\n", 1)[1].split("\n## Code map", 1)[0]
    context = [
        "# Portable chat context", "",
        "Generated from repository memory; re-export after every completed or interrupted stage.",
        "This is a context snapshot, not proof of access to source files or prior conversations.",
        "Repository: " + ROOT.as_posix(),
        "Read AGENTS.md / CLAUDE.md / DESIGN.md and the numbered prompt when repository access exists.",
        "Without filesystem access, request required source files; never claim to have edited local files.",
        "Implement one selected step only; preserve dirty work, paper-only mode and frozen test policy.",
        "Keep implementation, candidate acceptance and future evidence statuses separate.",
        "Update canonical STATE/DECISIONS/HANDOFF and stage evidence, then export/check again.", "",
        "## Compact architecture", architecture, "",
        read(MEMORY / "STATE.md"), "", read(MEMORY / "DECISIONS.md"), "",
        read(MEMORY / "HANDOFF.md"), "",
    ]
    if step == "none":
        following = "# No numbered step queued\n\nRead STATE.md for pending prospective observations and remaining limits.\n"
    else:
        s = steps[step]
        following = (
            "# Next prompt — generated from STATE.md\n\n"
            "Select the model/effort below, then paste the block in a fresh project chat.\n\n"
            + read(ROOT / s["file"]) + "\n"
        )
    pack = (
        f"# All {manifest()['stage_count']} implementation prompts\n\n"
        "Choose the model and thinking setting above a prompt, then paste its text into a fresh project chat.\n"
        "Read START_HERE.md first. This long work-order collection is not a compact memory file.\n\n"
    )
    pack += "\n---\n\n".join(read(ROOT / s["file"]) for s in manifest()["steps"]) + "\n"
    return {MEMORY / "CHAT_CONTEXT.md": "\n".join(context), MEMORY / "NEXT_PROMPT.md": following,
            DOCS / "ALL_PROMPTS.md": pack}

def export() -> None:
    problems = readiness_errors(manifest(), read(MEMORY / "STATE.md"))
    if problems:
        raise ValueError("Cannot export an unready handoff: " + "; ".join(problems))
    outputs = generated_texts()
    for path, content in outputs.items():
        if path in GENERATED and len(content.splitlines()) > LIMIT:
            raise ValueError(f"{path.name} would exceed {LIMIT} lines; compact STATE/DECISIONS/HANDOFF first.")
    for path, content in outputs.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content.rstrip() + "\n", encoding="utf-8")
        temporary.replace(path)
        print(f"Exported {path.relative_to(ROOT).as_posix()} ({len(content.splitlines())} lines)")

def manifest_order_errors(data: dict) -> list[str]:
    """Allow append-only follow-up stages while rejecting gaps and malformed counts."""
    count = data.get("stage_count")
    if type(count) is not int or not 18 <= count <= 99:
        return ["Manifest stage_count must be an integer from 18 to 99."]
    ids = [s.get("id") for s in data.get("steps", [])]
    if ids != [f"{i:02d}" for i in range(1, count + 1)]:
        return [f"Manifest must contain exactly ordered steps01..{count:02d}."]
    return []


def check() -> bool:
    errors = []
    files = CORE + GENERATED + sorted((MEMORY / "stages").glob("*.md"))
    files += [DOCS / "NEW_CHAT.md", DOCS / "START_HERE.md", DOCS / "MODELS.md"]
    for optional in ("CONTRACTS.md", "EVALUATION_PROTOCOL.md"):
        path = DOCS / optional
        if path.exists():
            files.append(path)
    for path in files:
        if not path.exists():
            errors.append(f"Missing {path.relative_to(ROOT)}")
            continue
        count = len(read(path).splitlines())
        print(f"{count:3d} lines  {path.relative_to(ROOT).as_posix()}")
        if count > LIMIT:
            errors.append(f"{path.name}: {count} lines exceeds {LIMIT}")
    data = manifest()
    errors.extend(readiness_errors(data, read(MEMORY / "STATE.md")))
    steps = data["steps"]
    ids = [s["id"] for s in steps]
    errors.extend(manifest_order_errors(data))
    for s in steps:
        path = ROOT / s["file"]
        if not path.is_file():
            errors.append(f"Missing prompt {s['id']}")
            continue
        content = read(path)
        if f"**Model:** {s['model']}" not in content or f"**Thinking level:** {s['effort']}" not in content:
            errors.append(f"Model/effort mismatch in prompt {s['id']}")
        if content.count(chr(96) * 3) != 2:
            errors.append(f"Prompt {s['id']} must have one complete copyable code block.")
        for dep in s["depends"] + s.get("review_of", []):
            if dep not in ids or int(dep) >= int(s["id"]):
                errors.append(f"Invalid dependency {s['id']} -> {dep}")
    rows = re.findall(r"^\|\s*(\d{2})\s*\|[^|]*\|\s*([A-Z_]+)\s*\|", read(MEMORY / "STATE.md"), re.M)
    if [r[0] for r in rows] != ids:
        errors.append("STATE ledger must contain exactly the ordered manifest steps.")
    for sid, status in rows:
        if status not in ALLOWED:
            errors.append(f"Invalid step status {sid}: {status}")
        if status in {"DONE", "EVALUATED_NO_GAIN"} and not (MEMORY / "stages" / f"{sid}.md").exists():
            errors.append(f"Completed step {sid} has no stage evidence file.")
    step = next_step()
    handoff = re.search(r"^Next step:\s*(\d{2}|none)\s*$", read(MEMORY / "HANDOFF.md"), re.M)
    if not handoff or handoff.group(1) != step:
        errors.append("HANDOFF Next step disagrees with STATE.")
    for path, expected in generated_texts().items():
        if not path.exists() or read(path) != expected.rstrip():
            errors.append(f"{path.name} stale; run export after updating memory.")
    for name, expected in ARCHIVES.items():
        path = ROOT / "docs" / "archive" / "memory-setup-20260918" / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest().upper() != expected:
            errors.append(f"Original archive missing or altered: {name}")
    if errors:
        for error in errors:
            print("FAIL:", error)
        return False
    print(f"PASS: compact memory, {len(steps)} prompts, dependencies, exports and original archives.")
    return True

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "check"))
    args = parser.parse_args()
    if args.action == "export":
        export()
    else:
        raise SystemExit(0 if check() else 1)

if __name__ == "__main__":
    main()
