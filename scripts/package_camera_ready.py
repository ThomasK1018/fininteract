"""Package only the active ACL LaTeX sources and figures, after building the PDF."""
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

def source_files():
    pending = [PAPER / "main_acl.tex"]
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        if not path.is_relative_to(PAPER.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unexpected source: {path}")
        found.add(path)
        source = path.read_text(encoding="utf-8")
        for name in re.findall(r"\\input\{([^}]+)\}", source):
            pending.append(ROOT / (name if name.endswith(".tex") else name + ".tex"))
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source):
            graphic = (PAPER / name).resolve()
            if not graphic.is_relative_to(PAPER.resolve()) or not graphic.is_file():
                raise ValueError(f"Missing or unexpected figure: {graphic}")
            found.add(graphic)
    return found

def main():
    target = PAPER / "fininteract_camera_ready_sources.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_files()):
            archive.write(path, path.relative_to(ROOT).as_posix())
        for name in ("acl.sty", "acl_natbib.bst", "refs.bib"):
            archive.write(PAPER / name, name)
        archive.write(PAPER / "build_acl" / "main_acl.bbl", "main.bbl")
        archive.writestr("main.tex", "\\input{paper/main_acl.tex}\n")
        archive.writestr("README.txt", (
            "FinInteract - Findings of ACL: EACL 2027\n\n"
            "Compile main.tex from this directory with pdfLaTeX and BibTeX:\n"
            "  pdflatex main.tex\n  bibtex main\n  pdflatex main.tex\n  pdflatex main.tex\n"
            "Alternatively: tectonic main.tex\n\n"
            "The official ACL style is included. main.bbl is also supplied.\n"
            "The corresponding PDF is provided separately.\n"
            "This source archive contains only active paper sources, bibliography, style, and figures.\n"
        ))
    print(f"Packaged {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
