"""Build the EACL paper from the repository root; preserve logs in paper/build_acl.

Uses Tectonic when available, otherwise pdfLaTeX + BibTeX. No shell escape.
Run: python scripts/build_camera_ready.py
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD = PAPER / "build_acl"

def main():
    BUILD.mkdir(exist_ok=True)
    tectonic = shutil.which("tectonic")
    bundled = Path(sys.prefix) / "Library" / "bin" / "tectonic.exe"
    if not tectonic and bundled.is_file():
        tectonic = str(bundled)
    if tectonic:
        commands = [[tectonic, "--keep-logs", "--keep-intermediates",
                     "-Z", "search-path=.", "--outdir", "paper/build_acl",
                     "paper/main_acl.tex"]]
        run_cwd = ROOT
        env = os.environ.copy()
    elif shutil.which("pdflatex") and shutil.which("bibtex"):
        env = os.environ.copy()
        for key in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS"):
            env[key] = str(PAPER) + "//" + os.pathsep + str(ROOT) + "//" + os.pathsep + env.get(key, "")
        tex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
               "-file-line-error", str(PAPER / "main_acl.tex")]
        commands = [tex, ["bibtex", "main_acl"], tex, tex]
        run_cwd = BUILD
    else:
        raise SystemExit("Install Tectonic, or pdfLaTeX and BibTeX, then run again.")
    with (BUILD / "build-output.txt").open("w", encoding="utf-8") as log:
        for command in commands:
            result = subprocess.run(command, cwd=run_cwd, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise SystemExit("Build failed. See paper/build_acl/build-output.txt")
    latex_log = (BUILD / "main_acl.log").read_text(encoding="utf-8", errors="replace")
    errors = [line for line in latex_log.splitlines()
              if ("undefined" in line and ("Citation" in line or "Reference" in line))
              or "Missing character" in line or "multiply defined" in line]
    if errors:
        raise SystemExit("Build needs attention:\n" + "\n".join(errors))
    shutil.copy2(BUILD / "main_acl.pdf", PAPER / "main_acl.pdf")
    print("Built paper/main_acl.pdf")
    print("Logs and bibliography: paper/build_acl/")

if __name__ == "__main__":
    main()
