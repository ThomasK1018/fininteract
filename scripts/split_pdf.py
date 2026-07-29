#!/usr/bin/env python3
"""Split a compiled paper PDF into a main-submission PDF and a supplementary PDF.

The AAAI-27 main submission is the first N pages (7 pages of content + up to 2 pages
of references); everything after page N is supplementary material that ships with the
code/data package.

Examples
--------
# Pages 1-8 -> main_paper.pdf, pages 9-end -> supplementary.pdf
python scripts/split_pdf.py --input paper/main.pdf --split-page 8

# Also bundle the supplementary PDF together with the code/data into one zip
python scripts/split_pdf.py --input paper/main.pdf --split-page 8 \
    --bundle-zip submission/supplement.zip \
    --bundle-path scripts data/final SUPPLEMENTARY.md requirements.txt configs/openrouter.example.json
"""
import argparse
import os
import sys
import zipfile

from pypdf import PdfReader, PdfWriter

# Never ship these in a supplementary bundle (secrets / junk / non-anonymized).
EXCLUDE_NAMES = {"__pycache__", ".git", ".DS_Store", "openrouter.json"}
EXCLUDE_SUFFIX = (".pyc",)


def _excluded(path: str) -> bool:
    base = os.path.basename(path.rstrip("/\\"))
    return base in EXCLUDE_NAMES or path.endswith(EXCLUDE_SUFFIX)


def split_pdf(inp: str, split_page: int, main_out: str, suppl_out: str,
              suppl_start: int = None) -> str:
    reader = PdfReader(inp)
    n = len(reader.pages)
    if not (1 <= split_page < n):
        sys.exit(f"--split-page must be between 1 and {n - 1} (the PDF has {n} pages)")
    # by default the supplement begins on the page after the main cutoff; pass
    # --suppl-start to overlap (e.g. when references/appendix share the cutoff page)
    if suppl_start is None:
        suppl_start = split_page + 1
    if not (1 <= suppl_start <= n):
        sys.exit(f"--suppl-start must be between 1 and {n} (the PDF has {n} pages)")

    main_w, suppl_w = PdfWriter(), PdfWriter()
    for i in range(split_page):              # pages 1 .. split_page
        main_w.add_page(reader.pages[i])
    for i in range(suppl_start - 1, n):      # pages suppl_start .. n
        suppl_w.add_page(reader.pages[i])

    for path, writer in ((main_out, main_w), (suppl_out, suppl_w)):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "wb") as f:
            writer.write(f)

    print(f"input : {inp} ({n} pages)")
    print(f"  main  -> {main_out}   (pages 1-{split_page}, {split_page} pages)")
    print(f"  suppl -> {suppl_out}   (pages {suppl_start}-{n}, {n - suppl_start + 1} pages)")
    return suppl_out


def bundle_zip(zip_path: str, suppl_pdf: str, extra_paths: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)) or ".", exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(suppl_pdf, os.path.basename(suppl_pdf))
        count += 1
        for p in extra_paths:
            if _excluded(p):
                continue
            if os.path.isdir(p):
                top = os.path.dirname(p.rstrip("/\\")) or "."
                for root, dirs, files in os.walk(p):
                    dirs[:] = [d for d in dirs if d not in EXCLUDE_NAMES]
                    for fn in files:
                        full = os.path.join(root, fn)
                        if _excluded(full):
                            continue
                        z.write(full, os.path.relpath(full, top))
                        count += 1
            elif os.path.isfile(p):
                z.write(p, os.path.basename(p))
                count += 1
            else:
                print(f"  warning: skipping missing path {p}")
    print(f"  bundle-> {zip_path}   ({count} files)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", default="paper/main.pdf", help="compiled paper PDF")
    ap.add_argument("--split-page", type=int, default=8,
                    help="last page of the main submission (default: 8 = 7 content + 1 references)")
    ap.add_argument("--suppl-start", type=int, default=None,
                    help="first page of the supplement (default: split-page+1; set equal to "
                         "split-page to overlap when a page is shared with the main body)")
    ap.add_argument("--main-out", default="submission/main_paper.pdf")
    ap.add_argument("--suppl-out", default="submission/supplementary.pdf")
    ap.add_argument("--bundle-zip", help="if set, zip the supplementary PDF with the --bundle-path items")
    ap.add_argument("--bundle-path", nargs="*", default=[],
                    help="files/dirs to include alongside the supplementary PDF in the bundle zip")
    a = ap.parse_args()

    suppl = split_pdf(a.input, a.split_page, a.main_out, a.suppl_out, a.suppl_start)
    if a.bundle_zip:
        bundle_zip(a.bundle_zip, suppl, a.bundle_path)


if __name__ == "__main__":
    main()
