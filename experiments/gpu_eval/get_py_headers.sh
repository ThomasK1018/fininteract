#!/usr/bin/env bash
# Fetch the CPython dev headers (Python.h, pyconfig.h) WITHOUT sudo, for a box where
# Triton's runtime compile fails with "fatal error: Python.h: No such file or directory".
# Extracts the matching Ubuntu libpython3.X-dev .deb into ~/pyinclude and prints the
# C_INCLUDE_PATH to export. Usage: bash get_py_headers.sh [python-exe]
set -uo pipefail
PYEXE="${1:-python3}"
MM=$("$PYEXE" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
VER=$(dpkg -s "libpython${MM}-stdlib" 2>/dev/null | awk '/^Version:/{print $2}')
[ -n "$VER" ] || { echo "cannot read libpython${MM}-stdlib version"; exit 1; }
MIRROR="${MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/ubuntu}"
DEB="libpython${MM}-dev_${VER}_amd64.deb"
mkdir -p "$HOME/pyinclude" && cd "$HOME/pyinclude" || exit 1
if [ ! -f "usr/include/python${MM}/Python.h" ]; then
  for pool in "pool/main/p/python${MM}" "pool/main/p/python3.10" "pool/main/p/python3.12"; do
    URL="$MIRROR/$pool/$DEB"
    echo "trying $URL"
    if curl -fsSL --max-time 300 -o "$DEB" "$URL"; then break; fi
  done
  [ -s "$DEB" ] || { echo "download failed for $DEB"; exit 2; }
  dpkg-deb -x "$DEB" . || exit 3
fi
ls "usr/include/python${MM}/Python.h" "usr/include/x86_64-linux-gnu/python${MM}/pyconfig.h" || exit 4
echo "export C_INCLUDE_PATH=$HOME/pyinclude/usr/include/python${MM}:$HOME/pyinclude/usr/include/x86_64-linux-gnu/python${MM}"
