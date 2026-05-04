#!/bin/bash
# Move every file in data/labeled/_quarantine_lowq/<class>/ back to
# data/labeled/embedder/<class>/. Run on the box as root.
set -e
QROOT=/opt/rfcai/repo/training/data/labeled/_quarantine_lowq
EROOT=/opt/rfcai/repo/training/data/labeled/embedder
sudo -u rfcai bash -c '
for cls_dir in '"$QROOT"'/*/; do
    cls=$(basename "$cls_dir")
    moved=0
    for f in "$cls_dir"*; do
        [ -f "$f" ] || continue
        dst="'"$EROOT"'/$cls/$(basename "$f")"
        mv "$f" "$dst"
        moved=$((moved+1))
    done
    echo "  $cls: restored $moved"
done
rmdir '"$QROOT"'/* 2>/dev/null || true
rmdir '"$QROOT"' 2>/dev/null || true
'
