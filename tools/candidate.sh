#!/bin/bash
# Safe candidate: the best arena version's book (tools/book_v115.txt) + lines proven by probes
# (tools/probe_book.txt, first). Checked by book_guard.py. -> out/cand.cpp
cd "$(dirname "$0")/.."
{ grep -v '^#' tools/probe_book.txt; cat tools/book_v115.txt; } > /tmp/cand_book.txt
python3 tools/book_guard.py --book /tmp/cand_book.txt --base tools/book_v115.txt > /tmp/cand_guard.txt; r=$?; tail -3 /tmp/cand_guard.txt; [ $r = 0 ] || exit 1
python3 tools/pack.py -n tools/nets/n7t640.bin -H 640 --huff --book /tmp/cand_book.txt -o out/cand.cpp bot.cpp | tail -1
echo "probed lines: $(grep -c '^#' tools/probe_book.txt), entries: $(grep -vc '^#' tools/probe_book.txt)"
