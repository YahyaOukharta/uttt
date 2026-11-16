#!/bin/bash
# Safe refresh loop: after each placement, change only positions the live version lost every game in.
# State in $SC: subs.txt (all our submission ids), books/<sub>.txt (book of each submission), next_version.txt.
# Revert to the previous version if the new one places > 0.2 below it.
cd "$(dirname "$0")/.."
SC=${SC:-/tmp/uttt_loop}; mkdir -p $SC/books
SUBS_FILE=$SC/subs.txt
prog() { python3 tools/cg.py progress 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['submission_id'], int(d['in_progress']), d['score'])"; }
PREV=""; PREV_SCORE=""
for k in $(seq 1 ${1:-12}); do
  while true; do set -- $(prog); [ "$2" = "0" ] && break; sleep 120; done
  CUR=$1; SCORE=$3
  echo "== cycle $k $(date +%H:%M) live $CUR placed $SCORE (previous $PREV $PREV_SCORE)"
  if [ -n "$PREV" ] && python3 -c "import sys; sys.exit(0 if $SCORE < $PREV_SCORE - 0.2 else 1)"; then
    echo "WORSE: reverting to $PREV"
    python3 tools/pack.py -n tools/nets/n7t640.bin -H 640 --huff --book $SC/books/$PREV.txt -o out/revert.cpp bot.cpp | tail -1
    python3 tools/cg.py submit out/revert.cpp > /dev/null 2>&1
    sleep 60; set -- $(prog); cp $SC/books/$PREV.txt $SC/books/$1.txt; echo "$(cat $SUBS_FILE) $1" > $SUBS_FILE
    echo "REVERTED as $1"; PREV=""; continue
  fi
  python3 tools/arena.py fetch --logs $CUR 2>&1 | tail -1
  python3 tools/book.py positions $(cat $SUBS_FILE) --logs
  T=r$k$(date +%H%M)
  python3 tools/kaggle_book.py push $T 8 --src out/c24_n8t.cpp --kernels 1-3 2>&1 | grep -E "positions over|rror" | head -2
  UT_KP=2 python3 tools/kaggle_book.py push $T 8 --src out/c24_n8t.cpp --kernels 4-8 2>&1 | grep -E "rror" | head -2
  for w in $(seq 1 60); do a=$(python3 tools/kaggle_gen.py status $T 8 2>/dev/null | tail -1); b=$(UT_KP=2 python3 tools/kaggle_gen.py status $T 8 2>/dev/null | tail -1); n=$(echo "$a $b" | grep -oE "RUNNING|QUEUED" | wc -l | tr -d ' '); [ "$n" = "0" ] && break; sleep 60; done
  python3 tools/kaggle_book.py fetch $T 8 2>&1 | tail -1; UT_KP=2 python3 tools/kaggle_book.py fetch $T 8 2>&1 | tail -1
  python3 tools/book.py build --gain 0.0 --draw-gain 0.02 | tail -1
  python3 tools/refresh.py $CUR --base $SC/books/$CUR.txt --out /tmp/next_book.txt
  NEW=$(python3 tools/refresh.py $CUR --base $SC/books/$CUR.txt --out /tmp/next_book.txt | grep -o "; [0-9]* new" | grep -o "[0-9]*")
  if [ "${NEW:-0}" -lt 10 ]; then echo "only ${NEW:-0} new entries: no submit"; continue; fi
  if ! python3 tools/book_guard.py --book /tmp/next_book.txt --base $SC/books/$CUR.txt > /tmp/guard.txt; then echo "GUARD BLOCKED"; tail -3 /tmp/guard.txt; continue; fi
  python3 tools/pack.py -n tools/nets/n7t640.bin -H 640 --huff --book /tmp/next_book.txt -o out/next.cpp bot.cpp | tail -1
  g++ -std=c++17 -O2 -o /tmp/nextbin out/next.cpp 2>/dev/null || { echo "compile failed"; continue; }
  python3 tools/cg.py submit out/next.cpp > /dev/null 2>&1
  sleep 60; set -- $(prog)
  if [ "$1" != "$CUR" ]; then
    cp /tmp/next_book.txt $SC/books/$1.txt; echo "$(cat $SUBS_FILE) $1" > $SUBS_FILE
    V=$(cat $SC/next_version.txt); cp out/next.cpp versions/v1.$V.cpp; echo $((V + 1)) > $SC/next_version.txt
    echo "SUBMITTED v1.$V = $1 $(date +%H:%M) (+$NEW entries)"; PREV=$CUR; PREV_SCORE=$SCORE
  else
    echo "submit did not go through"
  fi
done
echo "loop done $(date +%H:%M)"
