#!/bin/sh
# run a queue of configs sequentially: args = TAG:B:F ...
for c in "$@"; do
  T=${c%%:*}; R=${c#*:}; B=${R%%:*}; F=${R#*:}
  V16_B=$B V16_F=$F python3 -W ignore run_v16.py fit $T
done
