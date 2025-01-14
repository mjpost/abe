#!/usr/bin/env python
import sys

cur_line_number = 1
lines = []

joining_string = " <eos> "

for line in sys.stdin:
    line = line.split('\t')
    if int(line[0]) != cur_line_number:
        print(joining_string.join(lines))
        lines = []
        #if int(line[0]) - cur_line_number != 1:
        #    print(line)
    cur_line_number = int(line[0])
    lines.append(line[1].strip())
print(joining_string.join(lines)) 
