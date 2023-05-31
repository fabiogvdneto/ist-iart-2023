#!/bin/bash

for FILE in $(ls instances); do
    if [[ "$FILE" == *.txt ]]; then
        python3 bimaru.py < instances/$FILE > instances/${FILE:0:10}.result
    fi
done