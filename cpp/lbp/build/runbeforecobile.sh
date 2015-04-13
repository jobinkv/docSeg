#!/bin/bash

rm docSeg
cmake ..
make
./docSeg $1 $2 $3 $4
eog image.jpg
