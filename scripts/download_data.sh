#!/usr/bin/env bash
set -euo pipefail # stop on any error

mkdir -p data
cd data
curl -L -o tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
python -m zipfile -e tiny-imagenet-200.zip .
rm tiny-imagenet-200.zip
echo "Done: data/tiny-imagenet-200"