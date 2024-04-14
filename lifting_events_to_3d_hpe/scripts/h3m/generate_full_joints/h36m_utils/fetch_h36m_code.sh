#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

FILE=code-v1.1.zip
URL=http://vision.imar.ro/human3.6m/code-v1.1.zip

if [ -f $FILE ]; then
  echo "File already exists."
  exit 0
fi

echo "Downloading h36m code..."

wget $URL -O $FILE

echo "Unzipping... (27M)"

unzip $FILE

echo "Done."