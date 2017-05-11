#!/bin/bash

if [ -z "$1" ]
  then
    echo "No AWS hostname supplied"
fi

HOST=$1

rm data.tar.gz
tar cfz data.tar.gz data/ 
scp data.tar.gz carnd@$HOST:~/CarND-Behavioral-Cloning-P3
