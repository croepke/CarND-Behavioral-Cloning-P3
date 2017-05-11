#!/bin/bash

if [ -z "$1" ]
  then
    echo "No AWS hostname supplied"
fi

HOST=$1

rm model.h5
scp carnd@$1:~/CarND-Behavioral-Cloning-P3/model.h5 .
