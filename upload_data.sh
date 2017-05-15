#!/bin/bash

if [ -z "$1" ]
  then
    echo "No AWS hostname supplied"
fi

HOST=$1

zip -r data.zip /data
scp data.zip carnd@$HOST:~/CarND-Behavioral-Cloning-P3
