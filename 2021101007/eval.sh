#!/bin/bash

test_file=$1
if [ ! -n "$test_file" ]; then
    echo 'Error: Provide test file name'
    exit 1
fi

if [ ! -f "$test_file" ]; then
    echo 'Error: Test file does not exists'
    exit 1
fi

train_file=$2
if [ ! -n "$train_file" ]; then
    train_file='data.npy'
fi

if [ ! -f "$train_file" ]; then
    echo 'Error: Train file does not exists'
    echo 'Place data.npy as train file or give correct file name as second argument'
    exit 1
fi

python check.py $train_file $test_file