#!/bin/bash

ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log
# rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

# create log/cache
CACHE_DIR=$LOG_DIR/cache
mkdir -p $CACHE_DIR

# copy to log/cache
cp -r /root/.triton/cache/* $CACHE_DIR
cp -r /tmp/* $CACHE_DIR
chmod -R 777 $CACHE_DIR
