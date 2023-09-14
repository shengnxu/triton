#!/bin/bash

CACHED_FILES=$(find /root/.triton/cache/ -type f -name "*.*")

ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log
# rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

CACHE_DIR=$LOG_DIR/cache


for file in ${CACHED_FILES[@]}; do
	echo "$file"
	if [[ $file == *.so ]]; then
		echo "Skipping printing .so file"
	elif [[ $file == *.cubin ]]; then
		echo "Skipping printing .cubin file"
	else
		sed -i -e '$a\' $file
		cat $file
		cp $file $CACHE_DIR
	fi
done

cp /tmp/* $CACHE_DIR
chmod -R 777 $CACHE_DIR
