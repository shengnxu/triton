#!/bin/bash

# CACHED_DIRS=$(find /root/.triton/cache/ -type d)

ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log
# rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

CACHE_DIR=$LOG_DIR/cache
mkdir -p $CACHE_DIR


# for dir in ${CACHED_DIRS[@]}; do
# 	echo "$dir"
# 	# if [[ $file == *.so ]]; then
# 	# 	echo "Skipping printing .so file"
# 	# elif [[ $file == *.cubin ]]; then
# 	# 	echo "Skipping printing .cubin file"
# 	# else
# 		# sed -i -e '$a\' $file
# 		# cat $file
# 	cp -rf $dir $CACHE_DIR
# 	# fi
# done
cp -r /root/.triton/cache/* $CACHE_DIR

cp -r /tmp/* $CACHE_DIR
chmod -R 777 $CACHE_DIR
