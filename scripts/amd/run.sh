clear

set -x


ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR


bash scripts/amd/clean.sh 2>&1 |tee $LOG_DIR/clean.log
bash scripts/amd/build.sh 2>&1 |tee $LOG_DIR/build.log
# bash scripts/amd/test.sh 2>&1 |tee $LOG_DIR/test.log
bash scripts/amd/test.sh backtrace 2>&1 |tee $LOG_DIR/backtrace.log