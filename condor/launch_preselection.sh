#!/bin/bash

export WORK_DIR=$PWD
export TAG=${1:-BST_preselection}
export SCRIPT=$WORK_DIR/do_bootstrap_preselection.py
export LOG_DIR=$WORK_DIR/condor/logs

X509_USER_PROXY_OR=$(voms-proxy-info --path)
cp $X509_USER_PROXY_OR $PWD/userProxy
export X509_USER_PROXY=$PWD/userProxy

mkdir -p $LOG_DIR

echo "Work dir : $WORK_DIR"
echo "Script   : $SCRIPT"
echo "Tag      : $TAG"

condor_submit -spool $WORK_DIR/condor/submit_preselection.sub
