#!/bin/bash


expdir=$1

maxstep=500000
pw="";
pi="";
while true
do
    ckpt=`perl utils/find_max_ckpt.pl $expdir 1 $maxstep`;
    if [ $? == 1 ]; then
        echo "Find max ckpt \"$ckpt\", max step=$maxstep, monitor finish now!";
        break;
    fi
    
    pw=`perl utils/watch_process.pl -w $pw -i $pi`;
    if [ $? == 1 ]; then
        echo "Restart task ...";
        if [ x$ckpt == "x" ]; then
            echo "./run.sh";
            rm -rf $expdir/*; 
            ./run.sh &
        else
            echo "./run.sh --resume \"$ckpt\"";
            ./run.sh --resume "$ckpt" &
        fi
        echo "Restart task done!";
    else
        echo "Task is running, watch process is: $pw, ignore process is: $pi";
    fi
    sleep 900;
done
