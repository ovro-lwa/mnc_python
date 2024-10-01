#!/bin/bash

workingdir="/lustre/pipeline/lastnight/"

mkdir -p $workingdir

#msdate="20240127"

last=$(date --date='yesterday' '+%Y%m%d')
msdate=$last

hrs=(06 07 08 09 10 11 12)
for hr in ${hrs[@]}
do 
    python3 extract_autocor.py -p /lustre/pipeline/slow/ -d $msdate -t $hr -w $workingdir
done

workingdir=$workingdir$last"/"
python3 plot_autocor.py -p $workingdir

# clean up old data
old=$(date --date='last week' '+%Y%m%d')
olddir=$workingdir$old"/"
rm -rf $olddir

scp /tmp/lastnight/$last/fig/${last}_antenna_status_*.png lwacalim10:/data10/pipeline/anthealth/
scp /tmp/lastnight/$last/stats/${last}.txt lwacalim10:/data10/pipeline/anthealth/
