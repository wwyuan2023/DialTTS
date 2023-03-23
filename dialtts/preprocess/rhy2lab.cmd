cat polo.rhy | perl -ne '
if(/^Wav=(.+)\.wav$/){ print "$1    " }
if(/^Pinyin=(.+)$/){ $py=$1; $py=~s/\s*\*\s*/ /g; $py=~s/(\)\d)\s+/$1-/g; $py=~s/([a-z]+)\s+/${1}_/g; $py=~s/(\d+)([a-z]+)/$1 $2/g; }
if(/^Pos=(.+)$/){ $cx=$1; $cx=~s/\/sent/\/w/g;  print "$cx|$py\n" }
'