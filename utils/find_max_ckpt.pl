#/usr/bin/perl
# 
#

$max_step = 1000000;
$delete = 0;

$dir = $ARGV[0]; 
if((!defined $dir) or ($dir eq "") or ($dir eq "/")){
    exit(0);
}else{
    $dir =~ s/\/$//g;
}

$delete = $ARGV[1] if(defined $ARGV[1]);
$max_step = $ARGV[2] if(defined $ARGV[2]);

opendir DIR, $dir;
@a = ();
foreach(readdir DIR){
    next if(!/checkpoint-(\d+)steps.pkl/);
    push @a, [int($1), $_];
}
exit(0) if(scalar(@a) <= 0);

@a = sort {$b->[0] <=> $a->[0]} @a;
for($i = 2; $delete and $i < $#a; $i ++){
    unlink($dir . "/" . $a[$i][1]) if($a[$i][0] % 100000 != 0); # 删除ckpt，除了前两个和最后一个且不是100k的整数倍
}
if(@a > 0){
    print $dir . "/" . $a[0][1];
}else{
    print "";
}
exit(1) if($a[0][0] >= $max_step);
exit(0);

