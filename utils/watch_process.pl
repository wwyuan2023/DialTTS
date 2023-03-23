#/usr/bin/perl
# 
#


#-------GLOBAL--------#

# @pwatch: process which need monitor
@pwatch = ();

# @pignor: preocess which ignoring
@pignor = ();

# @prun: process which runing
@prun = ();

#--------MAIN---------#

&parse_opt();
&get_run();

# $f: @pwatch is belong to @prun ?
$f = 1;
$f = 0 if($#pwatch > $#prun);
for($i = 0; $f && $i <= $#pwatch; $i ++){
    for($j = $i; $f && $j <= $#prun; $j ++){
        last if($pwatch[$i] == $prun[$j]);
    }
    $f = 0 if($j > $#prun);
}

# kill @prun all if $f is false and @pwatch is not empty
if(!$f){
    for($n = 0; $n <= $#pwatch; $n ++){
        $s = join(" ", @pwatch);
        `kill -9 $s`;
        sleep 1;
    }
    @pwatch = ();
    sleep 30;
}else{
    @pwatch = @prun if($#pwatch <= 0);
}
print join(" ", @pwatch);

exit(0) if(@pwatch > 0);
exit(1);

#--------END----------#

sub parse_opt{
    @args = @ARGV;
    for($i = 0; $i <= $#args; $i++){
        if($args[$i] =~ /^-i$/){
            for($i++; $i <= $#args && $args[$i] !~ /^-/; $i++){
                push @pignor, $args[$i];
            }
            $i--;
        }elsif($args[$i] =~ /^-w$/){
            for($i++; $i <= $#args && $args[$i] !~ /^-/; $i++){
                push @pwatch, $args[$i];
            }
            $i--;
        }elsif($args[$i] =~ /^\d+$/){
            push @pwatch, $args[$i];
        }else{
        }
    }
}

sub get_run{
    @a=`nvidia-smi`;
    while(@a>0){
        $_ = shift @a;
        last if(/GPU\s+GI\s+CI\s+PID\s+Type\s+Process\s+name/);
    }

    foreach(@a){
         if(/\s+(\d{3,5})\s+/){
             $p = $1;
             $g = 0;
             for($i = 0; $i <= $#pignor; $i ++){
                 $g = 1 if($pignor[$i] == $p);
             }
             push @prun, $p if(!$g);
         }
    }
    @prun = sort @prun;
}




