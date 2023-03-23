#!/usr/bin/perl
#
#

use strict;
use warnings;
use utf8;
use Encode;

binmode STDIN,  "utf8";
binmode STDOUT, "utf8";
binmode STDERR, "utf8";

#*************************#
my %hpunc = ("。" => "sil0",
    "！" => "sil0",
    "？" => "sil0",
    "！？" => "sil0",
    "？！" => "sil0",
    "，" => "sil0",
    "；" => "sil0",
    "：" => "sil0",
    "…" => "sil0",
    "——" => "sil0");
my @gpos = qw /a c d f i m n nr nx nz o p q r t u v w y/;
my %hgpos;
foreach(@gpos){ $hgpos{$_} = 1 }

my $max_wd = "";
my $max_wlen = 0;
#*************************#

my $mrk = shift @ARGV;

open F, $mrk or die "$mrk: $!\n";
binmode F, "utf8";
my @all;
while(<F>){
    s/^\s+|\s+$//g;
    next if(/^$/);
    my $mrk = $_;
    my $err = "";
    die if(!/^(\S+)\s+(.+)\|(.+)$/);
    my ($id, $seg, $py) = ($1, $2, $3);
    
    $id =~ s/\.wav$//i;
    $py=~s/(\d+)([a-z]+|\[)/$1 $2/g;
    
    my @py = split /\s+/, lc($py);
    my @seg = split /\s+/, $seg;
    my @out = ();
    for(my $i = 0; $i < scalar(@seg); ++$i){
        die "$id, $seg[$i]" if($seg[$i] !~ /^(.+)\/(.+)$/);
        my ($wd, $cx) = ($1, $2);
        # 判断儿化音
        if($wd eq "儿" && scalar(@out) > 0){
            die if($out[-1] !~ /^(.+)\/([^;]+)(.+)$/);
            my ($wd_, $py_, $left) = ($1, $2, $3);
            if($py_ !~ /^er\d$/ && $py_ =~ /r\d$/){
                $out[-1] = "${wd_}$wd/${py_}$left";
                next;
            }
        }
        # print $wd, $cx, "\n";
        my $wp = "";
        my $lang = 'CN'; # CN=Chinese; EN=English
        if($wd !~ /^\p{Han}+$/ && $wd !~ /[a-zA-Z]+/){
            # 句末标点符号的对应标记
            $wd =~ s/……/…/g;
            $wd =~ s/\.{3,}/…/g;
            $wd =~ s/-{2,}/——/g;
            $wd =~ tr/\.!\?,;:/。！？，；：/;
            if(exists $hpunc{$wd}){ # 属于句末标点
                $wp = "$wd/sil0;w;$lang;;;";
            }else{
                $wp = "$wd/;w;$lang;;;";
            }
        }elsif($wd =~ /^(n\'t|\'ll|\'ve|\'re|\'s|\'m|\'d|\'em|\')$/i){
            # 不发音的英文单词
            $lang = 'EN';
            $cx = 'nx';
            $wp = "$wd/;$cx;$lang;;;";
        }else{
            #die "$id, @out: $wd/$cx" if(scalar(@py) == 0);
            if(scalar(@py) == 0){
                $err = "$id, @out: $wd/$cx\n";
                goto next_label;
            }
            my $py = shift @py;
            if($py =~ s/\[|\]//g){
                # 英文
                $lang = 'EN';
                $cx = "nx"; # 英文词性改成中文词性
                $py =~ s/2/1/g; # tone: 2 -> 1
                #die "$id, @out: $wd" if($wd !~ /^[a-zA-Z'\.\-]+$/);
                if($wd !~ /^[a-zA-Z'\.\-]+$/){
                    $err = "$id, @out: $wd\n";
                    goto next_label;
                }
            }else{
                $lang = 'CN';
                $py =~ s/6/2/g; # tone:6 -> 2
                my @wd = split //, $wd;
                for(my $j = 1; $j < @wd; ++$j){
                    if(0 && $wd[$j] eq "儿" && $py =~ /r\d{1}$/){
                        # 儿化音
                    }elsif($wd[$j] !~ /^\p{Han}+$/){
                        # 不是汉子
                    }else{
                        # 中文
                        #die "$id, @out: $wd" if(@py <= 0);
                        if(@py <= 0){
                            $err = "$id, @out: $wd";
                            goto next_label;
                        }
                        $_ = shift @py;
                        s/6/2/g;
                        $py .= "-$_";
                    }
                }
                if(scalar(@wd) > $max_wlen and $cx ne "m"){
                    $max_wlen = scalar(@wd);
                    $max_wd = $wd;
                }
            }
            #die "$id, $cx, $wd, @out" if(!exists $hgpos{$cx});
            if(!exists $hgpos{$cx}){
                $err = "$id, $cx, $wd, @out";
                goto next_label;
            }
            $wp = "$wd/$py;$cx;$lang;;;";
        }
        push @out, $wp;
    }
    if(scalar(@py) > 0){
        #die "$id, @py, @out";
        $err = "$id, @py, @out\n";
        goto next_label;
    }else{
        unshift @out, $id;
        push @all, join(" ", @out);
    }
    #die if($id eq "");
    next_label:
    print STDERR $err if($err ne "");
}
close F;

foreach(@all){
    print "$_\n";
}

                
