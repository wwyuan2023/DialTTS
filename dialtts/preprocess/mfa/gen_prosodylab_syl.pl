#
#
#

use strict;
use warnings;
use utf8;
use Encode;

binmode STDIN,  "utf8";
binmode STDOUT, "utf8";
binmode STDERR, "utf8";

my $outdir = $ARGV[0];
my $dictfn = $ARGV[1];

my %hs2p = ("sil" => "sil"); # mapping of syllable to phoneme
while(<DATA>){
    s/\d+//g;
    s/^\s+|\s+$//g;
    next if(/^$/);
    my @arr = split;
    my $syl = shift @arr;
    $hs2p{$syl} = \@arr;
}

my %hdict = ("sil sil" => 1);
while(<STDIN>){
    s/^\s+|\s+$//g;
    die $_ if(!/^(\S+)\s+(.+)$/);
    my $outfn = $outdir . "/$1.lab";
    my $txtln = $2;
    print "Process $outfn ...";
    my $str = "";
    foreach my $seg(split /\s+/, $txtln){
        if($seg =~ /^(.+)\/(.*?);/){
            my ($wds, $pys) = ($1, $2);
            my $dstr = "";
            if($pys eq ""){
                next;
            }elsif($pys =~ /sil/){
                $str .= "sil ";
                next;
            }elsif($pys =~ /\(/){ # English
                foreach(split /-/, $pys){
                    die $_ if(!/^\((.+)\)(\d+)$/);
                    my ($phns, $tone) = ($1, $2);
                    foreach my $phn(split /_/, $phns){
                        $dstr .= " EN$phn";
                    }
                }
            }else{ # Chinese
                foreach(split /-/, $pys){
                    die $_ if(!/^([a-z]+)(\d+)$/);
                    my ($py, $tone) = ($1, $2);
                    die "$outfn: $py is not in dict." if(!exists $hs2p{$py});
                    foreach my $phn(@{$hs2p{$py}}){
                        $dstr .= " CN$phn";
                    }
                }
            }
            $pys =~ s/\(|\)//g;
            $str .= "$pys ";
            $hdict{$pys . $dstr} = 1;
        }else{
            die "$outfn: $seg: format error";
        }
    }
    open F, ">$outfn" or die "$?: $outfn";
    $str =~ s/^\s+|\s+$//g;
    $str =~ s/^sil\s+|\s+sil$//g;
    print F $str;
    close F;
    print "Done!\n";
}

# output dictionary
open D, ">$dictfn" or die "$?: $dictfn";
foreach(sort keys %hdict){
    print D "$_\n";
}
close D;


exit(0);

__DATA__
a 1 aa
ai 2 ay ib
an 2 ae n
ang 2 ah ng
ao 2 ah ub
ba 2 b aa
bai 3 b ay ib
ban 3 b ae n
bang 3 b ah ng
bao 3 b ah ub
bei 3 b ei ib
ben 3 b ax n
beng 3 b oe ng
bi 2 b iy
bian 4 b il eh n
biao 4 b il ah ub
bie 3 b il ee
bin 3 b ih n
bing 3 b ih ng
bo 3 b ul ao
bu 2 b uw
ca 2 c aa
cai 3 c ay ib
can 3 c ae n
cang 3 c ah ng
cao 3 c ah ub
ce 2 c ea
cei 3 c ei ib
cen 3 c ax n
ceng 3 c oe ng
cha 2 ch aa
chai 3 ch ay ib
chan 3 ch ae n
chang 3 ch ah ng
chao 3 ch ah ub
che 2 ch ea
chen 3 ch ax n
cheng 3 ch oe ng
chi 2 ch izh
chong 3 ch oh ng
chou 3 ch oh ub
chu 2 ch uw
chua 3 ch ul aa
chuai 4 ch ul ay ib
chuan 4 ch ul ae n
chuang 4 ch ul ah ng
chui 4 ch ul ei ib
chun 4 ch ul ax n
chuo 3 ch ul ao
ci 2 c iz
cong 3 c oh ng
cou 3 c oh ub
cu 2 c uw
cuan 4 c ul ae n
cui 4 c ul ei ib
cun 4 c ul ax n
cuo 3 c ul ao
da 2 d aa
dai 3 d ay ib
dan 3 d ae n
dang 3 d ah ng
dao 3 d ah ub
de 2 d ea
dei 3 d ei ib
den 3 d ax n
deng 3 d oe ng
di 2 d iy
dia 3 d il aa
dian 4 d il eh n
diao 4 d il ah ub
die 3 d il ee
ding 3 d ih ng
diu 4 d il oh ub
dong 3 d oh ng
dou 3 d oh ub
du 2 d uw
duan 4 d ul ae n
dui 4 d ul ei ib
dun 4 d ul ax n
duo 3 d ul ao
e 1 ea
ei 2 ei ib
en 2 ax n
eng 2 oe ng
er 1 aar
fa 2 f aa
fan 3 f ae n
fang 3 f ah ng
fei 3 f ei ib
fen 3 f ax n
feng 3 f oe ng
fiao 4 f il ah ub
fo 3 f ul ao
fou 3 f oh ub
fu 2 f uw
ga 2 g aa
gai 3 g ay ib
gan 3 g ae n
gang 3 g ah ng
gao 3 g ah ub
ge 2 g ea
gei 3 g ei ib
gen 3 g ax n
geng 3 g oe ng
gong 3 g oh ng
gou 3 g oh ub
gu 2 g uw
gua 3 g ul aa
guai 4 g ul ay ib
guan 4 g ul ae n
guang 4 g ul ah ng
gui 4 g ul ei ib
gun 4 g ul ax n
guo 3 g ul ao
ha 2 h aa
hai 3 h ay ib
han 3 h ae n
hang 3 h ah ng
hao 3 h ah ub
he 2 h ea
hei 3 h ei ib
hen 3 h ax n
heng 3 h oe ng
hong 3 h oh ng
hou 3 h oh ub
hu 2 h uw
hua 3 h ul aa
huai 4 h ul ay ib
huan 4 h ul ae n
huang 4 h ul ah ng
hui 4 h ul ei ib
hun 4 h ul ax n
huo 3 h ul ao
ji 2 j iy
jia 3 j il aa
jian 4 j il eh n
jiang 4 j il ah ng
jiao 4 j il ah ub
jie 3 j il ee
jin 3 j ih n
jing 3 j ih ng
jiong 4 j il oh ng
jiu 4 j il oh ub
ju 2 j vw
juan 4 j vl eh n
jue 3 j vl ee
jun 4 j vl ih n
ka 2 k aa
kai 3 k ay ib
kan 3 k ae n
kang 3 k ah ng
kao 3 k ah ub
ke 2 k ea
kei 3 k ei ib
ken 3 k ax n
keng 3 k oe ng
kong 3 k oh ng
kou 3 k oh ub
ku 2 k uw
kua 3 k ul aa
kuai 4 k ul ay ib
kuan 4 k ul ae n
kuang 4 k ul ah ng
kui 4 k ul ei ib
kun 4 k ul ax n
kuo 3 k ul ao
lv 2 l vw
lve 3 l vl ee
lue 3 l vl ee
la 2 l aa
lai 3 l ay ib
lan 3 l ae n
lang 3 l ah ng
lao 3 l ah ub
le 2 l ea
lei 3 l ei ib
leng 3 l oe ng
li 2 l iy
lia 3 l il aa
lian 4 l il eh n
liang 4 l il ah ng
liao 4 l il ah ub
lie 3 l il ee
lin 3 l ih n
ling 3 l ih ng
liu 4 l il oh ub
lo 2 l ao
long 3 l oh ng
lou 3 l oh ub
lu 2 l uw
luan 4 l ul ae n
lun 4 l ul ax n
luo 3 l ul ao
ma 2 m aa
mai 3 m ay ib
man 3 m ae n
mang 3 m ah ng
mao 3 m ah ub
me 2 m ea
mei 3 m ei ib
men 3 m ax n
meng 3 m oe ng
mi 2 m iy
mian 4 m il eh n
miao 4 m il ah ub
mie 3 m il ee
min 3 m ih n
ming 3 m ih ng
miu 4 m il oh ub
mo 3 m ul ao
mou 3 m oh ub
mu 2 m uw
nv 2 n vw
nve 3 n vl ee
nue 3 n vl ee
na 2 n aa
nai 3 n ay ib
nan 3 n ae n
nang 3 n ah ng
nao 3 n ah ub
ne 2 n ea
nei 3 n ei ib
nen 3 n ax n
neng 3 n oe ng
ni 2 n iy
nian 4 n il eh n
niang 4 n il ah ng
niao 4 n il ah ub
nie 3 n il ee
nin 3 n ih n
ning 3 n ih ng
niu 4 n il oh ub
nong 3 n oh ng
nou 3 n oh ub
nu 2 n uw
nuan 4 n ul ae n
nun 4 n ul ax n
nuo 3 n ul ao
o 1 ao
ou 2 oh ub
pa 2 p aa
pai 3 p ay ib
pan 3 p ae n
pang 3 p ah ng
pao 3 p ah ub
pei 3 p ei ib
pen 3 p ax n
peng 3 p oe ng
pi 2 p iy
pian 4 p il eh n
piao 4 p il ah ub
pie 3 p il ee
pin 3 p ih n
ping 3 p ih ng
po 3 p ul ao
pou 3 p oh ub
pu 2 p uw
qi 2 q iy
qia 3 q il aa
qian 4 q il eh n
qiang 4 q il ah ng
qiao 4 q il ah ub
qie 3 q il ee
qin 3 q ih n
qing 3 q ih ng
qiong 4 q il oh ng
qiu 4 q il oh ub
qu 2 q vw
quan 4 q vl eh n
que 3 q vl ee
qun 4 q vl ih n
ran 3 r ae n
rang 3 r ah ng
rao 3 r ah ub
re 2 r ea
ren 3 r ax n
reng 3 r oe ng
ri 2 r izh
rong 3 r oh ng
rou 3 r oh ub
ru 2 r uw
rua 3 r ul aa
ruan 4 r ul ae n
rui 4 r ul ei ib
run 4 r ul ax n
ruo 3 r ul ao
sa 2 s aa
sai 3 s ay ib
san 3 s ae n
sang 3 s ah ng
sao 3 s ah ub
se 2 s ea
sei 3 s ei ib
sen 3 s ax n
seng 3 s oe ng
sha 2 sh aa
shai 3 sh ay ib
shan 3 sh ae n
shang 3 sh ah ng
shao 3 sh ah ub
she 2 sh ea
shei 3 sh ei ib
shen 3 sh ax n
sheng 3 sh oe ng
shi 2 sh izh
shou 3 sh oh ub
shu 2 sh uw
shua 3 sh ul aa
shuai 4 sh ul ay ib
shuan 4 sh ul ae n
shuang 4 sh ul ah ng
shui 4 sh ul ei ib
shun 4 sh ul ax n
shuo 3 sh ul ao
si 2 s iz
song 3 s oh ng
sou 3 s oh ub
su 2 s uw
suan 4 s ul ae n
sui 4 s ul ei ib
sun 4 s ul ax n
suo 3 s ul ao
ta 2 t aa
tai 3 t ay ib
tan 3 t ae n
tang 3 t ah ng
tao 3 t ah ub
te 2 t ea
tei 3 t ei ib
teng 3 t oe ng
ti 2 t iy
tian 4 t il eh n
tiao 4 t il ah ub
tie 3 t il ee
tin 3 t ih n
ting 3 t ih ng
tong 3 t oh ng
tou 3 t oh ub
tu 2 t uw
tuan 4 t ul ae n
tui 4 t ul ei ib
tun 4 t ul ax n
tuo 3 t ul ao
wa 2 w aa
wai 3 w ay ib
wan 3 w ae n
wang 3 w ah ng
wei 3 w ei ib
wen 3 w ax n
weng 3 w oe ng
wo 2 w ao
wu 2 w uw
xi 2 x iy
xia 3 x il aa
xian 4 x il eh n
xiang 4 x il ah ng
xiao 4 x il ah ub
xie 3 x il ee
xin 3 x ih n
xing 3 x ih ng
xiong 4 x il oh ng
xiu 4 x il oh ub
xu 2 x vw
xuan 4 x vl eh n
xue 3 x vl ee
xun 4 x vl ih n
ya 2 y aa
yan 3 y ae n
yang 3 y ah ng
yao 3 y ah ub
ye 2 y ee
yi 2 y iy
yin 3 y ih n
ying 3 y ih ng
yo 2 y ao
yong 3 y oh ng
you 3 y oh ub
yu 2 y vw
yuan 4 y vl eh n
yue 3 y vl ee
yun 4 y vl ih n
za 2 z aa
zai 3 z ay ib
zan 3 z ae n
zang 3 z ah ng
zao 3 z ah ub
ze 2 z ea
zei 3 z ei ib
zen 3 z ax n
zeng 3 z oe ng
zha 2 zh aa
zhai 3 zh ay ib
zhan 3 zh ae n
zhang 3 zh ah ng
zhao 3 zh ah ub
zhe 2 zh ea
zhei 3 zh ei ib
zhen 3 zh ax n
zheng 3 zh oe ng
zhi 2 zh izh
zhong 3 zh oh ng
zhou 3 zh oh ub
zhu 2 zh uw
zhua 3 zh ul aa
zhuai 4 zh ul ay ib
zhuan 4 zh ul ae n
zhuang 4 zh ul ah ng
zhui 4 zh ul ei ib
zhun 4 zh ul ax n
zhuo 3 zh ul ao
zi 2 z iz
zong 3 z oh ng
zou 3 z oh ub
zu 2 z uw
zuan 4 z ul ae n
zui 4 z ul ei ib
zun 4 z ul ax n
zuo 3 z ul ao
hm 2 h m
hng 2 h ng
m 1 m
ng 1 ng
sil 1 pau
oov 1 pau
pau 1 pau
m 1 m
kiu 4 k il oh ub
fai 3 f ay ib
bia 3 b il aa
biu 4 b il oh ub
gia 3 g il aa
giao 4 g il ah ub
giu 4 g il oh ub
hia 3 h il aa
hiao 4 h il ah ub
hiu 4 h il oh ub
kia 3 k il aa
kiao 4 k il ah ub
mia 3 m il aa
nia 3 n il aa
pia 3 p il aa
piu 4 p il oh ub
be 2 b ea
pe 2 p ea
bou 3 b oh ub
bua 3 b ul aa
cua 3 c ul aa
dua 3 d ul aa
lua 3 l ul aa
mua 3 m ul aa
nua 3 n ul aa
pua 3 p ul aa
sua 3 s ul aa
tua 3 t ul aa
bue 3 b vl ee
due 3 d vl ee
gue 3 g vl ee
hue 3 h vl ee
kue 3 k vl ee
mue 3 m vl ee
pue 3 p vl ee
sue 3 s vl ee
tue 3 t vl ee
wue 3 w vl ee
bui 4 b ul ei ib
jui 4 j ul ei ib
lui 4 l ul ei ib
mui 4 m ul ei ib
nui 4 n ul ei ib
pui 4 p ul ei ib
qui 4 q ul ei ib
xui 4 x ul ei ib
buo 3 b ul ao
juo 3 j ul ao
muo 3 m ul ao
puo 3 p ul ao
quo 3 q ul ao
gie 3 g il ee
hie 3 h il ee
kie 3 k il ee
tia 3 t il aa
tiu 4 t il oh ub


