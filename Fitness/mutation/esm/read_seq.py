import BLAT_ECOLX_Palzkill2012
print(BLAT_ECOLX_Palzkill2012.seq)
count=0
for i in range(len(BLAT_ECOLX_Palzkill2012.seq)):
    if(count==BLAT_ECOLX_Palzkill2012.seq[i][1:-1]):
        continue
    seq=BLAT_ECOLX_Palzkill2012.seq[i][0]
    count=BLAT_ECOLX_Palzkill2012.seq[i][1:-1]
    print(seq,end='')
    #print(BG_STRSQ_Abate2015.res[i])