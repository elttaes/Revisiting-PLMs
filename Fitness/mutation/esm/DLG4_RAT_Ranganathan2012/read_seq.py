import DLG4_RAT_Ranganathan2012
print(DLG4_RAT_Ranganathan2012.seq)
count=0
for i in range(len(DLG4_RAT_Ranganathan2012.seq)):
    if(count==DLG4_RAT_Ranganathan2012.seq[i][1:-1]):
        continue
    seq=DLG4_RAT_Ranganathan2012.seq[i][0]
    count=DLG4_RAT_Ranganathan2012.seq[i][1:-1]
    print(seq,end='')
    #print(BG_STRSQ_Abate2015.res[i])