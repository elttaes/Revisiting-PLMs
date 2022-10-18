import os
from Bio import SeqIO


if __name__ == '__main__':
    path = 'a3m_list'
    for file in os.listdir(path):
        
        print('*' * 100, flush=True)
        print(file, flush=True)
        print('*' * 100, flush=True)
        ori_name = file.replace('.a3m', '')
        
        record = next(SeqIO.parse(f"{path}/{file}", 'fasta'))
       # if len(str(record.seq)) >= 200:
        #    continue

        if os.path.exists(f"out/{ori_name}/ranking_debug.json"):
            continue
        
        if not os.path.exists(f"out/{ori_name}"):
            os.mkdir(f"out/{ori_name}")
            os.mkdir(f"out/{ori_name}/msas")
        os.system(f"cp {path}/{file} out/{ori_name}/msas/bfd_uniclust_hits.a3m")
            
        with open(f"fasta_list/{ori_name}.fasta", 'w') as f:
            f.write(f">{record.description}\n{str(record.seq)}\n")
        
        os.system(f"python3 run_alphafold.py --fasta_paths=fasta_list/{ori_name}.fasta --max_template_date=2020-05-14")
        
