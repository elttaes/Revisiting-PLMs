import os
import pickle
import time
from absl import app
from alphafold.data import pipeline

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

DOWNLOAD_DIR='./'
data_predata_save = './pkl'

if not (os.path.exists(data_predata_save)):
    os.makedirs(data_predata_save)

if not (os.path.exists(data_predata_save + '/msa')):
     os.makedirs(data_predata_save + '/msa')
if not (os.path.exists(data_predata_save + '/fasta')):
     os.makedirs(data_predata_save + '/fasta')
if not (os.path.exists(data_predata_save + '/feature')):
     os.makedirs(data_predata_save + '/feature')

data_predata_save_finish = data_predata_save+'/finish'

if not (os.path.exists(data_predata_save_finish)):
    os.makedirs(data_predata_save_finish)
# the all ouptput will be saved into data_predata_save
feature_path = os.path.join(data_predata_save, 'feature')

def predict_structure(
        fasta_path: str,
        fasta_name: str,
        output_dir_base: str,
        data_pipeline: pipeline.DataPipeline,label):
    """Predicts structure using AlphaFold for the given sequence."""
    feature_dir = os.path.join(feature_path, fasta_name + '.pkl')
    timings = {}
    output_dir = os.path.join(output_dir_base, fasta_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    msa_output_dir = os.path.join(output_dir)
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir)
    # Get features.
    t_0 = time.time()
    feature_dict = data_pipeline.process(
        fasta_name=fasta_name,
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)
    feature_dict['label'] = label
    timings['features'] = time.time() - t_0
    
    # Write out features as a pickled dictionary.
    print(feature_dir)
    features_output_path = os.path.join(feature_dir)
    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)

def main(argv):

    data_pipeline = pipeline.DataPipeline()
    #seq folder including all MSA file such as 1.a3m,2.a3m,3.a3m
    #label folder including correspond label file such as 1.label, 2.label, 3.label
    for name in os.listdir('data_seq/seq'):
        print(name)
        hit_pdb_code = name

        f = open('data_seq/seq/'+name)
        seqence = f.readline()
        print(seqence)
        f.close()
        f = open('data_seq/label/'+name)
        label = f.readline()
        f.close()

        fasta_dir = os.path.join(data_predata_save, 'fasta')
        filename = os.path.join(fasta_dir, hit_pdb_code + '.fasta')
        f = open(filename, "w")
        a = ">" + hit_pdb_code + "\n" + seqence
        f.write(a)
        f.close()
        
        fasta_dir = os.path.join(data_predata_save, 'msa')
        filename = os.path.join(fasta_dir, hit_pdb_code + '.a3m')
        f = open(filename, "w")
        f.write(a)
        f.close()

        output_dir = os.path.join(data_predata_save, 'msa')
        predict_structure(
            fasta_path=filename,
            fasta_name=hit_pdb_code,
            output_dir_base=output_dir,
            data_pipeline=data_pipeline,label=label)


if __name__ == '__main__':
    while(1):
        app.run(main)
