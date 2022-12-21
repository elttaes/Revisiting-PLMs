import os
import pickle
import time
from absl import app
from alphafold.data import pipeline

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

DOWNLOAD_DIR='./'
data_predata_save = './data/af2_data'

if not (os.path.exists(data_predata_save)):
    os.makedirs(data_predata_save)

if not (os.path.exists(data_predata_save + '/fasta')):
    os.makedirs(data_predata_save + '/fasta')
if not (os.path.exists(data_predata_save + '/feature')):
    os.makedirs(data_predata_save + '/feature')

# the all ouptput will be saved into data_predata_save
feature_path = os.path.join(data_predata_save, 'feature')

def predict_structure(
        a3m_path: str,
        fasta_name: str,
        data_pipeline: pipeline.DataPipeline, label):
    """Predicts structure using AlphaFold for the given sequence."""
    feature_dir = os.path.join(feature_path, fasta_name + '.pkl')
    timings = {}
    # Get features.
    t_0 = time.time()
    feature_dict = data_pipeline.process(
        fasta_name=fasta_name,
        input_a3m_path=a3m_path)
    feature_dict['label'] = label
    timings['features'] = time.time() - t_0
    
    # Write out features as a pickled dictionary.
    print(feature_dir)
    features_output_path = os.path.join(feature_dir)
    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)

def main(argv):
    data_pipeline = pipeline.DataPipeline()
    #a3m folder including all MSA file such as 1.a3m,2.a3m,3.a3m
    #label folder including correspond label file such as 1.label, 2.label, 3.label
    for name in os.listdir('data/a3m'):
        print(name)
        hit_pdb_code = name[:-4]
        a3m_path='data/a3m/'+name
        f = open(a3m_path)
        seqence = f.readline()
        seqence = f.readline()
        print(seqence)
        f.close()
        f = open('data/label/'+name[:-4])
        label = f.readline()
        f.close()

        fasta_dir = os.path.join(data_predata_save, 'fasta')
        filename = os.path.join(fasta_dir, hit_pdb_code + '.fasta')
        f = open(filename, "w")
        a = ">" + hit_pdb_code + "\n" + seqence
        f.write(a)
        f.close()
        predict_structure(
            a3m_path = a3m_path,
            fasta_name=hit_pdb_code,
            data_pipeline=data_pipeline,label=label)


if __name__ == '__main__':
    while(1):
        app.run(main)
