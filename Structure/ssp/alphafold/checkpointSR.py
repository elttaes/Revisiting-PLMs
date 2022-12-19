import pickle
def save_checkpoint(v, filename):
    f=open(filename,'wb')
    pickle.dump(v, f)
    f.close()
    return filename
def load_checkpoint(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r