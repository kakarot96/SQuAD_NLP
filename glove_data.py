from preprocess import *
import os
import zipfile
glovedata_url = 'http://nlp.stanford.edu/data/'

if __name__=='__main__':

    data_dir = os.path.join('data','download','glove')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_name = 'glove.6B.zip'
    download_from_url(data_dir, glovedata_url, file_name, 862182613L)
    
    zip_ref = zipfile.ZipFile(os.path.join(data_dir, file_name), 'r')
    zip_ref.extractall(data_dir)
    zep_ref.close()
