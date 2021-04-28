import io
import os
import requests
import pathlib
import gzip
import imageio
import pandas as pd
from multiprocessing import Pool, cpu_count

def tif_gzip_to_png(tif_path):
    '''Function to convert .tif.gz to .png and put it in the same folder
    Eg. for working in local work station
    '''
    png_path = pathlib.Path(tif_path.replace('.tif.gz','.png'))
    tf = gzip.open(tif_path).read()
    img = imageio.imread(tf, 'tiff')
    imageio.imwrite(png_path, img)
    
def download_and_convert_tifgzip_to_png(paths):    
    '''Function to convert .tif.gz to .png and put it in the same folder
    Eg. in Kaggle notebook
    '''
    try:
        url,target_path = paths
        r = requests.get(url)
        print(len(r.content))
        f = io.BytesIO(r.content)
        tf = gzip.open(f).read()
        img = imageio.imread(tf, 'tiff')
        imageio.imwrite(target_path, img)
        print(f'Downloaded: {url} as {target_path}')  
    except:
        print(f'Failed: {url}')
    
# All label names in the public HPA and their corresponding index. 
all_locations = dict({
    "Nucleoplasm": 0,
    "Nuclear membrane": 1,
    "Nucleoli": 2,
    "Nucleoli fibrillar center": 3,
    "Nuclear speckles": 4,
    "Nuclear bodies": 5,
    "Endoplasmic reticulum": 6,
    "Golgi apparatus": 7,
    "Intermediate filaments": 8,
    "Actin filaments": 9,
    "Focal adhesion sites": 9,
    "Microtubules": 10,
    "Mitotic spindle": 11,
    "Centrosome": 12,
    "Centriolar satellite": 12,
    "Plasma membrane": 13,
    "Cell Junctions": 13,
    "Mitochondria": 14,
    "Aggresome": 15,
    "Cytosol": 16,
    "Vesicles": 17,
    "Peroxisomes": 17,
    "Endosomes": 17,
    "Lysosomes": 17,
    "Lipid droplets": 17,
    "Cytoplasmic bodies": 17,
    "No staining": 18
})


def add_label_idx(df, all_locations):
    '''Function to convert label name to index
    '''
    df["Label_idx"] = None
    for i, row in df.iterrows():
        labels = row.Label.split(',')
        idx = []
        for l in labels:
            if l in all_locations.keys():
                idx.append(str(all_locations[l]))
        if len(idx)>0:
            df.loc[i,"Label_idx"] = "|".join(idx)
            
        print(df.loc[i,"Label"], df.loc[i,"Label_idx"])
    return df

public_hpa_df = pd.read_csv('./data/kaggle_2021.tsv')
# Remove all images overlapping with Training set
public_hpa_df = public_hpa_df[public_hpa_df.in_trainset == False]

# Remove all images with only labels that are not in this competition
public_hpa_df = public_hpa_df[~public_hpa_df.Label_idx.isna()]

colors = ['blue', 'red', 'green', 'yellow']
celllines = ['A-431', 'A549', 'EFO-21', 'HAP1', 'HEK 293', 'HUVEC TERT2', 'HaCaT', 'HeLa', 'PC-3', 'RH-30', 'RPTEC TERT1', 'SH-SY5Y', 'SK-MEL-30', 'SiHa', 'U-2 OS', 'U-251 MG', 'hTCEpi']
public_hpa_df_17 = public_hpa_df[public_hpa_df.Cellline.isin(celllines)]
print(len(public_hpa_df), len(public_hpa_df_17))
public_hpa_17_copy = public_hpa_df_17

to_download = ['8', '6', '10', '9', '1', '15', '11']

public_hpa_df_17['Label_idx'] = public_hpa_df_17['Label_idx'].str.split('|')
public_hpa_df_17 = public_hpa_df_17.explode('Label_idx').reset_index().drop(columns=['index'])

public_hpa_to_download = public_hpa_df_17[public_hpa_df_17['Label_idx'].isin(to_download)].reset_index().drop(columns=['index'])

print(len(public_hpa_to_download))
unique = public_hpa_to_download['Image'].unique()

save_dir = os.path.join('../data','public_hpa')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
download_files = []

for i, img in enumerate(unique):       
    for color in colors:
        img_url = f'{img}_{color}.tif.gz'
        save_path = os.path.join(save_dir,  f'{os.path.basename(img)}_{color}.png')
        download_files.append((img_url,save_path))  

#print("There are {} CPUs on this machine ".format(cpu_count()))
pool = Pool(2)

results = pool.imap(download_and_convert_tifgzip_to_png, download_files)
pool.close()
pool.join()


public_hpa_train = public_hpa_17_copy[public_hpa_17_copy['Image'].isin(unique)]
public_hpa_train['ID'] = public_hpa_train['Image'].apply(lambda x: os.path.basename(x))
public_hpa_train = public_hpa_train.drop(columns=['Image', 'Label', 'Cellline','in_trainset'], axis=1)
public_hpa_train = public_hpa_train.rename(columns={'Label_idx':'Label'}).reset_index().drop(columns=['index'])

original_train_df = pd.read_csv('../data/train.csv')
full_train_df = original_train_df.append(public_hpa_train).reset_index().drop(columns=['index'])
full_train_df.to_csv('../data/train_additional.csv', index=False)
print(len(full_train_df))
