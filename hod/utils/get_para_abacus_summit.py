#!/usr/bin/env python
import pandas as pd
import os
import glob

#### cosmo_c***_ph***_z0p***.param files seem unreliable. 
#### read directly from the csv file
def get_cosmo_para(cosmo_id_wanted):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    loc = os.path.join(BASE_DIR, '../../repo/latin_hypercube/')

    df = pd.read_csv(loc+'parameters/cosmologies.csv', sep=',')
    df.columns = df.columns.str.replace(' ', '')
    #print(df.columns)
    nrows = df.shape[0]
    
    cosmo_dict = None # update if the cosmology exsits. otherwise return None

    for irow in range(nrows):
        # retrieve one cosmology at a time
        row = df.iloc[irow]
        root = row['root'].replace(' ', '')
        cosmo_id = int(root[-3:])

        if cosmo_id == cosmo_id_wanted:
            #print('cosmo_id', cosmo_id)
            hubble = row['h']
            OmegaB = row['omega_b']/hubble**2
            if len(row['omega_ncdm']) == 12:
                Oncdm = float(row['omega_ncdm'])/hubble**2
            else:
                Oncdm = float(row['omega_ncdm'][0:10])/hubble**2

            OmegaM = row['omega_cdm']/hubble**2 + OmegaB + Oncdm
            OmegaL = 1 - OmegaM
            #sigma8 = row['sigma8_cb'] # baryons-plus-cdm-only (CLASS)
            sigma8 = row['sigma8_m']
            ns = row['n_s']
        
            cosmo_dict = {'cosmo_id': cosmo_id, 'OmegaM': OmegaM, 'OmegaL': OmegaL,
                'hubble': hubble,'sigma8': sigma8,'OmegaB': OmegaB,'ns': ns, 
                'w0': row['w0_fld'], 'wa': row['wa_fld'], 'alpha_s': row['alpha_s'],
                'Nur': row['N_ur']}

            

    return cosmo_dict


def get_hod_para(hod_id_wanted):#, miscen=False):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    loc = os.path.join(BASE_DIR, '../../repo/latin_hypercube/')

    # collect all the csv files
    # if miscen == True:
    #     fname_list = glob.glob(loc+'parameters_miscen/*.csv')
    # else:
    #     fname_list = glob.glob(loc+'parameters/*.csv')
    fname_list = glob.glob(loc+'parameters/hod_*.csv')
    # put all of them in a big data frame
    df_list = []
    for fname in fname_list:
        df_in = pd.read_csv(fname, sep=',')
        df_list.append(df_in)

    df = pd.concat(df_list, axis=0)  # Vertical stacking (rows)
    
    #print('largest hod_id', max(df['hod_id']))
    nrows = df.shape[0]
    for irow in range(nrows):
        row = df.iloc[irow]
        hod_id = row['hod_id']
        if hod_id == hod_id_wanted:
            row_output = row
            break
    return row_output.to_dict() # converting a data frame to a dictionary
    
if __name__ == "__main__":
    print(get_cosmo_para(0))


    print(get_hod_para(0))
    # print(get_hod_para(14))
    # print(get_hod_para(18))
    #print(get_hod_para(201))
    #print(get_hod_para(2051))