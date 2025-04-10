#!/usr/bin/env python
import pandas as pd
import os

#### cosmo_c***_ph***_z0p***.param files seem unreliable. 
#### read directly from the csv file
def get_cosmo_para(cosmo_id_wanted):
    df = pd.read_csv('/projects/hywu/cluster_sims/cluster_finding/data/AbacusSummit_base/cosmologies.csv', sep=',')
    df.columns = df.columns.str.replace(' ', '')
    #print(df.columns)
    nrows = df.shape[0]
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
            break

    cosmo_dict = {'cosmo_id': cosmo_id, 'OmegaM': OmegaM, 'OmegaL': OmegaL,
        'hubble': hubble,'sigma8': sigma8,'OmegaB': OmegaB,'ns': ns, 
        'w0': row['w0_fld'], 'wa': row['wa_fld'], 'alpha_s': row['alpha_s'],
        'Nur': row['N_ur']}
    return cosmo_dict


def get_hod_para(hod_id_wanted):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    loc = os.path.join(BASE_DIR, '../../repo/abacus_summit/')
    df1 = pd.read_csv(loc+'/hod_params.csv', sep=',')
    df2 = pd.read_csv(loc+'hod_plusminus.csv', sep=',')
    df3 = pd.read_csv(loc+'hod_latin1.csv', sep=',')
    df4 = pd.read_csv(loc+'hod_latin2.csv', sep=',')
    df5 = pd.read_csv(loc+'hod_latin3.csv', sep=',')
    dfs = [df1, df2, df3, df4, df5]
    df = pd.concat(dfs, axis=0)  # Vertical stacking (rows)
    #print(df)

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
    print(get_hod_para(201))