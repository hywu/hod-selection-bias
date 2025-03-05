#!/usr/bin/env python
import glob
import os
import pandas as pd
from astropy.io import fits
from astropy.table import Table

def merge_files(in_fname='x*.dat', out_fname='x.fit', nfiles_expected=100):
    # merge all temp file and output a fits file with the same header
    fname_list = glob.glob(in_fname)
    nfiles = len(fname_list)
    if nfiles < nfiles_expected:
        print('missing ', nfiles_expected - nfiles, 'files, not merging')
    else:
        df_list = [pd.read_csv(file, sep=r'\s+') for file in fname_list]
        merged_df = pd.concat(df_list, ignore_index=True)
        # Turn it into a fits file
        table = Table.from_pandas(merged_df)
        # Create FITS Header based on CSV column names
        hdr = fits.Header()
        column_names = merged_df.columns.tolist()
        for idx, col in enumerate(column_names):
            hdr[f'COLUMN{idx+1}'] = col
        primary_hdu = fits.PrimaryHDU(header=hdr)
        table_hdu = fits.BinTableHDU(table)
        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(out_fname, overwrite=True)

        os.system('rm -rf ' + in_fname)

if __name__ == '__main__':
    out_path = '/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/base_c000_ph000/z0p300/model_hod000/'
    merge_files(in_fname=f'{out_path}/temp/gals_*.dat', out_fname=f'{out_path}/gals.fit', nfiles_expected=100)


