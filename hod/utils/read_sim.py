#!/usr/bin/env python

## a wrapper for READXXX

def read_sim(para):

    if para['nbody'] == 'mini_uchuu':
        from hod.utils.read_mini_uchuu import ReadMiniUchuu
        readcat = ReadMiniUchuu(para['nbody_loc'], para['redshift'])

    if para['nbody'] == 'uchuu':
        from hod.utils.read_uchuu import ReadUchuu
        readcat = ReadUchuu(para['nbody_loc'], para['redshift'])

    if para['nbody'] == 'abacus_summit':
        from hod.utils.read_abacus_summit import ReadAbacusSummit
        readcat = ReadAbacusSummit(para['nbody_loc'], para['redshift'], cosmo_id=para['cosmo_id'])

    if para['nbody'] == 'tng_dmo':
        from hod.utils.read_tng_dmo import ReadTNGDMO
        halofinder = para.get('halofinder', 'rockstar')
        readcat = ReadTNGDMO(para['nbody_loc'], para['halofinder'], para['redshift'])
        print('halofinder', halofinder)

    if para['nbody'] == 'flamingo':
        from hod.utils.read_flamingo import ReadFlamingo
        readcat = ReadFlamingo(para['nbody_loc'], para['redshift'], subsample_loc=para['output_loc'])

    return readcat