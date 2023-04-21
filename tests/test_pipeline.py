#!/usr/bin/env python

import unittest
import os

os.chdir('../repo/pipeline')

yml_fname = '../yml/mini_uchuu/mini_uchuu_fid_hod_part.yml'

class TestPipeline(unittest.TestCase):
    def test_pipeline_step_by_step(self):
        cmd = f'./make_gal_cat.py {yml_fname}'
        os.system(cmd)

        cmd = f'./calc_richness.py {yml_fname}'
        os.system(cmd)

        cmd = f'./plot_lensing.py {yml_fname}'
        os.system(cmd)

        cmd = f'./plot_counts_richness.py {yml_fname}'
        os.system(cmd)

    def test_pipeline(self):
        cmd = f'./pipeline.py {yml_fname}'
        os.system(cmd)

if __name__ == '__main__':
    unittest.main()