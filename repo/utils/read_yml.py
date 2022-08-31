#!/usr/bin/env python
import yaml

class ReadYML(object):
    def __init__(self, yml_fname):
        with open(yml_fname, 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
                #print(parsed_yaml)
            except yaml.YAMLError as exc:
                print(exc)
        self.model_set = parsed_yaml['model_set']
        self.output_loc = parsed_yaml['output_loc']
        self.which_sim = parsed_yaml['which_sim']
        self.redshift = parsed_yaml['redshift']
