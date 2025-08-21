import configparser

class AbacusConfigFile(object):
	def __init__(self,filename):
	
		self.filename = filename
		self.cp = configparser.ConfigParser(strict=False)
		
		with open(self.filename, mode='r', encoding='utf-8') as f:
			file_string = '\n'.join( [i.strip() for i in f.read().splitlines()] )
			config_string = '[config]\n' + file_string
			
		self.cp.read_string(config_string)
		
		self.boxSize = self.cp.getfloat('config','boxsizehmpc')
		self.Omega_M = self.cp.getfloat('config','Omega_M')
		self.H0 = self.cp.getfloat('config','H0')
		self.redshift = self.cp.getfloat('config','Redshift')
