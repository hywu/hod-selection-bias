import fitsio
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('MNRAS')

fname = 'dr8_run_redmapper_v5.10_lgt5_catalog.fit'
data, header = fitsio.read(fname, header=True)
#print(header)
z_BCG = data['BCG_SPEC_Z'] #added
z_lambda = data['Z_LAMBDA']
Lambda = data['LAMBDA_ZRED'] #added
sel = (z_BCG != -1) & (0.08 <= z_lambda) & (z_lambda <= 0.12) & (Lambda >= 5)
ID_cl = data['MEM_MATCH_ID'][sel] #added
z_BCG = z_BCG[sel]
z_lambda = z_lambda[sel]
Lambda = Lambda[sel]

fname = 'dr8_run_redmapper_v5.10_lgt5_catalog_members_mod.fit'
data, header = fitsio.read(fname, header=True)
print(header)
z_mem = data['ZSPEC']
sel = (z_mem != -1)
z_mem = z_mem[sel]
ID_mem = data['MEM_MATCH_ID'][sel]
mag_i = data['IMAG'][sel]
CHISQ_MOD = data['CHISQ_MOD'][sel]

#applying the Lstar filter
exp = (mag_i - 22.44-3.36*np.log10(z_mem)-0.273*np.log10(z_mem)**2+0.0618*np.log10(z_mem)**3+0.0227*np.log10(z_mem)**4) / -2.5
lstar_frac = 10**exp
sel = (lstar_frac >= 0.55)
z_mem = z_mem[sel]
ID_mem = ID_mem[sel]
CHISQ_MOD = CHISQ_MOD[sel]

#filtering out galaxies that don't belong to clusters with z_BCG
sel = np.isin(ID_mem, ID_cl)
ID_mem = ID_mem[sel]
z_mem = z_mem[sel]
CHISQ_MOD = CHISQ_MOD[sel]

delz = np.array([])
delzov1pz = np.array([])
RovRLambda = np.array([])
CHISQ = np.array([])

for i in range(len(ID_cl)):
    sel = (ID_mem == ID_cl[i])
    delz = np.concatenate((delz, z_mem[sel] - z_BCG[i]))
    delzov1pz = np.concatenate((delzov1pz, (z_mem[sel] - z_BCG[i]) / (1 + z_BCG[i]) ))  
    CHISQ = np.concatenate((CHISQ, CHISQ_MOD[sel]))

z = 0.1
Om = 0.3089
Ez = np.sqrt(Om * (1 + z)**3 + (1 - Om))
c = 3e5

def dchi2dz(dchi):
    return dchi * Ez / 3000.

def dz2dchi(dz):
    return dz * 3000. / Ez

def plot_hexbin():
	sel = (abs(delz)<=.1) & (CHISQ > 0)

	fig, ax = plt.subplots()
	hexbin = ax.hexbin(dz2dchi(delz[sel]), CHISQ[sel], mincnt=1, yscale='log',norm=matplotlib.colors.LogNorm(), \
	                  cmap='gray_r')
	ax.grid(False)
	fig.colorbar(mappable=hexbin)
	ax.set_xlim([dz2dchi(-0.1),dz2dchi(0.1)])
	ax.set_ylim([10**(-1.5), ax.get_ylim()[1]])
	ax.set_xlabel(r'$\Delta \chi$ (L.O.S.)')
	ax.set_ylabel(r'$\chi^2_{\rm colour}$')
	ax.tick_params(axis='x', which='both', top=False, bottom=True)
	secax = ax.secondary_xaxis('top', functions=(dchi2dz, dz2dchi))
	secax.set_xlabel(r'$\Delta z$')
	fig.savefig('hexbin.png')

if __name__=='__main__':
	plot_hexbin()