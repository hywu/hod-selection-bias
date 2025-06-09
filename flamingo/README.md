

# Extract halo and galaxy files from Flamingo
- get_host_halos.py
- make_gal_cat_mstar.py

(also subsampling particles)
- `lensing/subsample_particles_flamingo_step1.py`
- `lensing/subsample_particles_flamingo_step2.py`

# Select galaxies 
- color selection (TODO: clean up the code)

# clusters around halos (for Shulei's project and for benchmarking)
- calc_richness_halos.py
- plot_lensing.py

# Cluster finding
- see `pipeline_rank.py`
- first calculate the rank (calc_richness_rank.py)
- then create the rank file 
