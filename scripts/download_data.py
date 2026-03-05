import mirdata

# Initialize the loader
mdb_stem_synth = mirdata.initialize('mdb_stem_synth', data_home='data/mdb_stem_synth')

# Download the dataset
mdb_stem_synth.download()