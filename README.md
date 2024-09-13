This is just a basic set of python functions for running Spectral Energy Density (SED) and Spectral Heat Flux (SHF) analysis. SED gives you a phonon dispersion from Molecular Dynamics (with linewidths!) and SHF shows the vibrational frequencies carrying heat (which may be different from the vibrational frequencies present, e.g. vDOS!). 

Phonon linewidths indicate scattering rates, so if you're feeling fancy, you can fit a lorentzian to the linewidths and get scattering rates. 

extra functions are included: a dump file reader using ovito's python tools, a position averager (with wrapping for periodic boundary conditions), neighbor finding, etc

i also included an example script showing how to use the SED and SHF functions, and example outputs. example script relies on https://github.com/ExSiTE-Lab/niceplot

shoutout to https://github.com/tyst3273/phonon-sed for the excellent introduction and explanation into the math
