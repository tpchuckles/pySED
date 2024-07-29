from pySED import *
import sys ; sys.path.insert(1,"../niceplot") ; from nicecontour import *

# SED arguments include: atomic positions (indices: atom,xyz), velocities (indices: timestep,atom,xyz), wave direction, velocity direction (gives longitudinal vs transverse). optionally, you can specify k-space sampling. bs is used to select specific atoms to include (exclude all others). 


positions,velocities,timesteps,types=qdump("../../MD/projects/AlN_SED_long/NVE.qdump")

lx=255.1144 ; ly=11.04677756356124 ; lz=10.384714 # my simulation was already orthogonalized (I think avgPos only handles gamma skewed cells?)
nx,ny,nz=80,4,2
a=lx/nx ; b=ly/ny ; c=lz/nz

avg,disp=avgPos(positions,nx*a,ny*b,nz*c) # we'll use average atomic positions for SED (can also use lattice-defined points)

lattice=np.loadtxt("../../MD/projects/AlN_SED_long/lattice.dat") # in my case, columns are: atom ID, unit cell ID x, y, z, atom ID within unit cell

p_direc=0 # simulation is a long bar in the x direction, so 0 index for x. can also pass vectors, e.g. [1,0,0] for x direction. simulation size is inverse kspace resolution! (long simulations required for good k-space)
latconst=a/2 ; nk=160 # this defines k-spacing, and max k (BZ edge). too-high k-space resolution will just give you junk! (more modes may not exist in your tiny simulation!)

v_direcs=[0,1,2] # want all 3 branches, L (kx,vx), T (kx,vy), T (kx,vz)

Zs=[]
for b in range(4): # AlN has a 4 atom basis. two Al atoms, two N atoms
	bs=lattice[lattice[:,4]==b,0].astype(int) # selects column 0, based on value in column 4 (atom ID within unit cell). yours may vary! 
	for v in v_direcs:
		Z,ks,ws=SED(avg,velocities,p_direc,v,latconst,nk=nk,bs=bs) # if a list of indices for bs is passed, we only select those atoms! 
		Zs.append(Z)

ks/=np.pi # convert to 1/wavelength
ws/=(0.0005*30) # convert to THz: .002 picosecond timesteps, every 10th timestep logged

# sum over branches, square it for visualization purposes
contour(np.sqrt(np.sum(Zs,axis=0)),ks,ws,xlabel="inverse wavelength ($\AA$^-1)",ylabel="frequency (THz)",title="AlN - 4 atom basis",filename="SED_4atombasis.png")


# what's the practical difference between 4-atom-basis and monatomic? 
# 1. for certain optical modes (atoms vibrate out-of-phase with their neighbors), they sum to zero and the branch disappears if you use the monatomic basis. if you want to see ALL phonon branches, you must analyze each atom within the unit cell separately!) 
# 2. monatomic basis "unwraps" the folded brillioun zone! (a long wavelength wave (low k) affecting every-other-atom (bs argument specify those atom IDs) may exactly mathematically satisfy the same motion as a short wavelength wave (high k) affecting every atom. this is where optic branches come from!) 
Zs=[]
for v in v_direcs:
	Z,ks,ws=SED(avg,velocities,p_direc,v,latconst,nk=nk,bs=bs) # if a list of indices for bs is NOT passed, all atoms are selected. 
	Zs.append(Z)

ks/=np.pi # convert to 1/wavelength
ws/=(0.0005*30) # convert to THz: .002 picosecond timesteps, every 10th timestep logged

contour(np.sqrt(np.sum(Zs,axis=0)),ks,ws,xlabel="inverse wavelength ($\AA$^-1)",ylabel="frequency (THz)",title="AlN - monatomic basis",filename="SED_monatomic.png")

