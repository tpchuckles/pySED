import sys ; sys.path.insert(1,"../") ; from pySED import *
import sys ; sys.path.insert(1,"../../niceplot") ; from nicecontour import *

# SED arguments include: atomic positions (indices: atom,xyz), velocities (indices: timestep,atom,xyz), wave direction, velocity direction (gives longitudinal vs transverse). optionally, you can specify k-space sampling. bs is used to select specific atoms to include (exclude all others). 

positions,velocities,timesteps,types=qdump("../../../MD/projects/Si_SED_04/NVE.qdump") # columns include: id type x y z vx vy vz

a,b,c=5.43729,5.43729,5.43729
nx,ny,nz=120,5,5

avg,disp=avgPos(positions,nx*a,ny*b,nz*c) # we'll use average atomic positions for SED (can also use lattice-defined points)

atomIDs=np.arange(len(avg)) # better to import a lattice file (ties atoms by ID (0 through N) to atom ID within the unit cell (2 atoms for Si)

p_direc=0 # simulation is a long bar in the x direction, so 0 index for x. can also pass vectors, e.g. [1,0,0] for x direction. simulation size is inverse kspace resolution! (long simulations required for good k-space)
latconst=a/2 ; nk=120 # this defines k-spacing, and max k (BZ edge). too-high k-space resolution will just give you junk! (more modes may not exist in your tiny simulation!)

v_direcs=[0,1,2] # want all 3 branches, L (kx,vx), T (kx,vy), T (kx,vz)

Zs=[] ; 
for b in range(2): # AlN has a 4 atom basis. two Al atoms, two N atoms
	# primitive cell has 2 atoms, conventional cell has 8. mine are defined as first 4 being 1st primitive position, next 4 are 2nd primitive position
	bs=sum([ list(atomIDs[b*8+i::8]) for i in range(4)],[]) # if a list of indices for bs is passed, we only select those atoms! 
	for v in v_direcs:
		Z,ks,ws=SED(avg,velocities,p_direc,v,latconst,nk=nk,bs=bs) 
		Zs.append(Z)

ks/=np.pi # convert to 1/wavelength
ws/=(0.002*10) # convert to THz: .002 picosecond timesteps, every 10th timestep logged

# sum over branches, square it for visualization purposes
contour(np.sqrt(np.sum(Zs,axis=0)),ks,ws,xlabel="inverse wavelength ($\AA$^-1)",ylabel="frequency (THz)",title="Si - 2 atom basis",filename="Si_2atombasis.png")


# what's the practical difference between 2-atom-basis and monatomic? 
# 1. for certain optical modes (atoms vibrate out-of-phase with their neighbors), they sum to zero and the branch disappears if you use the monatomic basis. if you want to see ALL phonon branches, you must analyze each atom within the unit cell separately!) 
# 2. monatomic basis "unwraps" the folded brillioun zone! (a long wavelength wave (low k) affecting every-other-atom (bs argument specify those atom IDs) may exactly mathematically satisfy the same motion as a short wavelength wave (high k) affecting every atom. this is where optic branches come from!) 
Zs=[]
for v in v_direcs:						# if a list of indices for bs is NOT passed, all atoms are selected. 
	Z,ks,ws=SED(avg,velocities,p_direc,v,latconst/2,nk=nk)	# using a/4 instead of a/2, to get folded BZ
	Zs.append(Z)

ks/=np.pi # convert to 1/wavelength
ws/=(0.0005*30) # convert to THz: .002 picosecond timesteps, every 10th timestep logged

contour(np.sqrt(np.sum(Zs,axis=0)),ks,ws,xlabel="inverse wavelength ($\AA$^-1)",ylabel="frequency (THz)",title="Si - monatomic basis",filename="Si_monatomic.png")

