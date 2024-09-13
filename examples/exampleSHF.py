import sys ; sys.path.insert(1,"../") ; from pySED import *
import sys ; sys.path.insert(1,"../../niceplot") ; from niceplot import *
import time

# YOU WILL NEED:
# 1. An MD run with positions and velocities dumped as a function of time
# (e.g. "dump dumpName groupName custom 10 fileName.qdump id type x y z vx vy vz")
# 2. a function which accepts a list of atoms' x,y,z positions ([na,3]) and returns the potential energy for the potential used. I have provided one for stillinger-weber, see potentialFunc_SW() below. if you use a different potential, you will need to provide this function.
# 3. some sane way to generate the candidate sets of atoms to consider. Below I am considering a simple solid Si-Ge interface, so i can run findNeighbors on the average positions, then I can eliminate ijk triplets where all of the atoms are on the same side of the interface. you could in theory check all permutations of atoms but this is a bit nuts and the forces will be zero for a huge majority of sets. 


positions,velocities,timesteps,types=qdump("../../../MD/projects/SiGeBase-A/NEMD.qdump") # columns include: id type x y z vx vy vz

a,b,c=5.43729,5.43729,5.43729
nx,ny,nz=24,5,5
dt=0.002*10	# 0.002 ps timesteps, dumped every 10th

avg,disp=avgPos(positions,nx*a,ny*b,nz*c) # we'll use average atomic positions for neighbor-finding

sliceX=(nx/2-1/8)*a # beware! an atom at x=0 in the local unit cell coordinate system will actually be at (and around) nx/2*a! 

# for speed, we'll only use the last 1000 timesteps (more timesteps = higher frequency resolution)
positions=positions[:1000,:,:] ; velocities=velocities[:1000,:,:] ; disp=disp[:1000,:,:]

# STEP 1: Neighbor finding. e.g. for silicon, each atom should have 4 neighbors within the cut-off distances. first atom in each list is "central" atom
print("finding neighbors") ; start=time.time()
neighbors=findNeighbors(avg,r_min=0,r_max=3.7)#334682)
print("took",time.time()-start,"s")

# STEP 2: turn these into triplets of atoms, because that's what the stillinger-weber potential considers (i,j,k, where i is a central atom (j-i-k))
# e.g. 0 has neighbors 1,2,3,4, so we want: [0,1,2],[0,1,3],[0,1,4],[0,2,1],[0,2,3],[0,2,4],[0,3,1],[0,3,2],[0,3,4]
print("calculating permutations") ; start=time.time()
permuted=[] ; from itertools import combinations
for ijk in neighbors:
	if len(ijk)<3:
		continue
	i=ijk[0] ; jk=ijk[1:] # keep first atom first, shuffle the rest
	for shuffled in combinations(jk,2):
		permuted.append([i]+list(shuffled))
print("took",time.time()-start,"s")

# STEP 3: filter these. we only care about triples for whom at least one atom is on each side
As=np.arange(len(avg)) ; As=As[avg[:,0]<sliceX]
Bs=np.arange(len(avg)) ; Bs=Bs[avg[:,0]>sliceX]
print("filtering by side") ; start=time.time()
neighbors=filterNeighborsByGroup(permuted,As,Bs)
print("took",time.time()-start,"s")

# STEP 4: calculate interatomic forces: Fi,Fj,Fk as a function of time. this creates a folder with Fx,Fy,Fz files for each triplet
print("calculating interatomic forces") ; start=time.time()
potential=potentialFunc_SW("../../../MD/projects/SiGeBase-A/SiGe.sw")
calculateInteratomicForces(avg+disp,potential,neighbors,perturbBy=.0001)
print("took",time.time()-start)

# STEP 5: Spectral heat flux
print("calculating Q")
Qw=SHF(velocities,As,Bs) # this will read in the files from the folder created via calculateInteratomicForces

ws=np.fft.fftfreq(len(Qw)) ; ws/=dt
Qw=Qw[:len(Qw)//2] ; ws=ws[:len(ws)//2]
Qw=np.absolute(Qw.T) # w,xyz --> xyz,w

x="Frequency (THz)" ; t="spectral heat flux" ; m=["-"]*3 ; l=["L (F_x*v_x)","T (F_y*v_y)","T (F_z*v_z)"]
plot([ws]*3,Qw,xlabel=x,ylabel="Q (unscaled)",title=t,markers=m,labels=l,filename="SHF.png")
plot([ws]*3,np.cumsum(Qw,axis=1),xlabel=x,ylabel="accumulated Q (unscaled)",title=t,markers=m,labels=l,filename="SHF_accumulated.png")
