import sys ; sys.path.insert(1,"../") ; from pySED import *
import sys ; sys.path.insert(1,"../../niceplot") ; from nicecontour import *
import os

# SED arguments include: atomic positions (indices: atom,xyz), velocities (indices: timestep,atom,xyz), wave direction, velocity direction (gives longitudinal vs transverse). optionally, you can specify k-space sampling. bs is used to select specific atoms to include (exclude all others). 

# Low resolution (smaller simulation volume = fewer k points)
path="../../../MD/projects/Si_SED_09/NVE.qdump"
a,b,c=5.43729,5.43729,5.43729
nx,ny,nz=25,25,5

# High res (bigger volume = more k points, but also an absolutely enormous qdump file! pro-tip: add "convert=False" argument to qdump() to prevent it from saving off npy file copies of the qdump file, if disk space is limited. also uncomment "positions=0" below if the avg file exists already, if ram is limited)
#path="../../../MD/projects/Si_SED_08/NVE.qdump"
#a,b,c=5.43729,5.43729,5.43729
#nx,ny,nz=50,50,5

# don't bother reading in positions/averagepositions/velocities/etc if SED has been run for all already! (we're just importing and plotting)
if not ( os.path.exists("GXKG/GX_Z.npy") and os.path.exists("GXKG/GK_Z.npy") and os.path.exists("GXKG/XK_Z.npy") ):
	positions,velocities,timesteps,types=qdump(path) # columns include: id type x y z vx vy vz
	# positions=0 ; timesteps=0 ; types=0

	avgFile="/".join(path.split("/")[:-1]+["avg.npy"])
	if os.path.exists(avgFile):
		avg=np.load(avgFile)
	else:
		avg,disp=avgPos(positions,nx*a,ny*b,nz*c) # we'll use average atomic positions for SED (can also use lattice-defined points)
		np.save(avgFile,avg)

	atomIDs=np.arange(len(avg)) # better to import a lattice file (ties atoms by ID (0 through N) to atom ID within the unit cell (2 atoms for Si)
	bs=[]
	for n in range(2): # primitive cell has 2 atoms, conventional cell has 8. mine are defined as first 4 being 1st primitive position, next 4 are 2nd primitive position
		b=[ list(atomIDs[n*8+i::8]) for i in range(4) ] # [ [0,8,16...],[1,9,17...],[2,10,18...],[3,11,19...] ]
		b=sum(b,[]) 	# "flatten" the above list, which gives us (unsorted) [0,1,2,3,8,9,10,11,16,17,18,19...] (chunks of 4)
		bs.append(b)

ZS=[] ; KS=[] ; WS=[]

os.makedirs("GXKG",exist_ok=True)

# build (or read in) Γ --> X
if os.path.exists("GXKG/GX_Z.npy"):
	Zs=np.load("GXKG/GX_Z.npy") ; ks=np.load("GXKG/GX_k.npy") ; ws=np.load("GXKG/GX_w.npy")
else:
	p_direc=0 ; latconst=a/2		# wave direction is x
	nk=50 ; Zs=[]
	for v in [0,1,2]: 			# want all 3 branches, L (kx,vx), T (kx,vy), T (kx,vz)
		for b in bs:
			Z,ks,ws=SED(avg,velocities,p_direc,v,latconst,nk=nx*2,bs=b) # Z has ω,k indices
			Zs.append(Z) # appended --> branch,ω,k indices
	Zs=np.sum(Zs,axis=0) # branch,ω,k indices
	np.save("GXKG/GX_Z.npy",Zs) ; np.save("GXKG/GX_k.npy",ks) ; np.save("GXKG/GX_w.npy",ws)
print(np.shape(Zs))
ks/=np.pi # convert to 1/wavelength
ws/=(0.002*10) # convert to THz: .002 picosecond timesteps, every 10th timestep logged
ZS.append(Zs) ; KS.append(ks) ; WS.append(ws)

# build (or read in) Γ --> K
if os.path.exists("GXKG/GK_Z.npy"):
	Zs=np.load("GXKG/GK_Z.npy") ; ks=np.load("GXKG/GK_k.npy") ; ws=np.load("GXKG/GK_w.npy")
else:
	p_direc=[1,1,0] ; latconst=a/2*np.sqrt(2)	# wave direction is diagonally in x,y [1,1,0]
	nk=50 ; Zs=[]
	for v in [[1,1,0],[-1,1,0],[0,0,1]]:	# 3 branches: L [1,1,0], and two perpendicular directions [-1,1,0] (other diagonal) and [0,0,1] (up)
		for b in bs:
			Z,ks,ws=SED(avg,velocities,p_direc,v,latconst,nk=nx*2,bs=b) # Z has ω,k indices
			Zs.append(Z)	# appended --> branch,ω,k indices
	Zs=np.sum(Zs,axis=0) # branch,ω,k indices
	np.save("GXKG/GK_Z.npy",Zs) ; np.save("GXKG/GK_k.npy",ks) ; np.save("GXKG/GK_w.npy",ws)
ks/=np.pi # convert to 1/wavelength
ws/=(0.002*10) # convert to THz: .002 picosecond timesteps, every 10th timestep logged
ZS.append(Zs) ; KS.append(ks) ; WS.append(ws)


# build (or read in) X --> K: more tricky: we need to sweep through angles, and run for just one k point at the BZ edge at each angle! 
if os.path.exists("GXKG/XK_Z.npy"):
	Zs=np.load("GXKG/XK_Z.npy") ; angles=np.load("GXKG/XK_k.npy") ; ws=np.load("GXKG/XK_w.npy")
else:
	# what IS the X-K dispersion showing? it's marching along the BZ edge from X to K, i.e. BZ edge across all angles between 0 and 45°
	angles=np.linspace(0,np.pi/4,nx*2) ; Zs=[]
	for n,theta in enumerate(angles):
		print("THETA",n,theta)
		cos=np.cos(theta) ; sin=np.sin(theta)
		p_direc=[cos,sin,0] ; latconst=a/2/cos ; Zs.append([])		# vector at the given angle 
		for v in [[cos,sin,0],[-sin,cos,0],[0,0,1]]:			# L is parallel, T1 is perp. (swap sin/cos), T2 is z
			print("p",p_direc,"v",v)
			for b in bs:
				Z,ks,ws=SED(avg,velocities,p_direc,v,latconst,nk=2,bs=b)	# only technically need 1 k point at BZ edge
				Zs[-1].append(Z[:,-1]) # every ω, last k (BZ edge)
		Zs[-1]=np.sum(Zs[-1],axis=0) # branch,ω indices
	Zs=np.asarray(Zs).T # a,ω indices --> ω,a
	np.save("GXKG/XK_Z.npy",Zs) ; np.save("GXKG/XK_k.npy",angles) ; np.save("GXKG/XK_w.npy",ws)
ws/=(0.002*10)

ks=np.linspace(0,2/a,len(angles))
ZS.append(Zs) ; KS.append(ks) ; WS.append(ws)
#contour(Zs,angles,ws)

# Assemble into a dispersion Γ X K Γ
Zs=[] ; ks=[]
for i in range(len(KS[0])):						# start with Γ X (first one we calculated)
	Zs.append(ZS[0][:,i]) ; ks.append(KS[0][i])
for i in range(1,len(KS[2])):						# we calculated X K last, but *add* k to last k from Γ X
	Zs.append(ZS[2][:,i]) ; ks.append(KS[0][-1]+KS[2][i])
for i in reversed(range(len(KS[1]))):					# we calculated Γ K, so need to reverse the order for K Γ
	Zs.append(ZS[1][:,i]) ; ks.append(KS[0][-1]+KS[2][-1]+KS[1][i])

print([ len(z) for z in Zs ])
print(np.shape(Zs),np.shape(ks),np.shape(ws))

Zs=np.asarray(Zs).T # k,ω indices --> ω,k

k_BZ=0 ,KS[0][-1],KS[0][-1]+KS[2][-1],KS[0][-1]+KS[2][-1]+KS[1][-1]
def axisTicks(plt):
	axs=plt.gca()
	print( axs.get_xticklabels() )
	lbls=["Γ","X"      ,"K"                ,"Γ"                           ]
	axs.set_xticks(k_BZ,lbls)
overplot=[ {"xs":[k,k],"ys":[0,ws[-1]],"kind":"line","linestyle":":","c":"r"} for k in k_BZ ]

#for Z,k,w in zip(ZS,KS,WS):
#	contour(np.sqrt(Z),k,w,heatOrContour="pix")

contour(np.sqrt(Zs),ks,ws,heatOrContour="pix",extras=[axisTicks],overplot=overplot,xlabel="",ylabel="Frequency (THz)",title="Silicon phonon dispersion - Γ X K Γ",zlabel="",filename="GXKG/GXKG.png",ylim=[0,20])

#p_direc=0 # simulation is a long bar in the x direction, so 0 index for x. can also pass vectors, e.g. [1,0,0] for x direction. simulation size is inverse kspace resolution! (long simulations required for good k-space)
 # this defines k-spacing, and max k (BZ edge). too-high k-space resolution will just give you junk! (more modes may not exist in your tiny simulation!)

#v_direcs=[0,1,2] 
