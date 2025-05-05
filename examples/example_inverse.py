import sys
sys.path.insert(1,"../../../Various Code/pySED")
from pySED import *
sys.path.insert(1,"../../../Various Code/niceplot")
from nicecontour import *

a,b,c=5.43729,5.43729,5.43729
nx,ny,nz=120,5,5

positions,velocities,timesteps,types=qdump("NVE.qdump") # columns include: id type x y z vx vy vz
if os.path.exists("avg.npy"):
	avg=np.load("avg.npy") ; disp=np.load("disp.npy")
else:
	avg,disp=avgPos(positions,nx*a,ny*b,nz*c) # we'll use average atomic positions for SED (can also use lattice-defined points)
	np.save("avg.npy",avg) ; np.save("disp.npy",disp)

#avg,types=scrapePos("positions.pos")

indices=np.arange(len(avg))
mlID=np.round(avg[:,0]/(a/4)).astype(int)
ml0=indices[ mlID%2==0 ]
ml1=indices[ mlID%2==1 ]
bs=[ ml0 , ml1 ]

#k=.6 ; w=.0880
#k=.3486 ; w=.0880
#k=1.05 ; w=.2405
#k=1.51; w=.3086
k=.874 ; w=.299

iSED(avg,disp,[1,0,0],k=k,w=w,a=a/4,nk=nx*4,bs=bs,ks='',rescaling="auto",types=types)

Zs=np.sum([ np.absolute( np.load( "dSED_b0_p100_v"+str(i)+".npy" )) for i in range(3) ],axis=0)
ks=np.load("dSED_ks_p100.npy") ; ws=np.load("dSED_ws_p100.npy")
Zs[:,:]*=ws[:,None]
contour(Zs**.25,ks,ws,heatOrContour="pix",overplot=[{"xs":[k],"ys":[w],"kind":"scatter"}])