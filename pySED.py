import numpy as np
import glob,os
from tqdm import tqdm

def qdump(filename,timescaling=1,convert=True,safemode=False): # OBSCENELY FAST IN COMPARISON TO scrapeDump()
	if os.path.exists(filename+"_ts.npy"):
		print("ignoring qdump, reading npy files instead")
		ts=np.load(filename+"_ts.npy")
		pos=np.load(filename+"_pos.npy")
		vel=np.load(filename+"_vel.npy")
		types=np.load(filename+"_typ.npy")
		return pos,vel,ts,types

	from ovito.io import import_file # TODO WEIRD BUG, AFTER THIS RUNS, WE CAN'T PLOT STUFF WITH NICEPLOT. WE GET THE FOLLOWING ERROR: ImportError: Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'qt' is currently running
	# WHY TF DOES OVITO LOAD qt AND HOW DO WE UNLOAD IT??
	print("reading qdump")
	pipeline = import_file(filename)
	nt=pipeline.source.num_frames
	data=pipeline.compute(0)
	na,nxyz=np.shape(data.particles.positions.array)
	pos=np.zeros((nt,na,3))
	vel=np.zeros((nt,na,3))
	types=data.particles.particle_type.array
	ts=np.arange(nt)*timescaling
	for n in tqdm(range(nt)):
		if safemode:
			try:
				data=pipeline.compute(n)
			except:
				print("safemode == True. failure on timestep",n)
				continue
		else:
			data=pipeline.compute(n)
		pos[n,:,:]=data.particles.position.array
		vel[n,:,:]=data.particles.velocities.array
	if convert:
		np.save(filename+"_ts.npy",ts)
		np.save(filename+"_pos.npy",pos)
		np.save(filename+"_vel.npy",vel)
		np.save(filename+"_typ.npy",types)
	return pos,vel,ts,types

def avgPos(pos,sx,sy,sz,alpha=90,beta=90,gamma=90): # takes "pos" as from: pos,timesteps=scrapeDump(trange="-"+str(avgOver)+":"), [nS,nA,xyz]
	nS,nA,na=np.shape(pos)
	displacements=np.zeros((nS,nA,na))
	for t in tqdm(range(nS)):
		# distance between initial position and position at time t (for each atom, along each axis), including wrapping
		displacements[t,:,:]=dxyz(pos[0,:,:],pos[t,:,:],sx,sy,sz,alpha,beta,gamma) 
	# average position = initial position + average of all displacements away from initial position
	# time-dependent displacements *from that average position* also requires subtracting mean(displacements)
	return pos[0,:,:]+np.mean(displacements[:,:,:],axis=0),displacements-np.mean(displacements[:,:,:],axis=0)

def getWrapAndRange(size,axis):
	wrap=np.zeros(3)
	if size is None:
		wrap[axis]=1 ; r=[0]
	else:
		wrap[axis]=size ; r=[-1,0,1]
	return wrap,r

def dxyz(pos1,pos2,sx,sy,sz,alpha=90,beta=90,gamma=90): # given two snapshots of positions [[xa,ya,za],[xb,yb,zb],...] x2, don't just do dxyz=xyz1-xyz2. must include wrapping!
	dxyz_0=pos2-pos1
	# we use these for wrapping
	wx,i_range=getWrapAndRange(sx,0) ; wy,j_range=getWrapAndRange(sy,1) ; wz,k_range=getWrapAndRange(sz,2)
	
	# for pos_1 in the 27 surrounding positions (original, and 26 neighbors), keep only the smallest (absolute) distance found
	for i in i_range:
		for j in j_range:
			for k in k_range:

				# e.g. [.1,.2,-.1] + 1*[25,0,0]+0*[0,10,0]+0*[0,0,10] # to wrap +x for a hypothetical 25x10x10 simulation volume
				shift_xyz=i*wx+j*wy+k*wz
				if gamma!=90: # TODO WE SHOULD BE ABLE TO HANDLE NON-90 ALPHA AND BETA TOO
					skew=np.eye(3) ; skew[0,1]=-np.sin(gamma*np.pi/180-np.pi/2) ; skew[1,1]=np.cos(gamma*np.pi/180-np.pi/2)
					shift_xyz=np.matmul(skew,shift_xyz)
				#print(i,j,k,shift_xyz)

				dxyz_w=pos2+shift_xyz-pos1 # if an atom crossed the x axis (from +x to -x ie L) it'll register as closer if we take (x0+L)-xf
				dxyz_0=absMin(dxyz_0,dxyz_w)
	return dxyz_0

def absMin(dxyz_a,dxyz_b): # use this for getting absolute distances with wrapping. Nx3 [[1dx,1dy,1dz],[2dx,2dy,2dz],...] vs [[1dx,1dy,1dz],[2dx,2dy,2dz],...] with different wrapping, we want to keep the closest (not max, not min)
	abs1=np.absolute(dxyz_a) # absolute distances (still in x,y,z separately) for first comparison
	abs2=np.absolute(dxyz_b) # and second comparison (Wrapped)
	minabs=np.minimum(abs1,abs2) # lowest distances, between two comparisons (wrapped and unwrapped)
	keeps=np.zeros((len(dxyz_a),3)) # next, we'll "select" distances from the approriate dxyz*, including sign, by comparing minabs (lowests) vs each comparison's
	keeps[abs1==minabs]=dxyz_a[abs1==minabs]
	keeps[abs2==minabs]=dxyz_b[abs2==minabs]
	return keeps

# Spectral Energy Density: phonon dispersions!
# avg - average positions [a,xyz] (import using scrapeDump or qdump. average via avgPos)
# velocities - time-dependent atom velocities [t,a,xyz] (as imported via scrapeDump or qdump)
# p_xyz - 0,1,2 indicating if we should track positions in x,y or z (this is your wave-vector direction)
# v_xyz - like v_xyz, but for which velocities to track. L vs T modes
# a - this is your specified periodicity (or lattice constant for crystals)
# nk - resolution in k-space. note your resolution in ω is inherited from ts
# bs - optional: should be a list of atom indices to include. this allows the caller to sum over crystal cell coordinates (see discussion on Σb below)
# TODO: currently k_max=π/a. this is convention. so if you want your x axis to be wavelength⁻¹, you need to divide by π? should we do this for you? idk
# TODO: ditto for ω, which is rad/timestep. you need to scale it accordingly (timesteps to time units) and include 2π to get to Hz vs rad/s
def SED(avg,velocities,p_xyz,v_xyz,a,nk=100,bs='',perAtom=False,ks='',keepComplex=False):
	nt,na,nax=np.shape(velocities)
	if len(bs)==0:
		bs=np.arange(na)
	else:
		na=len(bs)
	nt2=int(nt/2) #; nt2=nt # this is used to trim off negative frequencies
	if len(ks)==0:
		ks=np.linspace(0,np.pi/a,nk)
	else:
		nk=len(ks)

	ws=np.fft.fftfreq(nt)[:nt2] ; Zs=np.zeros((nt2,nk))
	if keepComplex:
		Zs=np.zeros((nt2,nk),dtype=complex)

	# Φ(k,ω) = Σb | ∫ Σn u°(n,b,t) exp( i k r̄(n,0) - i ω t ) dt |² # https://github.com/tyst3273/phonon-sed/blob/master/manual.pdf
	# b is index *within* unit cell, n is index *of* unit cell. pairs of n,b
	# can be thought of as indices pointing to a specific atom.
	# u° is the velocity of each atom as a function of time. r̄(n,b=0) is the
	# equilibrium position of the unit cell. (if we assume atom 0 in a unit
	# cell is at the origin, we can use x̄(n,b=0)). looping over n inside the 
	# integral, but not b, means we are effectively picking up "every other 
	# atom", which means short-wavelength optical modes will register as 
	# their BZ-folded longer-wavelength selves*. using r̄(n,b=0) rather than
	# x̄(n,b) means a phase-shift is applied for b≠0 atoms. this means if we
	# ignore b,n ("...Σb | ∫ Σn...") and sum across all atoms instead
	# ("... | ∫ Σi..."), we lose BZ folding. if we don't care (fine, so long
	# as a is small enough / k is large is small enough to "unfold" the 
	# dispersion), we simplify summing, and can use x̄ instead. we thus no
	# longer require a perfect crystal to be analyzed. the above equation
	# also simplified to:
	# Φ(k,ω) = | ∫ Σn u°(n,t) exp( i k x̄(n) - i ω t ) dt |²
	# and noting the property: e^(A+B)=e^(A)*e^(B)
	# Φ(k,ω) = | ∫ Σn u°(n,t) exp( i k x̄(n) ) * exp( - i ω t ) dt |²
	# and noting that the definition of a fourier transform:
	# F(w) = ∫ f(t) * exp( -i 2 π ω t ) dt
	# we can reformulate the above eqaution as:
	# f(t) = u°(n,t) * exp( i k x )
	# Φ(k,ω) = | FFT{ Σn f(t) } |²
	# of course, this code still *can* analyze crystals with folding: the 
	# user should simply call this function multiple times, passing the "bs"
	# argument with a list of atom indices for us to use

	if isinstance(p_xyz,(int,float)): # 0,1,2 --> x,y,z
		xs=avg[bs,p_xyz] # a,xyz --> a
	else:	# [1,0,0],[1,1,0],[1,1,1] and so on
		# https://math.stackexchange.com/questions/1679701/components-of-velocity-in-the-direction-of-a-vector-i-3j2k
		# project a vector A [i,j,k] on vector B [I,J,K], simply do: A•B/|B| (dot, mag)
		# for at in range(na): x=np.dot(avg[at,:],p_xyz)/np.linalg.norm(p_xyz)
		# OR, use np.einsum. dots=np.einsum('ij, ij->i',listOfVecsA,listOfVecsB)
		p_xyz=np.asarray(p_xyz)
		d=p_xyz[None,:]*np.ones((na,3)) # xyz --> a,xyz
		xs=np.einsum('ij, ij->i',avg[bs,:],d) # pos • vec, all at once
		xs/=np.linalg.norm(p_xyz)

	# TODO mongo ram usage for rotation step vs simply using a reference to the existing matrix, if velocities is huge, e.g. simulations with a 100000 atoms, e.g. 50x50x5 UC silicon (8 atoms per UC), as required for 110 SED
	if isinstance(v_xyz,(int,float)):
		vs=velocities[:,bs,v_xyz] # t,a,xyz --> t,a
	else: 
		# for handling velocities, there's just one more step from above: "flattening" first two axes t,a,xyz --> t*a,x,y,z
		vflat=np.reshape(velocities[:,bs,:],(nt*na,3)) # t,a,xyz --> t*a,xyz
		v_xyz=np.asarray(v_xyz)
		d=v_xyz[None,:]*np.ones((nt*na,3))
		vs=np.einsum('ij, ij->i',vflat,d)
		vs=np.reshape(vs,(nt,na)) # and unflattening at the end: t*a,xyz --> t,a,xyz
		vs/=np.linalg.norm(v_xyz)

	if perAtom:
		Zs=np.zeros((nt2,nk,na),dtype=complex)
		for j,k in enumerate(tqdm(ks)):	
			# f(t) = u°(a,t) * exp( i k x̄ )
			F=vs[:,:]*np.exp(1j*k*xs[None,:]) # t,a
			# Σn u°(a,t) * exp( i k x̄ )
			# F=np.sum(F,axis=1)/na # t,a --> t
			# ∫ { Σn u°(a,t) exp( i k x) } * exp( - i ω t ) dt AKA FFT{ Σn u°(a,t) exp( i k x) }
			integrated=np.fft.fft(F,axis=0)[:nt2,:] # t,a --> ω,t. trim off negative ω
			# | ∫ Σn u°(a,t) exp( i k x) * exp( - i ω t ) dt |²
			Zs[:,j,:]+=integrated # np.absolute(integrated)**2
		return Zs,ks,ws

	for j,k in enumerate(tqdm(ks)):
		# f(t) = u°(a,t) * exp( i k x̄ )
		F=vs[:,:]*np.exp(1j*k*xs[None,:]) # t,a
		# Σn u°(a,t) * exp( i k x̄ )
		F=np.sum(F,axis=1)/na # t,a --> t
		# ∫ { Σn u°(a,t) exp( i k x) } * exp( - i ω t ) dt AKA FFT{ Σn u°(a,t) exp( i k x) }
		integrated=np.fft.fft(F)[:nt2] # t --> ω. trim off negative ω
		# | ∫ Σn u°(a,t) exp( i k x) * exp( - i ω t ) dt |²
		if keepComplex:
			Zs[:,j]+=integrated
		else:
			Zs[:,j]+=np.absolute(integrated)**2
	return Zs,ks,ws

# save positions (na,xyz) to a positions file. useful for visualizing avg and stuff
def outToPositionsFile(filename,pos,types,sx,sy,sz,masses):
	import datetime
	now=datetime.datetime.now()
	lines=["########### lammpsScapers.py > outToPositionsFile() "+now.strftime("%Y/%m/%d %H:%M:%S")+" ###########"]
	lines=lines+[str(len(pos))+" atoms","",str(len(masses))+" atom types",""]
	for s,xyz in zip([sx,sy,sz],["x","y","z"]):
		lines.append("0.0 "+str(s)+" "+xyz+"lo "+xyz+"hi")
	lines=lines+["","Masses",""]+[ str(i+1)+" "+str(m) for i,m in enumerate(masses) ]+["","Atoms",""]
	for t,xyz in zip(types,pos):
		t=int(t) ; atxyz=[ str(v) for v in [1,t,*xyz] ]
		lines.append(" ".join(atxyz))
	with open(filename,'w') as f:
		for l in lines:
			f.write(l+"\n")