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
	# WHY DOES OVITO LOAD qt AND HOW DO WE UNLOAD IT??
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
# p_xyz - 0,1,2 indicating if we'll track positions in x,y or z (this is your wave-vector direction). vector also allowed: e.g. [1,1,0] for waves in 110
# v_xyz - like p_xyz, but for which velocities to track (L vs T modes). e.g. p_xyz=[1,1,0] v_xyz=[-1,1,0] are transverse modes in 110
# a - this is your specified periodicity (or lattice constant for crystals). 1/a --> highest k value
# nk - resolution in k-space. note your resolution in ω is inherited from your time sampling
# bs - optional: should be a list of atom indices to include. this allows the caller to sum over crystal cell coordinates (see discussion on Σb below)
# TODO: currently k_max=π/a. this is convention. so if you want your x axis to be wavelength⁻¹, you need to divide by π? should we do this for you? idk
# TODO: ditto for ω, which is rad/timestep. you need to scale it accordingly (timesteps to time units) and include 2π to get to Hz vs rad/s
def SED(avg,velocities,p_xyz,v_xyz,a,nk=100,bs='',perAtom=False,ks='',keepComplex=False,masses=None,hannFilter=True):
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
		vs=np.reshape(vs,(nt,na)) # and unflattening at the end: t*a --> t,a
		vs/=np.linalg.norm(v_xyz)

	#if vacf:
	#	for a in range(na):
	#		v=np.correlate(vs[:,a],vs[:,a],mode="full") # https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
	#		vs[:,a]=v[:nt]-np.mean(v[:nt])
	
	if masses is not None:
		vs=vs[:,:]*masses[None,:] # t,a indices for velocities

	if hannFilter:
		ts=np.linspace(0,np.pi,len(vs)) ; hann=np.sin(ts)**2
		#import sys; sys.path.insert(1,"../../niceplot") ; from niceplot import plot ; plot([ts],[hann],markers=['-'])
		vs*=hann[:,None]

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
		#F=np.correlate(F,F,mode="full") ; F=F[:nt] # Harrison says this gives less noise (for fluids)
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

# returns a function which can be passed a triplet of three atoms' positions, [ijk,xyz], and return the potential energy
def potentialFunc_SW(swfile):
	r_cut,potential2Body,potential3Body=SW(swfile)
	print("r_cut",r_cut)
	def potential(pos):
		vij=pos[1,:]-pos[0,:]
		vik=pos[2,:]-pos[0,:]
		rij=lVec(vij) ; rik=lVec(vik)
		if rij>r_cut or rik>r_cut:
			return 0
		tijk=angBetween(vij,vik)
		return potential3Body(rij,rik,tijk)+potential2Body(rij)
	return potential

#E=ΣᵢΣⱼϕ₂(rᵢⱼ)+ΣᵢΣⱼΣₖϕ₃(rᵢⱼ,rᵢₖ,θᵢⱼₖ)
#ϕ₂ is 2-body component, ϕ₂(rᵢⱼ)=A ϵ *[ B*(σ/r)^p-(σ/r)^q ] * exp( σ / r-a*σ )
#ϕ₃ is 3-body component, ϕ₃(rᵢⱼ,rᵢₖ,θᵢⱼₖ)=λ ϵ (cos(θᵢⱼₖ)-cos₀)² exp( γᵢⱼσᵢⱼ / r-aᵢⱼ*σᵢⱼ ) exp( γᵢₖσᵢₖ / r-aᵢₖ*σᵢₖ )
#calculate potential energy E for each atom in A as a result of its interaction with each atom in B.
def SW(swfile):
	sw=readSW(swfile)
	e=sw["e"];s=sw["s"];a=sw["a"];l=sw["l"];g=sw["g"];c=sw["c"];A=sw["A"];B=sw["B"];p=sw["p"];q=sw["q"];t=sw["t"]
	r_cut=a*s*.99 #BEWARE: check this if you change sw potentials! use sw2LJ()
	def potential2Body(rij):
		return A*e*(B*(s/rij)**p-(s/rij)**q)*np.exp(s/(rij-a*s))
	def potential3Body(rij,rik,tijk):
		return l*e*(np.cos(tijk)-c)**2*np.exp(g*s/(rij-a*s))*np.exp(g*s/(rik-a*s))
	return r_cut,potential2Body,potential3Body

def readSW(swfile):
	f=open(swfile,'r',errors='ignore') #some versions of python choke on readlines() if there are unicode characters in the file being read (eg, umlauts because some germans developed your SW potential)
	entries={}
	lines=f.readlines()
	for l in lines:
		if len(l)<1 or l[0]=="#" or len(l.split())<14:
			continue
		#print(l)			#                  0       1     2 3      4     5         6 7 8 9 10
		l=list(map(float,l.split()[3:]))	#elem1,elem2,elem3,epsilon,sigma,a,lambda,gamma,costheta0,A,B,p,q,tol
		names="e,s,a,l,g,c,A,B,p,q,t"
		for n,v in zip(names.split(','),l):
			entries[n]=float(v)
		return entries	

# length of a vector
def lVec(v1):
	return np.sqrt( np.sum( v1**2 ) )

# angle between two vectors
def angBetween(v1,v2): # cos(θ)=(vᵢⱼ•vᵢₖ)/(|vᵢⱼ|*|vᵢₖ|)
	return np.arccos(np.dot(v1,v2)/(lVec(v1)*lVec(v2))) #cos(θ)=(vᵢⱼ•vᵢₖ)/(|vᵢⱼ|*|vᵢₖ|)

# For each atom i, find atoms j,k,l,m,n,o within the radii limits, return i,j,k,l...
# avgs - [na,xyz]
def findNeighbors(avgs,r_min=0,r_max=np.inf):
	neighbors=[]
	na=len(avgs) ; indices=np.arange(na)
	print("calculating neighbor distances")
	d0=np.sqrt( np.sum( (avgs[:,None,:]-avgs[None,:,:])**2 , axis=2) ) # √(dx²+dy²+dz²), yields an na x na "lookup" matrix of distances
	print("iterating")
	for i in range(na):
		mask=np.zeros(na)+1 ; mask[d0[i,:]<r_min]=0 ; mask[d0[i,:]>r_max]=0 ; mask[i]=0
		neighbors.append([i]+list(indices[mask==1])) # ensure atom i is first in the list!
	return neighbors

# only keep neighbors when at least one is in both groups
# neighbors - a potentially-ragged list of lists, each list containing atom IDs
# As, Bs - lists of atom IDs
def filterNeighborsByGroup(neighbors,As,Bs):
	filtered=[]
	for ijk in neighbors:
		inA=[ i for i in ijk if i in As ]
		inB=[ i for i in ijk if i in Bs ]
		if len(inA)>0 and len(inB)>0:
			filtered.append(ijk)
	return filtered

# positions - [nt,na,xyz]
# potential - a function which can be passed a list of atom's positions [na,xyz] and return the potential energy
# atomSets - lists of atomIDs to feed into potential. e.g. i,j,k triplets for a 3-body potential. note that some potentials differentiate between i,j,k and i,k,j (stillinger-weber does), so make sure both are passed if you care.
# HOW DOES THIS WORK? 
# for any potential, you can find the force on an atom by perturbing its position and recalculating the potential energy. 
# Fₓ=-(Eₚ-Eₒ)/dx (if we perturbed in the x direction)
# for a pair-wise potential, forces are equal-and-opposite. calculate a force on one atom, this *must be* the force from the other atom.
# Fᵢ=Fᵢⱼ=-Fⱼᵢ
# for a many-body potential, we can assume satellite atoms do not interact (they do, but we'll consider those forces when we consider those atoms as the central atom. the potential does not specify forces, it specifies potential energy. for anything unspecified, we can assume, and so long as the potential is satisfied, then the assumption is valid). 
# Fᵢ=Fᵢⱼ+Fᵢₖ+Fᵢₗ+Fᵢₘ+Fᵢₙ+Fᵢₒ+... , Fⱼₖ=Fₖⱼ=0 , Fⱼₗ=Fₗⱼ=0, .... for central atom i, and satellite atoms j,k,l,m,n,o...
# which means we can simply perturb the satellites, finding Fⱼ, which is guaranteed to be Fⱼᵢ (since Fⱼₖ=Fⱼₗ=Fⱼₘ=Fⱼₙ=Fⱼₒ=...=0)
def calculateInteratomicForces(positions,potential,atomSets,perturbBy=.0001):
	os.makedirs("calculateInteratomicForces",exist_ok=True)
	nt=len(positions)
	nBody=len(atomSets[0])
	for ijk in tqdm(atomSets):
		for xyz in range(3):
			forces=np.zeros((nt,nBody)) # will hold force at each timestep: total on i, then contribution from j,k,etc
			for t in range(nt):
				atoms=positions[t,ijk,:]
				Vo=potential(atoms)					# potential energy for unperturbed atomic configuration
				for j in range(nBody):					# for each atom in the set
					dx=np.zeros((nBody,3)) ; dx[j,xyz]=perturbBy	# perturbation matrix: perturb the jth atom, in x or y or z
					Vi=potential(atoms+dx)				# recalculate potential, perturbing atom j in x y or z
					forces[t,j]=-(Vi-Vo)/perturbBy			# Fₓ=-dE/dx (if Eₚ > Eₒ, Force is in negative direction)
			xyzString=["x","y","z"][xyz]
			ijkString=",".join([ str(j) for j in ijk ])
			fileout="calculateInteratomicForces/F"+xyzString+"_"+ijkString+".txt"
			np.savetxt(fileout,forces)

# Power = Force * velocity , Fₓ=dU/dx. for two atoms interacting: Qᵢⱼ=dUᵢ/drᵢⱼ *  vᵢ - dUⱼ/drⱼᵢ * vⱼ , "Qnet: power on i by j minus power on j by i"
# In the time domain: Qᴬᴮ(t)=ΣᵢΣⱼ( dUᵢ(t)/drᵢⱼ * vᵢ(t) - dUⱼ(t)/dⱼᵢ * vⱼ(t) )
# In the frequency domain: Qᴬᴮ(ω)=ΣᵢΣⱼ( dUᵢ(ω)/drᵢⱼ * vᵢ(ω) - dUⱼ(ω)/dⱼᵢ * vⱼ(ω) )
# Note that these are NOT the same! ℱ[ f(t) * g(t) ] ≠ ℱ[ f(t) ] * ℱ[ g(t) ] (or f(ω)*g(ω))
# another way to think about this is: our fundamental question is "what (frequency of) oscillatory forces result in energy flow", which is a slightly separate question from simply "what oscillatory forces are there" (or oscillatory velocities, as in vDOS). Imagine the simplest case where F=-sin(ωt) and v=cos(ωt). The force "leads" the velocity slightly, and this is a case where the there is there clearly ought to be a net Q at ω. mathematically though, Q(t)=sin(ωt)*cos(ωt) is a function with *half* the periodicity (function is positive when F and v are both positive, or when F and v are both negative). clearly we want F(ω) and v(ω) separate, i.e. ℱ[ F(t) ] * ℱ[ v(t) ]
# So if ℱ[ f(t) * g(t) ] ≠ ℱ[ f(t) ] * ℱ[ g(t) ], then what IS ℱ[ f(t) ] * ℱ[ g(t) ] in the time domain?
# ℱ[ f(t) ] * ℱ[ g(t) ] = ℱ[ ⟨ f(t),g(t) ⟩ ] (where ⟨-⟩ is a cross-correlation) 
# So Q(ω) is NOT simply ℱ[ Q(t) ], but either ℱ[ F(t) ] * ℱ[ v(t) ] or ℱ[ ⟨ F(t),v(t) ⟩ ]
# And expanding this to "power between sides A and B", we sum over atoms in A and B, only including instances where i is in A and j is in B
# And practically in the code, how do we get forces? and what about the case of a 3-body potential?
# Consider atoms j-i-k where i is the central atom. 
# Let's start with "forces on j" (Fⱼ). compute energy (Eₒ), perturn atom j by dx and recalculate (Eₚⱼ). Fⱼₓ=-(Eₚⱼ-Eₒ)/dx (if Eₚⱼ > Eₒ, force is in negative direction). If we perturbed j, this is net force on j (Fⱼ)
# For a many-body potential, we will make the assumption that satellite atoms only experience a force from the central atom: Fⱼₖ=0, thus Fⱼᵢ=Fⱼ ("force on j by i is the same as the net force on j"). repeat for Fₖᵢ=Fₖ=(Eₚₖ-Eₒ)/dx. For forces on i, we can say interactions are "equal and opposite", Fᵢⱼ=-Fⱼᵢ, Fᵢ=-Fₖᵢ, and then for net force on i, we can sum: Fᵢ=Fᵢⱼ+Fᵢₖ
# Is this assumption of Fⱼₖ=Fₖⱼ=0 allowed?? the potential does not define forces, it defines energy. so we can make whatever statements we want so long as the energy expressions are satisfied! 
# How does this code work? you'll want to pre-calculate forces (using the function calculateInteratomicForces, which will create a folder by the same name, with files for forces in x,y,z for groups of atomd). pass velocities, and lists of indices for A and B. SHF code is quite simple, we just cycle through and sum as appropriate (taking note that, for example, Fj only contributes if atoms i,j are on opposite sides and so on). cross-correlate velocities and forces (in the time domain), and FFT.
def SHF(velocities,As,Bs):
	forceFiles=glob.glob("calculateInteratomicForces/*")
	nt=len(velocities)
	Qt=np.zeros((nt,3))
	for f in tqdm(forceFiles):
		xyz=["Fx","Fy","Fz"].index(f.split("/")[-1].split("_")[0])
		ijk=f.split("/")[-1].split("_")[-1].replace(".txt","").split(",")
		ijk=[ int(i) for i in ijk ]
		inA=[ i for i in ijk if i in As ]
		inB=[ i for i in ijk if i in Bs ]
		forces=np.loadtxt(f)		# columns are: Fᵢ, Fⱼ, Fₖ... these came from perturbing atoms i,j,k and so on.
		# if atom i on the left, j,k,etc gaining energy is +Q and i gaining energy is -Q. flipped if i on right
		i=ijk[0]
		sign={True:1,False:-1}[ i in As ]
		# we're going to sum up forces acting on by only atoms on the other side of the boundary
		Fi=np.zeros(nt)
		for c,j in enumerate(ijk):
			if c==0:
				continue
			if ( i in As and j in Bs ) or ( i in Bs and j in As ): # i on left, j on right, or vice-versa. IF SO, Qⱼ counts, Qⱼ=Fⱼ*vⱼ
				# Fⱼ was found by perturbing j, and we assumed Fⱼₖ=Fₖⱼ=0, therefore: Fⱼ=Fⱼᵢ, i.e. "F on j by i"
				Qt[:,xyz]+=sign*np.correlate(forces[:,c],velocities[:,j,xyz],mode="same")
				# Fᵢⱼ=-Fⱼᵢ ("F on i by j, is equal-and-opposite, of F on j by i"). and Fᵢ=Fᵢⱼ+Fᵢₖ (ifof j and k are across the boundary)
				Fi-=forces[:,c] 
		Qt[:,xyz]-=sign*np.correlate(Fi,velocities[:,i,xyz],mode="same") # Qᵢ=(Fᵢⱼ+Fᵢₖ+Fᵢₗ+Fᵢₘ...)*vᵢ (if j,k,l,m are all across the boundary)
	return np.fft.fft(Qt,axis=0)
