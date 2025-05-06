import tkinter as tk
import matplotlib,os,ast
from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
matplotlib.use("Agg") # FOR SOME REASON, WE HAD "TkAgg" BEFORE, AND IT SUDDENLY DECIDED TO START OPENING NEW WINDOWS FOR EACH PLOT? comparing to gui.py > TDTR_fitting.py > niceplot.py, which works correctly (updated plot opens within window), gui.py specifies TkAgg, BUT, niceplot specifies Agg
import matplotlib.pyplot as plt
from pySED import *

#  ____________________________
# |   |            |           |
# | B | d          |   i       |
# | U |  i         |  n m      |
# | T |   s        | a   a     |
# | T |    p     n |      t    |
# | O |     e   o  |       i   |
# | N |      r i   |        o  |
# | S |       s    |         n |
# |___|____________|___________|

# button panel: 
# [ select file ]
# [ file entry  ]
# [ initialize  ]

figsize=(6,6)

window = tk.Tk()
frame=Frame(master=window) ; frame.grid(row=0,column=0,sticky="NSEW")

dumpfile="NVE.qdump" ; direc="/media/Alexandria/U Virginia/Research/MD/projects/Si_SED_04/"
Zs=[[]] ; ks=[] ; ws=[]
dispersionPlot={}

def main():
	addButtons()


def addButtons():
	global entry_filename
	button_file=tk.Button(master=frame,text="select file")
	button_file.bind("<Button-1>",selectFile)
	button_file.grid(row=0,column=0)
	entry_filename=tk.Entry(master=frame)
	entry_filename.grid(row=1,column=0)
	button_init=tk.Button(master=frame,text="initialize")
	button_init.bind("<Button-1>",initialize)
	button_init.grid(row=2,column=0)

def selectFile(event):
	global entry_filename ; global dumpfile,direc
	selected=tk.filedialog.askopenfilename(initialdir="../../MD/projects/",title="select qdump file") 	# TODO RESTORE INITIALIZE DIREC FROM LOG?
	direc=selected.split("/")		# "path/to/dumpfile.qdump" -> ["path","to","dumpfile.qdump"]
	dumpfile=selected[-1]
	if len(direc)>1:			# (dumpfile might be in current directory)
		direc="/".join(direc[:-1])
	else:
		direc="./"

def initialize(event):
	global atoms,avg,disp,a,nx,types
	a,b,c=5.43729,5.43729,5.43729	# TODO THESE SHOULD BE USER-ENTERABLE FIELDS
	nx,ny,nz=120,5,5

	positions,velocities,timesteps,types=qdump(direc+"/"+dumpfile) # columns include: id type x y z vx vy vz
	if os.path.exists(direc+"/avg.npy"):
		avg=np.load(direc+"/avg.npy") ; disp=np.load(direc+"/disp.npy")
	else:
		avg,disp=avgPos(positions,nx*a,ny*b,nz*c) # we'll use average atomic positions for SED (can also use lattice-defined points)
		np.save(direc+"/avg.npy",avg) ; np.save(direc+"/disp.npy",disp)

	atoms=iSED(avg,disp,[1,0,0],k=0,w=0,a=a/4,nk=nx*4,bs='',ks='',rescaling=1,types=types,store=False) # TODO p_xyz SHOULD BE USER-ENTERABLE
	
	showDispersion()


def showDispersion():
	# LOAD DISPERSION FROM FILE
	Zs=np.sum([ np.absolute( np.load( "dSED_b0_p100_v"+str(i)+".npy" )) for i in range(3) ],axis=0)
	ks=np.load("dSED_ks_p100.npy") ; ws=np.load("dSED_ws_p100.npy")
	# PLOT IT
	fig,ax=plt.subplots(figsize=figsize)			# new empty matplotlib figure/axes objects
	plt.imshow(Zs[::-1,:]**.25,cmap="inferno",extent=(min(ks),max(ks),min(ws),max(ws)),aspect=max(ks)/max(ws))
	# HANDLE GUI GLOBALS
	dispersionPlot["fig"],dispersionPlot["ax"]=fig,ax	# store the fig,ax objects to our dict
	if "canvas" in dispersionPlot.keys():			# if this isn't the first time, destroy the old "canvas" object
		dispersionPlot["canvas"].get_tk_widget().destroy()
	canvas = FigureCanvasTkAgg(fig, master=window)		# make a new canvas object to hold the figure
	canvas.get_tk_widget().grid(row=0,column=1,sticky='ew')	# pack it
	dispersionPlot["canvas"]=canvas				# and store off the canvas object too
	canvas.mpl_connect('button_press_event',regeniSED)	# link callback function to clicks

def regeniSED(event):
	# DETECT CLICK LOCATION
	x,y=event.xdata,event.ydata				# when the user clicks on the plot...
	print("clicked at", x,y)
	# REGEN ISED
	atoms=iSED(avg,disp,[1,0,0],k=x,w=y,a=a/4,nk=nx*4,bs='',ks='',rescaling=2,types=types,store=False) # TODO p_xyz SHOULD BE USER-ENTERABLE
	# PLOT IT (STILL PLOT)
	fig,ax=plt.subplots(figsize=figsize)			# new empty matplotlib figure/axes objects
	xs=atoms[0,:,0] ; ys=atoms[0,:,1]
	mn=max(min(xs),min(ys)) ; mx=min(max(xs),max(ys))
	ys=ys[xs>=mn] ; xs=xs[xs>=mn] ; ys=ys[xs<=mx] ; xs=xs[xs<=mx]
	xs=xs[ys>=mn] ; ys=ys[ys>=mn] ; xs=xs[ys<=mx] ; ys=ys[ys<=mx]
	ax.plot(xs, ys, marker=".",linewidth=0,c="k")
	# HANDLE GUI GLOBALS
	dispersionPlot["fig2"],dispersionPlot["ax2"]=fig,ax	# store the fig,ax objects to our dict
	if "canvas2" in dispersionPlot.keys():			# if this isn't the first time, destroy the old "canvas" object
		dispersionPlot["canvas2"].get_tk_widget().destroy()
	canvas = FigureCanvasTkAgg(fig, master=window)		# make a new canvas object to hold the figure
	canvas.get_tk_widget().grid(row=0,column=2,sticky='ew')	# pack it
	dispersionPlot["canvas2"]=canvas				# and store off the canvas object too
	#canvas.mpl_connect('button_press_event',regeniSED)	# link callback function to clicks


def quit_me():				# weird thing, when we generate a matplotlib plot, tkinter main loop doesn't exit when we click the x.
	#log("Quitting")		# to deal with that, we detect a "delete window" and then use that to quit.
	window.quit()			# https://stackoverflow.com/questions/55201199/the-python-program-is-not-ending-when-tkinter-window-is-closed
	window.destroy()
	#global done
	#done=True
#genFig(0,0)				# to initialize, start with just the 100-110-111 fig, and the user options panel
main()
window.protocol("WM_DELETE_WINDOW", quit_me)

window.mainloop()
