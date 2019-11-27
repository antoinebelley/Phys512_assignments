#Code to find gravitational waves from LIGO data
#Requires python 3.6 or higher (because of the use of f-strings)
#Author: Antoine Belley
#12/11/19


import numpy as np
from Read import read_file, read_template
from Gravitational_Wave_Finder import grav_wave_finder


directory = 'LOSC_Event_tutorial'


#Creates an array of the different files
fnames_H  = ['H-H1_LOSC_4_V2-1126259446-32.hdf5','H-H1_LOSC_4_V2-1128678884-32.hdf5','H-H1_LOSC_4_V2-1135136334-32.hdf5','H-H1_LOSC_4_V1-1167559920-32.hdf5'] 
fnames_L  = ['L-L1_LOSC_4_V2-1126259446-32.hdf5','L-L1_LOSC_4_V2-1128678884-32.hdf5','L-L1_LOSC_4_V2-1135136334-32.hdf5','L-L1_LOSC_4_V1-1167559920-32.hdf5'] 
strains_H = np.zeros([len(fnames_H),131072])
strains_L = np.zeros([len(fnames_L),131072])
dt = np.zeros([len(fnames_H)])
utc = []

#Creates array with all the filters files and arrays for the filtes at Hanford and Livingston
templates = np.array(['GW150914_4_template.hdf5','LVT151012_4_template.hdf5','GW151226_4_template.hdf5','GW170104_4_template.hdf5'])
th = np.zeros([len(templates), 131072])
tl = np.zeros([len(templates), 131072])

#Array of the event names
names = ['GW150914','LVT151012','GW151226','GW170104']

#Loop over the files to save all the strains utc's and dt's in arrays as well as the filters
for i in range(len(fnames_H)):
    strains_H[i,:], dt[i], utc_i = read_file(f'{directory}/{fnames_H[i]}')
    strains_L[i,:], dt[i], utc_i = read_file(f'{directory}/{fnames_L[i]}')
    utc.append(utc_i)
    th[i:,],tl[i:,]=read_template(f'{directory}/{templates[i]}')
np.array(utc)


#Window to be used in the noise model
x=np.arange(len(strains_H[0]))
x=x-1.0*x.mean()
window=0.5*(1+np.cos(x*np.pi/np.max(x)))

#Loop over the 4 events to find gravitational waves
results = []
for i in range(len(fnames_H)):
    res = grav_wave_finder(names[i], strains_H[i], strains_L[i], th[i], tl[i], window, dt[i])
    results.append(res)

#regex for the f-strings
nl = '\n'
tab = '\t'
#Write the answer to the questions asked in the problem set in the summary file.
summary = open('summary.txt', 'w')
summary.write('THE PLOTS CAN BE FOUND IN THE FOLDER NAMED AFTER THE EVENT\n\n')
summary.write('-------Answer to a)------------\n')
summary.write('To filter our noise we consider a gaussian stationary model. This is justified by the fact that the noise should not depend on time.\n')
summary.write('We use a cosine window to make sure our edge have a value of 0 to avoid trouble when taking the FT of the strains, and compute our\n')
summary.write('power spectrum and then convolve our spectrum with a gaussian filter. This is done in the fucntion noise found in util.py\n\n')

summary.write('-------Answer to b)------------\n')
summary.write('To filter our noise we consider a gaussian stationary model. This is justified by the fact that the noise should not depend on time.\n')
summary.write('We use a cosine window to make sure our edge have a value of 0 to avoid trouble when taking the FT of the strains, and compute our\n')
summary.write('power spectrum and then convolve our spectrum with a gaussian filter. This is done in the fucntion noise found in util.py\n\n')

summary.write('-------Answer to c)------------\n')
summary.write('The signal to noise ratio is given by m*sqrt((N^0.5*A)^T*(N^0.5*A)).\n')
summary.write('More detail can be found in util.py and the plots can be seen in the folders for each events.\n\n')

summary.write('-------Answer to d)------------\n')
summary.write('We can see that the two ways to find our signal to noise ratios give the same placement for the events.\n')
summary.write('However we can see that the amplitude is not exactly the same, and that the shape of the distribution is slightly different.\n')
summary.write('This is due to the fact that our noise is probaly not perfectly a gaussian stationary noise.\n\n')

summary.write('-------Answer to e)------------\n')
summary.write('Here are the frequency for each obseravtory by event:\n')
summary.write(f'{tab} {results[0]["name"]} {tab} {results[1]["name"]} {tab} {results[2]["name"]} {tab} {results[3]["name"]} {nl}')
summary.write(f'H{tab} {results[0]["freq_H"]}  {tab} {results[1]["freq_H"]} {tab} {results[2]["freq_H"]}{tab} {results[3]["freq_H"]} {nl}')
summary.write(f'L{tab} {results[0]["freq_L"]}{tab} {results[1]["freq_L"]} {tab} {results[2]["freq_L"]} {tab} {results[3]["freq_L"]} {nl}{nl}')

summary.write('-------Answer to e)------------\n')
summary.write('To find the time of the event, we fit a gaussian to our data and consider the mean to be the time\n')
summary.write('of the event and the standard deviation to be the error. We assumed 5000km between the detectors and used the speed of light to find\n')
summary.write('the error. Here are the resutls:\n')
summary.write(f'{tab}{tab}{tab} {results[0]["name"]} {tab} {results[1]["name"]} {tab} {results[2]["name"]} {tab} {results[3]["name"]} {nl}')
summary.write(f'time_H{tab}{tab} {results[0]["time_H"]} {tab} {results[1]["time_H"]} {tab} {results[2]["time_H"]}{tab} {results[3]["time_H"]} {nl}')
summary.write(f'err_H{tab}{tab} {results[0]["err_H"]} {tab} {results[1]["err_H"]} {tab} {results[2]["err_H"]}{tab} {results[3]["err_H"]} {nl}')
summary.write(f'time_L{tab}{tab} {results[0]["time_L"]} {tab} {results[1]["time_L"]} {tab} {results[2]["time_L"]}{tab} {results[3]["time_L"]} {nl}')
summary.write(f'err_L{tab}{tab} {results[0]["err_L"]} {tab} {results[1]["err_L"]} {tab} {results[2]["err_L"]}{tab} {results[3]["err_L"]} {nl}')
summary.write(f'angle_err{tab} {results[0]["err_tot"]}{tab} {results[1]["err_tot"]} {tab} {results[2]["err_tot"]} {tab} {results[3]["err_tot"]} {nl}{nl}')
summary.close()




