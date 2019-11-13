#Functions to read the data and template from LIGO
#Requires python 3.6 or higher (because of the use of f-strings)
#Author: Antoine Belley
#12/11/19


import h5py


def read_template(filename):
    """Read the template from a file.
    -Arguments: -filename (string): The path to the file containing the template
    -Returns  : -th (array)       : Template for Hanford
                -tl (array)       : Template for Livingston """
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl


def read_file(filename):
    """Read the data from a file for the Gravitational wave
    -Arguments: -filename (string): The path to the file containing the data
    -Returns  : -strain (array)   : Data from the interferometers
                -dt (array)       : Time step between data
                -utc (array)      : Time stamp """
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]
    meta=dataFile['meta']
    gpsStart=meta['GPSstart'].value
    utc=meta['UTCstart'].value
    duration=meta['Duration'].value
    strain=dataFile['strain']['Strain'].value
    dt=(1.0*duration)/len(strain)
    dataFile.close()
    return strain,dt,utc