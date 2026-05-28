#!/usr/bin/env python3

'''
Functions of Paranal Airglow Line and Continuum Emission (PALACE) model
Copyright (C) 2024  Stefan Noll

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>


List of PALACE functions:

Input:

Top:
parlist = parlist(cline, parfile='')

Called by parlist:
parlist = getparlist(cline, parfile='')
err = checkparlist(**parlist)

Output:

Top:
err = output(spec, **parlist)

Called by output:
err = writeparlist(**parlist)
err = writespec(spec, **parlist)
err = plotspec(spec, **parlist)

Model:

Top:
spec = model(**parlist)

Called by model:
lintab, conttab, vartab = readdata(**parlist)
scalfac = calcscalfac(vartab, **parlist)
slintab = scalelines(lintab, scalfac, **parlist)
sconttab = scalecont(conttab, scalfac, **parlist)
cslintab = corratmlines(slintab, **parlist)
csconttab = corratmcont(sconttab, **parlist)
linspec = calclinspec(cslintab, **parlist)
contspec = calccontspec(csconttab, **parlist)
addspec = addlinescont(linspec, contspec, **parlist)
spec = convolvelsf(addspec, **parlist)

Low-level functions:
isnight = isnight(month, time)
n = refindex(lam)
gauss = gauss(x, m, s)
fvr = vanrhijn(z, hlayer)
tabs = calcabs(tabs0, z, fwv, pwv, pwv0)
tscat = calcscat(lam, z)
yout, w = rebin(xout, xin, yin)
y = erf(x)
'''

import sys
import re
from pathlib import Path
try:
    import importlib_resources as resources
except ImportError:
    from importlib import resources
import numpy as np
from astropy.table import Table, join
import matplotlib.pyplot as plt
import matplotlib as mpl
import cython


def parlist(cline, parfile=''):

    '''
    Handles parameter list.
    Reads parameters from file and command line and checks their validity.
    An empty input list (or a list with only one item without '=' sign) or 
    an empty string will cause that all parameters are taken from parfile if
    a file name is given. Otherwise config/palace_default.par of the palace 
    package is used. The required syntax of the list items (except for the 
    first one) is 'name=value'. The valid parameter names and requirements 
    for the values are provided in palace_default.par.   
    Parameters:
    cline: list of command line arguments
    parfile: path and name of parameter file (if not given by cline)
    Returns:
    parlist: parameter list as dictionary
             (empty dictionary in the case of errors)
    '''

    parlist = getparlist(cline, parfile)
    if len(parlist) == 0:
        return parlist

    err = checkparlist(**parlist)
    if err > 0:
        parlist = {}

    return parlist    


def getparlist(cline, parfile=''):

    '''
    Gets parameter list.
    Takes command line arguments as provided by sys.argv and substitutes the
    corresponding parameter values in the parameter dictionary derived from
    the selected parameter file, which is either the default parameter file
    of the package (config/palace_default.par), the argument of parfile if 
    given, or the file name provided by the optional command line argument 
    starting with 'parfile='. The parameter value strings are converted into 
    int, float, or bool type if required.
    Parameters:
    cline: list of command line arguments
    parfile: path and name of parameter file (if not given by cline)
    Returns:
    parlist: parameter list as dictionary
             (empty dictionary in the case of a file read error)
    '''

    #Basic definitions
    defparfile = 'palace_default.par'
    inparnames = []
    inparvals = []
    parnames = []
    parvals = []
    regexpnum = r'^[-+]?[0-9]*\.?[0-9]*([eE][-+]?[0-9]+)?$'
    regexpint = r'^[-+]?[0-9]+$'

    #Use default parameter file if parfile is not provided
    if parfile == '':
        try:            
            parfile = resources.files('palace.config').joinpath(defparfile)
        except BaseException as errtext:
            print('No default parameter file: %s' % (errtext))
            parlist = {}
            return parlist
    
    #Read parameters from input list
    ninpar = len(cline)
    if ninpar >= 1:
        for i in range(ninpar):
            inpardata = cline[i].split('=')
            if len(inpardata) == 2:
                if inpardata[0] == 'parfile':
                    parfile = inpardata[1]
                else:    
                    inparnames.append(inpardata[0]) 
                    inparvals.append(inpardata[1])
            else:
                if i > 0:
                    print('Invalid input parameter %s -> skipped' %
                          (cline[i]))
                ninpar = ninpar - 1  

    #Read parameter file                
    try:
        file = open(parfile, 'rt')
        for fline in file:
            pardata = fline.split('=')
            if (len(pardata) != 2) & (fline[0] != '#') & (len(fline) > 1):
                print('Invalid line in parameter file %s:' % (parfile))
                print('"%s" -> skipped' % (fline.strip()))
            elif (len(pardata) == 2) & (fline[0] != '#'):    
                parnames.append(pardata[0].strip()) 
                parvals.append(pardata[1].strip())
        file.close()
    except BaseException as errtext:
        print('No valid parameter file: %s' % (errtext))
        parlist = {}
        return parlist

    #Substitute parameter values and warn for unknown parameters   
    check = [False for _ in range(ninpar)]
    npar = len(parnames)    
    for j in range(npar):
        for i in range(ninpar):
            if inparnames[i] == parnames[j]:
                parvals[j] = inparvals[i]
                check[i] = True
    for i in range(ninpar):
        if not check[i]:
            print('Command-line parameter %s: not known' % (inparnames[i]))
                
    #Convert non-string parameters            
    for j in range(npar):
        val = parvals[j]
        if re.match(regexpnum, val):
            if re.match(regexpint, val):
                parvals[j] = int(val)
            elif (not re.match(regexpint, val)) & (val != '.'):
                parvals[j] = float(val)
        else:
            if val == 'True':
                parvals[j] = True
            elif val == 'False':
                parvals[j] = False

    #Return parameter list as dictionary 
    parlist = dict(zip(parnames, parvals))    
    return parlist


def checkparlist(**parlist):

    '''
    Checks parameter list with respect to validity of values.
    Parameters:
    parlist: dictionary with model parameters
    Returns:
    err: number of error(s) (0 -> OK)
    '''

    err = 0

    #Valid parameter list?
    if len(parlist) == 0:
        print('No valid parameter list')
        err = 1
        return err
    
    #species
    if not 'species' in parlist.keys():
        print('Parameter species: not found')
        err = err + 1
    else:
        species = ['all', 'OH', 'O2', 'HO2', 'FeO', 'O', 'N', 'Na', 'K', 'H']
        match = 0
        for s in species:
            if parlist['species'] == s:
                match = 1
                break
        if match == 0:
            print('Parameter species: unknown species %s' % \
                  (parlist['species']))
            err = err + 1
        
    #z:
    if not 'z' in parlist.keys():
        print('Parameter z: not found')
        err = err + 1
    else:    
        if not isinstance(parlist['z'], (float, int)):
            print('Parameter z: not a number')
            err = err + 1
        else:
            if parlist['z'] >= 90.:
                print('Parameter z: line of sight below the horizon')
                err = err + 1
            elif parlist['z'] < 0.:
                print('Parameter z: impossible negative value')
                err = err + 1

    #mbin:
    if not 'mbin' in parlist.keys():
        print('Parameter mbin: not found')
        err = err + 1
    else:    
        if not isinstance(parlist['mbin'], int):
            print('Parameter mbin: not an integer')
            err = err + 1
        elif (parlist['mbin'] < 0) | (parlist['mbin'] > 12):
            print('Parameter mbin: valid range from 0 to 12')
            err = err + 1

    #tbin:    
    if not 'tbin' in parlist.keys():
        print('Parameter tbin: not found')
        err = err + 1
    else:    
        if not isinstance(parlist['tbin'], (int, float)):
            print('Parameter tbin: not a number')
            err = err + 1
        else:
            if isinstance(parlist['tbin'], int) & ((parlist['tbin'] < 0) |
                                                   (parlist['tbin'] > 12)):
                print('Parameter tbin (integer): valid range from 0 to 12')
                err = err + 1
            elif isinstance(parlist['tbin'], float) & \
                 ((parlist['tbin'] < -6.) |
                  ((parlist['tbin'] >= 6.) & (parlist['tbin'] < 18.)) |
                  (parlist['tbin'] >= 24.)):
                print('Parameter tbin (float): valid range [-6., 6.] or ' \
                      '[0., 6.] and [18., 24.]')
                err = err + 1

    #srf:
    if not 'srf' in parlist.keys():
        print('Parameter srf: not found')
        err = err + 1
    else:    
        if not isinstance(parlist['srf'], (float, int)):
            print('Parameter srf: not a number')
            err = err + 1
        elif parlist['srf'] < 0.:
            print('Parameter srf: impossible negative value')
            err = err + 1

    #isair:
    if not 'isair' in parlist.keys():
        print('Parameter isair: not found')
        err = err + 1
    else:    
        if not isinstance(parlist['isair'], bool):
            print('Parameter isair: not Boolean (True or False)')
            err = err + 1

    #isatm:
    if not 'isatm' in parlist.keys():
        print('Parameter isatm: not found')
        err = err + 1
    else:    
        if not isinstance(parlist['isatm'], bool):
            print('Parameter isatm: not Boolean (True or False)')
            err = err + 1

    #pwv:
    if not 'pwv' in parlist.keys():
        print('Parameter pwv: not found')
        err = err + 1
    else:    
        if not isinstance(parlist['pwv'], (float, int)):
            print('Parameter pwv: not a number')
            err = err + 1
        elif parlist['pwv'] < 0.:
            print('Parameter pwv: impossible negative value')
            err = err + 1

    #lammin:
    lamtol = 5e-3
    if not 'lammin' in parlist.keys():
        print('Parameter lammin: not found')
        err = err + 1
    else:    
        if not isinstance(parlist['lammin'], float):
            print('Parameter lammin: not a float')
            err = err + 1
        elif (parlist['lammin'] < 0.3 - lamtol) | \
             (parlist['lammin'] > 2.5 - lamtol):
            print('Parameter lammin: outside model range [0.3, 2.5]')
            err = err + 1

    #lammax:
    if not 'lammax' in parlist.keys():
        print('Parameter lammax: not found')
        err = err + 1
    else:    
        if not isinstance(parlist['lammax'], float):
            print('Parameter lammax: not a float')
            err = err + 1
        else:
            if (parlist['lammax'] < 0.3 + lamtol) | \
               (parlist['lammax'] > 2.5 + lamtol):
                print('Parameter lammax: outside model range [0.3, 2.5]')
                err = err + 1   
            if isinstance(parlist['lammin'], float):
                if parlist['lammax'] <= parlist['lammin']:
                    print('Parameter lammax: lower than lammin')
                    err = err + 1
                  
    #dlam:
    if not 'dlam' in parlist.keys():
        print('Parameter dlam: not found')
        err = err + 1
    else:    
        if not isinstance(parlist['dlam'], float):
            print('Parameter dlam: not a float')
            err = err + 1
        else:    
            if parlist['dlam'] < 1e-7:
                print('Parameter dlam: at least 0.1 pm required')
                err = err + 1
            elif isinstance(parlist['lammin'], float) & \
                 isinstance(parlist['lammax'], float):
                if parlist['dlam'] > (parlist['lammax'] - parlist['lammin']):
                    print('Parameter dlam: greater than range')
                    err = err + 1

    #resol:
    if not 'resol' in parlist.keys():
        print('Parameter resol: not found')
        err = err + 1
    else:    
        if not isinstance(parlist['resol'], (float, int)):
            print('Parameter resol: not a number')
            err = err + 1
        elif parlist['resol'] <= 0.:
            print('Parameter resol: positive value required')
            err = err + 1
                                    
    #outdir:
    if not 'outdir' in parlist.keys():
        print('Parameter outdir: not found')
        err = err + 1

    #outname:
    if not 'outname' in parlist.keys():
        print('Parameter outname: not found')
        err = err + 1
                  
    #specsuffix:
    if not 'specsuffix' in parlist.keys():
        print('Parameter specsuffix: not found')
        err = err + 1
                  
    #showplot:
    if not 'showplot' in parlist.keys():
        print('Parameter showplot: not found')
        err = err + 1
    else:    
        if not isinstance(parlist['showplot'], bool):
            print('Parameter showplot: not Boolean (True or False)')
            err = err + 1

    #Number of errors
    if err == 1:
        print('-> %d error' % (err))
    elif err > 1:    
        print('-> %d errors' % (err))
                  
    return err


def output(spec, **parlist):

    '''
    Writes output files and creates optional plot.
    Uses parameters outdir, outname, specsuffix, showplot, and saveplot.
    Parameters:
    spec: astropy table with output spectrum
    parlist: dictionary with model parameters
    Returns:
    err: 0 -> OK, 1 -> error
    '''

    #Valid parameter list?
    if len(parlist) == 0:
        print('No valid parameter list')
        err = 1
        return err
    
    #Write parameter file (suffix 'par') into output folder
    err = writeparlist(**parlist)
    if err == 1:
        return err

    #Write spectrum (default suffix: 'fits') into output folder
    err = writespec(spec, **parlist) 
    if err == 1:
        return err

    #Plot spectrum on screen if showplot=True
    if parlist['showplot']:
        err = plotspec(spec, **parlist) 

    return err


def writeparlist(**parlist):

    '''
    Writes parameter list into ASCII file.
    Directory and name are taken from the parameters outdir and outname. The 
    suffix is set to 'par'. The output directory is (and necessary parent 
    directories are) created if required. Comments as given in the default 
    parameter file are neglected.
    Parameters:
    parlist: dictionary with model parameters
    Returns:
    err: 0 -> OK, 1 -> error
    '''

    err = 0

    #Valid parameter list?
    if len(parlist) == 0:
        print('No valid parameter list')
        err = 1
        return err

    #Decompose parameter list
    npar = len(parlist)
    parnames = list(parlist.keys())
    parvals = list(parlist.values())

    #Create output directory (and parent directories) if required
    outdir = Path(parlist['outdir'])
    if not outdir.exists():
        try:
            outdir.mkdir(parents=True)
        except BaseException as errtext:
            print('Output directory: %s' % (errtext))
            err = 1
            return err
            
    #Create output parameter file and write parameters
    parfile = outdir.joinpath(parlist['outname'] + '.par')
    try:
        file = open(parfile, 'wt')
        for i in range(npar):
            file.write('%s=%s\n' % (parnames[i], str(parvals[i]))) 
        file.close()
    except BaseException as errtext:
        print('Output parameter file: %s' % (errtext))
        err = 1

    return err


def writespec(spec, **parlist):

    ''' 
    Writes spectrum into file.
    Directory and name are taken from the parameters outdir and outname. If
    'fits' is given for the parameter specsuffix, a FITS table is written. 
    In all other cases, an ASCII file with the given suffix is produced. The
    output file includes the columns 'lam' (wavelength in µm), 'flux' 
    (radiance in R/nm), and 'dflux' (residual radiance variability in R/nm). 
    In the case of ASCII files, these column names are provided in the first 
    line of the file. 
    Parameters:
    spec: astropy table with spectrum
    parlist: dictionary with model parameters
    Returns:
    err: 0 -> OK, 1 -> error   
    '''

    err = 0

    #Valid parameter list?
    if len(parlist) == 0:
        print('No valid parameter list')
        err = 1
        return err

    #Check existence of data
    if len(spec) == 0:
        print('Spectrum: no data to be written')
        err = 1
        return err

    #Create output directory (and parent directories) if required
    outdir = Path(parlist['outdir'])
    if not outdir.exists():
        try:
            outdir.mkdir(parents=True)
        except BaseException as errtext:
            print('Output directory: %s' % (errtext))
            err = 1
            return err

    #Write FITS or ASCII file
    specfile = outdir.joinpath(parlist['outname'] + '.' + \
                               parlist['specsuffix'])
    try:
        if parlist['specsuffix'] == 'fits':
            spec.write(specfile, overwrite=True)
        else:
            spec.write(specfile, overwrite=True, format='ascii.basic')
    except BaseException as errtext:
        print('Output FITS file: %s' % (errtext))
        err = 1
           
    return err


def plotspec(spec, figid=1, **parlist):

    '''
    Plots spectrum and its residual variability in window on screeen. A
    black curve shows the radiance in R/nm as a function of wavelength in 
    µm. Moreover, the residual variability of the model is provided by a
    green curve.
    Parameters:
    spec: astropy table with spectrum
    figid: ID of figure window (None: opens new plot)
    parlist: dictionary with model parameters
    Returns:
    err: 0 -> OK, 1 -> error   
    '''

    err = 0

    #Valid parameter list?
    if len(parlist) == 0:
        print('No valid parameter list')
        err = 1
        return err

    #Check existence of data
    if len(spec) == 0:
        print('Spectrum: no data to be plotted')
        err = 1
        return err

    #Plot data
    
    try:
    
        plt.style.use('default')
        mpl.rcParams['mathtext.default'] = 'rm'
        mpl.rcParams['font.size'] = 10.0

        fig = plt.figure(figid, figsize=[9.6,4.8])
        plt.clf()
        ax = plt.axes()

        ax.plot(spec['lam'], spec['flux'], 'k-')
        ax.plot(spec['lam'], spec['dflux'], 'g-')
        
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.tick_params(which='major', length=6)
        ax.tick_params(which='minor', length=3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10,
                                                    steps=[1,2,5,10]))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6,
                                                    steps=[1,2,5,10]))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        lammin = spec['lam'].min()
        lammax = spec['lam'].max()
        dellam = lammax - lammin
        delx = 0.02
        xmin = lammin - delx * dellam
        xmax = lammax + delx * dellam
        ax.set_xlim(xmin,xmax)
        
        ax.set_xlabel(r'Wavelength [$\mu$m]')
        ax.set_ylabel('Radiance [R nm$^{-1}$]')

        fig.tight_layout()
        plt.show()
        mpl.rcdefaults()

    except BaseException as errtext:
        
        print('Plot of spectrum: %s' % (errtext))
        err = 1

    return err


def model(**parlist):

    '''
    Calculates airglow model.
    Uses FITS tables with line list, continuum components, and variability 
    data from data folder of the palace package. The line intensities and 
    continuum fluxes are scaled depending on the observing conditions 
    described by the parameters z, mbin, tbin, and srf. Doppler broadening 
    is assumed for the calculation of Gaussian line profiles. The added 
    emissions from lines and continuum components, which are calculated for 
    the given wavelength grid defined by lammin, lammax, and dlam, are 
    corrected for atmospheric absorption and scattering if isatm=True is 
    chosen. Then, the parameter pwv affects the transmission in wavelength 
    ranges with water vapour absorption. The spectrum is given for air 
    (instead of vacuum) if isair=True. In the end, the resolving power from 
    parameter resol is used for the convolution of the spectrum with a 
    Gaussian line spread function. The resulting spectrum is returned. If a 
    specific species (instead of 'all') is selected, only the corresponding 
    lines and continuum components are considered for the calculations.
    Parameters:
    parlist: dictionary with model parameters
    Returns:
    spec: astropy table with output spectrum consisting of columns lam
          (wavelength in µm), flux (in R/nm), and dflux (uncertainty by
          residual variability in R/nm) 
          (empty table in the case of errors)
    '''

    #Valid parameter list?
    if not parlist:
        raise ValueError('No valid parameter list')

    #Read model data
    lintab, conttab, vartab = readdata(**parlist)

    if len(vartab) == 0:
        raise ValueError('No variability data')

    #Calculate climatological scaling factors for each species
    scalfac = calcscalfac(vartab, **parlist)

    #Scale emission lines and continuum components (also considering
    #van Rhijn effect)
    slintab = scalelines(lintab, scalfac, **parlist)
    sconttab = scalecont(conttab, scalfac, **parlist)

    #Apply extinction correction for lines and continua if isatm=True
    cslintab = corratmlines(slintab, **parlist)
    csconttab = corratmcont(sconttab, **parlist)

    #Calculate line spectrum considering Doppler broadening for given
    #wavelength grid (also considering parameter isair)
    linspec = calclinspec(cslintab, **parlist)

    #Regrid continuum spectra for given wavelength grid
    contspec = calccontspec(csconttab, **parlist)

    #Add line and continuum spectra
    addspec = addlinescont(linspec, contspec, **parlist)

    #Convolve resulting spectrum with Gaussian line spread function
    spec = convolvelsf(addspec, **parlist)
    spec_cont = convolvelsf(contspec, **parlist)
    spec_line = convolvelsf(linspec, **parlist)
    
    #Return final spectrum
    return spec, spec_cont, spec_line


def readdata(**parlist):

    '''
    Reads model data.
    FITS tables with line list (palace_lines.fits), continua 
    (palace_cont.fits), and variability data (palace_var.fits) are read in 
    the data folder of the palace package. The data are then returned as 
    astropy tables. The first two tables are limited with respect to the 
    selected wavelength range (parameters lammin and lammax). If a special 
    atom or molecule is selected with the parameter species, only the 
    relevant rows and columns are provided. If the desired species is not in 
    the line list or the continuum table, an empty table (length of zero) is 
    returned. As variability data are always needed, an empty vartab implies 
    an error in the reading procedure.
    Parameters:
    parlist: dictionary with model parameters
    Returns:
    lintab: astropy table with line data (empty if not relevant)
    conttab: astropy table with continuum data (empty if not relevant)
    vartab: astropy table with variability data (errors if empty)
    '''

    err = []

    #Names of model files in data folder
    linfile = 'palace_lines.fits'
    contfile = 'palace_cont.fits'
    varfile = 'palace_var.fits'

    #Get wavelength limits for selection
    maxsig = 5.0 #cut of Gaussians in multiples of sigma
    fwhm2sig = 1 / (2 * np.sqrt(2 * np.log(2)))
    reldel = maxsig * fwhm2sig / parlist['resol']
    ndelmin = int(np.ceil(parlist['lammin'] * reldel / parlist['dlam']))
    lammin = parlist['lammin'] - (ndelmin + 0.5) * parlist['dlam']
    ndelmax = int(np.ceil(parlist['lammax'] * reldel / parlist['dlam']))
    lammax = parlist['lammax'] + (ndelmax + 0.5) * parlist['dlam']    
    if parlist['isair']:
        lammin = lammin * refindex(lammin) 
        lammax = lammax * refindex(lammax)
                  
    #Line list
    try:
        file = resources.files('palace.data').joinpath(linfile)
        tlin = Table.read(file)
        #Wavelength selection
        tlin = tlin[(tlin['lam'] >= lammin) & (tlin['lam'] <= lammax)]
        if parlist['species'] != 'all':
            #Seclect table rows for specific species
            lintab = tlin[tlin['chem'] == parlist['species']]
        else:
            lintab = tlin

    except BaseException as errtext:
        err.append('linfile: %s' % (errtext))

                  
    #Continuum components    
    try:
        file = resources.files('palace.data').joinpath(contfile)        
        conttab = Table.read(file)
        #Wavelength selection
        conttab = conttab[(conttab['lam'] >= lammin) & \
                          (conttab['lam'] <= lammax)]
        if parlist['species'] != 'all':
            #Select columns and keywords for specific species
            imin = 10
            imax = 0
            for i in range(1, conttab.meta['NCONT']+1):
                stri = str(i)
                if conttab.meta['CHEM' + stri] != parlist['species']:
                    conttab.remove_column('fcont' + stri)
                    conttab.meta.pop('CHEM' + stri)
                    conttab.meta.pop('VARID' + stri)
                    conttab.meta.pop('HLAYER' + stri)
                else:
                    if i < imin:
                        imin = i
                    if i > imax:
                        imax = i
            if imin <= imax:
                ncont = imax - imin + 1
                conttab.meta['NCONT'] = ncont
                for h in range(1, ncont+1):
                    i = imin + h - 1
                    stri = str(i)
                    strh = str(h)                
                    if i != h:
                        conttab.rename_column('fcont' + stri, 'fcont' + strh)
                        conttab.meta['CHEM' + strh] = \
                            conttab.meta['CHEM' + stri]
                        conttab.meta['VARID' + strh] = \
                            conttab.meta['VARID' + stri]
                        conttab.meta['HLAYER' + strh] = \
                            conttab.meta['HLAYER' + stri]                     
                        conttab.meta.pop('CHEM' + stri)
                        conttab.meta.pop('VARID' + stri)
                        conttab.meta.pop('HLAYER' + stri)
            else:
                conttab = Table()

    except BaseException as errtext:
        err.append('contfile: %s' % (errtext))


    #Variability data 
    try:
        file = resources.files('palace.data').joinpath(varfile)        
        vartab = Table.read(file)
        if parlist['species'] != 'all':
            #Select columns and keywords for specific species
            jmin = 100
            jmax = 0
            for j in range(1, vartab.meta['NVARID']+1):
                strj = '%02d' % (j)
                if vartab.meta['CHEM' + strj] != parlist['species']:
                    varid = vartab.meta['VARID' + strj]
                    vartab.remove_column('rI_' + varid)
                    vartab.remove_column('SCE_' + varid)
                    vartab.remove_column('rdI_' + varid)
                    vartab.meta.pop('CHEM' + strj)
                    vartab.meta.pop('VARID' + strj)
                else:
                    if j < jmin:
                        jmin = j
                    if j > jmax:
                        jmax = j
            nvarid = jmax - jmin + 1        
            vartab.meta['NVARID'] = nvarid
            for k in range(1, nvarid+1):
                j = jmin + k - 1
                strj = '%02d' % (j)
                strk = '%02d' % (k)                
                if j != k:
                    vartab.meta['CHEM' + strk] = vartab.meta['CHEM' + strj]
                    vartab.meta['VARID' + strk] = vartab.meta['VARID' + strj]
                    vartab.meta.pop('CHEM' + strj)
                    vartab.meta.pop('VARID' + strj)
    except BaseException as errtext:
        err.append('varfile: %s' % (errtext))

    #Return empty tables in the case of errors    
    if err:
        raise ValueError(err)

    #Add meta data for temporal extension of output wavelength grid
    lintab.meta['MAXSIG'] = maxsig
    lintab.meta['NDELMIN'] = ndelmin
    lintab.meta['NDELMAX'] = ndelmax
    conttab.meta['MAXSIG'] = maxsig
    conttab.meta['NDELMIN'] = ndelmin
    conttab.meta['NDELMAX'] = ndelmax
                      
    return lintab, conttab, vartab


def calcscalfac(vartab, **parlist):

    '''
    Calculates scaling factors for the relevant variability classes.
    For a single bin, the relative intensity, solar cycle effect, and
    residual variability are combined to produce the corresponding scaling
    factors for intensity/flux and its variation for the given solar radio
    flux. If all months and/or local time bins are combined (mbin=0 and/or
    tbin=0), the effective quantities are estimated by means of the
    nighttime-weighted summation of Gaussians with the centres and widths
    taken from the values of the individual bins.
    Parameters:
    vartab: astropy table with variability data
    parlist: dictionary with model parameters
    Returns:
    scalfac: astropy table with scaling factors for intensities (lines) and
             fluxes (continua)
    '''

    #Constants
    lt0 = -6.5 #in h
    dlt = 1.   #in h
    tol = 1e-6
    srf0 = vartab.meta['SRF0'] #in sfu
    maxsig = 5.
    ndec = 2
    delri = np.power(10., -ndec)

    #Arrays for scaling factors    
    nv = vartab.meta['NVARID']
    varids = np.empty(nv, dtype='U4')
    ri = np.ones(nv, dtype='f4')
    rdi = np.ones(nv, dtype='f4')
                  
    #Conversion of local time in hours to bin number if required 
    mbin = parlist['mbin']
    if isinstance(parlist['tbin'], float):
        lt = parlist['tbin']
        if lt > 12.:
            lt = lt - 24.
        tbin = round((lt - lt0 + tol) / dlt)
        isnoctlt = isnight(mbin, lt)
        if not isnoctlt:
            print('mbin and tbin: no night at local time -> extrapolation')
    else:
        tbin = parlist['tbin']
        isnoctlt = True
        
    #Selection of single or all months and times
    if (mbin == 0) & (tbin == 0):
        tvar = vartab.copy()
    elif mbin == 0:
        tvar = vartab[(vartab['tbin'] == tbin)]
    elif tbin == 0:
        tvar = vartab[(vartab['mbin'] == mbin)]
    else:    
        tvar = vartab[(vartab['mbin'] == mbin) & (vartab['tbin'] == tbin)]
    nvar = len(tvar)    

    #Check nighttime contribution of selected climatological bin(s)
    w = np.array(tvar['wbin'])
    wsum = w.sum()
    if wsum == 0.:
        if isnoctlt:
            print('mbin and tbin: no nighttime in bin -> extrapolation')
        w = w + 1.

    #Get scaling factors for intensities/fluxes and their residual variability
    #for each variability class
        
    for i in np.arange(nv):
    
        varids[i] = tvar.meta['VARID%02d' % (i+1)] 
    
        ris = tvar['rI_' + str(varids[i])]
        sces = tvar['SCE_' + str(varids[i])]
        rdis = tvar['rdI_' + str(varids[i])]

        #Linear solar activity correction for intensity/flux
        mris = np.array(ris * (1 + sces * 0.01 * (parlist['srf'] - srf0)))
        ri[i] = np.average(mris, weights=w)

        #Intensity/flux times standard deviation
        mrdis = np.array(ris * rdis)
        rdi[i] = np.average(mrdis, weights=w)

        #Multiple months and/or local times: effective standard deviation
        #from combination of Gaussian functions with mean and sigma values
        #from the individual bins (under consideration of the nighttime
        #contribution)

        if nvar > 1:
    
            rimin = np.round((mris - maxsig * mrdis).min(), decimals=ndec)
            rimax = np.round((mris + maxsig * mrdis).max(), decimals=ndec)
            ria = np.arange(rimin, rimax + delri, delri)

            wa = np.zeros(len(ria))
            for j in np.arange(nvar):
                wa = wa + w[j] * gauss(ria, mris[j], mrdis[j])
            wa = wa / wa.sum()

            ri[i] = np.average(ria, weights=wa)
            rdi[i] = np.sqrt(np.sum(wa * np.square(ria - ri[i])))

    #Create output table with scaling factors
    scalfac = Table([varids, ri, rdi], names=['varID','rI','rdI'],
                    dtype=['U4','f4','f4'])
    scalfac['rI'].format = '{:6.3f}'
    scalfac['rdI'].format = '{:6.3f}'

    return scalfac
  

def scalelines(lintab, scalfac, **parlist):

    '''
    Scales line intensities and their residual variability based on the
    class-dependent input factors and the van Rhijn factors for the given
    zenith angle and the listed layer heights.
    Parameters:
    lintab: astropy table with line data
    scalfac: astropy table with scaling factors
    parlist: dictionary with model parameters (only z needed)
    Returns:
    slintab: astropy table with scaled line intensities
    '''

    if len(lintab) == 0:
        slintab = Table()
        return slintab

    slintab = lintab[['lam','trans','fH2O','ID','chem','molmass','Tkin',
                      'hlayer','varID','I']]
    slintab.rename_column('I', 'I0')
    
    slintab['fvR'] = vanrhijn(parlist['z'], slintab['hlayer'])
    slintab['fvR'].format = '{:5.3f}'

    slintab = join(slintab, scalfac, keys='varID', join_type='left')
    slintab.sort('lam') #optional

    slintab['I'] = slintab['I0'] * slintab['rI'] * slintab['fvR']
    slintab['I'].format = '{:10.3e}'
    slintab['I'].unit = 'R'
    
    slintab['dI'] = slintab['I0'] * slintab['rdI'] * slintab['fvR']
    slintab['dI'].format = '{:10.3e}'
    slintab['dI'].unit = 'R'

    return slintab


def scalecont(conttab, scalfac, **parlist):

    '''
    Scales continuum fluxes and their residual variability based on the
    class-dependent input factors and the van Rhijn factors for the given
    zenith angle and the listed layer heights.
    Parameters:
    conttab: astropy table with continuum data
    scalfac: astropy table with scaling factors
    parlist: dictionary with model parameters (only z needed)
    Returns:
    sconttab: astropy table with scaled continuum fluxes
    '''

    if len(conttab) == 0:
        sconttab = Table()
        return sconttab

    sconttab = conttab.copy()
    ncont = sconttab.meta['NCONT']

    for i in range(1, ncont+1):

        stri = str(i)
        varid = sconttab.meta['VARID' + stri]
        hlayer = sconttab.meta['HLAYER' + stri]

        fvr = vanrhijn(parlist['z'], hlayer)
        
        j = np.where(scalfac['varID'] == varid)[0][0]
        ri = scalfac['rI'][j]
        rdi = scalfac['rdI'][j]

        sconttab['fc' + stri] = ri * fvr * sconttab['fcont' + stri]
        sconttab['dfc' + stri] = rdi * fvr * sconttab['fcont' + stri]

        sconttab.meta['FVR' + stri] = fvr
        sconttab.meta['RI' + stri] = ri
        sconttab.meta['RDI' + stri] = rdi
    
    return sconttab


def corratmlines(slintab, **parlist):

    '''
    Corrects line intensities for atmospheric absorption and scattering.
    The input table contains line-dependent reference transmission values
    for zenith and a reference PWV (provided in the meta data in mm), which
    are modified for the given zenith angle and PWV under consideration of
    the optical depth fraction of water vapour absorption. The latter is 
    also provided by a column of the input table. The effective extinction 
    by scattering at molecules (Rayleigh scattering) and aerosol particles 
    (Mie scattering) is estimated based on the recipes of Noll et al. 
    (2012), which depend on wavelength and zenith angle. The combined 
    absorption and scattering effects are applied to the input line 
    intensities. The intensity changes are skipped if isatm=False.
    Parameters:
    slintab: astropy table with scaled line intensities
    parlist: dictionary with model parameters
    Returns:
    cslintab: astropy table with applied atmospheric extinction
    '''
    
    if len(slintab) == 0:
        cslintab = Table()
        return cslintab

    cslintab = slintab[['lam','trans','fH2O','ID','chem','molmass','Tkin',
                        'hlayer','varID','I','dI']]

    #Skip calculations if parameter isatm=False
    if not parlist['isatm']:
        return cslintab
    
    cslintab.rename_column('I', 'I0')
    cslintab.rename_column('dI', 'dI0')

    cslintab['tabs'] = calcabs(cslintab['trans'], parlist['z'],
                               cslintab['fH2O'], parlist['pwv'],
                               cslintab.meta['PWV0'])
    cslintab['tscat'] = calcscat(cslintab['lam'],
                                 parlist['z']).astype('float32')
    cslintab['tscat'].format = '{:6.4f}'
    cslintab['tscat'].unit = ''
    cslintab['I'] = cslintab['I0'] * cslintab['tabs'] * cslintab['tscat']
    cslintab['dI'] = cslintab['dI0'] * cslintab['tabs'] * cslintab['tscat']

    return cslintab


def corratmcont(sconttab, **parlist):

    '''
    Corrects continuum fluxes for atmospheric absorption and scattering.
    The input table contains reference transmission values for zenith and a
    reference PWV (provided in the meta data in mm), which are modified for
    the given zenith angle and PWV under consideration of the optical depth
    fraction of water vapour absorption. The latter is also provided by a
    column of the input table. The effective extinction by scattering at
    molecules (Rayleigh scattering) and aerosol particles (Mie scattering) 
    is estimated based on the recipes of Noll et al. (2012), which depend on
    wavelength and zenith angle. The combined absorption and scattering
    effects are applied to the input fluxes of the different continuum
    components. The flux changes are skipped if isatm=False.
    Parameters:
    sconttab: astropy table with scaled continuum fluxes
    parlist: dictionary with model parameters
    Returns:
    csconttab: astropy table with applied atmospheric extinction
    '''
    
    if len(sconttab) == 0:
        csconttab = Table()
        return csconttab

    csconttab = sconttab.copy()

    #Skip calculations if parameter isatm=False
    if not parlist['isatm']:
        return csconttab

    csconttab['tabs'] = calcabs(csconttab['trans'], parlist['z'],
                                csconttab['fH2O'], parlist['pwv'],
                                csconttab.meta['PWV0'])
    csconttab['tscat'] = calcscat(csconttab['lam'],
                                 parlist['z']).astype('float32')
    csconttab['tscat'].format = '{:6.4f}'
    csconttab['tscat'].unit = ''
    
    ncont = csconttab.meta['NCONT']

    for i in range(1, ncont+1):

        stri = str(i)

        csconttab.remove_column('fcont' + stri)    
        csconttab.rename_column('fc' + stri, 'fc' + stri + '_0')
        csconttab.rename_column('dfc' + stri, 'dfc' + stri + '_0')

        csconttab['fc' + stri] = csconttab['fc' + stri + '_0'] * \
            csconttab['tabs'] * csconttab['tscat']
        csconttab['dfc' + stri] = csconttab['dfc' + stri + '_0'] * \
            csconttab['tabs'] * csconttab['tscat']

    return csconttab


def calclinspec(cslintab, **parlist):

    '''
    Calculates line spectrum.
    The input line intensities are converted into line fluxes under the
    assumption of Doppler broadening. The widths of the related Gaussians 
    are derived from table columns containing a reference kinetic 
    temperature and the molecular weight. The central wavelengths depend on 
    the parameter isair, i.e. whether air or vacuum wavelengths are desired. 
    The output wavelength grid is defined by the parameters lammin, lammax, 
    and dlam (plus additional temporal margins for the subsequent 
    smoothing). The line-specific Gaussians are averaged for each bin of the 
    grid. Finally, the spectra of all lines are summed. The residual line 
    variability from the input data is also converted into a spectrum. As 
    the contributions of the different lines are also summed, perfect 
    correlation of the variations of the blended lines is assumed. This is a 
    simple but reasonable assumption for most wavelength bins as lines with 
    similar wavelengths tend to have a similar origin. Otherwise the 
    variability tends to be overestimated. For fast calculations, a Cython 
    helper function is used if the module is compiled.
    Parameters:
    cslintab: astropy table with scaled line intensities
    parlist: dictionary with model parameters
    Returns:
    linspec: astropy table with line spectrum (and its variation)
    '''

    #Empty output table without selected lines
    nlin = len(cslintab)
    if nlin == 0:
        linspec = Table()
        return linspec

    #Constants
    tol = 1e-7       #tolerance (Gaussian)
    const = 9.618e-9 #constant for Doppler width [sqrt(k_B*N_A/c^2)]
    maxsig = 5.0     #maximum integration range for Gaussian in sigma
    dsig = 0.01      #bin width in sigma for integration of Gaussian
    umtonm = 1e-3    #µm to nm

    #Get wavelength limits of extended wavelength grid
    lammin = parlist['lammin'] - cslintab.meta['NDELMIN'] * parlist['dlam']
    lammax = parlist['lammax'] + cslintab.meta['NDELMAX'] * parlist['dlam']
    
    #Create arrays for output table
    lamgrid = np.arange(lammin, lammax + tol, parlist['dlam'],
                        dtype='float64')
    nlam = len(lamgrid)
    fluxes = np.zeros(nlam, dtype='float32')
    dfluxes = np.zeros(nlam, dtype='float32')

    #Calculate basic Gaussian 
    x = np.arange(-maxsig, maxsig + tol, dsig, dtype='float32')
    y = gauss(x, 0., 1.).astype('float32')
    y = y / y.sum()

    #Central wavelengths and widths of lines
    if parlist['isair']:
        lam0 = np.array(cslintab['lam'] / refindex(cslintab['lam']),
                        dtype='float64')
    else:
        lam0 = np.array(cslintab['lam'], dtype='float64')        
    siglam = np.array(const * \
                      np.sqrt(cslintab['Tkin'] / cslintab['molmass']) * lam0)

    #Gaussian parameters in coordinates of output wavelength grid
    pos0 = (lam0 - lamgrid[0]) / parlist['dlam'] + 0.5
    rsig = siglam / parlist['dlam']

    #Conversion from R/bin to R/nm
    bintonm = umtonm / parlist['dlam']
    ynm = y * bintonm

    #Arrays for input
    ia = np.array(cslintab['I'], dtype='float32')
    dia = np.array(cslintab['dI'], dtype='float32')
       
    #Calculate spectrum for each line and add them to the total spectrum
    if cython.compiled:
        #with Cython
        fluxes, dfluxes = _addlines_c(fluxes, dfluxes, ia, dia, x, ynm, pos0,
                                      rsig)
    else:
        #without Cython 
        fluxes, dfluxes = _addlines_p(fluxes, dfluxes, ia, dia, x, ynm, pos0,
                                      rsig)
        
    #Create output table
    lamform = '{:9.7f}'
    fluxform = '{:11.4e}'              
    linspec = Table([lamgrid, fluxes, dfluxes], names=['lam','flux','dflux'],
                    dtype=['f8','f4','f4'])
    linspec['lam'].format = lamform
    linspec['lam'].unit = 'um'
    linspec['flux'].format = fluxform
    linspec['flux'].unit = 'R/nm'    
    linspec['dflux'].format = fluxform
    linspec['dflux'].unit = 'R/nm'    
    
    #Return resulting spectrum
    return linspec


def calccontspec(csconttab, **parlist):

    '''
    Calculates continuum spectrum.
    First, the spectra of all continuum components and their residual
    variabilities are summed. Although the latter would require perfect
    correlations of the variations to avoid an overestimation of the
    variability, this is a minor issue for most wavelengths as there is a
    clearly dominating component in most cases. Then, the wavelength grid of
    the input data is mapped to the output wavelength grid defined by the
    parameters lammin, lammax, and dlam (plus additional temporal margins 
    for the subsequent smoothing). Depending on the bin sizes in the input 
    and output data, the procedure involves either rebinning or 
    interpolation of the fluxes. The parameter isair (air or vacuum 
    wavelengths) is considered. However, it only matters for the atmospheric 
    absorption of the continuum as the continuum templates are highly 
    smoothed. 
    Parameters:
    csconttab: astropy table with scaled continuum fluxes
    parlist: dictionary with model parameters
    Returns:
    contspec: astropy table with continuum spectrum (and its variation)
    '''

    #Empty output table without selected continuum components
    if len(csconttab) == 0:
        contspec = Table()
        return contspec

    #Constants
    tol = 1e-8     #tolerance (wavelength)
    rdlamsep = 1.5 #limiting wavelength step ratio (out vs. in) for rebinning
                   #(higher) and interpolation (lower)

    #Get wavelength limits of extended wavelength grid
    lammin = parlist['lammin'] - csconttab.meta['NDELMIN'] * parlist['dlam']
    lammax = parlist['lammax'] + csconttab.meta['NDELMAX'] * parlist['dlam']
    
    #Create arrays for output table
    lamgrid = np.arange(lammin, lammax + tol, parlist['dlam'],
                        dtype='float64')
    llamgrid = lamgrid - 0.5 * parlist['dlam']
    nlam = len(lamgrid)
    fluxes = np.zeros(nlam, dtype='float32')
    dfluxes = np.zeros(nlam, dtype='float32')

    #Create output table
    lamform = '{:9.7f}'
    fluxform = '{:11.4e}'              
    contspec = Table([lamgrid, fluxes, dfluxes], names=['lam','flux','dflux'],
                     dtype=['f8','f4','f4'])
    contspec['lam'].format = lamform
    contspec['lam'].unit = 'um'

    #Array of input wavelengths depending on isair
    if parlist['isair']:
        inlam = np.array(csconttab['lam'] / refindex(csconttab['lam']))
    else:
        inlam = np.array(csconttab['lam'])

    #Step size of input wavelengths    
    nin = len(csconttab)
    if nin == 1:
        indlam = 1 / tol
    else:    
        indlam = (inlam[-1] - inlam[0]) / (nin - 1)   

    #Array with fluxes and variations of the summed continuum components    
    influx = np.zeros(nin, dtype='float32')     
    indflux = np.zeros(nin, dtype='float32')         
    ncont = csconttab.meta['NCONT']
    for i in range(1, ncont+1):
        stri = str(i)
        influx = influx + csconttab['fc' + stri]
        indflux = indflux + csconttab['dfc' + stri]

    #Rebinning or interpolation?    
    rdlam = parlist['dlam'] / indlam
    if rdlam < rdlamsep:
        #Linear interpolation for lower step size in output spectrum
        contspec['flux'] = np.interp(lamgrid, inlam, influx)
        contspec['dflux'] = np.interp(lamgrid, inlam, indflux)   
    else:
        contspec['flux'], w = rebin(lamgrid, inlam, influx)
        contspec['dflux'], w = rebin(lamgrid, inlam, indflux)

    #Format for output table columns
    contspec['flux'] = contspec['flux'].astype('float32')
    contspec['flux'].format = fluxform
    contspec['flux'].unit = 'R/nm'    
    contspec['dflux'] = contspec['dflux'].astype('float32')
    contspec['dflux'].format = fluxform
    contspec['dflux'].unit = 'R/nm'
        
    #Return resulting spectrum
    return contspec


def addlinescont(linspec, contspec, **parlist):

    '''
    Adds line and continuum spectrum.
    Both spectra need to have the same wavelength grid. If one of the 
    spectrum tables is empty, it is ignored. If there are no data in both
    cases, the requested wavelength grid defined by the parameters lammin,
    lammax, and dlam is returned with a flux of zero.
    Parameters:
    linspec: astropy table with line spectrum
    contspec: astropy table with continuum spectrum
    parlist: dictionary with model parameters
    Returns:
    addspec: astropy table with added line and continuum spectrum
    '''

    #Constants
    tol = 1e-8

    #Prepare check of input wavelength grids
    nlamlin = len(linspec)
    nlamcont = len(contspec)
    if (nlamlin > 0) & (nlamcont > 0):
        delmin = np.abs(linspec['lam'].min() - contspec['lam'].min())
        delmax = np.abs(linspec['lam'].max() - contspec['lam'].max())
    else:
        delmin = 0.
        delmax = 0.   

    #Check input wavelength grids
    if (nlamlin > 0) & (nlamcont > 0):
        if (nlamlin != nlamcont) | (delmin > tol) | (delmax > tol):
            print('Line and continuum spectra: different wavelength grids!')
            match = False
        else:
            match = True
    else:
        match = True

    #Add spectra or handle cases with empty tables
    if (match == False) | ((nlamlin == 0) & (nlamcont == 0)):
        lamgrid = np.arange(parlist['lammin'], parlist['lammax'] + tol,
                            parlist['dlam'], dtype='float64')
        nlam = len(lamgrid)
        fluxes = np.zeros(nlam, dtype='float32')
        dfluxes = np.zeros(nlam, dtype='float32')
        lamform = '{:9.7f}'
        fluxform = '{:11.4e}'              
        addspec = Table([lamgrid, fluxes, dfluxes],
                        names=['lam','flux','dflux'],
                        dtype=['f8','f4','f4'])
        addspec['lam'].format = lamform
        addspec['lam'].unit = 'um'
        addspec['flux'].format = fluxform
        addspec['flux'].unit = 'R/nm'    
        addspec['dflux'].format = fluxform
        addspec['dflux'].unit = 'R/nm'
    elif (match == True) & ((nlamlin > 0) & (nlamcont == 0)):
        addspec = linspec
    elif (match == True) & ((nlamlin == 0) & (nlamcont > 0)):   
        addspec = contspec
    elif (match == True) & ((nlamlin > 0) & (nlamcont > 0)):
        addspec = linspec.copy()
        addspec['flux'] = addspec['flux'] + contspec['flux']
        addspec['dflux'] = addspec['dflux'] + contspec['dflux']

    #Return resulting spectrum
    return addspec


def convolvelsf(addspec, **parlist):

    '''
    Convolves Gaussian line-spread function.
    The relative full width at half maximum of the Gaussian is derived from 
    the given resolving power (parameter resol). The absolute width 
    therefore scales with the wavelength. The wavelength-dependent 
    line-spread functions are applied to the flux and flux variability of 
    the input spectrum in order to simulate limitations in the instrumental 
    resolution. For fast calculations, a Cython helper function is used.
    Parameters:
    addspec: astropy table with combined line and continuum spectrum 
             (showing natural resolution in the case of the line emission)
    parlist: dictionary with model parameters
    Returns:
    spec: astropy table with convolved spectrum
    '''

    #Constants
    maxresol = 1e7 #maximum resolving power to be considered
    maxsig = 5.0 #maximum sigma of Gaussian
    dsig = 1e-5 #step size for Gaussian
    conv = 0.5 / np.sqrt(2 * np.log(2)) #FWHM to sigma
    tol = 1e-8 #tolerance for wavelength
    
    #Copy spectrum
    spec = addspec.copy()
    
    #Case of no data
    if spec['dflux'].max() == 0:
        return spec

    #No convolution in the case of extremely high resolving power
    if parlist['resol'] > maxresol:
        return spec

    #Create input and output arrays for calculations
    lam = np.array(addspec['lam'], dtype='float64')
    flux0 = np.array(addspec['flux'], dtype='float32')
    dflux0 = np.array(addspec['dflux'], dtype='float32')

    #Get relevant adjacent pixels for each pixel 
    ni = len(spec)
    siglam = conv * lam / parlist['resol']
    rsig = (parlist['dlam'] / siglam).astype('float32')
    ia = np.arange(ni)
    maxrsig = maxsig / rsig
    imin = (np.floor(ia - maxrsig)).astype('int32')
    imin = np.maximum(imin, 0)
    imax = (np.ceil(ia + maxrsig)).astype('int32')
    imax = np.minimum(imax, ni-1)
    
    #Approximate error function for integration of Gaussian
    xerf = np.arange(-maxsig, maxsig + dsig, dsig, dtype='float64')
    xerf0 = xerf[0].astype('float32')
    yerf = 0.5 * erf(xerf).astype('float32')
    dxerf = dsig
    
    #Calculate flux contributions to each bin from other bins using weights
    #from Gaussian kernel
    #(slight approximation: fixed width of Gaussian for each contributing bin)
    if cython.compiled:
        #if compiled
        flux, dflux = _convolve_c(flux0, dflux0, imin, imax, rsig, xerf0,
                                  dxerf, yerf)
    else:
        #if not compiled
        flux, dflux = _convolve_p(flux0, dflux0, imin, imax, rsig, xerf0,
                                  dxerf, yerf)
                
    #Write convolved fluxes into output table    
    spec['flux'] = flux
    spec['flux'].format = addspec['flux'].format
    spec['flux'].unit = addspec['flux'].unit
    spec['dflux'] = dflux
    spec['dflux'].format = addspec['dflux'].format
    spec['dflux'].unit = addspec['dflux'].unit
        
    #Cut margins in final spectrum
    spec = spec[(spec['lam'] > parlist['lammin'] - tol) & \
                (spec['lam'] < parlist['lammax'] + tol)]
        
    #Return final spectrum
    return spec


def isnight(month, time):

    '''
    Checks for nighttime (solar zenith angle > 100°) at Cerro Paranal.
    Depending on month and local time (solar mean time), True or False is
    returned.
    Parameters:
    month: month number (1 to 12)
    time: local time in hours (valid ranges: -12 to 12 or 0 to 24)
    Returns:
    isnight: True or False
    '''

    ltlim = [-4.36, -4.57, -5.01, -5.52, -5.86, -5.96, -5.83, -5.63, -5.46,
             -5.24, -4.91, -4.52]
    utlim = [ 4.61,  5.03,  5.32,  5.55,  5.75,  5.95,  6.01,  5.79,  5.33,
              4.81,  4.40,  4.31]

    if time >= 12.:
        time = time - 24.

    i = month - 1            
    if (time >= ltlim[i]) & (time <= utlim[i]):
        isnight = True
    else:  
        isnight = False

    return isnight   

 
def refindex(lam):

    '''
    Refractive index of air under standard conditions for given wavelength
    according to Edlen (1966).
    Parameters:
    lam: wavelength(s) in µm
    Returns:
    n: refractive index/indices of air at given wavelength(s)
    '''

    sig2 = 1 / np.square(lam)
    n = 8342.13 + 2406030 / (130 - sig2) + 15997 / (38.9 - sig2)
    n = 1 + 1e-8 * n
    
    return n


def gauss(x, m, s):

    '''
    Calculates Gaussian.
    Parameters:
    x: independent coordinate(s)
    m: mean (i.e. centre)
    s: sigma (i.e. width)
    Returns:
    gauss: Gaussian
    '''

    norm = 1 / (s * np.sqrt(2 * np.pi))
    gauss = norm * np.exp(-0.5 * np.square((x - m) / s))
    
    return gauss


def vanrhijn(z, hlayer):

    '''
    Calculates van Rhijn correction factor depending on zenith angle and
    layer height.
    Parameters:
    z: zenith angle in deg (single value or numpy array)
    hlayer: layer height in km (single value or numpy array)
    Returns:
    fvr: correction factor (>= 1) (single value or numpy array)
    '''

    re = 6371. #average Earth radius in km
    
    zrad = z * np.pi / 180.
    fvr = 1 / np.sqrt(np.where(hlayer < 0., 1.,
                               1. - np.square((re / (re + hlayer)) *
                                               np.sin(zrad))))

    if fvr.size == 1:
        fvr = fvr.item()
    
    return fvr


def calcabs(tabs0, z, fwv, pwv, pwv0):

    '''
    Calculates atmospheric absorption depending on zenith angle and amount
    of precipitable water vapour (PWV) based on a reference transmission
    curve for zenith and a given PWV. The quality of the approximation
    depends on the deviation of the conditions from the reference. All
    parameters can be provided as single values or as numpy arrays with the
    same size.
    Parameters:
    tabs0: transmission at zenith for reference PWV (pwv0) (value or array) 
    z: zenith angle in deg (value or array)
    fwv: fraction of optical depth by water vapour (value or array)
    pwv: PWV in mm (value or array)
    pwv0: reference PWV in mm (value or array)
    Returns:
    tabs: modified transmission (value or array)
    '''

    #Airmass according to Rozenberg (1966)
    zrad = z * np.pi / 180.
    xz = 1 / (np.cos(zrad) + 0.025 * np.exp(-11 * np.cos(zrad)))

    #Relative PWV difference compared to reference
    rpwv = (pwv - pwv0) / pwv0

    #Approximative transmission by absorption
    tabs = np.power(tabs0, (1. + rpwv * fwv) * xz)

    return tabs

                    
def calcscat(lam, z):

    '''
    Calculates airglow transmission by scattering for given wavelength and
    zenith angle based on the recipes of Noll et al. (2012), which were
    derived for typical conditions at Cerro Paranal and an emission layer at
    an altitude of 90 km.
    Parameters:
    lam: wavelength in µm (value or array)
    z: zenith angle in deg (value or array)
    Returns:
    tscat: transmission by scattering (value or array)
    '''

    #Constants
    
    #Rayleigh scattering (Liou 2002, P. 352; Noll et al. 2012) 
    a = 0.00864 #coefficient
    b = 6.5e-6 #coefficient
    c = 3.916 #coefficient
    d = 0.074 #coefficient
    e = 0.050 #coefficient
    h = 2.64 #reference height in km (Cerro Paranal)
    p = 744 #reference pressure in hPa (Cerro Paranal)
    ps = 1013.25 #standard pressure in hPa    
    
    #Aerosol scattering (Patat et al. 2011; Noll et al. 2012)
    k0mag = 0.014 #aerosol extinction coefficient at 1 µm in magnitudes
    lam0 = 0.4 #cut for aerosol extinction power law in µm
    alpha = -1.38 #Angstrom coefficient for aerosol extinction
    
    #Dependence of extinction reduction by scattering into line of sight on
    #airmass (Noll et al. 2012)
    cr = -0.146 #constant of fit for Rayleigh scattering
    mr = 1.669 #slope of fit for Rayleigh scattering
    cm = -0.318 #constant of fit for aerosol (Mie) scattering
    mm = 1.732 #constant of fit for aerosol (Mie) scattering

    #Calculations
    
    #Magnitudes into natural units
    k0 = 0.4 * np.log(10) * k0mag

    #Rayleigh scattering (Liou 2002, P. 352):
    taur = (a + b * h) * (lam ** (-c - d * lam - e / lam)) * p / ps

    #Aerosol (Mie) scattering (Patat et al. 2011):
    taum = k0 * np.power(np.where(lam < lam0, lam0, lam), alpha)

    #Airmass (Rozenberg 1966):
    zrad = z * np.pi / 180.
    x = 1 / (np.cos(zrad) + 0.025 * np.exp(-11 * np.cos(zrad)))

    #"Airmass" for layer at 90 km (van Rhijn)
    xvr = 1 / np.sqrt(1 - 0.972 * np.sin(zrad) ** 2)

    #Airglow extinction reduction (Noll et al. 2012):
    lgx = np.log10(xvr)
    fextr = mr * lgx + cr
    fextm = mm * lgx + cm

    #Transmission by airglow scattering
    tr = np.exp(-fextr * taur * x)
    tm = np.exp(-fextm * taum * x)
    tscat = tr * tm
    
    return tscat


def rebin(xout, xin, yin):

    '''
    Rebin yin with grid xin to grid xout.
    xout needs to have a constant step size. This is not required for xin. 
    Moreover, the step size of xout needs to be larger than all step sizes 
    of xin. The function returns the weighted average of the input bins
    contributing to the different output bins. If the extent of xout is
    larger than in the case of xin, constant extrapolation is performed. For
    fast calculations, a Cython helper function is used.
    Parameters:
    xout: output grid
    xin: input grid
    yin: values to be rebinned
    Returns:
    yout: rebinned values
    w: coverage of xout bins by xin grid (0 to 1) 
    '''

    #Tolerance
    tol = 1e-7

    #Output grid
    xout = np.array(xout)
    nout = xout.size
    yout = np.zeros(nout)
    w = np.zeros(nout)

    #Input grid
    xin = np.array(xin)
    yin = np.array(yin)
    nin = xin.size

    #Check sizes of xin and yin
    if yin.size != nin:
        print('Rebinning: sizes of xin and yin do not match!') 
        return yout, w

    #Check overlap of xin and xout grids
    if (xin.max() < xout.min()) | (xin.min() > xout.max()):
        print('Rebinning: no overlap of input and output grids!') 
        return yout, w

    #Get step sizes of xin and xout grids
    dxout = (xout[-1] - xout[0]) / (nout - 1) 
    pos = (xin - xout[0]) / dxout + 0.5
    dxin = xin[1:] - xin[:-1]
    dxinl = np.append(dxin[-1], dxin)
    dxinu = np.append(dxin, dxin[-1])

    #Check whether step size of xout grid is sufficiently large
    if dxin.max() > dxout:
        print('Rebinning: output step size too small!')
        return yout, w

    #Link positions in input and output grids
    #(xin bin intervals as positions in xout grid)
    rdl = 0.5 * dxinl / dxout
    rdu = 0.5 * dxinu / dxout
    posl = pos - rdl
    posu = pos + rdu
    posli = np.floor(posl).astype('int')
    posui = np.floor(posu).astype('int')

    #Get weights for input values contributing to each xout bin
    wl = np.where(posui == posli+1, posui - posl, rdl)
    wu = np.where(posui == posli+1, posu - posui, rdu)
    wly = wl * yin
    wuy = wu * yin

    #Perform weighted averaging for yout values using a cython function
    #(also calculate summed weights for each xout bin)
    yout, w = _rebin_c(yout, w, posli, wl, wly)
    yout, w = _rebin_c(yout, w, posui, wu, wuy)
    yout = np.array(yout)
    w = np.array(w)
       
    #Find xout bins without full coverage by the xin grid        
    jfull = np.where(w > 1. - tol)[0]
    jmin = jfull[0]
    jmax = jfull[-1]

    #Correct or extrapolate yout at lower margin
    if jmin > 0:
        if w[jmin-1] > tol:
            yout[jmin-1] = yout[jmin-1] / w[jmin-1]
        else:
            yout[jmin-1] = yout[jmin]
        if jmin > 1:
            yout[:jmin-1] = yout[jmin-1]
            
    #Correct or extrapolate yout at upper margin
    if jmax < nout - 1:
        if w[jmax+1] > tol:
            yout[jmax+1] = yout[jmax+1] / w[jmax+1]
        else:
            yout[jmax+1] = yout[jmax]
        if jmax < nout - 2:
            yout[jmax+2:] = yout[jmax+1]        

    #Return rebinned output values and summed weights
    return yout, w


def erf(x):

    '''
    Simple approximation of the error function erf using hyperbolic 
    tangent. According to Wikipedia, the maximum deviation amounts to
    0.000358.
    Parameters:
    x: input value x
    Returns:
    y: erf(x)
    '''

    term1 = 2. / np.sqrt(np.pi)
    term2 = 11. / 123.
    y = np.tanh(term1 * (x + term2 * np.power(x, 3)))

    return y


@cython.cfunc
def _addlines_c(fluxes: cython.float[:], dfluxes: cython.float[:],
                ia: cython.float[:], dia: cython.float[:], x: cython.float[:],
                ynm: cython.float[:], pos0: cython.double[:],
                rsig: cython.double[:]):

    '''
    Adds Gaussian lines to spectrum. Uses Cython for higher speed.
    Parameters:
    fluxes: array of zeros for spectrum
    dfluxes: array of zeros for variability of spectrum
    ia: array of line intensities
    dia: array of variability of line intensities
    x: array of coordinates for Gaussian
    ynm: array of values of Gaussian combined with a factor for the 
         conversion from intensity (R) to flux (R/nm)
    pos0: array of line positions in pixels of output spectrum
    rsig: array of Gaussian width (sigma) in pixels of output spectrum
    Returns:
    fluxes: array with output spectrum
    dfluxe: array with variability of output spectrum         
    '''

    nlin: cython.int = ia.shape[0]
    nx: cython.int = x.shape[0]
    nlam: cython.int = fluxes.shape[0]
    i: cython.int = 0
    j: cython.int
    k: cython.int
    flux: cython.float
    dflux: cython.float

    while i < nlin:
        j = 0
        while j < nx:
            flux = ynm[j] * ia[i]
            dflux = ynm[j] * dia[i]
            k = (int) ((rsig[i] * x[j] + pos0[i]) // 1)
            if (k >= 0) & (k < nlam):
                fluxes[k] += flux
                dfluxes[k] += dflux
            j += 1    
        i += 1        

    return fluxes, dfluxes


def _addlines_p(fluxes, dfluxes, ia, dia, x, ynm, pos0, rsig):

    '''
    Adds Gaussian lines to spectrum. Version without Cython.
    Parameters:
    fluxes: array of zeros for spectrum
    dfluxes: array of zeros for variability of spectrum
    ia: array of line intensities
    dia: array of variability of line intensities
    x: array of coordinates for Gaussian
    ynm: array of values of Gaussian combined with a factor for the 
         conversion from intensity (R) to flux (R/nm)
    pos0: array of line positions in pixels of output spectrum
    rsig: array of Gaussian width (sigma) in pixels of output spectrum
    Returns:
    fluxes: array with output spectrum
    dfluxe: array with variability of output spectrum         
    '''

    nlin = ia.shape[0]
    nx = x.shape[0]
    nlam = fluxes.shape[0]

    #Loop over each line

    for i in np.arange(nlin):

        #Fluxes for grid of Gaussian
        flux = ynm * ia[i]
        dflux = ynm * dia[i]

        #Combine grid of Gaussian and output wavelength grid by derivation
        #of related indices of both grids
        karr = np.floor(rsig[i] * x + pos0[i]).astype('int')
        ks, js = np.unique(karr, return_index=True)
        js = np.append(js, nx)

        #Add flux and its uncertainty for each relevant bin of the output
        #wavelength grid
        for h, k in enumerate(ks):
            if (k >= 0) & (k < nlam):
                fluxes[k] = fluxes[k] + flux[js[h]:js[h+1]].sum()
                dfluxes[k] = dfluxes[k] + dflux[js[h]:js[h+1]].sum()

    return fluxes, dfluxes


@cython.cfunc
def _rebin_c(yout: cython.double[:], wout: cython.double[:],
             posi: cython.long[:], w: cython.double[:], wy: cython.double[:]):

    '''
    Helper function for rebinning of a spectrum. Uses Cython for higher 
    speed.
    Parameters:
    yout: array for output spectrum
    wout: array for summed weights
    posi: array connecting input and output pixels
    w: array of input weights
    wy: array of multiplied input weights and fluxes 
    Returns:
    yout: array of output spectrum
    wout: array of summed weights
    '''

    i: cython.int = 0
    nin: cython.int = posi.shape[0]
    nout: cython.int = yout.shape[0]
    
    while i < nin:
        if (posi[i] >= 0) & (posi[i] < nout):
            yout[posi[i]] = yout[posi[i]] + wy[i]
            wout[posi[i]] = wout[posi[i]] + w[i]
        i = i + 1

    return yout, wout


@cython.cfunc
def _convolve_c(flux0: cython.float[:], dflux0: cython.float[:],
                imin: cython.int[:], imax: cython.int[:],
                rsig: cython.float[:], xerf0: cython.float,
                dxerf: cython.float, yerf: cython.float[:]):

    '''
    Convolves a spectrum and its variability with a Gaussian kernel with 
    variable width. Uses Cython for higher speed. The approach is to 
    calculate flux contributions to each bin from other bins using weights
    from the Gaussian integrated within the bin-related limits. The 
    integration is performed by means of a precalculated error function. 
    Parameters:
    flux0: array of input flux
    dflux0: array of input flux variability
    imin: array of lower limits of the pixel intervals for each bin
    imax: array of upper limits of the pixel intervals for each bin
    rsig: array of Gaussian widths (sigma) relative to bin size
    xerf0: first coordinate of precalculated error function
    dxerf: step size of coordinates of precalculated error function
    yerf: array of half values of precalculated error function
    Returns:
    flux: array of convolved flux
    dflux: array of convolved flux variability
    '''

    ni: cython.int = flux0.shape[0]
    flux: cython.float[:] = np.zeros(ni, dtype='float32')
    dflux: cython.float[:] = np.zeros(ni, dtype='float32')
    nerf: cython.int = yerf.shape[0]
    i: cython.int
    j: cython.int
    dij: cython.float
    x: cython.float
    errfac: cython.float = np.sqrt(0.5)
    xerfmin: cython.float = xerf0 - 0.5 * dxerf
    ierf: cython.int
    ierfl: cython.int = 0
    ierfu: cython.int = 0
    p: cython.float

    for i in range(ni):

        #Consider only relevant pixels for convolution

        for j in range(imin[i], imax[i]+2):
            
            #Pixel positions for i-centred convolution but negative definition
            #of x (decreasing order) provides correct results
            dij = (float) (i - j) 
            x = (dij + 0.5) * rsig[i]
            
            #Get position in array of error function values
            ierf = (int) ((x * errfac - xerfmin) // dxerf)
            if ierf < 0:
                ierf = 0
            elif ierf >= nerf:
                ierf = nerf - 1
                
            #Get weights for each relevant pixel from integration of error
            #function and add weighted fluxes for each output pixel 
            if j == imin[i]:
                ierfl = ierf
            else:
                ierfu = ierf
                p = yerf[ierfl] - yerf[ierfu]
                flux[i] += flux0[j-1] * p
                dflux[i] += dflux0[j-1] * p
                ierfl = ierfu
                
    return flux, dflux


def _convolve_p(flux0, dflux0, imin, imax, rsig, xerf0, dxerf, yerf):

    '''
    Convolves a spectrum and its variability with a Gaussian kernel with 
    variable width. Version without Cython. The approach is to calculate 
    flux contributions to each bin from other bins using weights from the 
    Gaussian integrated within the bin-related limits. The integration is 
    performed by means of a precalculated error function. 
    Parameters:
    flux0: array of input flux
    dflux0: array of input flux variability
    imin: array of lower limits of the pixel intervals for each bin
    imax: array of upper limits of the pixel intervals for each bin
    rsig: array of Gaussian widths (sigma) relative to bin size
    xerf0: first coordinate of precalculated error function
    dxerf: step size of coordinates of precalculated error function
    yerf: array of half values of precalculated error function
    Returns:
    flux: array of convolved flux
    dflux: array of convolved flux variability
    '''

    ni = flux0.shape[0]
    flux = np.zeros(ni, dtype='float32')
    dflux = np.zeros(ni, dtype='float32')
    nerf = yerf.shape[0]
    errfac = np.sqrt(0.5)
    xerfmin = xerf0 - 0.5 * dxerf

    for i in np.arange(ni):

        #Consider only relevant pixels for convolution
            
        #Pixel positions for i-centred convolution but negative definition
        #of x (decreasing order) provides correct results
        dij = (i - np.arange(imin[i], imax[i]+2)).astype('float32') 
        x = (dij + 0.5) * rsig[i]
            
        #Get position in array of error function values
        ierf = ((x * errfac - xerfmin) // dxerf).astype('int32')
        ierf = np.clip(ierf, 0, nerf - 1)
                
        #Get weights for each relevant pixel from integration of error
        #function
        p = yerf[ierf[:-1]] - yerf[ierf[1:]]

        #Add weighted fluxes for each output pixel
        flux[i] = np.dot(flux0[imin[i]:imax[i]+1], p) 
        dflux[i] = np.dot(dflux0[imin[i]:imax[i]+1], p) 
                       
    return flux, dflux
