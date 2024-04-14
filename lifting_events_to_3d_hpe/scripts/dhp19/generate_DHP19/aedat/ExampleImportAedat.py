# -*- coding: utf-8 -*-
"""
Example script for how to invoke the ImportAedat function
"""


import ImportAedat

# Create a dict with which to pass in the input parameters.
args = {}

# Put the filename, including full path, in the 'file' field.
args['filePathAndName'] = '/home/project/example3.aedat'  # Linux
# args['filePathAndName'] = 'N:\project\example3.aedat' # Windows

# Put the filename, including full path, in the 'file' field.
# Alternatively, make sure the file is already on the matlab path.
import sys

# sys.path.append('/home/project/')

# Add any restrictions on what to read out.
# This example limits readout to the first 1M events (aedat fileFormat 1 or 2 only):
# args['startEvent'] = 10e6;
# args['endEvent'] = 20e6;

# This example limits readout to a time window between 48.0 and 48.1 s:
# args['startTime'] = 48;
# args['endTime'] = 48.1;

# This example only reads out from packets 1000 to 2000 (aedat3.x only)
# args['startPacket'] = 1000;
args['endPacket'] = 2000

# This example limits the read out to certain types of event only
# args['dataTypes'] = {'polarity', 'special'};
args['dataTypes'] = {'special'}
# args['dataTypes'] = {'frame'};

# Setting the dataTypes empty tells the function to not import any data;
# You get the header info, plus packet indices info for Aedat3.x

# AEDAT3.x only - exclude non-valid events from the output
args['validOnly'] = True

# Working with a file where the source hasn't been declared - do this explicitly:
# args['source'] = 'Davis240b';

# Invoke the function
output = ImportAedat.ImportAedat(args)
