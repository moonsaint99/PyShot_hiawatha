# Active Source Seismic Processing Toolbox

This toolbox contains a number of python scripts designed to let you take active source shot records stored in Seismic
Unix format and do amplitude analysis on them in various ways.

## Summary of Contents

* load_segy.py: This can turn a folder of SU files into a dictionary containing ObsPy stream objects, each containing
traces corresponding to each trace in the shot record. It includes SU headers. 