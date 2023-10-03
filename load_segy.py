import os
import obspy as op


def loadshots(directory_path):
    """
    Loads all .su files in a directory into a dictionary of ObsPy Stream objects.
    """
    streamdict = {}  # Initialize an empty dictionary.
    # Loop over all files in the directory.
    for filename in os.listdir(directory_path):
        if filename.endswith(".su"):  # Make sure we're only reading .su files.

            # Generate the full file path.
            full_path = os.path.join(directory_path, filename)

            # Read the .su file into an ObsPy Stream object.
            stream = op.read(full_path, format="SU", unpack_trace_headers=True)

            # Store the Stream object in the dictionary.
            streamdict[filename] = stream

    return streamdict
