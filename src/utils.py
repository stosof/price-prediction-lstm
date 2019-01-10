import numpy as np

def write_3d_np_array_to_file(output_filepath, data):
    with open(output_filepath, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(data.shape))
        for data_slice in data:
            np.savetxt(outfile, data_slice, fmt='%-7.2f')
            outfile.write('# New slice\n')