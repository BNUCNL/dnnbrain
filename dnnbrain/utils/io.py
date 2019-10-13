import numpy as np

from collections import OrderedDict


def read_stim_csv(fpath):
    """
    Read pre-designed .stim.csv file.

    Parameters:
    -----------
    fpath[str]: Path of .stim.csv file.
        Format of .stim.csv of picture stimuli is
        --------------------------
        title:picture stimuli
        type:picture
        path:parent_dir_to_pictures
        [Several optional keys] (eg., hrf_tr:2)
        stim:stimID,[onset],[duration],[condition]
        meas:accuracy, reaction time
        pic1_path,0,1,cat,0.4,0.5
        pic2_path,1,1,dog,0.6,0.4
        pic3_path,2,1,cat,0.7,0.5
        ...,...,...,...,...,...

        Format of .stim.csv of video stimuli is
        --------------------------
        title:video stimuli
        type:video
        path:path_to_video_file
        [Several optional keys] (eg., hrf_tr:2)
        stim:stimID,[onset],[duration],[condition]
        meas:accuracy, reaction time
        1,0,1,cat,0.4,0.5
        2,1,1,dog,0.6,0.4
        3,2,1,cat,0.7,0.5
        ...,...,...,...,...,...

    Return:
    -------
    stim_dict[dict]: Dictionary of the output variable
    """
    # -load csv data-
    assert fpath.endswith('.stim.csv'), "File suffix must be .stim.csv"
    with open(fpath) as rf:
        lines = rf.read().splitlines()
    # remove null line
    while '' in lines:
        lines.remove('')
    stim_idx = [line.startswith('stim=') for line in lines].index(True)
    meas_idx = stim_idx + 1
    meta_lines = lines[:stim_idx]
    var_lines = lines[meas_idx+1:]

    # -handle csv data-
    # --operate meta_lines--
    stim_dict = {}
    for line in meta_lines:
        k, v = line.split('=')
        stim_dict[k] = v
    assert 'title' in stim_dict.keys(), "'title' needs to be included in meta data."
    assert 'type' in stim_dict.keys(), "'type' needs to be included in meta data."
    assert 'path' in stim_dict.keys(), "'path' needs to be included in meta data."

    # --operate var_lines--
    # prepare keys
    stim_keys = lines[stim_idx].split('=')[1].split(',')
    assert 'stimID' in stim_keys, "'stimID' must be included in 'stim=' line."
    n_stim_key = len(stim_keys)
    meas_keys = lines[meas_idx].split('=')[1].split(',')
    while '' in meas_keys:
        meas_keys.remove('')

    # prepare variable data
    var_data = [line.split(',') for line in var_lines]
    var_data = list(zip(*var_data))
    if stim_dict['type'] == 'picture':
        # data type for stimID or condition is str, others float.
        for i, v in enumerate(var_data[:n_stim_key]):
            dtype = np.str if stim_keys[i] in ['stimID', 'condition'] else np.float
            var_data[i] = np.asarray(v, dtype=dtype)
    elif stim_dict['type'] == 'video':
        for i, v in enumerate(var_data[:n_stim_key]):
            if stim_keys[i] == 'stimID':
                dtype = np.int
            elif stim_keys[i] == 'condition':
                dtype = np.str
            else:
                dtype = np.float
            var_data[i] = np.asarray(v, dtype=dtype)
    else:
        raise ValueError('not supported stimulus type: {}'.format(stim_dict['type']))

    # get stimulus variable dict
    stim_var_dict = OrderedDict()
    for idx, key in enumerate(stim_keys):
        stim_var_dict[key] = var_data[idx]
    stim_dict['stim'] = stim_var_dict

    # get measurement variable dict
    if meas_keys:
        for i, v in enumerate(var_data[n_stim_key:], n_stim_key):
            var_data[i] = np.asarray(v, dtype=np.float)
        meas_var_dict = OrderedDict()
        for idx, key in enumerate(meas_keys, n_stim_key):
            meas_var_dict[key] = var_data[idx]
        stim_dict['meas'] = meas_var_dict
    else:
        stim_dict['meas'] = None

    return stim_dict


def save_stim_csv(fpath, title, type, path, stim_var_dict,
                  meas_var_dict=None, opt_meta=None):
    """
    Generate .stim.csv

    Parameters:
    ------------
    fpath[str]: output file path, ending with .stim.csv
    title[str]: customized title
    type[str]: stimulus type in ('picture', 'video')
    path[str]: path_to_stimuli
        If type is 'picture', the path is the parent directory of the pictures.
        If type is 'video', the path is the file path of the video.
    stim_var_dict[dict]: dictionary of stimulus variables
    meas_var_dict[dcit]: dictionary of measurement variables
    opt_meta[dict]: some other optional meta data
    """
    assert fpath.endswith('.stim.csv'), "File suffix must be .stim.csv"
    with open(fpath, 'w') as wf:
        # write the tile, type and path
        wf.write('title={}\n'.format(title))
        wf.write('type={}\n'.format(type))
        wf.write('path={}\n'.format(path))

        # write optional meta data
        if opt_meta is not None:
            for k, v in opt_meta.items():
                wf.write('{0}={1}\n'.format(k, v))

        # write stim and meas
        wf.write('stim={}\n'.format(','.join(stim_var_dict.keys())))
        var_data = np.array(list(stim_var_dict.values()), dtype=np.str).T
        if meas_var_dict is None:
            wf.write('meas=\n')
        else:
            wf.write('meas={}\n'.format(','.join(meas_var_dict.keys())))
            var_data = np.c_[var_data, np.array(list(meas_var_dict.values()), dtype=np.str).T]

        var_data = [','.join(row) for row in var_data]
        wf.write('\n'.join(var_data))
