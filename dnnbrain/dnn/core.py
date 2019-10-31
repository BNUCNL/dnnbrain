import numpy as np

from copy import deepcopy
from dnnbrain.io import file as iofile
from dnnbrain.dnn import io as dio
from dnnbrain.dnn.models import dnn_truncate
from dnnbrain.utils.util import array_fe
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
from scipy.signal import convolve, periodogram
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


class Stimulus:
    """
    Store and handle stimulus-related information
    """
    def __init__(self, path=None):
        """
        Parameter:
        ---------
        path[str]: file path with suffix as .stim.csv
        """
        self.meta = dict()
        self._data = dict()
        if path is not None:
            self.load(path)

    def load(self, path):
        """
        Load stimulus-related information

        Parameter:
        ---------
        path[str]: file path with suffix as .stim.csv
        """
        stim_file = iofile.StimulusFile(path)
        stimuli = stim_file.read()
        self._data = stimuli.pop('data')
        self.meta = stimuli

    def save(self, path):
        """
        Save stimulus-related information

        Parameter:
        ---------
        path[str]: file path with suffix as .stim.csv
        """
        stim_file = iofile.StimulusFile(path)
        meta = self.meta.copy()
        stim_file.write(meta.pop('type'), meta.pop('path'),
                        self._data, **meta)

    def get(self, item):
        """
        Get a column of data according to the item

        Parameter:
        ---------
        item[str]: item name of each column

        Return:
        ------
        col[array]: a column of data
        """
        return self._data[item]

    def set(self, item, value):
        """
        Set a column of data according to the item

        Parameters:
        ----------
        item[str]: item name of the column
        value[array_like]: an array_like data with shape as (n_stim,)
        """
        self._data[item] = np.asarray(value)

    def delete(self, item):
        """
        Delete a column of data according to item

        Parameter:
        ---------
        item[str]: item name of each column
        """
        self._data.pop(item)

    @property
    def items(self):
        return list(self._data.keys())

    def __getitem__(self, indices):
        """
        Get part of the Stimulus object by imitating 2D array's subscript index

        Parameter:
        ---------
        indices[int|tuple|slice]: subscript indices

        Return:
        ------
        stim[Stimulus]: a part of the self.
        """
        # parse subscript indices
        if isinstance(indices, (int, slice, list)):
            # regard it all as row indices
            # get all columns
            rows = indices
            cols = self.items
        elif isinstance(indices, tuple):
            if len(indices) == 0:
                # get all rows and columns
                rows = slice(None, None, None)
                cols = self.items
            elif len(indices) == 1:
                # regard the only element as row indices
                # get all columns
                rows = indices[0]
                cols = self.items
            elif len(indices) == 2:
                # regard the first element as row indices
                # regard the second element as column indices
                rows, cols = indices
                if isinstance(cols, int):
                    # get a column according to an integer
                    cols = [self.items[cols]]
                elif isinstance(cols, str):
                    # get a column according to an string
                    cols = [cols]
                elif isinstance(cols, list):
                    if np.all([isinstance(i, int) for i in cols]):
                        # get columns according to a list of integers
                        cols = [self.items[i] for i in cols]
                    elif np.all([isinstance(i, str) for i in cols]):
                        # get columns according to a list of strings
                        pass
                    else:
                        raise IndexError("only integer [list], string [list] and slices (`:`) "
                                         "are valid column indices")
                elif isinstance(cols, slice):
                    # get columns according to a slice
                    cols = self.items[cols]
                else:
                    raise IndexError("only integer [list], string [list] and slices (`:`) "
                                     "are valid column indices")
            else:
                raise IndexError("This is a 2D data, "
                                 "and can't support more than 3 subscript indices!")
        else:
            raise IndexError("only integer, slices (`:`), list and tuple are valid indices")

        # get part of self
        stim = Stimulus()
        stim.meta = self.meta.copy()
        for item in cols:
            stim.set(item, self.get(item)[rows])

        return stim


class DNN:
    """DNN neural network"""
    def __init__(self, net=None):
        """
        Parameter:
        ---------
        net[str]: deep neural network name
        """
        if net is not None:
            self.load(net)

    def load(self, net):
        """
        load DNN and its information

        Parameter:
        ---------
        net[str]: deep neural network name
        """
        pass

    def save(self, fpath):
        """
        Save DNN

        Parameter:
        ---------
        fpath[str]: file path
        """

    def get_act(self, data_loader, dmask):
        """
        Extract DNN activation

        Parameters:
        ----------
        data_loader[DataLoader]: Pytorch DataLoader
        dmask[Mask]: The mask includes layers/channels/columns of interest.

        Return:
        ------
        act[Activation]: DNN activation
        """
        pass


class Activation:
    """DNN activation"""

    def __init__(self, fpath=None, dmask=None):
        """
        Parameters:
        ----------
        fpath[str]: DNN activation file
        dmask[Mask]: The mask includes layers/channels/columns of interest.
        """
        self.act = dict()
        if fpath is not None:
            self.load(fpath, dmask)

    def load(self, fpath, dmask=None):
        """
        Load DNN activation

        Parameters:
        ----------
        fpath[str]: DNN activation file
        dmask[Mask]: The mask includes layers/channels/columns of interest.
        """
        self.act = iofile.ActivationFile().read(fpath, dmask.mask)

    def save(self, fpath):
        """
        Save DNN activation

        Parameter:
        ---------
        fpath[str]: output file of DNN activation
        """
        iofile.ActivationFile().write(fpath, self.act)

    def set(self, layer, data=None, **kwargs):
        """
        Set DNN activation or its attribution
        If the layer doesn't exist, initiate it with the data.

        Parameters:
        ----------
        layer[str]: layer name
        data[array]: 3D DNN activation array with shape (n_stim, n_chn, n_col)
        kwargs[dict]: the layer's attributions
        """
        if layer not in self.act:
            if data is None:
                raise ValueError("The data can't be None when initiating a new layer!")
            self.act[layer] = {'data': data, 'attrs': kwargs}
        else:
            if data is not None:
                self.act[layer]['data'] = data
            self.act[layer]['attrs'].update(kwargs)

    def delete(self, layer, *args):
        """
        Delete DNN activation or its attribution

        Parameters:
        ----------
        layer[str]: layer name
        args[tuple]: attribution names
            If is empty, delete the whole layer.
            else, only delete these attributions.
        """
        if args:
            for attr in args:
                self.act[layer]['attrs'].pop(attr)
        else:
            self.act.pop(layer)

    def mask(self, dmask):
        """
        Mask DNN activation

        Parameter:
        ---------
        dmask[Mask]: The mask includes layers/channels/columns of interest.

        Return:
        ------
        act[Activation]: DNN activation
        """
        act = Activation()
        for layer, d in dmask.items():
            data = self.act[layer]['data']
            if d['chn'] != 'all':
                channels = [chn-1 for chn in d['chn']]
                data = data[:, channels, :]
            if d['col'] != 'all':
                columns = [col-1 for col in d['col']]
                data = data[:, :, columns]
            act.set(layer, data, **self.act[layer]['attrs'])

        return act


class Mask:
    """DNN mask"""

    def __init__(self, fpath=None):
        """
        Parameter:
        ---------
        fpath[str]: DNN mask file
        """
        self._mask = dict()
        if fpath is not None:
            self.load(fpath)

    def load(self, fpath):
        """
        Load DNN mask, the whole mask will be overrode.

        Parameter:
        ---------
        fpath[str]: DNN mask file
        """
        self._mask = iofile.MaskFile().read(fpath)

    def update(self, dmask_dict):
        """
        Update DNN mask, the existed layer with the same name will be overrode.

        Parameter:
        ---------
        dmask_dict[dict]: Dictionary of the DNN mask information
        """
        self._mask.update(dmask_dict)

    def set(self, layer, channels=None, columns=None):
        """
        Set DNN mask.
        If the layer doesn't exist, initiate it with all channels and columns.

        Parameters:
        ----------
        layer[str]: layer name
        channels[list|str]:
            If is list, it contains sequence numbers of channels of interest.
            If is str, it must be 'all' that means all channels in the layer.
            If is None, do nothing.
        columns[list|str]:
            If is list, it contains sequence numbers of columns of interest.
            If is str, it must be 'all' that means all columns in the layer.
            If is None, do nothing.
        """
        if layer not in self._mask:
            self._mask[layer] = {'chn': 'all', 'col': 'all'}
        if channels is not None:
            self._mask[layer]['chn'] = channels
        if columns is not None:
            self._mask[layer]['col'] = columns

    def copy(self):
        """
        Make a copy of the DNN mask
        """
        dmask = Mask()
        dmask._mask = deepcopy(self._mask)

        return dmask

    def pop(self, layer):
        """
        Delete a layer

        Parameter:
        ---------
        layer[str]: layer name
        """
        return self.mask.pop(layer)

    def clear(self):
        """
        Empty the DNN mask
        """
        self.mask.clear()

    def save(self, fpath):
        """
        Save DNN mask

        Parameter:
        ---------
        fpath[str]: output file path of DNN mask
        """
        iofile.MaskFile().write(fpath, self.mask)

    @property
    def mask(self):
        """Get mask"""
        if not self._mask:
            raise AttributeError("The mask in this instance is empty. "
                                 "Please set it through load(), update(), or set().")
        return self._mask


def save_activation(activation, outpath):
    """
    Save activaiton data as a csv file or mat format file to outpath
         csv format save a 2D.
            The first column is stimulus indexs
            The second column is channel indexs
            Each row is the activation of a filter for a image
         mat format save a 2D or 4D array depend on the activation from
             convolution layer or fully connected layer.
            4D array Dimension:sitmulus x channel x pixel x pixel
            2D array Dimension:stimulus x activation
    Parameters:
    ------------
    activation[4darray]: sitmulus x channel x pixel x pixel
    outpath[str]:outpath and outfilename
    """
    imgname = os.path.basename(outpath)
    imgsuffix = imgname.split('.')[-1]

    if imgsuffix == 'csv':
        if len(activation.shape) == 4:
            activation2d = np.reshape(
                    activation, (np.prod(activation.shape[0:2]), -1,),
                    order='C')
            channelline = np.array(
                    [channel + 1 for channel
                     in range(activation.shape[1])] * activation.shape[0])
            stimline = []
            for i in range(activation.shape[0]):
                a = [i + 1 for j in range(activation.shape[1])]
                stimline = stimline + a
            stimline = np.array(stimline)
            channelline = np.reshape(channelline, (channelline.shape[0], 1))
            stimline = np.reshape(stimline, (stimline.shape[0], 1))
            activation2d = np.concatenate(
                    (stimline, channelline, activation2d), axis=1)
        elif len(activation.shape) == 2:
            stim_indexs = np.arange(1, activation.shape[0] + 1)
            stim_indexs = np.reshape(stim_indexs, (-1, stim_indexs[0]))
            activation2d = np.concatenate((stim_indexs, activation), axis=1)
        np.savetxt(outpath, activation2d, delimiter=',')
    elif imgsuffix == 'mat':
        scipy.io.savemat(outpath, mdict={'activation': activation})
    else:
        np.save(outpath, activation)


class NetLoader:
    def __init__(self, net=None):
        """
        Load neural network model

        Parameters:
        -----------
        net[str]: a neural network's name
        """
        netlist = ['alexnet', 'vgg11', 'vggface']
        if net in netlist:
            if net == 'alexnet':
                self.model = torchvision.models.alexnet()
                self.model.load_state_dict(torch.load(
                        os.path.join(DNNBRAIN_MODEL_DIR, 'alexnet_param.pth')))
                self.layer2indices = {'conv1': (0, 0), 'conv1_relu': (0, 1), 'conv1_maxpool': (0, 2), 'conv2': (0, 3),
                                      'conv2_relu': (0, 4), 'conv2_maxpool': (0, 5), 'conv3': (0, 6), 'conv3_relu': (0, 7),
                                      'conv4': (0, 8), 'conv4_relu': (0, 9),'conv5': (0, 10), 'conv5_relu': (0, 11),
                                      'conv5_maxpool': (0, 12), 'fc1': (2, 1), 'fc1_relu': (2, 2),
                                      'fc2': (2, 4), 'fc2_relu': (2, 5), 'fc3': (2, 6), 'prefc': (2,)}
                self.layer2loc = {'conv1': ('features', '0'), 'conv1_relu': ('features', '1'),
                                  'conv1_maxpool': ('features', '2'), 'conv2': ('features', '3'),
                                  'conv2_relu': ('features', '4'), 'conv2_maxpool': ('features', '5'),
                                  'conv3': ('features', '6'), 'conv3_relu': ('features', '7'),
                                  'conv4': ('features', '8'), 'conv4_relu': ('features', '9'),
                                  'conv5': ('features', '10'), 'conv5_relu': ('features', '11'),
                                  'conv5_maxpool': ('features', '12'), 'fc1': ('classifier', '1'),
                                  'fc1_relu': ('classifier', '2'), 'fc2': ('classifier', '4'),
                                  'fc2_relu': ('classifier', '5'), 'fc3': ('classifier', '6')}
                self.img_size = (224, 224)
            elif net == 'vgg11':
                self.model = torchvision.models.vgg11()
                self.model.load_state_dict(torch.load(
                        os.path.join(DNNBRAIN_MODEL_DIR, 'vgg11_param.pth')))
                self.layer2indices = {'conv1': (0, 0), 'conv2': (0, 3),
                                      'conv3': (0, 6), 'conv4': (0, 8),
                                      'conv5': (0, 11), 'conv6': (0, 13),
                                      'conv7': (0, 16), 'conv8': (0, 18),
                                      'fc1': (2, 0), 'fc2': (2, 3),
                                      'fc3': (2, 6), 'prefc':(2,)}
                self.img_size = (224, 224)
            elif net == 'vggface':
                self.model = Vgg_face()
                self.model.load_state_dict(torch.load(
                        os.path.join(DNNBRAIN_MODEL_DIR, 'vgg_face_dag.pth')))
                self.layer2indices = {'conv1': (0,), 'conv2': (2,),
                                      'conv3': (5,), 'conv4': (7,),
                                      'conv5': (10,), 'conv6': (12,),
                                      'conv7': (14,), 'conv8': (17,),
                                      'conv9': (19,), 'conv10': (21,),
                                      'conv11': (24,), 'conv12': (26,),
                                      'conv13': (28,), 'fc1': (31,),
                                      'fc2': (34,), 'fc3': (37,), 'prefc':(31,)}
                self.img_size = (224, 224)
        else:
            print('Not internal supported, please call netloader function'
                  'to assign model, layer2indices and image size.')
            self.model = None
            self.layer2indices = None
            self.img_size = None

    def load_model(self, dnn_model, model_param=None,
                   layer2indices=None, input_imgsize=None):
        """
        Load DNN model

        Parameters:
        -----------
        dnn_model[nn.Modules]: DNN model
        model_param[string/state_dict]: Parameters of DNN model
        layer2indices[dict]: Comparison table between layer name and
            DNN frame layer.
            Please make dictionary as following format:
                {'conv1': (0, 0), 'conv2': (0, 3), 'fc1': (2, 0)}
        input_imgsize[tuple]: the input image size
        """
        self.model = dnn_model
        if model_param is not None:
            if isinstance(model_param, str):
                self.model.load_state_dict(torch.load(model_param))
            else:
                self.model.load_state_dict(model_param)
        self.layer2indices = layer2indices
        self.img_size = input_imgsize
        print('You had assigned a model into netloader.')


class ActReader:
    def __init__(self, fpath):
        """
        Get DNN activation from .act.h5 file

        Parameters:
        ----------
        fpath[str]: DNN activation file
        """
        assert fpath.endswith('.act.h5'), "the file's suffix must be .act.h5"
        self._file = h5py.File(fpath, 'r')

    def close(self):
        self._file.close()

    def get_act(self, layer, to_numpy=True):
        """
        Get a layer's activation

        Parameters:
        ----------
        layer[str]: layer name
        to_numpy[bool]:
            If False, return HDF5 dataset directly.
            If True, transform to numpy array.

        Return:
        ------
        act: DNN activation
        """
        act = self._file[layer]
        if to_numpy:
            act = np.array(act)

        return act

    def get_attr(self, layer, attr):
        """
        Get an attribution of a layer's activation

        Parameters:
        ----------
        layer[str]: layer name
        attr[str]: attribution name

        Return:
        ------
            attribution
        """
        return self._file[layer].attrs[attr]

    @property
    def title(self):
        """
        Get the title of the file

        Return:
        ------
            a string
        """
        return self._file.attrs['title']

    @property
    def cmd(self):
        """
        Get the command used to generate the file

        Return:
        ------
            a string
        """
        return self._file.attrs['cmd']

    @property
    def date(self):
        """
        Get the date when the file was generated

        Return:
        ------
            a string
        """
        return self._file.attrs['date']

    @property
    def layers(self):
        """
        Get all layer names in the file

        Return:
        ------
            a list
        """
        return list(self._file.keys())


class ActWriter:
    def __init__(self, fpath, title):
        """
        Save DNN activation into .act.h5 file

        Parameters:
        ----------
        fpath[str]: DNN activation file
        title[str]: a simple description for the file
        """
        assert fpath.endswith('.act.h5'), "the file's suffix must be .act.h5"
        self._file = h5py.File(fpath, 'w')
        self._file.attrs['title'] = title

    def close(self):
        """
        Write some information and close the file
        """
        self._file.attrs['cmd'] = ' '.join(sys.argv)
        self._file.attrs['date'] = time.asctime()
        self._file.close()

    def set_act(self, layer, act):
        """
        Set a layer's activation

        Parameters:
        ----------
        layer[str]: layer name
        act[array]: DNN activation
        """
        self._file.create_dataset(layer, data=act)

    def set_attr(self, layer, attr, value):
        """
        Set an attribution of a layer's activation

        Parameters:
        ----------
        layer[str]: layer name
        attr[str]: attribution name
        value: the value of the attribution
        """
        self._file[layer].attrs[attr] = value


def read_dmask_csv(fpath):
    """
    Read pre-designed .dmask.csv file.

    Parameters:
    ----------
    fpath: path of .dmask.csv file

    Return:
    ------
    dmask_dict[OrderedDict]: Dictionary of the DNN mask information
    """
    # -load csv data-
    assert fpath.endswith('.dmask.csv'), 'File suffix must be .dmask.csv'
    with open(fpath) as rf:
        lines = rf.read().splitlines()

    # extract layers, channels and columns of interest
    dmask_dict = OrderedDict()
    for l_idx, line in enumerate(lines):
        if '=' in line:
            # layer
            layer, axes = line.split('=')
            dmask_dict[layer] = {'chn': None, 'col': None}

            # channels and columns
            axes = axes.split(',')
            while '' in axes:
                axes.remove('')
            assert len(axes) <= 2, \
                "The number of a layer's axes must be less than or equal to 2."
            for a_idx, axis in enumerate(axes, 1):
                assert axis in ('chn', 'col'), 'Axis must be from (chn, col).'
                numbers = [int(num) for num in lines[l_idx+a_idx].split(',')]
                dmask_dict[layer][axis] = numbers

    return dmask_dict


def save_dmask_csv(fpath, dmask_dict):
    """
    Generate .dmask.csv

    Parameters
    ---------
    fpath[str]: output file path, ending with .dmask.csv
    dmask_dict[dict]: Dictionary of the DNN mask information
    """
    assert fpath.endswith('.dmask.csv'), 'File suffix must be .dmask.csv'
    with open(fpath, 'w') as wf:
        for layer, axes_dict in dmask_dict.items():
            axes = []
            num_lines = []
            assert len(axes_dict) <= 2, \
                "The number of a layer's axes must be less than or equal to 2."
            for axis, numbers in axes_dict.items():
                assert axis in ('chn', 'col'), 'Axis must be from (chn, col).'
                if numbers is not None:
                    axes.append(axis)
                    num_line = ','.join(map(str, numbers))
                    num_lines.append(num_line)

            wf.write('{0}={1}\n'.format(layer, ','.join(axes)))
            for num_line in num_lines:
                wf.write(num_line+'\n')



def dnn_activation_deprecated(input, netname, layer, channel=None, column=None,
                              fe_axis=None, fe_meth=None):
    """
    Extract DNN activation

    Parameters:
    ------------
    input[dataloader]: input image dataloader	
    netname[str]: DNN network
    layer[str]: layer name of a DNN network
    channel[list]: specify channel in layer of DNN network, channel was counted from 1 (not 0)
    column[list]: column of interest
    fe_axis{str}: axis for feature extraction
    fe_meth[str]: feature extraction method, max, mean, median

    Returns:
    ---------
    dnnact[numpy.array]: DNN activation, A 3D array with its shape as (n_picture, n_channel, n_column)
    """
    assert (fe_axis is None) == (fe_meth is None), 'Please specify fe_axis and fe_meth at the same time.'

    # get dnn activation
    loader = dio.NetLoader(netname)
    actmodel = dnn_truncate(loader, layer)
    actmodel.eval()
    dnnact = []
    count = 0  # count the progress
    for picdata, _ in input:
        dnnact_part = actmodel(picdata)
        dnnact.extend(dnnact_part.detach().numpy())
        count += dnnact_part.shape[0]
        print('Extracted acts:', count)
    dnnact = np.array(dnnact)
    dnnact = dnnact.reshape((dnnact.shape[0], dnnact.shape[1], -1))

    # mask the data
    if channel is not None:
        dnnact = dnnact[:, channel, :]
    if column is not None: 
        dnnact = dnnact[:, :, column]
        
    # feature extraction
    if fe_axis is not None:
        fe_meths = {
            'max': np.max,
            'mean': np.mean,
            'median': np.median
        }

        if fe_axis == 'layer':
            dnnact = dnnact.reshape((dnnact.shape[0], -1))
            dnnact = fe_meths[fe_meth](dnnact, -1)[:, np.newaxis, np.newaxis]
        elif fe_axis == 'channel':
            dnnact = fe_meths[fe_meth](dnnact, 1)[:, np.newaxis, :]
        elif fe_axis == 'column':
            dnnact = fe_meths[fe_meth](dnnact, 2)[:, :, np.newaxis]
        else:
            raise ValueError('fe_axis should be layer, channel or column')
    
    return dnnact


def dnn_activation(data, model, layer_loc, channels=None):
    """
    Extract DNN activation from the specified layer

    Parameters:
    ----------
    data[tensor]: input stimuli of the model with shape as (n_stim, n_chn, height, width)
    model[model]: DNN model
    layer_loc[sequence]: a sequence of keys to find the location of
        the target layer in the DNN model. For example, the location of the
        fifth convolution layer in AlexNet is ('features', '10').
    channels[list]: channel indices of interest

    Return:
    ------
    dnn_acts[array]: DNN activation
        a 4D array with its shape as (n_stim, n_chn, n_r, n_c)
    """
    # change to eval mode
    model.eval()

    # prepare dnn activation hook
    dnn_acts = []

    def hook_act(module, input, output):
        act = output.detach().numpy().copy()
        if channels is not None:
            act = act[:, channels]
        dnn_acts.append(act)

    module = model
    for k in layer_loc:
        module = module._modules[k]
    hook_handle = module.register_forward_hook(hook_act)

    # extract dnn activation
    model(data)
    dnn_acts = dnn_acts[0]

    hook_handle.remove()
    return dnn_acts


def dnn_mask(dnn_acts, chn=None, col=None):
    """
    Extract DNN activation

    Parameters:
    ------------
    dnn_acts[array]: DNN activation, A 3D array with its shape as (n_stim, n_chn, n_col)
    chn[list]: channel indices of interest
    col[list]: column indices of interest

    Returns:
    ---------
    dnn_acts[array]: DNN activation after mask
        a 3D array with its shape as (n_stim, n_chn, n_col)
    """
    if chn is not None:
        dnn_acts = dnn_acts[:, chn, :]
    if col is not None:
        dnn_acts = dnn_acts[:, :, col]

    return dnn_acts


def dnn_pooling(dnn_acts, method):
    """
    Pooling DNN activation for each channel

    Parameters:
    ------------
    dnn_acts[array]: DNN activation, A 3D array with its shape as (n_stim, n_chn, n_col)
    method[str]: pooling method, choices=(max, mean, median)

    Returns:
    ---------
    dnn_acts[array]: DNN activation after pooling
        a 3D array with its shape as (n_stim, n_chn, 1)
    """
    return array_fe(dnn_acts, method, 2, True)


def dnn_fe(dnn_acts, meth, n_feat, axis=None):
    """
    Extract features of DNN activation

    Parameters:
    ----------
    dnn_acts[array]: DNN activation
        a 3D array with its shape as (n_stim, n_chn, n_col)
    meth[str]: feature extraction method, choices=(pca, hist, psd)
        pca: use n_feat principal components as features
        hist: use histogram of activation as features
            Note: n_feat equal-width bins in the given range will be used!
        psd: use power spectral density as features
    n_feat[int]: The number of features to extract
    axis{str}: axis for feature extraction, choices=(chn, col)
        If it's None, extract features from the whole layer. Note:
        The result of this will be an array with shape (n_stim, n_feat, 1), but
        We also regard it as (n_stim, n_chn, n_col)

    Returns:
    -------
    dnn_acts_new[array]: DNN activation
        a 3D array with its shape as (n_stim, n_chn, n_col)
    """
    # adjust iterative axis
    n_stim, n_chn, n_col = dnn_acts.shape
    if axis is None:
        dnn_acts = dnn_acts.reshape((n_stim, 1, -1))
    elif axis == 'chn':
        dnn_acts = dnn_acts.transpose((0, 2, 1))
    elif axis == 'col':
        pass
    else:
        raise ValueError('not supported axis:', axis)
    _, n_iter, _ = dnn_acts.shape

    # extract features
    dnn_acts_new = np.zeros((n_stim, n_iter, n_feat))
    if meth == 'pca':
        pca = PCA(n_components=n_feat)
        for i in range(n_iter):
            dnn_acts_new[:, i, :] = pca.fit_transform(dnn_acts[:, i, :])
    elif meth == 'hist':
        for i in range(n_iter):
            for j in range(n_stim):
                dnn_acts_new[j, i, :] = np.histogram(dnn_acts[j, i, :], n_feat)[0]
    elif meth == 'psd':
        for i in range(n_iter):
            for j in range(n_stim):
                f, p = periodogram(dnn_acts[j, i, :])
                dnn_acts_new[j, i, :] = p[:n_feat]
    else:
        raise ValueError('not supported method:', meth)

    # adjust iterative axis
    if axis is None:
        dnn_acts_new = dnn_acts_new.transpose((0, 2, 1))
    elif axis == 'chn':
        dnn_acts_new = dnn_acts_new.transpose((0, 2, 1))

    return dnn_acts_new


def db_uva(dnn_acts, resp, model, iter_axis=None, cvfold=3):
    """
    Use DNN activation to predict responses of brain or behavior
    by univariate analysis.'

    Parameters:
    ----------
    dnn_acts[array]: DNN activation
        A 3D array with its shape as (n_stim, n_chn, n_col)
    resp[array]: response of brain or behavior
        A 2D array with its shape as (n_samp, n_meas)
    model[str]: the name of model used to do prediction
    iter_axis[str]: iterate along the specified axis
        channel: Summarize the maximal prediction score for each channel.
        column: Summarize the maximal prediction score for each column.
        default: Summarize the maximal prediction score for the whole layer.
    cvfold[int]: cross validation fold number

    Return:
    ------
    pred_dict[dict]:
        score_arr: max score array
        channel_arr: channel position of the max score
        column_arr: column position of the max score
        model_arr: fitted model of the max score
    """
    n_stim, n_chn, n_col = dnn_acts.shape
    n_samp, n_meas = resp.shape  # n_sample x n_measures
    assert n_stim == n_samp, 'n_stim != n_samp'

    # transpose axis to make dnn_acts's shape as (n_stimulus, n_iterator, n_element)
    if iter_axis is None:
        dnn_acts = dnn_acts.reshape((n_stim, 1, n_chn * n_col))
    elif iter_axis == 'column':
        dnn_acts = dnn_acts.transpose((0, 2, 1))
    elif iter_axis == 'channel':
        pass
    else:
        raise ValueError("Unspported iter_axis:", iter_axis)
    n_stim, n_iter, n_elem = dnn_acts.shape

    # prepare model
    if model in ('lrc', 'svc'):
        score_evl = 'accuracy'
    elif model in ('glm', 'lasso'):
        score_evl = 'explained_variance'
    else:
        raise ValueError('unsupported model:', model)

    if model == 'lrc':
        model = LogisticRegression()
    elif model == 'svc':
        model = SVC(kernel='linear', C=0.025)
    elif model == 'lasso':
        model = Lasso()
    else:
        model = LinearRegression()

    # prepare container
    score_arr = np.zeros((n_iter, n_meas), dtype=np.float)
    channel_arr = np.zeros_like(score_arr, dtype=np.int)
    column_arr = np.zeros_like(score_arr, dtype=np.int)
    model_arr = np.zeros_like(score_arr, dtype=np.object)

    # start iteration
    for meas_idx in range(n_meas):
        for iter_idx in range(n_iter):
            score_tmp = []
            for elem_idx in range(n_elem):
                cv_scores = cross_val_score(model, dnn_acts[:, iter_idx, elem_idx][:, None],
                                            resp[:, meas_idx], scoring=score_evl, cv=cvfold)
                score_tmp.append(np.mean(cv_scores))

            # find max score
            max_elem_idx = np.argmax(score_tmp)
            max_score = score_tmp[max_elem_idx]
            score_arr[iter_idx, meas_idx] = max_score

            # find position for the max score
            if iter_axis is None:
                chn_idx = max_elem_idx // n_col
                col_idx = max_elem_idx % n_col
            elif iter_axis == 'channel':
                chn_idx, col_idx = iter_idx, max_elem_idx
            else:
                chn_idx, col_idx = max_elem_idx, iter_idx

            channel_arr[iter_idx, meas_idx] = chn_idx + 1
            column_arr[iter_idx, meas_idx] = col_idx + 1

            # fit the max-score model
            model_arr[iter_idx, meas_idx] = model.fit(dnn_acts[:, iter_idx, max_elem_idx][:, None],
                                                      resp[:, meas_idx])
            print('Meas: {0}/{1}; iter:{2}/{3}'.format(meas_idx + 1, n_meas,
                                                       iter_idx + 1, n_iter,))
    pred_dict = {
        'score': score_arr,
        'chn_pos': channel_arr,
        'col_pos': column_arr,
        'model': model_arr
    }
    return pred_dict


def db_mva(dnn_acts, resp, model, iter_axis=None, cvfold=3):
    """
    Use DNN activation to predict responses of brain or behavior
    by multivariate analysis.'

    Parameters:
    ----------
    dnn_acts[array]: DNN activation
        A 3D array with its shape as (n_stim, n_chn, n_col)
    resp[array]: response of brain or behavior
        A 2D array with its shape as (n_samp, n_meas)
    model[str]: the name of model used to do prediction
    iter_axis[str]: iterate along the specified axis
        channel: Do mva using all units in each channel.
        column: Do mva using all units in each column.
        default: Do mva using all units in the whole layer.
    cvfold[int]: cross validation fold number

    Return:
    ------
    pred_dict[dict]:
        score_arr: prediction score array
        model_arr: fitted model
    """
    n_stim, n_chn, n_col = dnn_acts.shape
    n_samp, n_meas = resp.shape  # n_sample x n_measures
    assert n_stim == n_samp, 'n_stim != n_samp'

    # transpose axis to make dnn_acts's shape as (n_stimulus, n_iterator, n_element)
    if iter_axis is None:
        dnn_acts = dnn_acts.reshape((n_stim, 1, n_chn * n_col))
    elif iter_axis == 'column':
        dnn_acts = dnn_acts.transpose((0, 2, 1))
    elif iter_axis == 'channel':
        pass
    else:
        raise ValueError("Unspported iter_axis:", iter_axis)
    n_stim, n_iter, n_elem = dnn_acts.shape

    # prepare model
    if model in ('lrc', 'svc'):
        score_evl = 'accuracy'
    elif model in ('glm', 'lasso'):
        score_evl = 'explained_variance'
    else:
        raise ValueError('unsupported model:', model)

    if model == 'lrc':
        model = LogisticRegression()
    elif model == 'svc':
        model = SVC(kernel='linear', C=0.025)
    elif model == 'lasso':
        model = Lasso()
    else:
        model = LinearRegression()

    score_arr = []
    model_arr = []
    # start iteration
    for iter_idx in range(n_iter):
        # cross validation
        score_tmp = [cross_val_score(model, dnn_acts[:, iter_idx, :], resp[:, i],
                                     scoring=score_evl, cv=cvfold) for i in range(n_meas)]
        score_arr.append(np.array(score_tmp).mean(-1))

        # fit model
        model.fit(dnn_acts[:, iter_idx, :], resp)
        model_arr.append(model)

        print('Finish iteration{0}/{1}'.format(iter_idx + 1, n_iter))
    score_arr = np.array(score_arr)
    model_arr = np.array(model_arr)

    pred_dict = {
        'score': score_arr,
        'model': model_arr
    }
    return pred_dict


def convolve_hrf(X, onsets, durations, n_vol, tr, ops=100):
    """
    Convolve each X's column iteratively with HRF and align with the timeline of BOLD signal

    parameters:
    ----------
    X[array]: [n_event, n_sample]
    onsets[array_like]: in sec. size = n_event
    durations[array_like]: in sec. size = n_event
    n_vol[int]: the number of volumes of BOLD signal
    tr[float]: repeat time in second
    ops[int]: oversampling number per second

    Returns:
    ---------
    X_hrfed[array]: the result after convolution and alignment
    """
    assert np.ndim(X) == 2, 'X must be a 2D array'
    assert X.shape[0] == len(onsets) and X.shape[0] == len(durations), 'The length of onsets and durations should ' \
                                                                       'be matched with the number of events.'
    assert ops in (10, 100, 1000), 'Oversampling rate must be one of the (10, 100, 1000)!'

    # unify the precision
    decimals = int(np.log10(ops))
    onsets = np.round(np.asarray(onsets), decimals=decimals)
    durations = np.round(np.asarray(durations), decimals=decimals)
    tr = np.round(tr, decimals=decimals)

    n_clipped = 0  # the number of clipped time points earlier than the start point of response
    onset_min = onsets.min()
    if onset_min > 0:
        # The earliest event's onset is later than the start point of response.
        # We supplement it with zero-value event to align with the response.
        X = np.insert(X, 0, np.zeros(X.shape[1]), 0)
        onsets = np.insert(onsets, 0, 0, 0)
        durations = np.insert(durations, 0, onset_min, 0)
        onset_min = 0
    elif onset_min < 0:
        print("The earliest event's onset is earlier than the start point of response.\n"
              "We clip the earlier time points after hrf_convolution to align with the response.")
        n_clipped = int(-onset_min * ops)

    # do convolution in batches for trade-off between speed and memory
    batch_size = int(100000 / ops)
    bat_indices = np.arange(0, X.shape[-1], batch_size)
    bat_indices = np.r_[bat_indices, X.shape[-1]]

    vol_t = (np.arange(n_vol) * tr * ops).astype(int)  # compute volume acquisition timing
    n_time_point = int(((onsets + durations).max()-onset_min) * ops)
    X_hrfed = np.zeros([n_vol, 0])
    for idx, bat_idx in enumerate(bat_indices[:-1]):
        X_bat = X[:, bat_idx:bat_indices[idx+1]]
        # generate X raw time course
        X_tc = np.zeros((n_time_point, X_bat.shape[-1]), dtype=np.float32)
        for i, onset in enumerate(onsets):
            onset_start = int(onset * ops)
            onset_end = int(onset_start + durations[i] * ops)
            X_tc[onset_start:onset_end, :] = X_bat[i, :]

        # generate hrf kernel
        hrf = spm_hrf(tr, oversampling=tr*ops)
        hrf = hrf[:, np.newaxis]

        # convolve X raw time course with hrf kernal
        X_tc_hrfed = convolve(X_tc, hrf, method='fft')
        X_tc_hrfed = X_tc_hrfed[n_clipped:, :]

        # downsample to volume timing
        X_hrfed = np.c_[X_hrfed, X_tc_hrfed[vol_t, :]]

        print('hrf convolution: sample {0} to {1} finished'.format(bat_idx+1, bat_indices[idx+1]))

    return X_hrfed
