#! /usr/bin/env python

"""
Multivarate analysis(mva) to explore relations between CNN activation and responses of brain or behavior.

CNL @ BNU
"""
import os
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from os.path import join as pjoin
from dnnbrain.dnn import analyzer
from dnnbrain.dnn import io as dio
from dnnbrain.brain import io as bio
from sklearn import linear_model, model_selection, decomposition, svm


def main():
    parser = argparse.ArgumentParser(description='Use CNN activation from \
                                     multiple units to predict responses from \
                                     brain or behavior.')

    parser.add_argument('-net',
                        type=str,
                        required=True,
                        metavar='NetName',
                        help='pretained convolutional network name')

    parser.add_argument('-layer',
                        type=str,
                        required=True,
                        metavar='LayerName',
                        help='The layer whose activation is used \
                        to predict brain/behavior response. conv and fc indicate \
                        convolution and fullly connected layers:conv1, \
                        conv2,...,conv5, fc1, ...,fc3.')

    parser.add_argument('-axis',
                        type=str,
                        default='layer',
                        required=True,
                        choices=['layer', 'channel', 'column'],
                        metavar='AxisName',
                        help='Target axis to organize the predictors. \
                        layer: do mva using all unit within a layer\
                        channel：do mva using all units within each channel. \
                        column: do mva using all units within each column. \
                        default is layer.')

    parser.add_argument('-dmask',
                        type=str,
                        required=False,
                        metavar='DnnMaskFile',
                        help='a db.csv file in which channles and columns of \
                        intereset ae listed.')

    parser.add_argument('-dfe',
                        type=str,
                        required=False,
                        metavar='DnnFeatureExtraction',
                        choices=['hist', 'max', 'mean', 'median'],
                        help='Feature extraction for dnn activation in \
                        the specified axis. \
                        max: use max activiton as feature, \
                        median: use median activation as feature, \
                        mean: use mean activtion as feature, \
                        hist: use hist proflies as features.')

    parser.add_argument('-dpca',
                        type=int,
                        required=False,
                        metavar='PCA',
                        help='The number of PC to be kept.')

    parser.add_argument('-stim',
                        type=str,
                        required=False,
                        metavar='StimuliInfoFile',
                        help='a stim.db.csv file provides stimuli information')

    parser.add_argument('-movie',
                        type=str,
                        required=False,
                        action='append',
                        metavar='MoiveStimulusFile',
                        help='a mp4 video file')

    parser.add_argument('-response',
                        type=str,
                        required=True,
                        metavar='ResponseFile',
                        help='a resp.db.csv file to provide target response. \
                        The target reponse could be behavior measures or \
                        brain response from some rois')

    parser.add_argument('-hrf',
                        action='store_true',
                        required=False,
                        help='The canonical HRF is used. Default no hrf is used')

    parser.add_argument('-bmask',
                        type=str,
                        required=False,
                        metavar='BrainMaskFile',
                        help='Brain mask(nii or nii.gz) to indicate \
                        voxels of interest. It works only when response \
                        is nii or nii.gz file')

    parser.add_argument('-model',
                        type=str,
                        required=True,
                        metavar='Model',
                        choices=['glm', 'lasso', 'svc', 'lrc'],
                        help='glm: general linear regression \
                        lasso: lasso regression \
                        svc: support vector machine classification \
                        lrc: logistic regression for classification.')

    parser.add_argument('-cvfold',
                        default=2,
                        type=int,
                        required=False,
                        metavar='FoldNumber',
                        help='Fold number of cross validation')

    parser.add_argument('-outdir',
                        type=str,
                        required=True,
                        metavar='OutputDir',
                        help='Output directory. Model coef, accuracy score, and \
                        predicted responss for each stimlus will be saved \
                        in the output dir.')

    args = parser.parse_args()

    # %% Brain/behavior response(i.e.,Y)
    """
    First, we prepare the response data for exploring relations between
    the CNN activation and brain/behavior responses.

    """
    if args.response.endswith('.db.csv'):
        resp_dict = dio.read_dnn_csv(args.response)
        Y = np.asarray(list(resp_dict['variable'].values())).T

        tr = float(resp_dict['tr'])

    elif args.response.endswith('.nii') or args.response.endswith('.nii.gz'):
        resp, header = bio.load_brainimg(args.response)
        bshape = resp.shape

        # Get resp data within brain mask
        resp = resp.reshape(resp.shape[0], -1)  # n_stim x n_vox
        if args.bmask is None:
            bmask = np.any(resp, 0)
        else:
            bmask, _ = bio.load_brainimg(args.bmask, ismask=True)
            bmask = bmask.reshape(-1).astype(np.int)
            assert bmask.shape[0] == resp.shape[1], (
                              'brain mask and brain response mismatched in space')
            Y = resp[:, bmask != 0]  # n_stim x n_roi or n_vox

        # Get tr from nii header
        tr = header['pixdim'][4]
        if header.get_xyzt_units()[-1] == 'ms':
            tr = tr / 1000
    else:
        raise Exception('Only db.csv and nii vloume are supported')

    print('response data successfully loaded')

    # %% CNN activation
    """
    Second, we prepare CNN activation(i.e., X) for exploring relations between
    the CNN activation and brain/behavior responses.

    """
    # Load CNN
    netloader = dio.NetLoader(args.net)
    transform = transforms.Compose([transforms.Resize(netloader.img_size),
                                    transforms.ToTensor()])
    # Load stimulus
    stim = dio.read_dnn_csv(args.stim)
    picdataset = dio.PicDataset(
            stim['picPath'], stim['variable'], transform=transform)
    picdataloader = DataLoader(picdataset, batch_size=8, shuffle=False)

    # calculate dnn activation: n_stim * n_channel * unit * unit
    dnn_act = analyzer.dnn_activation(picdataloader, args.net, args.layer)
    # n_stim * n_channel * n_unit
    dnn_act = dnn_act.reshape(dnn_act.shape[0], dnn_act.shape[1], -1)

    print('extraction of dnn activation finished')

    # define dnn mask
    if args.dmask is not None:
        dmask = dio.read_dnn_csv(args.dmask)
        chnoi = list(dmask['variable']['chn'])
        coloi = list(dmask['variable']['col'])
        dnn_act = dnn_act[:, chnoi][:, :, coloi]  # n_stim x n_chnoi x n_coloi

        print('dmask successfully applied')

    # transpose axis accoring to user specified axis
    # the specified axis in 2nd dimension
    if args.axis == 'layer':
        dnn_act = dnn_act.reshape(
                dnn_act.shape[0], 1, dnn_act.shape[1]*dnn_act.shape[2])
    elif args.axis == 'channel':
        pass
    elif args.axis == 'column':
        dnn_act = dnn_act.transpose(0, 2, 1)
    else:
        raise Exception('Axis should be layer, channel or column.')
        
    # dnn feature extraction
    if args.dfe is not None:
        assert args.axis != 'layer', (
                'dnn feacure extraction can not be applied in axis layer.')
        if args.axis == 'channel':
            if args.dfe == 'mean':
                dnn_act = np.mean(dnn_act, axis=-1)[..., None]
            elif args.dfe == 'max':
                dnn_act = np.max(dnn_act, axis=-1)[..., None]
            elif args.dfe == 'median':
                dnn_act = np.median(dnn_act, axis=-1)[..., None]
#            elif args.dfe == 'hist':
#                dnn_act = np.histogramdd(dnn_act)

        print('dnn feature extraction finished')

    # PCA on dnn features
    if args.dpca is not None:
        assert args.dpca < dnn_act.shape[-1], (
                'number of PC in PCA can not be larger than sample size.')
        pca = decomposition.PCA(n_components=args.dpca)

        X = np.zeros((dnn_act.shape[0], dnn_act.shape[1], args.dpca))
        for i in range(dnn_act.shape[1]):
            X[:, i, :] = pca.fit_transform(dnn_act[:, i, :])

        print('PCA on dnn activation finished')

    # size of cnn activation (i.e, X)
    n_stim, n_axis, n_element = X.shape

    # Convert dnn activtaion to bold response.
    if args.hrf is not None:
        onset = stim['variable']['onset']
        duration = stim['variable']['duration']

        X = analyzer.generate_bold_regressor(
                X.reshape(n_stim, -1), onset, duration, Y.shape[0], tr)
        X = X.reshape(Y.shape[0], n_axis, n_element)

        print('hrf convolution on dnn activation finished')

    # %% multivariate analysis
    """
    Third, we use multivariate model to explore the relations between
    CNN activation and brain/behavior responses.

    """

    if args.model == 'glm':
        model = linear_model.LinearRegression()
    elif args.model == 'lasso':
        model = linear_model.Lasso()
    elif args.model == 'svc':
        model = svm.SVC(kernel="linear", C=0.025)
    elif args.model == 'lrc':
        model = linear_model.LogisticRegression()
    else:
        raise Exception('Please use glm or lasso for linear regression and \
                        svc or lrc for linear classification')

    # set parameters of mv model validation
    if args.model == 'glm' or args.model == 'lasso':
        scoring = 'explained_variance'
    else:
        scoring = 'accuracy'

    # run and validate mv models
#    pred = []
    coef = [], score = []
    for i in range(n_axis):
        # run mv model to do prediction
        model.fit(X[:, i, :], Y)
#        pred.append(model.predict(X[:, i, :]))
        coef.append(model.coef_.T)
        # validate the model
        score_chn = [model_selection.cross_val_score(
                model, X[:, i, :], Y[:, j], scoring=scoring, cv=args.cvfold
                ) for j in range(Y.shape[1])]
        score.append(np.asarray(score_chn).mean(-1)[np.newaxis, :])
        print('model fitting and validation on axis {0} finished'.format(
                i+1))
#    pred = np.asarray(pred).transpose(1, 0, 2)
    coef = np.asarray(coef).transpose(1, 0, 2)
    score = np.asarray(score).transpose(1, 0, 2)

    print('model fitting and validation finished')

    # %% save the results to disk
    """
    Finally, we save the related results to the outdir.

    """
    if os.path.exists(args.outdir) is not True:
        os.makedirs(args.outdir)

    # save files
    if args.response.endswith('db.csv'):

        # save model score
        dict_score = {keys: score[0, :, i] for i, keys
                      in enumerate(resp_dict['variable'].keys())}
        score_op = pjoin(args.outdir, 'model_score.db.csv')
        dio.save_dnn_csv(score_op, 'model_score', 'test', 'col', dict_score)

    else:

        def wrap_nii(model_data, bmask, bshape):
            """ wrap model_data into nii data

            parameters:
            -----------
            model_data[array]: [n_axis, n_voxel]
            bmask[array]: [n_voxel] reshaped bmask,without geometrical info
            bshape[array]: size = 3
            """
            n_axis = model_data.shape[0]
            img = np.zeros((n_axis, bmask.shape[0]))
            img[:, bmask != 0] = model_data
            img = img.reshape((n_axis, bshape[1], bshape[2], bshape[3]))

            return img

        # save model score
        score_op = pjoin(args.outdir, 'model_score.nii.gz')
        score_oimg = wrap_nii(score[0, :, :], bmask, bshape)
        bio.save_brainimg(score_op, score_oimg, header)


if __name__ == '__main__':
    main()
