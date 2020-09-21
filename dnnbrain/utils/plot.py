import matplotlib.pyplot as plt
import numpy as np


def imgarray_show(x, nrows=1, ncols=1, row_label=None, vmin=None, vmax=None,
                  figsize=(10, 6), cmap='coolwarm', cbar=False, frame_on=True,
                  img_names=None, show=True, save_path=None):
    """
    create a figure showing multiple images.
    
    Parameters
    ----------
    x : list 
        List of image array (2d or 3d-RGB[A]).
    nrows, ncols : int 
        Number of rows/columns of the subplot grid.
    row_label : list 
        Row names.
    vmin, vmax : scalar 
        **vmin** and **vmax** define the value range of 
        colormap applied to all images. By default, colormaps
        adapt to each image's value range.
    figsize : float, float
        *width*, height of figure in inches.
    cmap : str 
        The Colormap instance or registered colormap name used 
        to map scalar data to colors. 
    frame_on : bool 
        Set whether the axes rectangle patch is drawn.
    img_names : list 
        Image names with the same length as x.
    show : bool 
        Set whether the figure is displayed.
    save_path : str 
        File path to save the figure.
        
    Return
    ------
    matplotlib.figure : object
    """

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            subplot_kw={'xticks': [], 'yticks': [],
                                        'frame_on': frame_on},
                            figsize=figsize)

    for i, ax in enumerate(axs.flat[:len(x)]):
        if row_label is not None and np.mod(i, ncols) == 0:
            ax.set_ylabel(row_label[i//ncols])
        if img_names is not None:
            ax.set_xlabel(img_names[i])
        img = ax.imshow(x[i], cmap=cmap, vmin=vmin, vmax=vmax)
        if cbar:
            fig.colorbar(img, ax=ax)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)

    if show is True:
        plt.show()
