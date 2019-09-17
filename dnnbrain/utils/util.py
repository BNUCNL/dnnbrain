import cv2


def get_frame_time_info(vid_file, original_onset, interval=1, before_vid=0, after_vid=0):
    """
    Extract frames of interest from a video with their onsets and durations,
    according to the experimental design.

    Parameters:
    -----------
    vid_file[str]: video file path
    original_onset[float]: the first stimulus' time point relative to the beginning of the response
        For example, if the response begins at 14 seconds after the first stimulus, the original_onset is -14.
    interval[int]: Get one frame per 'interval' frames
    before_vid[float]: Display the first frame as a static picture for 'before_vid' seconds before video.
    after_vid[float]: Display the last frame as a static picture for 'after_vid' seconds after video.

    Returns:
    --------
    frame_nums[list]: sequence numbers of the frames of interest
    onsets[list]: onsets of the frames of interest
    durations[list]: durations of the frames of interest
    """
    assert isinstance(interval, int) and interval > 0, "Parameter 'interval' must be a positive integer!"

    # load video information
    vid_cap = cv2.VideoCapture(vid_file)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    n_frame = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # generate sequence numbers
    frame_nums = list(range(1, n_frame+1, interval))

    # generate durations
    duration = 1 / fps * interval
    durations = [duration] * len(frame_nums)
    durations[0] = durations[0] + before_vid
    durations[-1] = durations[-1] + after_vid

    # generate onsets
    onsets = [original_onset]
    for d in durations[:-1]:
        onsets.append(onsets[-1] + d)

    return frame_nums, onsets, durations
