import rerun as rr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML as HTML_Display
from jtap_mice.utils.stimuli import discrete_obs_to_rgb

def rerun_jtap_stimulus(rgb_video = None, discrete_obs = None, stimulus_name = None, rerun_url = "rerun+http://127.0.0.1:9876/proxy", is_lr = False):


    if rgb_video is None and discrete_obs is None:
        raise ValueError("Must provide either rgb_video or discrete_obs")
    if rgb_video is not None:
        assert discrete_obs is None, "Cannot provide both rgb_video and discrete_obs"
    if discrete_obs is not None:
        assert rgb_video is None, "Cannot provide both rgb_video and discrete_obs"

    if discrete_obs is not None:
        rgb_video = discrete_obs_to_rgb(discrete_obs)

    viz_name = "jtap_stimulus"
    if stimulus_name is not None:
        viz_name = f"jtap_stimulus_{stimulus_name}"
    rr.init(viz_name, spawn=False)
    rr.connect_grpc(url=rerun_url)
    rr.log("/", rr.Clear(recursive=True))

    for i in range(len(rgb_video)):
        rr.set_time(timeline = "frame", sequence = i)
        rr.log("stimulus", rr.Image(rgb_video[i]))


def matplotlib_jtap_stimulus(
    stimulus,
    fps=None,
    image_scale=4,
    return_html=True
):
    """
    Visualize stimulus as an animated matplotlib figure.
    
    Parameters:
    -----------
    stimulus : JTAPMiceStimulus
        Stimulus object from load_left_right_stimulus
    fps : float, optional
        Frames per second for the animation. If None, uses stimulus.fps / stimulus.skip_t
    image_scale : float, default=4
        Scale factor for figure size
    return_html : bool, default=True
        If True, return HTML5 video. If False, return JSHTML.
    
    Returns:
    --------
    IPython.display.HTML
        HTML display of the animation
    """
    # Get RGB video from stimulus (following pattern from jtap_viz.py)
    if hasattr(stimulus, "rgb_video_highres"):
        rgb_video = np.asarray(stimulus.rgb_video_highres)
    elif hasattr(stimulus, "rgb_video"):
        rgb_video = np.asarray(stimulus.rgb_video)
    elif hasattr(stimulus, "discrete_obs"):
        rgb_video = discrete_obs_to_rgb(stimulus.discrete_obs)
    else:
        raise ValueError("Could not find rgb frames in stimulus.")
    
    num_frames, H, W = rgb_video.shape[:3]
    
    # Get fps from stimulus if not provided
    if fps is None:
        fps = stimulus.fps / stimulus.skip_t
    
    # Create figure with no margins/padding - just the video
    fig = plt.figure(
        figsize=(image_scale * W / 100, image_scale * H / 100),
        facecolor='w'
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    ax = fig.add_subplot(111, facecolor='w')
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    
    # Initialize image artist
    img_artist = ax.imshow(
        np.zeros_like(rgb_video[0]),
        origin="upper",
        animated=True,
        zorder=0,
        extent=(0, W, H, 0)
    )
    
    def animate(t):
        img = rgb_video[t]
        img_artist.set_data(img)
        return [img_artist]
    
    anim = animation.FuncAnimation(
        fig, animate, frames=num_frames, interval=(1000/fps), blit=True, repeat=True
    )
    plt.close()
    
    if return_html:
        return HTML_Display(anim.to_html5_video())
    else:
        return HTML_Display(anim.to_jshtml())