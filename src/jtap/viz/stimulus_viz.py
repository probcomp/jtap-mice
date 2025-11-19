import rerun as rr
from jtap.utils.stimuli import discrete_obs_to_rgb, discrete_lr_obs_to_rgb

def rerun_jtap_stimulus(rgb_video = None, discrete_obs = None, stimulus_name = None, rerun_url = "rerun+http://127.0.0.1:9876/proxy", is_lr = False):


    if rgb_video is None and discrete_obs is None:
        raise ValueError("Must provide either rgb_video or discrete_obs")
    if rgb_video is not None:
        assert discrete_obs is None, "Cannot provide both rgb_video and discrete_obs"
    if discrete_obs is not None:
        assert rgb_video is None, "Cannot provide both rgb_video and discrete_obs"

    if discrete_obs is not None:
        if is_lr:
            rgb_video = discrete_lr_obs_to_rgb(discrete_obs)
        else:
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