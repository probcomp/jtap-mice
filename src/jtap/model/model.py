from genjax import gen
from .stepper import stepper_model
from .initialization import init_model
from .likelihood import likelihood_model

def get_render_args(mi, x, y):
   return (mi.pix_x, mi.pix_y, mi.diameter,
                     x, y, mi.masked_barriers, mi.masked_occluders, 
                     mi.red_sensor, mi.green_sensor, mi.image_discretization)

@gen
def full_init_model(mi):
   init_mo = init_model(mi) @ "init"
   likelihood_model(get_render_args(mi, init_mo.x, init_mo.y), mi.pixel_corruption_prob, mi.tile_size_arr, mi.σ_pixel_spatial, mi.image_power_beta) @ "init_obs"
   return init_mo

@gen
def full_step_model(mo, mi, inference_mode_bool):
   mo = stepper_model(mo, mi, inference_mode_bool) @ "step"
   likelihood_model(get_render_args(mi, mo.x, mo.y), mi.pixel_corruption_prob, mi.tile_size_arr, mi.σ_pixel_spatial, mi.image_power_beta) @ "obs"
   return mo
