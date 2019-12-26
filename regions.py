_basin_bounds = {
    'GL': ((0, 360, -90, 90),),
    'PA': ((290, 276, -80,  9),
           (276, 270, -80, 14),
           (270, 260, -80, 18),
           (260, 180, -80, 66),
           (180, 145, -80, 66),
           (145, 100,   0, 66)),
    'IN': ((100,   20, -80, 31),
           (145,  100, -80,  0)),
    'AT': ((100,   20,  31, 90),
           (20,     0, -80, 90),
           (360,  290, -80, 90),
           (290,  276,   9, 90),
           (276,  270,  14, 90),
           (270,  260,  18, 90),
           (260,  100,  66, 90))}


def _basin_check(basin, lat, lon):
    """Returns a mask indicating which observation points (d) are
     within the given basin"""
    mask = None
    for box in _basin_bounds[basin]:
        m2 = (lon <= box[0]) & (lon > box[1]) & \
             (lat >= box[2]) & (lat < box[3])
        mask = m2 if mask is None else (mask | m2)
    return mask


ocean = {
    # by latitude band
    'r_gl': lambda d: (d[0] <= 60) & (d[0] >= -60),
    'r_np': lambda d: (d[0] > 60),
    'r_nh': lambda d: (d[0] <= 60) & (d[0] >= 20),
    'r_tp': lambda d: (d[0] < 20) & (d[0] > -20),
    'r_sh': lambda d: (d[0] >= -60) & (d[0] <= -20),
    'r_sp': lambda d: (d[0] < -60),

    # by basin
    'b_pa': lambda d: _basin_check('PA', d[0], d[1]),
    'b_in': lambda d: _basin_check('IN', d[0], d[1]),
    'b_at': lambda d: _basin_check('AT', d[0], d[1]),

    # by nino regions
    'r_nino3': lambda d: (d[1] >= 210) & (d[1] <= 270) & \
                         (d[0] >= -5) & (d[0] <= 5),
    'r_nino34': lambda d: (d[1] >= 190) & (d[1] <= 240) & \
                          (d[0] >= -5) & (d[0] <= 5),
    'r_nino4': lambda d: (d[1] >= 160) & (d[1] <= 210) & \
                         (d[0] >= -5) & (d[0] <= 5),
    'r_nino12': lambda d: (d[1] >= 270) & (d[1] <= 280) & \
                          (d[0] >= -10) & (d[0] <= 0),
}

ocean['r_nh_pa'] = lambda d: ocean['r_nh'](d) & ocean['b_pa'](d)
ocean['r_tp_pa'] = lambda d: ocean['r_tp'](d) & ocean['b_pa'](d)
ocean['r_sh_pa'] = lambda d: ocean['r_sh'](d) & ocean['b_pa'](d)

ocean['r_nh_at'] = lambda d: ocean['r_nh'](d) & ocean['b_at'](d)
ocean['r_tp_at'] = lambda d: ocean['r_tp'](d) & ocean['b_at'](d)
ocean['r_sh_at'] = lambda d: ocean['r_sh'](d) & ocean['b_at'](d)

ocean['r_nh_in'] = lambda d: ocean['r_nh'](d) & ocean['b_in'](d)
ocean['r_tp_in'] = lambda d: ocean['r_tp'](d) & ocean['b_in'](d)
ocean['r_sh_in'] = lambda d: ocean['r_sh'](d) & ocean['b_in'](d)
