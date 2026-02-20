# Sample Upload Reference Files

These files are ready to upload in the web UI tab: `Upload Reference`.

Supported schemas:
- `s,offset`: normalized progress `s` in `[0,1]`, lateral `offset` in meters
- `x,y`: world coordinates in meters
- `x,y,heading,velocity,time` (or aliases `t`, `v`, `vx`)
  - `heading` can be radians or degrees
  - `time` must be strictly increasing (seconds)
  - `velocity` is in m/s

Suggested test order:
1. `s_offset_centerline.csv` (very easy sanity test)
2. `s_offset_smooth_lane_change.csv` (easy)
3. `xy_heading_t_timeprofile.csv` (medium, radians heading)
4. `xy_heading_deg_timeprofile.csv` (medium, heading in degrees)
5. `s_offset_s_curve.csv` (medium-hard)
6. `s_offset_zigzag.csv` (hard/aggressive)
7. `xy_manual_points.csv` (direct points only)
