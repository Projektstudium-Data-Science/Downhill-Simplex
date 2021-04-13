import csv
import bpy

data = []

bpy.ops.object.empty_add(type='ARROWS', radius=5)

with open('/Users/fuerstchristian/nelder_mead.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        # data.append(row)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(float(row[0]), float(row[1]), float(row[2])))
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=10, time_limit=0.2)
