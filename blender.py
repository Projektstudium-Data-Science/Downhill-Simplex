import csv
import bpy

data = []

# bpy.ops.mesh.primitive_z_function_surface(equation = "(1 - x)**2 + 100 * ((y - (x**2))**2)", div_x = 15, div_y = 15, size_x = 2, size_y = 2) 
bpy.ops.mesh.primitive_z_function_surface(equation = "(x**2 + y - 11)**2 + (x + y**2 - 7)**2", div_x = 15, div_y = 15, size_x = 10, size_y = 10)

bpy.ops.object.empty_add(type='ARROWS', radius=5)

with open('/Users/fuerstchristian/nelder_mead.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        # data.append(row)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(float(row[0]), float(row[1]), float(row[2])))
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=10, time_limit=0.2)
