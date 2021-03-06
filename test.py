import csv
import bpy
import mathutils
import bmesh

data = []
"""
# bpy.ops.mesh.primitive_z_function_surface(equation = "(1 - x)**2 + 100 * ((y - (x**2))**2)", div_x = 15, div_y = 15, size_x = 2, size_y = 2) 
bpy.ops.mesh.primitive_z_function_surface(equation = "(x**2 + y - 11)**2 + (x + y**2 - 7)**2", div_x = 15, div_y = 15, size_x = 10, size_y = 10)

bpy.ops.object.empty_add(type='ARROWS', radius=5)
"""
vert = bpy.ops.mesh.primitive_vert_add()

# bpy.ops.object.editmode_toggle()
bpy.ops.object.mode_set(mode='EDIT')

obj = bpy.context.object
me = obj.data
bm = bmesh.from_edit_mesh(me)

v1 = bm.verts.new((-2.9, 2.7, 7))
v2 = bm.verts.new((-2.7, 2.5, 12.5))
v3 = bm.verts.new((-2.7, 2.5, 14.3))
v4 = bm.verts.new((-3.3, 3.2, 10.2))


bm.edges.new((v1, v2))
bm.edges.new((v2, v3))
bm.edges.new((v1, v3))
bm.edges.new((v1, v4))
bm.edges.new((v2, v4))
bm.edges.new((v3, v4))

bmesh.update_edit_mesh(obj.data)

bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=10, time_limit=0.3)

bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.delete(use_global=False)

"""
with open('/Users/fuerstchristian/nelder_mead_2.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        # data.append(row)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(float(row[0]), float(row[1]), float(row[2])))
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=10, time_limit=0.2)

"""

