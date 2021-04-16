import csv
import bpy
import mathutils
import bmesh

data = []

# bpy.ops.mesh.primitive_z_function_surface(equation = "(1 - x)**2 + 100 * ((y - (x**2))**2)", div_x = 15, div_y = 15, size_x = 2, size_y = 2) 
bpy.ops.mesh.primitive_z_function_surface(equation = "(x**2 + y - 11)**2 + (x + y**2 - 7)**2", div_x = 15, div_y = 15, size_x = 10, size_y = 10)

bpy.ops.object.empty_add(type='ARROWS', radius=5)

with open('/Users/fuerstchristian/nelder_mead_2.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        data.append(row)
        
    i = 0   
    
    while i <= 19 * 4:   

        vert = bpy.ops.mesh.primitive_vert_add()
        
        bpy.ops.object.mode_set(mode='EDIT')

        obj = bpy.context.object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        v1 = bm.verts.new((float(data[i][0]), float(data[i][1]), float(data[i][2])))
        i = i + 1
        v2 = bm.verts.new((float(data[i][0]), float(data[i][1]), float(data[i][2])))
        i += 1
        v3 = bm.verts.new((float(data[i][0]), float(data[i][1]), float(data[i][2])))
        i += 1
        v4 = bm.verts.new((float(data[i][0]), float(data[i][1]), float(data[i][2])))
        i += 1

        bm.edges.new((v1, v2))
        bm.edges.new((v2, v3))
        bm.edges.new((v1, v3))
        bm.edges.new((v1, v4))
        bm.edges.new((v2, v4))
        bm.edges.new((v3, v4))

        bmesh.update_edit_mesh(obj.data)

        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=10, time_limit=0.5)

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.delete(use_global=False)
