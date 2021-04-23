import csv
import bpy
import mathutils
import bmesh

data = []

bpy.ops.object.empty_add(type='ARROWS', radius=5)

with open('/Users/fuerstchristian/PycharmProjects/TestProject/Downhill-Simplex/nelder_mead.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        data.append(row)
    if row[4] == 'Himmelblau':
        bpy.ops.mesh.primitive_z_function_surface(equation = "(x**2 + y - 11)**2 + (x + y**2 - 7)**2", div_x = 200, div_y = 200, size_x = 5, size_y = 5)
    elif row[4] == 'Rosenbrock':
        bpy.ops.mesh.primitive_z_function_surface(equation = "(1 - x)**2 + 100 * ((y - (x**2))**2)", div_x = 200, div_y = 200, size_x = 5, size_y = 5) 

    mat = bpy.data.materials.new("PKHG")
    mat.diffuse_color = (1,1,1, 1)
    o = bpy.context.selected_objects[0] 
    o.active_material = mat
    i = 0   

spheres = bpy.data.collections["Sphere"].objects
offset = 0
data_length = len(data)
data_range = range(0, data_length, 4)
spheres[0].scale = [1,1,1]
spheres[1].scale = [1,1,1]
spheres[2].scale = [1,1,1]
spheres[3].scale = [1,1,1]

for d in data_range:
    spheres[0].location = [float(data[d][0]), float(data[d][1]), float(data[d][2])]
    spheres[0].keyframe_insert(data_path="location", frame=0+offset)
    spheres[1].location = [float(data[d+1][0]), float(data[d+1][1]), float(data[d+1][2])]
    spheres[1].keyframe_insert(data_path="location", frame=0+offset)
    spheres[2].location = [float(data[d+2][0]), float(data[d+2][1]), float(data[d+2][2])]
    spheres[2].keyframe_insert(data_path="location", frame=0+offset)
    spheres[3].location = [float(data[d+3][0]), float(data[d+3][1]), float(data[d+3][2])]
    spheres[3].keyframe_insert(data_path="location", frame=0+offset)
    offset += 10
