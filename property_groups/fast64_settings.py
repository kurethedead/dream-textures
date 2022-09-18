import bpy

fast64_texture_enum = [
    ("Texture 0", "Texture 0", "Texture 0"),
    ("Texture 1", "Texture 1", "Texture 1"),
]

class DreamTextureFast64Settings(bpy.types.PropertyGroup):
    enable : bpy.props.BoolProperty(name = "", default = False)
    dimensions : bpy.props.IntVectorProperty(size = 2, min = 1, max = 1024, default = (32, 32))
    texture_index : bpy.props.EnumProperty(items = fast64_texture_enum)
