from importlib.resources import path
import bpy
import asyncio
import os
import math

from ..preferences import StableDiffusionPreferences
from ..async_loop import *
from ..pil_to_image import *
from ..prompt_engineering import *
from ..absolute_path import WEIGHTS_PATH, absolute_path
from .install_dependencies import are_dependencies_installed

import tempfile

# A shared `Generate` instance.
# This allows the slow model loading process to happen once,
# and re-use the model on subsequent calls.
generator = None

def image_has_alpha(img):
    b = 32 if img.is_float else 8
    return (
        img.depth == 2*b or   # Grayscale+Alpha
        img.depth == 4*b      # RGB+Alpha
    )

class DreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture"
    bl_label = "Dream Texture"
    bl_description = "Generate a texture with AI"
    bl_options = {'REGISTER'}
    
    def invoke(self, context, event):
        weights_installed = os.path.exists(WEIGHTS_PATH)
        if not weights_installed or not are_dependencies_installed():
            self.report({'ERROR'}, "Please complete setup in the preferences window.")
            return {"FINISHED"}
        else:
            return self.execute(context)

    def draw_fast64(self, context, layout):
        settings = context.scene.dream_textures_fast64
        fast64_box = layout.box()
        fast64_box_heading = fast64_box.row()
        fast64_box_heading.prop(settings, "enable")
        fast64_box_heading.label(text="Fast64")
        if settings.enable:
            fast64_box_content = fast64_box.column()
            fast64_box_content.row().prop(settings, "dimensions", text="Resize")
            fast64_box_content.prop(settings, "texture_index", text="Index")
            fast64_box_content.label(text="If in 3D Viewport, an object must be selected.")
            fast64_box_content.label(text="The texture will be set on the active material.")
            fast64_box_content.label(text="Visuals may not update correctly. (ex. clamp)", icon = "ERROR")
            fast64_box_content.label(text="In this case try toggling a texture setting.")


    def draw(self, context):
        layout = self.layout
        
        scene = context.scene
        
        prompt_box = layout.box()
        prompt_box_heading = prompt_box.row()
        prompt_box_heading.label(text="Prompt")
        prompt_box_heading.prop(scene.dream_textures_prompt, "prompt_structure")
        structure = next(x for x in prompt_structures if x.id == scene.dream_textures_prompt.prompt_structure)
        for segment in structure.structure:
            segment_row = prompt_box.row()
            enum_prop = 'prompt_structure_token_' + segment.id + '_enum'
            is_custom = getattr(scene.dream_textures_prompt, enum_prop) == 'custom'
            if is_custom:
                segment_row.prop(scene.dream_textures_prompt, 'prompt_structure_token_' + segment.id)
            segment_row.prop(scene.dream_textures_prompt, enum_prop, icon_only=is_custom)
        
        size_box = layout.box()
        size_box.label(text="Configuration")
        size_box.prop(scene.dream_textures_prompt, "width")
        size_box.prop(scene.dream_textures_prompt, "height")
        size_box.prop(scene.dream_textures_prompt, "seamless")
        
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                if area.spaces.active.image is not None and image_has_alpha(area.spaces.active.image):
                    inpainting_box = layout.box()
                    inpainting_heading = inpainting_box.row()
                    inpainting_heading.prop(scene.dream_textures_prompt, "use_inpainting")
                    inpainting_heading.label(text="Inpaint Open Image")
                    break

        if not scene.dream_textures_prompt.use_inpainting:
            init_img_box = layout.box()
            init_img_heading = init_img_box.row()
            init_img_heading.prop(scene.dream_textures_prompt, "use_init_img")
            init_img_heading.label(text="Init Image")
            if scene.dream_textures_prompt.use_init_img:
                init_img_box.template_ID(context.scene, "init_img", open="image.open")
                init_img_box.prop(scene.dream_textures_prompt, "strength")
                init_img_box.prop(scene.dream_textures_prompt, "fit")

        self.draw_fast64(context, layout)

        advanced_box = layout.box()
        advanced_box_heading = advanced_box.row()
        advanced_box_heading.prop(scene.dream_textures_prompt, "show_advanced", icon="DOWNARROW_HLT" if scene.dream_textures_prompt.show_advanced else "RIGHTARROW_THIN", emboss=False, icon_only=True)
        advanced_box_heading.label(text="Advanced Configuration")
        if scene.dream_textures_prompt.show_advanced:
            advanced_box.prop(scene.dream_textures_prompt, "full_precision")
            advanced_box.prop(scene.dream_textures_prompt, "seed")
            # advanced_box.prop(self, "iterations") # Disabled until supported by the addon.
            advanced_box.prop(scene.dream_textures_prompt, "steps")
            advanced_box.prop(scene.dream_textures_prompt, "cfgscale")
            advanced_box.prop(scene.dream_textures_prompt, "sampler")
            advanced_box.prop(scene.dream_textures_prompt, "show_steps")

    async def dream_texture(self, context):
        history_entry = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.history.add()
        for prop in context.scene.dream_textures_prompt.__annotations__.keys():
            if hasattr(history_entry, prop):
                setattr(history_entry, prop, getattr(context.scene.dream_textures_prompt, prop))

        generated_prompt = context.scene.dream_textures_prompt.generate_prompt()

        # Support Apple Silicon GPUs as much as possible.
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        from ..stable_diffusion.ldm.generate import Generate
        from omegaconf import OmegaConf
        
        models_config  = absolute_path('stable_diffusion/configs/models.yaml')
        model   = 'stable-diffusion-1.4'

        models  = OmegaConf.load(models_config)
        config  = absolute_path('stable_diffusion/' + models[model].config)
        weights = absolute_path('stable_diffusion/' + models[model].weights)

        global generator
        if generator is None or generator.full_precision != context.scene.dream_textures_prompt.full_precision:
            generator = Generate(
                conf=models_config,
                model=model,
                # These args are deprecated, but we need them to specify an absolute path to the weights.
                weights=weights,
                config=config,
                full_precision=context.scene.dream_textures_prompt.full_precision
            )
            generator.load_model()

        node_tree = context.material.node_tree if hasattr(context, 'material') else None
        window_manager = context.window_manager
        screen = context.screen
        last_data_block = None
        scene = context.scene

        def get_f3d_material():
            obj = context.view_layer.objects.active

            material = None
            if hasattr(context, "material"):
                material = context.material
            elif obj is not None:
                material = obj.active_material
            else:
                print("Fast64: No active object.")
            
            f3d_material = None
            if material is not None and material.is_f3d:
                f3d_material = material.f3d_mat
            else:
                print("Fast64: No active fast64 material.")
            
            return material, f3d_material

        def apply_fast64(image : bpy.types.Image):
            if image is None:
                return
            
            settings = context.scene.dream_textures_fast64
            material, f3d_material = get_f3d_material()
            
            image.scale(settings.dimensions[0], settings.dimensions[1])

            if f3d_material is not None:
                if settings.texture_index == "Texture 1":
                    tex_index = 1
                    texture_prop = f3d_material.tex1
                else:
                    tex_index = 0
                    texture_prop = f3d_material.tex0
                texture_prop.tex = image

                # Note that calling bpy.ops.material.update_f3d_nodes does not work,
                # because overriding context here causes a crash for some reason.
                # Thus we manually set properties here for now.
                # This only handles everything that changes between image generations,
                # so the initial setting may not be correct.
                
                # Set shader nodes
                for i in range(1, 5):
                    nodeName = f"Tex{tex_index}_{i}"
                    if material.node_tree.nodes.get(nodeName):
                        material.node_tree.nodes[nodeName].image = image

                # Set dimensions
                nodes = material.node_tree.nodes
                uv_basis: bpy.types.ShaderNodeGroup = nodes["UV Basis"]
                inputs = uv_basis.inputs

                inputs[f"{tex_index} S TexSize"].default_value = image.size[0]
                inputs[f"{tex_index} T TexSize"].default_value = image.size[1]

            else:
                print("Fast64: No f3d material found.")

        def image_writer(image, seed, upscaled=False):
            nonlocal last_data_block
            # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
            if not upscaled:
                if last_data_block is not None:
                    bpy.data.images.remove(last_data_block)
                    last_data_block = None
                image = pil_to_image(image, name=f"{seed}")
                fast64_settings = context.scene.dream_textures_fast64
                if node_tree is not None and not fast64_settings.enable:
                    nodes = node_tree.nodes
                    texture_node = nodes.new("ShaderNodeTexImage")
                    texture_node.image = image
                    nodes.active = texture_node
                for area in screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        area.spaces.active.image = image
                if fast64_settings.enable:
                    apply_fast64(image)
                window_manager.progress_end()
        
        def view_step(samples, step):
            step_progress(samples, step)
            nonlocal last_data_block
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    step_image = pil_to_image(generator._sample_to_image(samples), name=f'Step {step + 1}/{scene.dream_textures_prompt.steps}')
                    area.spaces.active.image = step_image
                    if last_data_block is not None:
                        bpy.data.images.remove(last_data_block)
                    last_data_block = step_image
                    return # Only perform this on the first image editor found.
        
        def step_progress(samples, step):
            window_manager.progress_update(step)

        def save_temp_image(img, path=None):
            path = path if path is not None else tempfile.NamedTemporaryFile().name

            settings = scene.render.image_settings
            file_format = settings.file_format
            mode = settings.color_mode
            depth = settings.color_depth

            settings.file_format = 'PNG'
            settings.color_mode = 'RGBA'
            settings.color_depth = '8'

            img.save_render(path)

            settings.file_format = file_format
            settings.color_mode = mode
            settings.color_depth = depth

            return path

        def perform():
            window_manager.progress_begin(0, scene.dream_textures_prompt.steps)
            init_img = scene.init_img if scene.dream_textures_prompt.use_init_img else None
            if scene.dream_textures_prompt.use_inpainting:
                for area in screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        if area.spaces.active.image is not None and image_has_alpha(area.spaces.active.image):
                            init_img = area.spaces.active.image
            init_img_path = None
            if init_img is not None:
                init_img_path = save_temp_image(init_img)

            generator.prompt2image(
                # prompt string (no default)
                prompt=generated_prompt,
                # iterations (1); image count=iterations
                iterations=scene.dream_textures_prompt.iterations,
                # refinement steps per iteration
                steps=scene.dream_textures_prompt.steps,
                # seed for random number generator
                seed=None if scene.dream_textures_prompt.seed == -1 else scene.dream_textures_prompt.seed,
                # width of image, in multiples of 64 (512)
                width=scene.dream_textures_prompt.width,
                # height of image, in multiples of 64 (512)
                height=scene.dream_textures_prompt.height,
                # how strongly the prompt influences the image (7.5) (must be >1)
                cfg_scale=scene.dream_textures_prompt.cfgscale,
                # path to an initial image - its dimensions override width and height
                init_img=init_img_path,

                # generate tileable/seamless textures
                seamless=scene.dream_textures_prompt.seamless,

                fit=scene.dream_textures_prompt.fit,
                # strength for noising/unnoising init_img. 0.0 preserves image exactly, 1.0 replaces it completely
                strength=scene.dream_textures_prompt.strength,
                # strength for GFPGAN. 0.0 preserves image exactly, 1.0 replaces it completely
                gfpgan_strength=0.0, # 0 disables upscaling, which is currently not supported by the addon.
                # image randomness (eta=0.0 means the same seed always produces the same image)
                ddim_eta=0.0,
                # a function or method that will be called each step
                step_callback=view_step if scene.dream_textures_prompt.show_steps else step_progress,
                # a function or method that will be called each time an image is generated
                image_callback=image_writer,
                
                sampler_name=scene.dream_textures_prompt.sampler
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, perform)

    def execute(self, context):
        async_task = asyncio.ensure_future(self.dream_texture(context))
        # async_task.add_done_callback(done_callback)
        ensure_async_loop()

        return {'FINISHED'}
