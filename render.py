import moderngl
import numpy as np
import pyrr
import os
import skimage.filters
import skimage.io

from PIL import Image
from PIL import ImageFilter

import random
import io_utils

EXPERIMENTAL_2_BAND = False

def generate_mask(image_width, image_height, kernel_size):
	image_high_np = np.zeros((int(image_height), int(image_width)), dtype=np.unit8)
	image_high_np[kernel_size:-kernel_size,kernel_size:-kernel_size] = 255
	image_high = Image.fromarray(image_high_np.astype(np.uint8))
	image_low = image_high.filter(ImageFilter.GaussianBlur(kernel_size))
	return image_high, image_low

def split_image_frequencies(image, sigma_size, auto_mask_sigma_size=None):
	image_np = np.asarray(image).astype(np.float32)
	truncate = 3
	
	if auto_mask_sigma_size is None:
		auto_mask_sigma_size = sigma_size / 5
	
	# if there is an alpha channel, split into RGB and A components
	if image_np.shape[2] > 3:
		mask_np = image_np[:,:,3].astype(np.float32) / 255
		image_np = image_np[:,:,:3]
	else:
		# generate a mask
		mask_np = np.zeros(image_np.shape[:2])
		padding = int(sigma_size*truncate)
		mask_np[padding:-padding,padding:-padding] = 1
		mask_np = skimage.filters.gaussian(mask_np, auto_mask_sigma_size, truncate=truncate)
		#mask_np[2*kernel_size:-2*kernel_size,2*kernel_size:-2*kernel_size] = 0
	
	image_low_np = skimage.filters.gaussian(image_np, sigma_size, truncate=truncate, multichannel=True)
	mask_low_np = skimage.filters.gaussian(mask_np, sigma_size, truncate=truncate)
	
	# Laplacian for RGB, Gaussian for A
	image_high_np = image_np - image_low_np
	mask_high_np = mask_np
	
	# Renormalize
	image_high_np = (image_high_np + 255) / 2
	
	# Recombine
	full_high_np = np.dstack([image_high_np, mask_high_np * 255])
	full_low_np = np.dstack([image_low_np, mask_low_np * 255])
		
	image_high = Image.fromarray(full_high_np.astype(np.uint8))
	image_low = Image.fromarray(full_low_np.astype(np.uint8))
	return image_high, image_low
	
	
def split_image_frequencies_no_alpha(image, kernel_size):
	image_low = image.filter(ImageFilter.GaussianBlur(kernel_size))
	
	image_np = np.asarray(image).astype(np.float32)
	image_low_np = np.asarray(image_low).astype(np.float32)
	
	image_high_np = np.zeros(image_np.shape)
	image_high_np = ((image_np - image_low_np) + 255) / 2
	
	image_high = Image.fromarray(image_high_np.astype(np.uint8))
	
	return image_high, image_low
	
def combine_image_frequencies(image_high, image_low):
	image_low_np = np.asarray(image_low).astype(np.float32)
	image_high_np = np.asarray(image_high).astype(np.float32)
	
	image_np = image_low_np + (image_high_np * 2 - 255)
	image_np = np.clip(image_np, 0, 255)
	
	image = Image.fromarray(image_np.astype(np.uint8))
	return image

class Texture_Loader:
	
	def __init__(self, moderngl_context):
		self.ctx = moderngl_context
		
		self.loaded_textures = {}
		
	def load_texture(self, fname, model_matr):
		if fname in self.loaded_textures:
			print('Using cached texture for {}'.format(fname))
			return self.loaded_textures[fname]
		else:
			image = Image.open(fname)
			
			if EXPERIMENTAL_2_BAND:
				image_high, image_low = split_image_frequencies(image, 100)
				
				texture_high = pil_image_to_texture(self.ctx, image_high)
				texture_high.anisotropy = 16.0
				texture_high.build_mipmaps()
				
				texture_low = pil_image_to_texture(self.ctx, image_low)
				
				print('Loaded texture {}'.format(fname))
			else:
				texture_high = pil_image_to_texture(self.ctx, image)
				texture_high.anisotropy = 16.0
				texture_high.build_mipmaps()
				
				texture_low = None
			
			result = (texture_high, texture_low)
			self.loaded_textures[fname] = result
			return result
	
	def load_texture_cube(self, fname):
		if fname in self.loaded_textures:
			print('Using cached texture cube for {}'.format(fname))
			return self.loaded_textures[fname]
		else:
			face_fnames = [
				os.path.join(fname, 'posx.jpg'),
				os.path.join(fname, 'negx.jpg'),
				os.path.join(fname, 'posy.jpg'),
				os.path.join(fname, 'negy.jpg'),
				os.path.join(fname, 'posz.jpg'),
				os.path.join(fname, 'negz.jpg'),
			]
			
			face_images = [Image.open(x) for x in face_fnames]
			
			if EXPERIMENTAL_2_BAND:
				face_images_hl = [split_image_frequencies_no_alpha(x, 30) for x in face_images]
				face_images_high = [f[0] for f in face_images_hl]
				face_images_low = [f[1] for f in face_images_hl]
				
				
				face_bytes_high = [x.tobytes() for x in face_images_high]
				face_bytes_low = [x.tobytes() for x in face_images_low]
				
				texture_high = self.ctx.texture_cube(face_images_high[0].size, 3, b''.join(face_bytes_high))
				texture_low = self.ctx.texture_cube(face_images_low[0].size, 3, b''.join(face_bytes_low))
			else:
				face_bytes = [x.tobytes() for x in face_images]
				texture_high = self.ctx.texture_cube(face_images[0].size, 3, b''.join(face_bytes))
				texture_low = None
			
			print('Loaded texture {}'.format(fname))
			
			result = (texture_high, texture_low)
			self.loaded_textures[fname] = result
			return result
	
	def clear_all(self):
		self.loaded_textures.clear()

class PanoObj:
	
	def __init__(self, custom_name, texture_high, texture_low, model_matr, model_matr_rotation, is_skybox, source_fname, mask_image, mask_texture):
		self.custom_name = custom_name
		self.texture_high = texture_high
		self.texture_low = texture_low
		self.mask_texture = mask_texture
		self.mask_image = mask_image
		self.model_matr = model_matr
		self.model_matr_rotation = model_matr_rotation
		self.source_fname = source_fname
		self.is_skybox = is_skybox
		
	def renormalize_model_matrix(self):
		center_point = self.model_matr @ np.array([0.5, 0.5, 0.0, 1.0])
		self.model_matr[:3,:4] *= 10 / np.linalg.norm(center_point[:3])
	
	def apply_world_transform(self, matr):
		self.model_matr = matr @ self.model_matr
		self.renormalize_model_matrix()
		
	def apply_world_rotation(self, rot_matr):
		self.model_matr_rotation = rot_matr @ self.model_matr_rotation
		
	def update_mask_texture(self):
		self.mask_texture.write(np_array_to_pil_image(self.mask_image).tobytes())
		

class View_Params:
	def __init__(self, pitch_rad=0, yaw_rad=0, fov=90):
		self.pitch_rad = pitch_rad
		self.yaw_rad = yaw_rad
		self.fov = fov
		
	def compute_view_matr(self):
		matr_view = pyrr.matrix44.create_look_at((0, 0, 0), pitch_yaw_to_direction(self.pitch_rad, self.yaw_rad), (0, 1, 0)).T
		return matr_view

	def look_natural(self, anchor_dir, dest_dir):
		self.pitch_rad, self.yaw_rad = direction_to_pitch_yaw(anchor_dir)
		

def model_matr_from_orientation(origin_loc, axis_u, axis_v):
	matr = np.eye(4, )
	matr[:3,3] = origin_loc
	matr[:3,0] = axis_u
	matr[:3,1] = axis_v
	return matr
		
def pitch_yaw_to_direction(pitch_rad, yaw_rad):
	'''
	X = cos(yaw) * cos(pitch)
	Y = sin(pitch)
	Z = sin(yaw) * cos(pitch)
	'''
	
	return np.array([
		np.cos(yaw_rad) * np.cos(pitch_rad),
		np.sin(pitch_rad),
		np.sin(yaw_rad) * np.cos(pitch_rad),
	])
	
def direction_to_pitch_yaw(direction):
	x,y,z = direction[:3]
	yaw = np.arctan2(z, x)
	pitch = np.arcsin(y)
	return pitch, yaw

class Renderer:
	
	def __init__(self, opengl_context):
		self.ctx = opengl_context
		
		self.texture_loader = Texture_Loader(self.ctx)
		
		self._init_pano_obj_shader()
		self._init_skybox_shader()
		self._init_pano_obj_vao()
		self._init_skybox_vao()
		self._init_fbo(100, 100)
		
		self._debug_scalar = 1
		
		self.pano_objs = []
		
		self.view_params = View_Params()
		
	def add_pano_obj(self, fname_image, model_matr=None, model_matr_rot=None, custom_name=None, mask_image=None):
		
		texture_high, texture_low = self.texture_loader.load_texture(fname_image, model_matr)
		
		if model_matr is None:
			model_matr = np.eye(4)
		if custom_name is None:
			custom_name = '{} {}'.format(random.randrange(0,99999), fname_image)
		if model_matr_rot is None:
			model_matr_rot = np.eye(4)
		if mask_image is None:
			mask_image = np.ones((256, 256))
			mask_image[:,0] = 0
			mask_image[0,:] = 0
			mask_image[:,-1] = 0
			mask_image[-1,:] = 0
		
		mask_texture = pil_image_to_texture(self.ctx, np_array_to_pil_image(mask_image))
			
		pano_obj = PanoObj(custom_name, texture_high, texture_low, model_matr, model_matr_rot, False, fname_image, mask_image, mask_texture)
		self.pano_objs.append(pano_obj)
		
		return pano_obj
		
	def add_skybox(self, folder_skybox, model_matr=None, model_matr_rot=None, custom_name=None):
		
		texture_high, texture_low = self.texture_loader.load_texture_cube(folder_skybox)
		
		if model_matr is None:
			model_matr = np.eye(4)
		if custom_name is None:
			custom_name = '{} {}'.format(random.randrange(0,99999), folder_skybox)
		if model_matr_rot is None:
			model_matr_rot = np.eye(4)
		pano_obj = PanoObj(custom_name, texture_high, texture_low, model_matr, model_matr_rot, True, folder_skybox, None, None)
		self.pano_objs.append(pano_obj)
		
		return pano_obj
		
	def clone_pano_obj(self, pano_obj_or_skybox):
		obj = pano_obj_or_skybox
		
		base_name = get_base_name(obj.custom_name)
		
		# Find an appropriate name
		all_names = [x.custom_name for x in self.pano_objs]
		next_name = 1
		while True:
			custom_name = 'C{} {}'.format(next_name, base_name)
			if custom_name not in all_names:
				break
			next_name += 1
		
		is_skybox = obj.is_skybox
		model_matr = obj.model_matr.copy()
		if is_skybox:
			texture_high, texture_low = self.texture_loader.load_texture_cube(obj.source_fname)
		else:
			texture_high, texture_low = self.texture_loader.load_texture(obj.source_fname, model_matr)
		mask_texture = pil_image_to_texture(self.ctx, np_array_to_pil_image(obj.mask_image))
		mask_image = obj.mask_image.copy()
		source_fname = obj.source_fname
		model_matr_rot = obj.model_matr_rotation.copy()
		
		clone = PanoObj(custom_name, texture_high, texture_low, model_matr, model_matr_rot, is_skybox, source_fname, mask_image, mask_texture)
		
		self.pano_objs.append(clone)
		return clone
		
	def clear_all(self):
		self.pano_objs.clear()
		self.texture_loader.clear_all()
		
	def _init_fbo(self, width, height):
		self.fbo_high = self.ctx.simple_framebuffer((width, height))
		if EXPERIMENTAL_2_BAND:
			self.fbo_low = self.ctx.simple_framebuffer((width, height))
		
	def _init_pano_obj_shader(self):
		
		self.pano_obj_shader_program = self.ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 unif_mvp;

				in vec3 in_pos;
				in vec2 in_uv;

				out vec2 vert_uv;

				void main() {
					vert_uv = in_uv;
					gl_Position = unif_mvp * vec4(in_pos, 1);
				}
			''',
			fragment_shader='''
				#version 330

				uniform sampler2D unif_texture;
				uniform sampler2D unif_texture_mask;

				in vec2 vert_uv;

				out vec4 frag_color;

				void main() {
					frag_color = texture(unif_texture, vert_uv);
					frag_color.a *= texture(unif_texture_mask, vert_uv).r;
				}
			'''
		)
		self.pano_obj_shader_program['unif_texture'] = 0
		self.pano_obj_shader_program['unif_texture_mask'] = 1
		
	def _init_skybox_shader(self):
		
		self.skybox_shader_program = self.ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 unif_unprojection;

				in vec3 in_pos;
				
				out vec4 vert_view_dir;

				void main() {
					vert_view_dir = unif_unprojection * vec4(in_pos, 1);
					gl_Position = vec4(in_pos, 1);
				}
			''',
			fragment_shader='''
				#version 330

				uniform samplerCube unif_texture;

				in vec4 vert_view_dir;

				out vec4 frag_color;

				void main() {
					frag_color = texture(unif_texture, vert_view_dir.xyz);
				}
			'''
		)
		
	def _init_pano_obj_vao(self):
		vert_buff = np.array([
			[0, 0, 0, 0, 0],
			[1, 0, 0, 1, 0],
			[0, 1, 0, 0, 1],
			
			[1, 1, 0, 1, 1],
			[0, 1, 0, 0, 1],
			[1, 0, 0, 1, 0],
		])
		vbo = self.ctx.buffer(vert_buff.astype(np.float32).tobytes())
		self.pano_obj_vao = self.ctx.simple_vertex_array(self.pano_obj_shader_program, vbo, 'in_pos', 'in_uv')
		
	def _init_skybox_vao(self):
		vert_buff = np.array([
			[-1, -1, 0],
			[1, -1, 0],
			[-1, 1, 0],
			
			[1, 1, 1],
			[-1, 1, 1],
			[1, -1, 1],
		])
		vbo = self.ctx.buffer(vert_buff.astype(np.float32).tobytes())
		self.skybox_vao = self.ctx.simple_vertex_array(self.skybox_shader_program, vbo, 'in_pos')
		
	def compute_view_matr(self):
		return self.view_params.compute_view_matr()
		
	def compute_proj_matr(self):
		return pyrr.matrix44.create_perspective_projection_matrix(self.view_params.fov, self.get_width() / self.get_height(), 0.1, 100.0).T
		
	def compute_view_proj_matr(self):
		matr_proj = self.compute_proj_matr()
		matr_view = self.compute_view_matr()
		matr_view_proj = matr_proj @ matr_view
		return matr_view_proj
			
	def resize(self, new_width, new_height):
		self._init_fbo(new_width, new_height)
		
	def get_width(self):
		return self.fbo_high.width
	def get_height(self):
		return self.fbo_high.height

	def get_world_dir(self, canvas_x, canvas_y):
		raise NotImplementedError()
		matr_view_proj = self.compute_view_proj_matr()
		matr_view_proj_inv = np.linalg.inv(matr_view_proj)
		
		ndc = np.array([
			(canvas_x / self.fbo_high.width) * 2 - 1,
			-((canvas_y / self.fbo_high.height) * 2 - 1),
			0,
			1,
		])
		
		homo_world_coords = matr_view_proj_inv @ ndc
		homo_world_coords /= homo_world_coords[3] # Perspective divice
		homo_world_coords[:3] /= np.linalg.norm(homo_world_coords[:3])
		homo_world_coords[3] = 0
		
		return homo_world_coords
		
	def get_highlight_bounds_for(self, pano_obj):
		if pano_obj.is_skybox:
			return None
			
		img_width = self.get_width()
		img_height = self.get_height()
		
		num_samples = 10
		verts = []
		verts.extend([x, 0] for x in np.linspace(0, 1, num_samples))
		verts.extend([0, x] for x in np.linspace(0, 1, num_samples))
		verts.extend([x, 1] for x in np.linspace(0, 1, num_samples))
		verts.extend([1, x] for x in np.linspace(0, 1, num_samples))
		
		matr_view_proj = self.compute_view_proj_matr()
		matr_mvp = matr_view_proj @ pano_obj.model_matr_rotation @ pano_obj.model_matr
		
		frame_locs = []
		
		point = np.zeros(4)
		point[3] = 1
		for vert in verts:
			point[:2] = vert
			ndc = matr_mvp @ point
			if ndc[3] <= 0:
				continue
			ndc /= ndc[3]
			
			# yes, this mutates ndc
			frame_loc = ndc[:2]
			frame_loc[1] *= -1
			frame_loc += 1
			frame_loc /= 2
			frame_loc[0] *= img_width
			frame_loc[1] *= img_height
			frame_locs.append(frame_loc)
		
		if len(frame_locs) == 0:
			return None
		
		frame_locs = np.array(frame_locs)
		
		upper_bound = np.max(frame_locs, axis=0)
		lower_bound = np.min(frame_locs, axis=0)
		
		return upper_bound, lower_bound
				
		
	def get_corners_world_pos(self, pano_obj):
		raise NotImplementedError()
		if pano_obj.is_skybox:
			raise RuntimeError('Corners are undefined for skybox objects')
		
		vertices = np.array([
			[0, 0, 0, 1],
			[0, 1, 0, 1],
			[1, 0, 0, 1],
			[1, 1, 0, 1],
		])
		
		result = np.zeros((4, 4))
		
		matr = pano_obj.model_matr_rotation @ pano_obj.model_matr
		for idx, vertex in enumerate(vertices):
			result[idx] = matr @ vertex
			
		return result
		
	def get_canvas_loc(self, world_pos):
		raise NotImplementedError()
		matr_view_proj = self.compute_view_proj_matr()
		
		if len(world_pos) != 4:
			homo = np.zeros(4,)
			homo[:3] = world_pos
			homo[3] = 1
		
		ndc = matr_view_proj @ world_pos
		
		frame = ndc / ndc[3]
		frame[1] *= -1
		
		
		
		
	def _render_layer(self, high_freq):
		matr_view_proj = self.compute_view_proj_matr()
		matr_view_proj_inv = np.linalg.inv(matr_view_proj)
		
		if high_freq:
			self.fbo_high.use()
			self.fbo_high.clear(0.5, 0.5, 0.5, 1.0)
		else:
			self.fbo_low.use()
			self.fbo_low.clear(0.0, 0.0, 0.0, 1.0)
			
		self.ctx.enable(moderngl.BLEND)
		
		for pano_obj in self.pano_objs:
			if high_freq:
				pano_obj.texture_high.use(0)
			else:
				pano_obj.texture_low.use(0)
			
			if pano_obj.is_skybox:
				model_matr_inv = np.linalg.inv(pano_obj.model_matr) @ pano_obj.model_matr_rotation.T
				self.skybox_shader_program['unif_unprojection'].write((model_matr_inv @ matr_view_proj_inv).T.astype(np.float32).tobytes())
				self.skybox_vao.render(moderngl.TRIANGLES)
			else:
				pano_obj.mask_texture.use(1)
				matr_mvp = matr_view_proj @ pano_obj.model_matr_rotation @ pano_obj.model_matr
				self.pano_obj_shader_program['unif_mvp'].write(matr_mvp.T.astype(np.float32).tobytes())
				self.pano_obj_vao.render(moderngl.TRIANGLES)

		if high_freq:
			image = Image.frombytes('RGB', self.fbo_high.size, self.fbo_high.read(), 'raw', 'RGB', 0, -1)
		else:
			image = Image.frombytes('RGB', self.fbo_low.size, self.fbo_low.read(), 'raw', 'RGB', 0, -1)
		return image
		
	def render(self):
		if EXPERIMENTAL_2_BAND:
			image_high = self._render_layer(True)
			image_low = self._render_layer(False)
			return combine_image_frequencies(image_high, image_low)
		else:
			image = self._render_layer(True)
			return image

	def get_vanishing_point_on_canvas(self, direction):
		
		matr_view_proj = self.compute_view_proj_matr()
		
		large = 10000
		
		dir_far = np.zeros(4)
		dir_far[:3] = direction * large
		dir_far[3] = 1
		
		raw_coords = matr_view_proj @ dir_far
		w = raw_coords[3]
		
		if w <= 0:
			return None
		raw_coords /= w
		raw_coords[1] *= -1
		
		canvas_coords = ((raw_coords[:2] + 1) / 2) * np.array((self.get_width(), self.get_height()))
		return canvas_coords
		

class Single_Image_Renderer:
	
	def __init__(self, opengl_context, image, caching=True):
		self.ctx = opengl_context
		self.image = image
		self.texture = pil_image_to_texture(self.ctx, image)
		self.texture.build_mipmaps()
		
		self.use_caching = caching
		
		self.image_draw_point = np.zeros(2)
		self.image_draw_size = np.zeros(2)
		
		self._init_shader()
		self._init_vao()
		self._init_fbo(100, 100)
		
		self.cached_image = None
		
	def _init_fbo(self, width, height):
		self.fbo = self.ctx.simple_framebuffer((width, height))
		
	def _init_shader(self):
		
		self.shader_program = self.ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 unif_transform;

				in vec3 in_pos;
				in vec2 in_uv;

				out vec2 vert_uv;

				void main() {
					vert_uv = in_uv;
					gl_Position = unif_transform * vec4(in_pos, 1);
				}
			''',
			fragment_shader='''
				#version 330

				uniform sampler2D unif_texture;

				in vec2 vert_uv;

				out vec4 frag_color;

				void main() {
					frag_color = texture(unif_texture, vert_uv);
				}
			'''
		)
		
	def _init_vao(self):
		vert_buff = np.array([
			[0, 0, 0, 0, 0],
			[1, 0, 0, 1, 0],
			[0, 1, 0, 0, 1],
			
			[1, 1, 1, 1, 1],
			[0, 1, 1, 0, 1],
			[1, 0, 1, 1, 0],
		])
		vbo = self.ctx.buffer(vert_buff.astype(np.float32).tobytes())
		self.vao = self.ctx.simple_vertex_array(self.shader_program, vbo, 'in_pos', 'in_uv')
		
	def resize(self, new_width, new_height):
		if self.get_width() != new_width or self.get_height() != new_height:
			self._init_fbo(new_width, new_height)
			self.cached_image = None
		
	def get_width(self):
		return self.fbo.width
		
	def get_height(self):
		return self.fbo.height
		
	def set_image_draw_point(self, image_draw_point):
		if not np.all(self.image_draw_point == image_draw_point):
			self.image_draw_point = image_draw_point
			self.cached_image = None
		
	def set_image_draw_size(self, image_draw_size):
		if not np.all(self.image_draw_size == image_draw_size):
			self.image_draw_size = image_draw_size
			self.cached_image = None

	def render(self):
		
		if self.use_caching and self.cached_image is not None:
			return self.cached_image, False
		
		self.fbo.use()
		self.fbo.clear(0.5, 0.5, 0.5, 1.0)
		self.ctx.enable(moderngl.BLEND)
		
		matr_transform = np.eye(4)
		matr_transform[0, 3] = (self.image_draw_point[0] / self.get_width()) * 2 - 1
		matr_transform[1, 3] = (self.image_draw_point[1] / self.get_height()) * 2 - 1
		matr_transform[0, 0] = (self.image_draw_size[0] / self.get_width()) * 2
		matr_transform[1, 1] = (self.image_draw_size[1] / self.get_height()) * 2
		
		# Flip Y
		matr_transform[1] *= -1
		
		self.shader_program['unif_transform'].write((matr_transform).T.astype(np.float32).tobytes())
		self.texture.use()
		self.vao.render(moderngl.TRIANGLES)

		image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
		
		if self.use_caching:
			self.cached_image = image
		
		return image, True
	
def pil_image_to_texture(ctx, image):
	mode = image.mode
	if mode == 'L':
		return ctx.texture(image.size, 1, image.tobytes())
	elif mode == 'RGB':
		return ctx.texture(image.size, 3, image.tobytes())
	elif mode == 'RGBA':
		return ctx.texture(image.size, 4, image.tobytes())
	raise RuntimeError('Unsupported color depth: {}'.format(mode))
	
def np_array_to_pil_image(image):
	scaled = image * 255
	scaled = np.clip(scaled, 0, 255)
	return Image.fromarray(scaled.astype(np.uint8))

def get_base_name(name):
	if len(name) == 0:
		return name
	substr = name.split(' ')
	first = substr[0]
	if first[0] != 'C':
		return name
	
	try:
		int(first[1:])
	except ValueError:
		return name
	return ' '.join(substr[1:])

if __name__ == '__main__':
	
	test = get_base_name('C1234 my name')
	print(test)
	test = get_base_name('Counter')
	print(test)
	for _ in range(10):
		pitch = np.random.rand() * np.pi - (np.pi/2)
		yaw = np.random.rand() * np.pi * 2
		
		direct = pitch_yaw_to_direction(pitch, yaw)
		pitch2, yaw2 = direction_to_pitch_yaw(direct)
		
		print(pitch - pitch2, yaw - yaw2)
	
	# Profiling
	#import cProfile
	#r = Renderer()
	#cProfile.run('r.render()')
