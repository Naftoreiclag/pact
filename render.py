import moderngl
import numpy as np
import pyrr

from PIL import Image

import random
import io_utils

class Texture_Loader:
	
	def __init__(self, moderngl_context):
		self.ctx = moderngl_context
		
		self.loaded_textures = {}
		
	def load_texture(self, fname):
		if fname in self.loaded_textures:
			return self.loaded_textures[fname]
		else:
			image = Image.open(fname)
			texture = pil_image_to_texture(self.ctx, image)
			texture.anisotropy = 16.0
			texture.build_mipmaps()
			
			self.loaded_textures[fname] = texture
			
			return texture
	
	def load_texture_cube(self, fname):
		if fname in self.loaded_textures:
			return self.loaded_textures[fname]
		else:
			face_fnames = [
				fname + 'posx.jpg',
				fname + 'negx.jpg',
				fname + 'posy.jpg',
				fname + 'negy.jpg',
				fname + 'posz.jpg',
				fname + 'negz.jpg',
			]
			
			face_images = [Image.open(x) for x in face_fnames]
			face_bytes = [x.tobytes() for x in face_images]
			
			texture = self.ctx.texture_cube(face_images[0].size, 3, b''.join(face_bytes))
			
			self.loaded_textures[fname] = texture
			
			return texture
	
	def clear_all(self):
		self.loaded_textures.clear()

class PanoObj:
	
	def __init__(self, custom_name, texture, model_matr, is_skybox, source_fname):
		if custom_name is None:
			custom_name = '{} {}'.format(random.randrange(0,99999), source_fname)
			
		self.custom_name = custom_name
		self.texture = texture
		self.model_matr = model_matr
		self.source_fname = source_fname
		self.is_skybox = is_skybox
		
	def renormalize_model_matrix(self):
		center_point = self.model_matr @ np.array([0.5, 0.5, 0.0, 1.0])
		self.model_matr[:3,:4] *= 10 / np.linalg.norm(center_point[:3])
	
	def apply_world_transform(self, matr):
		self.model_matr = matr @ self.model_matr
		self.renormalize_model_matrix()

class View_Params:
	def __init__(self, pitch_rad=0, yaw_rad=0):
		self.pitch_rad = pitch_rad
		self.yaw_rad = yaw_rad
		
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
		
	def add_pano_obj(self, fname_image, model_matr=None, custom_name=None):
		
		texture = self.texture_loader.load_texture(fname_image)
		
		if model_matr is None:
			model_matr = np.eye(4)
		pano_obj = PanoObj(custom_name, texture, model_matr, False, fname_image)
		self.pano_objs.append(pano_obj)
		
		return pano_obj
		
	def add_skybox(self, folder_skybox, model_matr=None, custom_name=None):
		
		texture = self.texture_loader.load_texture_cube(folder_skybox)
		
		if model_matr is None:
			model_matr = np.eye(4)
		pano_obj = PanoObj(custom_name, texture, model_matr, True, folder_skybox)
		self.pano_objs.append(pano_obj)
		
		return pano_obj
		
	def look_natural(self, anchor_dir, canvas_x, canvas_y):
		raise NotImplementedError()
		matr_view = self._compute_view_matr()
		matr_proj = self._compute_proj_matr()
		matr_proj_inv = np.linalg.inv(matr_proj)
		
		ndc = np.array([
			(canvas_x / self.fbo.width) * 2 - 1,
			-((canvas_y / self.fbo.height) * 2 - 1),
			0,
			1,
		])
		
		homo_view_coords = matr_proj_inv @ ndc
		homo_view_coords /= homo_view_coords[3] # Perspective divice
		homo_view_coords[:3] /= np.linalg.norm(homo_view_coords[:3])
		homo_view_coords[3] = 0
		
		anchor_pitch, anchor_yaw = direction_to_pitch_yaw(anchor_dir)
		canvas_pitch, canvas_yaw = direction_to_pitch_yaw(homo_view_coords)
		
		self.view_params.pitch_rad = -anchor_pitch - canvas_pitch
		self.view_params.yaw_rad = -anchor_yaw - canvas_yaw
		
	def clear_all(self):
		self.pano_objs.clear()
		self.texture_loader.clear_all()
		
	def _init_fbo(self, width, height):
		self.fbo = self.ctx.simple_framebuffer((width, height))
		
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

				in vec2 vert_uv;

				out vec4 frag_color;

				void main() {
					frag_color = texture(unif_texture, vert_uv);
				}
			'''
		)
		
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
		
	def _compute_view_matr(self):
		return self.view_params.compute_view_matr()
		
	def _compute_proj_matr(self):
		return pyrr.matrix44.create_perspective_projection_matrix(90.0, self.get_width() / self.get_height(), 0.1, 100.0).T
		
	def _compute_view_proj_matr(self):
		matr_proj = self._compute_proj_matr()
		matr_view = self._compute_view_matr()
		matr_view_proj = matr_proj @ matr_view
		return matr_view_proj
			
	def resize(self, new_width, new_height):
		self._init_fbo(new_width, new_height)
		
	def get_width(self):
		return self.fbo.width
	def get_height(self):
		return self.fbo.height

	def get_world_dir(self, canvas_x, canvas_y):
		matr_view_proj = self._compute_view_proj_matr()
		matr_view_proj_inv = np.linalg.inv(matr_view_proj)
		
		ndc = np.array([
			(canvas_x / self.fbo.width) * 2 - 1,
			-((canvas_y / self.fbo.height) * 2 - 1),
			0,
			1,
		])
		
		homo_world_coords = matr_view_proj_inv @ ndc
		homo_world_coords /= homo_world_coords[3] # Perspective divice
		homo_world_coords[:3] /= np.linalg.norm(homo_world_coords[:3])
		homo_world_coords[3] = 0
		
		return homo_world_coords
		
	def render(self):
		
		matr_view_proj = self._compute_view_proj_matr()

		matr_view_proj_inv = np.linalg.inv(matr_view_proj)

		self.fbo.use()
		self.fbo.clear(0.0, 0.0, 0.0, 1.0)
		self.ctx.enable(moderngl.BLEND)
		
		for pano_obj in self.pano_objs:
			if pano_obj.is_skybox:
				model_matr_inv = np.linalg.inv(pano_obj.model_matr)
				self.skybox_shader_program['unif_unprojection'].write((model_matr_inv @ matr_view_proj_inv).T.astype(np.float32).tobytes())
				pano_obj.texture.use()
				self.skybox_vao.render(moderngl.TRIANGLES)
			else:
				matr_mvp = matr_view_proj @ pano_obj.model_matr
				self.pano_obj_shader_program['unif_mvp'].write(matr_mvp.T.astype(np.float32).tobytes())
				pano_obj.texture.use()
				self.pano_obj_vao.render(moderngl.TRIANGLES)

		image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
		return image

	def get_vanishing_point_on_canvas(self, direction):
		
		matr_view_proj = self._compute_view_proj_matr()
		
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

				out vec3 frag_color;

				void main() {
					frag_color = texture(unif_texture, vert_uv).xyz;
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
	if mode == 'RGB':
		return ctx.texture(image.size, 3, image.tobytes())
	elif mode == 'RGBA':
		return ctx.texture(image.size, 4, image.tobytes())
	raise RuntimeError('Unsupported color depth: {}'.format(mode))
	

if __name__ == '__main__':
	
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
