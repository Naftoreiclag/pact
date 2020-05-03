import moderngl
import numpy as np
import pyrr

from PIL import Image

import random

class PanoObj:
	
	def __init__(self, texture, model_matr, is_skybox):
		self.texture = texture
		self.model_matr = model_matr
		self.is_skybox = is_skybox

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
	
	def __init__(self):
		self.ctx = moderngl.create_standalone_context()
		
		self._init_pano_obj_shader()
		self._init_skybox_shader()
		self._init_pano_obj_vao()
		self._init_skybox_vao()
		self._init_fbo(100, 100)
		
		self.pano_objs = []
		
		self.view_params = View_Params()
		
		# TODO flip z axis afterwards
		
		if True:
			matr = model_matr_from_orientation([-1, 1, 1], [2, 0, 0], [0, 0, -2])
			skybox_textures = [
				Image.open('ignore/posx.jpg'),
				Image.open('ignore/negx.jpg'),
				Image.open('ignore/posy.jpg'),
				Image.open('ignore/negy.jpg'),
				Image.open('ignore/posz.jpg'),
				Image.open('ignore/negz.jpg'),
			]
			
			self.add_skybox(skybox_textures, matr)
			
		else:
			matr = model_matr_from_orientation([-1, 1, 1], [2, 0, 0], [0, 0, -2])
			self.add_pano_obj(Image.open('ignore/posy.jpg'), matr)
			matr = model_matr_from_orientation([-1, -1, -1], [2, 0, 0], [0, 0, 2])
			self.add_pano_obj(Image.open('ignore/negy.jpg'), matr)
			
			
			matr = model_matr_from_orientation([-1, 1, -1], [2, 0, 0], [0, -2, 0])
			self.add_pano_obj(Image.open('ignore/posz.jpg'), matr)
			matr = model_matr_from_orientation([1, 1, -1], [0, 0, 2], [0, -2, 0])
			self.add_pano_obj(Image.open('ignore/posx.jpg'), matr)
			matr = model_matr_from_orientation([1, 1, 1], [-2, 0, 0], [0, -2, 0])
			self.add_pano_obj(Image.open('ignore/negz.jpg'), matr)
			matr = model_matr_from_orientation([-1, 1, 1], [0, 0, -2], [0, -2, 0])
			self.add_pano_obj(Image.open('ignore/negx.jpg'), matr)
		
		
	def add_pano_obj(self, image, model_matr=None):
		texture = self.ctx.texture(image.size, 3, image.tobytes())
		texture.build_mipmaps()
		if model_matr is None:
			model_matr = np.eye(4)
		pano_obj = PanoObj(texture, model_matr, False)
		self.pano_objs.append(pano_obj)
		
	def add_skybox(self, images, model_matr=None):
		faces = [image.tobytes() for image in images]
		
		
		texture = self.ctx.texture_cube(images[0].size, 3, b''.join(faces))
		if model_matr is None:
			model_matr = np.eye(4)
		pano_obj = PanoObj(texture, model_matr, True)
		self.pano_objs.append(pano_obj)
		
		
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

				out vec3 frag_color;

				void main() {
					frag_color = texture(unif_texture, vert_uv).xyz;
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

				out vec3 frag_color;

				void main() {
					frag_color = texture(unif_texture, vert_view_dir.xyz).xyz;
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
		return pyrr.matrix44.create_perspective_projection_matrix(120.0, self.get_width() / self.get_height(), 0.1, 100.0).T
		
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
		
		for pano_obj in self.pano_objs:
			if pano_obj.is_skybox:
				self.skybox_shader_program['unif_unprojection'].write(matr_view_proj_inv.T.astype(np.float32).tobytes())
				pano_obj.texture.use()
				self.skybox_vao.render(moderngl.TRIANGLES)
			else:
				matr_mvp = matr_view_proj @ pano_obj.model_matr
				self.pano_obj_shader_program['unif_mvp'].write(matr_mvp.T.astype(np.float32).tobytes())
				pano_obj.texture.use()
				self.pano_obj_vao.render(moderngl.TRIANGLES)

		image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
		return image

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
