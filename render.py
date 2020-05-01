import moderngl
import numpy as np
import pyrr

from PIL import Image

import random

class PanoObj:
	
	def __init__(self, texture, model_matr):
		self.texture = texture
		self.model_matr = model_matr

def model_matr_from_orientation(origin_loc, axis_u, axis_v):
	matr = np.eye(4, )
	matr[:3,3] = origin_loc
	matr[:3,0] = axis_u
	matr[:3,1] = axis_v
	return matr

class View_Params:
	def __init__(self, pitch_rad=0, yaw_rad=0):
		self.pitch_rad = pitch_rad
		self.yaw_rad = yaw_rad
		
	def compute_view_matr(self):
		matr_view = pyrr.matrix44.create_look_at((0, 0, 0), (0, 0, 1), (0, 1, 0)).T
		return matr_view
		
	def compute_proj_matr(self, fovy, aspect, near, far):
		matr_proj = pyrr.matrix44.create_perspective_projection_matrix(fovy, aspect, near, far).T
		return matr_proj
		
		

class Renderer:
	
	def __init__(self):
		self.ctx = moderngl.create_standalone_context()
		self.shader_program = self.ctx.program(
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
		
		self.pano_objs = []
		
		self.view_params = View_Params()
		
		self._init_pano_obj_fbo()
		
		# TODO flip z axis afterwards
		
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
		
		self._init_fbo(100, 100)
		
	def add_pano_obj(self, image, model_matr=None):
		texture = self.ctx.texture(image.size, 3, image.tobytes())
		texture.build_mipmaps()
		if model_matr is None:
			model_matr = np.eye(4)
		pano_obj = PanoObj(texture, model_matr)
		self.pano_objs.append(pano_obj)
		
	def _init_fbo(self, width, height):
		self.fbo = self.ctx.simple_framebuffer((width, height))
		
	def _init_pano_obj_fbo(self):
		vert_buff = np.array([
			[0, 0, 0, 0, 0],
			[1, 0, 0, 1, 0],
			[0, 1, 0, 0, 1],
			
			[1, 1, 0, 1, 1],
			[0, 1, 0, 0, 1],
			[1, 0, 0, 1, 0],
		])
		vbo = self.ctx.buffer(vert_buff.astype(np.float32).tobytes())
		self.pano_obj_vao = self.ctx.simple_vertex_array(self.shader_program, vbo, 'in_pos', 'in_uv')
		
		
	def resize(self, new_width, new_height):
		self._init_fbo(new_width, new_height)

	def get_world_coordinates(self, canvas_x, canvas_y):
		pass

	def render(self):

		
		matr_proj = self.view_params.compute_proj_matr(120.0, 1.0, 0.1, 100.0)
		matr_view = self.view_params.compute_view_matr()
		mat_viewproj = matr_proj @ matr_view

		self.fbo.use()
		self.fbo.clear(0.0, 0.0, 0.0, 1.0)
		
		for pano_obj in self.pano_objs:
			mat_mvp = mat_viewproj @ pano_obj.model_matr
			self.shader_program['unif_mvp'].write(mat_mvp.T.astype(np.float32).tobytes())
			pano_obj.texture.use()
			self.pano_obj_vao.render(moderngl.TRIANGLES)

		image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
		return image

if __name__ == '__main__':
	
	# Profiling
	import cProfile
	r = Renderer()
	cProfile.run('r.render()')
