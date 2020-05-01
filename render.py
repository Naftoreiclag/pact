import moderngl
import numpy as np
import pyrr

from PIL import Image

import random

class PanoObj:
	
	def __init__(self, texture, model_matr):
		self.texture = texture
		self.model_matr = model_matr

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
		
		self.add_pano_obj(Image.open('test_texture.png'))
		
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
		
	def resize(self, new_width, new_height):
		self._init_fbo(new_width, new_height)

	def render(self):

		vert_buff = np.array([
			[0, 0, 0, 0, 0],
			[1, 0, 0, 1, 0],
			[0, 1, 0, 0, 1],
			
			[1, 1, 0, 1, 1],
			[0, 1, 0, 0, 1],
			[1, 0, 0, 1, 0],
		])

		mat_proj = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1.0, 0.1, 100.0).T
		mat_view = pyrr.matrix44.create_look_at((10, 10, 10), (0, 0, 0), (0, 1, 0)).T
		mat_viewproj = mat_proj @ mat_view
		
		vbo = self.ctx.buffer(vert_buff.astype(np.float32).tobytes())
		vao = self.ctx.simple_vertex_array(self.shader_program, vbo, 'in_pos', 'in_uv')

		self.fbo.use()
		self.fbo.clear(0.4, 0.5, 0.6, 1.0)
		
		for pano_obj in self.pano_objs:
			mat_mvp = mat_viewproj @ pano_obj.model_matr
			self.shader_program['unif_mvp'].write(mat_mvp.T.astype(np.float32).tobytes())
			pano_obj.texture.use()
			vao.render(moderngl.TRIANGLES)

		image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
		return image

if __name__ == '__main__':
	
	# Profiling
	import cProfile
	r = Renderer()
	cProfile.run('r.render()')
