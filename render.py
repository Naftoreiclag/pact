import moderngl
import numpy as np
import pyrr

from PIL import Image

import random

class Renderer:
	
	def __init__(self):
		self.ctx = moderngl.create_standalone_context()
		self.shader_program = self.ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 unif_mvp;

				in vec3 in_pos;
				in vec3 in_color;

				out vec3 vert_color;
				out vec2 vert_uv;

				void main() {
					vert_color = in_color;
					vert_uv = in_pos.xy;
					//gl_Position = vec4(in_pos, unif_mvp[0][0] * 0 + 1);
					gl_Position = unif_mvp * vec4(in_pos, 1);
				}
			''',
			fragment_shader='''
				#version 330

				uniform sampler2D unif_texture;

				in vec3 vert_color;
				in vec2 vert_uv;

				out vec3 frag_color;

				void main() {
					frag_color = vert_color * texture(unif_texture, vert_uv).xyz;
				}
			'''
		)
		texture_img = Image.open('test_texture.png')
		self.texture = self.ctx.texture(texture_img.size, 3, texture_img.tobytes())
		self.texture.build_mipmaps()
		
		self._init_fbo(100, 100)
		
	def _init_fbo(self, width, height):
		self.fbo = self.ctx.simple_framebuffer((width, height))
		
	def resize(self, new_width, new_height):
		self._init_fbo(new_width, new_height)

	def render(self):
		
		vert_buff = np.random.rand(300, 6)
		vert_buff[:,:3] = (np.round(vert_buff[:,:3]) * 2) - 1

		mat_proj = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1.0, 0.1, 100.0)
		#mat_view = pyrr.matrix44.create_look_at((0, 0, 0), (0, 0, -1), (0, 1, 0))
		#mat_view = pyrr.matrix44.create_look_at((10, 10, 10), (0, 0, 0), (0, 1, 0))
		mat_mvp = mat_proj# @ mat_view
		
		print(mat_proj)
		#mat_mvp = pyrr.matrix44.create_identity()
		
		vbo = self.ctx.buffer(vert_buff.astype(np.float32).tobytes())
		vao = self.ctx.simple_vertex_array(self.shader_program, vbo, 'in_pos', 'in_color')


		self.shader_program['unif_mvp'].write(mat_mvp.astype(np.float32).tobytes())

		self.fbo.use()
		self.texture.use()
		self.fbo.clear(0.4, 0.5, 0.6, 1.0)
		vao.render(moderngl.TRIANGLES)

		image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
		return image

if __name__ == '__main__':
	
	# Profiling
	import cProfile
	r = Renderer()
	cProfile.run('r.render()')
