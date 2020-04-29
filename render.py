from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from OpenGL.GLU import *
import glfw

from PIL import Image

import random

def make_shader(shader_src, shader_type):
	gl_shader = glCreateShader(shader_type)
	glShaderSource(gl_shader, shader_src)
	glCompileShader(gl_shader)
	compile_msg = glGetShaderInfoLog(gl_shader)
	
	return gl_shader, compile_msg

def make_shader_program(gl_vert_shader, gl_frag_shader):
	gl_prog = glCreateProgram()
	glAttachShader(gl_prog, gl_vert_shader)
	glAttachShader(gl_prog, gl_frag_shader)
	glLinkProgram(gl_prog)
	compile_msg = glGetProgramInfoLog(gl_prog)
	
	return gl_prog, compile_msg

class Renderer:
	
	def __init__(self):
		self._gl_fb_canvas = None
		self._init_buffers(500, 500)
		self._init_shaders()
		
	def __del__(self):
		self._cleanup_buffers()
		self._cleanup_shaders()
		
	def resize(self, new_width, new_height):
		self._cleanup_buffers()
		self._init_buffers(new_width, new_height)

	def _init_buffers(self, width, height):
		if self._gl_fb_canvas is not None:
			raise RuntimeError('Attempted to overwrite FBO!')
			
		self._buff_width = width
		self._buff_height = height
		
		# Create handles
		self._gl_fb_canvas = glGenFramebuffers(1)
		self._gl_rb_color = glGenRenderbuffers(1)
		self._gl_rb_depth = glGenRenderbuffers(1)

		# Select our FBO
		glBindFramebuffer(GL_FRAMEBUFFER, self._gl_fb_canvas)

		# Attach and initialize the color buffer
		glBindRenderbuffer(GL_RENDERBUFFER, self._gl_rb_color)
		glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, self._buff_width, self._buff_height) # Important, use 8-bit depth for Pillow
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self._gl_rb_color)

		# Attach and initialize the depth buffer
		glBindRenderbuffer(GL_RENDERBUFFER, self._gl_rb_depth)
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self._buff_width, self._buff_height)
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._gl_rb_depth)
	
	def _init_shaders(self):
		vert_shader_src = """
		#version 120
		void main() {
			gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
		}
		"""
		frag_shader_src = """
		#version 120
		void main() {
			gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
		}
		"""
		
		self._gl_vert_shader = shaders.compileShader(vert_shader_src, GL_VERTEX_SHADER)
		self._gl_frag_shader = shaders.compileShader(frag_shader_src, GL_FRAGMENT_SHADER)
		
		self._gl_shader_program = shaders.compileProgram(self._gl_vert_shader, self._gl_frag_shader)
		
	def _cleanup_shaders(self):
		glDeleteProgram(self._gl_shader_program)
		glDeleteShader(self._gl_vert_shader)
		glDeleteShader(self._gl_frag_shader)
		# Note, you can technically delete the shaders as soon as the 
		# shader program is linked, but I do it here just for consistency's
		# sake.
		
	
	def _cleanup_buffers(self):
		# Delete everything in opposite order
		glDeleteRenderbuffers(1, self._gl_rb_depth)
		glDeleteRenderbuffers(1, self._gl_rb_color)
		glDeleteFramebuffers(1, self._gl_fb_canvas)
		
		self._gl_fb_canvas = None
		self._gl_rb_color = None
		self._gl_fb_canvas = None

	def _download_buffer_contents(self):
		if self._gl_fb_canvas is None:
			raise RuntimeError('Attempted to download non-existent contents!')
		
		# Select our canvas
		glBindFramebuffer(GL_FRAMEBUFFER, self._gl_fb_canvas)
		
		# Alignment needs to be set in order for Pillow to read properly
		glPixelStorei(GL_PACK_ALIGNMENT, 1)
		
		# Load data and convert to Pillow image
		glReadBuffer(GL_COLOR_ATTACHMENT0)
		raw_data = glReadPixels(0, 0, self._buff_width, self._buff_height, GL_RGBA, GL_UNSIGNED_BYTE)
		image = Image.frombytes('RGBA', (self._buff_width, self._buff_height), raw_data)
		
		return image

	def render(self):
		self._draw_to_buffer()

		# Download from device and return
		image = self._download_buffer_contents()
		return image

	def _draw_to_buffer(self):
		# Select our canvas
		glBindFramebuffer(GL_FRAMEBUFFER, self._gl_fb_canvas)
		
		# Clear old contents
		glClearColor(0.5, 0.6, 0.7, 1.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		
		# Set up perspective
		gluOrtho2D(-1.0, 1.0, -1.0, 1.0)
		glViewport(0, 0, self._buff_width, self._buff_height)

		# Immediate mode draw
		glBegin(GL_TRIANGLES)
		def rand():
			return random.random()*2 - 1
		for i in range(1000):
			glColor(random.random(), random.random(), random.random())
			glVertex2d(rand(), rand())
			glVertex2d(rand(), rand())
			glVertex2d(rand(), rand())
		glEnd()
		glFlush()
		

def opengl_context_init():
	# Try init
	success = glfw.init()
	if not success:
		raise RuntimeError('GLFW init() failed!')
	
	# Not really important
	window_width = 100
	window_height = 100
	
	# This creates the OpenGL context and hidden window
	glfw.window_hint(glfw.VISIBLE, False)
	glfw_win = glfw.create_window(window_width, window_height, 'offscreen rendering window', None, None)
	
	# Check for errors
	if not glfw_win:
		glfw.terminate()
		raise RuntimeError('Failed to create GLFW offscreen rendering window / OpenGL context!')

	# Use context
	glfw.make_context_current(glfw_win)
	return glfw_win
	
def opengl_context_cleanup(glfw_win):
	glfw.destroy_window(glfw_win)
	glfw.terminate()


if __name__ == '__main__':
	
	# Profiling
	
	import cProfile
	
	ctx = opengl_context_init()
	
	r = Renderer()
	
	cProfile.run('r.render()')
	
	del r
	
	opengl_context_cleanup(ctx)
