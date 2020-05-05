
from PIL import Image
from PIL import ImageTk
import numpy as np
import render

class Control_Line:
	
	def __init__(self, start_pos, end_pos):
		self.start_pos = np.array(start_pos)
		self.end_pos = np.array(end_pos)

class Calibration:
	
	def __init__(self, tk_canvas, opengl_context):
		
		self.image = Image.open('ignore/bears.jpg')
		self.image_dimensions = np.array((self.image.width, self.image.height))
		self.tk_canvas = tk_canvas
		
		self.renderer = render.Single_Image_Renderer(opengl_context, self.image)
		
		self.control_lines = []
		self.scale_factor = 0.1
		
		self.control_lines.append(Control_Line([0, 0], [4000, 3000]))
		self.look_pos = np.zeros(2,)
		
		self.setup_binds()
		
	def setup_binds(self):
		self.tk_canvas.bind('<Configure>', self._on_canvas_reconfig)
		self.tk_canvas.bind('<ButtonPress-2>', self._on_canvas_press_m2)
		self.tk_canvas.bind('<B2-Motion>', self._on_canvas_drag_m2)
		self.tk_canvas.bind('<ButtonRelease-2>', self._on_canvas_release_m2)
		self.tk_canvas.bind('<MouseWheel>', self._on_canvas_mousewheel)

	def _on_canvas_mousewheel(self, event):
		new_scale = self.scale_factor*np.exp(event.delta / 30)
		new_scale = np.clip(new_scale, 0.01, 10)
		self.scale_factor = new_scale
		self.refresh_canvas()

	def refresh_canvas(self):
		
		canvas_size = np.array((self.renderer.get_width(), self.renderer.get_height()))
		origin_point = (canvas_size / 2) - self.look_pos
		
		self.tk_canvas.delete('all')
		
		image_draw_point = origin_point - (self.image_dimensions / 2) * self.scale_factor
		
		self.renderer.set_image_draw_point(image_draw_point)
		self.renderer.set_image_draw_size(self.image_dimensions * self.scale_factor)
		image, image_is_new = self.renderer.render()
		if image_is_new:
			self.tk_canvas.usr_image_ref = ImageTk.PhotoImage(image)
		self.tk_canvas.create_image(0, 0, image=self.tk_canvas.usr_image_ref, anchor='nw')
		
		for con_line in self.control_lines:
			
			start = image_draw_point + (con_line.start_pos * self.scale_factor)
			end = image_draw_point + (con_line.end_pos * self.scale_factor)
			
			
			
			self.tk_canvas.create_line(start[0], start[1], end[0], end[1])
	
	def _on_canvas_press_m2(self, event):
		self.anchor_pos = np.array((event.x, event.y))
		
	def _on_canvas_drag_m2(self, event):
		new_pos = np.array((event.x, event.y))
		
		delta = new_pos - self.anchor_pos
		
		self.look_pos -= delta
		
		print(self.look_pos)
		
		self.anchor_pos = new_pos
		self.refresh_canvas()
		
	def _on_canvas_release_m2(self, event):
		pass
	
	def _on_canvas_reconfig(self,event):
		self.canvas_size = np.array((event.width, event.height))
		self.renderer.resize(event.width, event.height)
		self.refresh_canvas()
