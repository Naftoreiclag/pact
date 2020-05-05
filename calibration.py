
from PIL import Image
from PIL import ImageTk
import numpy as np

class Calibrated_Image:
	
	def __init__(self, image):
		self.image = image

class Control_Line:
	
	def __init__(self, start_pos, end_pos):
		self.start_pos = np.array(start_pos)
		self.end_pos = np.array(end_pos)

class Calibration:
	
	def __init__(self, tk_canvas):
		
		self.control_lines = []
		self.zoom_level = 1
		self.cal_image = Calibrated_Image(Image.open('ignore/bears.jpg'))
		
		self.control_lines.append(Control_Line([0, 0], [4000, 3000]))
		
		self.tk_canvas = tk_canvas
		self._load_tk_image(0.1)
		self.setup_binds()
		
	def setup_binds(self):
		self.tk_canvas.bind('<Configure>', self._on_canvas_reconfig)
		self.tk_canvas.bind('<ButtonPress-2>', self._on_canvas_press_m2)
		self.tk_canvas.bind('<B2-Motion>', self._on_canvas_drag_m2)
		self.tk_canvas.bind('<ButtonRelease-2>', self._on_canvas_release_m2)
		
	def refresh_canvas(self):
		
		origin_point = self.canvas_size / 2
		
		self.tk_canvas.delete('all')
		
		image_draw_point = origin_point - (self.tk_image_size / 2)
		
		self.tk_canvas.create_image(image_draw_point[0], image_draw_point[1], image=self.tk_image, anchor='nw')
		
		for con_line in self.control_lines:
			
			start = image_draw_point + (con_line.start_pos * self.scale_factor)
			end = image_draw_point + (con_line.end_pos * self.scale_factor)
			
			
			
			self.tk_canvas.create_line(start[0], start[1], end[0], end[1])
		
	def _load_tk_image(self, scale_factor):
		image = self.cal_image.image
		image_size = np.array((image.width, image.height))
		self.scale_factor = scale_factor
		self.tk_image_size = np.array(image_size * scale_factor).astype(np.int)
		width, height = self.tk_image_size
		self.tk_image = ImageTk.PhotoImage(self.cal_image.image.resize((width, height), Image.ANTIALIAS))
	
	def _on_canvas_press_m2(self, event):
		pass
		
	def _on_canvas_drag_m2(self, event):
		pass
		
	def _on_canvas_release_m2(self, event):
		pass
	
	def _on_canvas_reconfig(self,event):
		self.canvas_size = np.array((event.width, event.height))
		self.refresh_canvas()
