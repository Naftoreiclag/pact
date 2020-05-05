import tkinter
from PIL import ImageTk
import render
import calibration
import time
import moderngl
import numpy as np

class Scene_Editor():
	def __init__(self, tk_canvas, opengl_context):
		self.anchor_xy = np.zeros((2, ))
		self.tk_canvas = tk_canvas
		self.renderer = render.Renderer(opengl_context)
		self.setup_binds()
		
	def setup_binds(self):
		self.tk_canvas.bind('<Configure>', self._on_canvas_reconfig)
		self.tk_canvas.bind('<ButtonPress-2>', self._on_canvas_press_m2)
		self.tk_canvas.bind('<B2-Motion>', self._on_canvas_drag_m2)
		self.tk_canvas.bind('<ButtonRelease-2>', self._on_canvas_release_m2)
		
	def refresh_canvas(self):
		image = self.renderer.render()
		
		tk_image = ImageTk.PhotoImage(image)
		self.tk_canvas.usr_image_ref = tk_image
		self.tk_canvas.delete('all')
		self.tk_canvas.create_image(0, 0, image=tk_image, anchor='nw')
	
	def _on_canvas_press_m2(self, event):
		self.anchor_xy = np.array((event.x, event.y))
		
	def _on_canvas_drag_m2(self, event):
		new_anchor_xy = np.array((event.x, event.y))
		
		diff = new_anchor_xy - self.anchor_xy
		self.renderer.view_params.yaw_rad -= (diff[0] / self.renderer.get_width()) * 2
		self.renderer.view_params.pitch_rad += (diff[1] / self.renderer.get_height()) * 2
		
		self.renderer.view_params.pitch_rad = np.clip(self.renderer.view_params.pitch_rad, -np.pi/2, np.pi/2)
		
		self.anchor_xy = new_anchor_xy
		self.refresh_canvas()
		
	def _on_canvas_release_m2(self, event):
		self.anchor_xy = None
	
	def _on_canvas_reconfig(self,event):
		width = event.width
		height = event.height
		
		self.renderer.resize(width, height)
		
		self.refresh_canvas()
	

def main():
	tk_root = tkinter.Tk()
	tk_root.title('hello world')
	
	tk_canvas = tkinter.Canvas(tk_root, width=800, height=800)
	tk_canvas.pack(expand=True, fill='both')
	
	
	ctx = moderngl.create_standalone_context()

	editor = Scene_Editor(tk_canvas, ctx)

	#prog = calibration.Calibration(tk_canvas, ctx)

	def clicky_1():
		renderer._debug_scalar *= 0.9
		editor.refresh_canvas()
	def clicky_2():
		renderer._debug_scalar /= 0.9
		editor.refresh_canvas()

	button1 = tkinter.Button(tk_root, text='clicky1', command=clicky_1)
	button1.pack()
	button2 = tkinter.Button(tk_root, text='clicky2', command=clicky_2)
	button2.pack()
	tk_root.mainloop()

	print('application closed')

if __name__ == '__main__':
	main()

