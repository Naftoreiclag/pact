import tkinter
from PIL import ImageTk
import render
import calibration
import time
import moderngl
from PIL import Image
import numpy as np
import os
import io_utils

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
		
	def add_pano_obj_from_file(self, fname_image):
		fname_leafless, fname_leaf = os.path.splitext(fname_image)
		fname_transform = fname_leafless + '.json'
		
		image = Image.open(fname_image)
		
		json_data = io_utils.json_load(fname_transform)
		matr = io_utils.load_matrix_from_json(json_data['image_plane_matrix'])
		
		flip_x = np.eye(4)
		flip_x[0,0] = -1
		flip_z = np.eye(4)
		flip_z[2,2] = -1
		self.renderer.add_pano_obj(image, matr)
		self.renderer.add_pano_obj(image, flip_x @ matr)
		self.renderer.add_pano_obj(image, flip_z @ matr)
		self.renderer.add_pano_obj(image, flip_z @ flip_x @ matr)
	
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

	example_fname = 'ignore/data/trash'
	img = Image.open(example_fname + '.jpg')
	if True:
		'''
		editor = Scene_Editor(tk_canvas, ctx)

		matr = calibration.solve_perspective(json_load('ignore/bears.json'), img.width, img.height)
		
		flip_x = np.eye(4)
		flip_x[0,0] = -1
		flip_z = np.eye(4)
		flip_z[2,2] = -1
		
		editor.renderer.add_pano_obj(img, matr)
		editor.renderer.add_pano_obj(img, flip_x @ matr)
		editor.renderer.add_pano_obj(img, flip_z @ matr)
		editor.renderer.add_pano_obj(img, flip_z @ flip_x @ matr)
		'''
		editor = Scene_Editor(tk_canvas, ctx)
		editor.add_pano_obj_from_file(example_fname + '.jpg')
	else:
		calib_tool = calibration.Calibration(tk_canvas, ctx, img)

	def on_button_save():
		json_data = calib_tool.save_to_json()
		io_utils.json_save(json_data, example_fname + '.json')
	def on_button_load():
		json_data = io_utils.json_load(example_fname + '.json')
		calib_tool.load_from_json(json_data)

	def clicky_1():
		renderer._debug_scalar *= 0.9
		editor.refresh_canvas()
	def clicky_2():
		renderer._debug_scalar /= 0.9
		editor.refresh_canvas()

	button_save = tkinter.Button(tk_root, text='save', command=on_button_save)
	button_save.pack()
	button_load = tkinter.Button(tk_root, text='load', command=on_button_load)
	button_load.pack()
	
	button1 = tkinter.Button(tk_root, text='clicky1', command=clicky_1)
	button1.pack()
	button2 = tkinter.Button(tk_root, text='clicky2', command=clicky_2)
	button2.pack()
	tk_root.mainloop()

	print('application closed')

if __name__ == '__main__':
	main()

