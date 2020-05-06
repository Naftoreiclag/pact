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
		
		rotate90 = np.eye(4)
		rotate90[0,0] = 0
		rotate90[2,0] = 1
		rotate90[0,2] = -1
		rotate90[2,2] = 0
		
		
		for asdf in [np.eye(4), rotate90]:
			self.renderer.add_pano_obj(image, asdf @ matr)
			self.renderer.add_pano_obj(image, asdf @ flip_x @ matr)
			self.renderer.add_pano_obj(image, asdf @ flip_z @ matr)
			self.renderer.add_pano_obj(image, asdf @ flip_z @ flip_x @ matr)
	
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

def create_scrollable(master):
	# Create the canvas to hold the scrollable contents
	canvas = tkinter.Canvas(master)
	
	# Create the scrollbar, and establish communication links
	scrollbar = tkinter.Scrollbar(master, orient='vertical', command=canvas.yview)
	canvas.configure(yscrollcommand=scrollbar.set)
	
	# Packing information
	scrollbar.pack(side='right', fill='y')
	canvas.pack(side='left', fill='both')
	
	# Create the child frame
	child_frame = tkinter.Frame(canvas)
	canvas.create_window((0,0), anchor='nw', window=child_frame)
	
	# Reset scroll region when reconfigured
	child_frame.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
	
	return child_frame
	
def main():
	tk_root = tkinter.Tk()
	tk_root.title('hello world')
	
	tk_canvas = tkinter.Canvas(tk_root, width=800, height=800)
	tk_canvas.grid(row=100, column=100, sticky='nsew')
	
	tk_root.columnconfigure(100, weight=1)
	tk_root.rowconfigure(100, weight=1)
	
	ctx = moderngl.create_standalone_context()

	example_fname = 'ignore/data/bears2'
	img = Image.open(example_fname + '.jpg')
	if False:
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
		return
		json_data = calib_tool.save_to_json()
		io_utils.json_save(json_data, example_fname + '.json')
	def on_button_load():
		return
		json_data = io_utils.json_load(example_fname + '.json')
		calib_tool.load_from_json(json_data)

	def clicky_1():
		editor.renderer._debug_scalar *= 0.9
		editor.refresh_canvas()
	def clicky_2():
		editor.renderer._debug_scalar /= 0.9
		editor.refresh_canvas()

	
	tk_frame = tkinter.Frame(tk_root)
	tk_frame.grid(row=99, column=100)
	
	parent_frame = tkinter.Frame(tk_root, relief=tkinter.GROOVE, bd=1)
	parent_frame.grid(row=100, column=101, sticky='ns')
	
	scrollable = create_scrollable(parent_frame)
	button_idx = 0
	
	def add_button():
		nonlocal button_idx
		button = tkinter.Button(scrollable, text='test', command=add_button)
		button.grid(row=button_idx, column=0)
		button_idx += 1
	add_button()
		
	
	button_save = tkinter.Button(tk_frame, text='save', command=on_button_save)
	button_save.grid(row=100, column=100)
	button_load = tkinter.Button(tk_frame, text='load', command=on_button_load)
	button_load.grid(row=100, column=101)
	
	if False:
		
		button1 = tkinter.Button(tk_root, text='clicky1', command=clicky_1)
		button1.pack()
		button2 = tkinter.Button(tk_root, text='clicky2', command=clicky_2)
		button2.pack()
	
	tk_root.mainloop()

	print('application closed')

if __name__ == '__main__':
	main()

