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
import draw_utils

class Editor_Tool:
		
	def on_press(self, event, editor):
		raise NotImplementedError()
		
	def on_drag(self, event, editor):
		raise NotImplementedError()
		
	def on_release(self, event, editor):
		raise NotImplementedError()

class Camera_Pan_Tool(Editor_Tool):
	
	def __init__(self):
		self.prev_xy = np.zeros((2, ))
		
	def on_press(self, event, editor):
		self.prev_xy = np.array((event.x, event.y))
		
	def on_drag(self, event, editor):
		new_prev_xy = np.array((event.x, event.y))
		
		diff = new_prev_xy - self.prev_xy
		renderer = editor.renderer
		renderer.view_params.yaw_rad -= (diff[0] / renderer.get_width()) * 2
		renderer.view_params.pitch_rad += (diff[1] / renderer.get_height()) * 2
		
		renderer.view_params.pitch_rad = np.clip(renderer.view_params.pitch_rad, -np.pi/2, np.pi/2)
		
		self.prev_xy = new_prev_xy
		editor.refresh_canvas()
		
	def on_release(self, event, editor):
		self.prev_xy = np.zeros((2, ))

class Axis_Aligned_Scaling_Tool(Editor_Tool):
	
	def __init__(self):
		self.prev_xy = np.zeros((2, ))
		
	def on_press(self, event, editor):
		self.prev_xy = np.array((event.x, event.y))
		
	def on_drag(self, event, editor):
		new_prev_xy = np.array((event.x, event.y))
		
		diff = new_prev_xy - self.prev_xy
		
		if editor.selected_object is not None:
			scaling = np.eye(4)
			axis = editor.selected_axis
			scaling[axis,axis] = np.exp((diff[0] / editor.renderer.get_width()))
			editor.selected_object.model_matr = scaling @ editor.selected_object.model_matr
			editor.selected_object.renormalize_model_matrix()
		
		self.prev_xy = new_prev_xy
		editor.refresh_canvas()
		
	def on_release(self, event, editor):
		self.prev_xy = np.zeros((2, ))
	

class Scene_Editor(tkinter.Frame):
	def __init__(self, tk_master, opengl_context):
		tkinter.Frame.__init__(self, tk_master)
		
		self.tk_master = tk_master
		self.renderer = render.Renderer(opengl_context)
		self.setup_interface()
		
		self.selected_m1_tool = Axis_Aligned_Scaling_Tool()
		self.selected_m2_tool = Camera_Pan_Tool()
		
		self.selected_axis = 0
		self.selected_object = None
		
		self.show_vanishing_points = True
		
	def setup_interface(self):
		
		tk_master = self.tk_master
		
		self.tk_menubar = tkinter.Menu(self.tk_master)
		tk_master.config(menu=self.tk_menubar)
		
		self.tk_menubar_file = tkinter.Menu(self.tk_menubar)
		self.tk_menubar_file.add_command(label='Exit', command=self._on_user_request_exit)
		self.tk_menubar.add_cascade(label='File', menu=self.tk_menubar_file)
		
		self.tk_canvas = tkinter.Canvas(tk_master, width=800, height=800)
		self.tk_canvas.grid(row=100, column=100, sticky='nsew')
		self.tk_canvas.bind('<Configure>', self._on_canvas_reconfig)
		self.tk_canvas.bind('<ButtonPress-2>', self._on_canvas_press_m2)
		self.tk_canvas.bind('<B2-Motion>', self._on_canvas_drag_m2)
		self.tk_canvas.bind('<ButtonRelease-2>', self._on_canvas_release_m2)
		self.tk_canvas.bind('<ButtonPress-1>', self._on_canvas_press_m1)
		self.tk_canvas.bind('<B1-Motion>', self._on_canvas_drag_m1)
		self.tk_canvas.bind('<ButtonRelease-1>', self._on_canvas_release_m1)
		self.tk_canvas.bind('<Key>', self._on_canvas_key)
		
		tk_master.columnconfigure(100, weight=1)
		tk_master.rowconfigure(100, weight=1)
		
		self.tk_selection_scrollable_parent_frame = tkinter.Frame(tk_master, relief=tkinter.GROOVE, bd=1)
		self.tk_selection_scrollable_parent_frame.grid(row=100, column=101, sticky='ns')
		
		self.tk_selection_scrollable = create_scrollable(self.tk_selection_scrollable_parent_frame)
		
		button_idx = 0
		
		def add_button():
			nonlocal button_idx
			button = tkinter.Button(self.tk_selection_scrollable, text='test', command=add_button)
			button.grid(row=button_idx, column=0)
			button_idx += 1
		add_button()
	
	def _on_user_request_exit(self):
		print('User requested exit')
	
	def refresh_canvas(self):
		image = self.renderer.render()
		
		self.tk_canvas.delete('all')
		
		tk_image = ImageTk.PhotoImage(image)
		self.tk_canvas.usr_image_ref = tk_image
		self.tk_canvas.create_image(0, 0, image=tk_image, anchor='nw')
		
		
		if self.show_vanishing_points:
			vanishing = np.array([
				[1, 0, 0],
				[-1, 0, 0],
				[0, 1, 0],
				[0, -1, 0],
				[0, 0, 1],
				[0, 0, -1],
			])
			colors = ['red', 'red', 'green', 'green', 'blue', 'blue']
			for color, point in zip(colors, vanishing):
				
				draw_at = self.renderer.get_vanishing_point_on_canvas(point)
				if draw_at is not None:
				
					draw_utils.draw_disk(self.tk_canvas, draw_at, 6, fill='black')
					draw_utils.draw_disk(self.tk_canvas, draw_at, 5, fill=color)
				
		
	def add_pano_obj_from_file(self, fname_image):
		fname_leafless, fname_leaf = os.path.splitext(fname_image)
		fname_transform = fname_leafless + '.json'
		
		image = Image.open(fname_image)
		
		json_data = io_utils.json_load(fname_transform)
		matr = io_utils.load_matrix_from_json(json_data['image_plane_matrix'])
		
		return self.renderer.add_pano_obj(image, matr)
		
		if False:
			flip_x = np.eye(4)
			flip_x[0,0] = -1
			flip_z = np.eye(4)
			flip_z[2,2] = -1
			
			rotate90 = np.eye(4)
			rotate90[0,0] = 0
			rotate90[2,0] = 1
			rotate90[0,2] = -1
			rotate90[2,2] = 0
			
			
			for asdf in [np.eye(4)]:
				self.renderer.add_pano_obj(image, asdf @ matr)
				self.renderer.add_pano_obj(image, asdf @ flip_x @ matr)
				self.renderer.add_pano_obj(image, asdf @ flip_z @ matr)
				self.renderer.add_pano_obj(image, asdf @ flip_z @ flip_x @ matr)
				
	def add_skybox_from_file(self, fname_image):
		skybox_textures = [
			Image.open('ignore/Bridge2/posx.jpg'),
			Image.open('ignore/Bridge2/negx.jpg'),
			Image.open('ignore/Bridge2/posy.jpg'),
			Image.open('ignore/Bridge2/negy.jpg'),
			Image.open('ignore/Bridge2/posz.jpg'),
			Image.open('ignore/Bridge2/negz.jpg'),
		]
		
		return self.renderer.add_skybox(skybox_textures)
	
	def _on_canvas_press_m2(self, event):
		self.tk_canvas.focus_set()
		self.selected_m2_tool.on_press(event, self)
		
	def _on_canvas_drag_m2(self, event):
		self.selected_m2_tool.on_drag(event, self)
		
	def _on_canvas_release_m2(self, event):
		self.selected_m2_tool.on_release(event, self)
		
	def _on_canvas_press_m1(self, event):
		self.tk_canvas.focus_set()
		self.selected_m1_tool.on_press(event, self)
		
	def _on_canvas_drag_m1(self, event):
		self.selected_m1_tool.on_drag(event, self)
		
	def _on_canvas_release_m1(self, event):
		self.selected_m1_tool.on_release(event, self)
	
	def _on_canvas_key(self, event):
		
		binds = {
			'1' : 0,
			'2' : 1,
			'3' : 2,
		}
		
		key = event.char
	
		if key in binds:
			self.selected_axis = binds[key]
			print('Selected axis {}'.format(self.selected_axis))
	
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
	
	tk_root.lift()
	tk_root.focus()
	tk_root.focus_force()
	tk_root.grab_set()
	tk_root.grab_release()
		
	ctx = moderngl.create_standalone_context()
	
	if True:
		editor = Scene_Editor(tk_root, ctx)
		editor.add_skybox_from_file(None)
		obj = editor.add_pano_obj_from_file('ignore/data/bench.png')
		editor.selected_object = obj
	else:
		editor = calibration.Calibration_Editor(tk_root, ctx, 'ignore/data/bench.png')
	
	tk_root.mainloop()

	print('application closed')

if __name__ == '__main__':
	main()

