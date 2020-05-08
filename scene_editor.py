import tkinter
import tkinter.filedialog
from PIL import ImageTk
import render
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
		
		self.current_scene_fname = None
		
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
		self.tk_menubar_file.add_command(label='Open', command=self._on_user_open)
		self.tk_menubar_file.add_command(label='Save', command=self._on_user_save)
		self.tk_menubar_file.add_command(label='Save as...', command=self._on_user_save_as)
		self.tk_menubar_file.add_separator()
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
		
		self.tk_quick_button_frame = tkinter.Frame(tk_master, relief=tkinter.GROOVE, bd=1)
		self.tk_quick_button_frame.grid(row=99, column=100, columnspan=2, sticky='ns')
		
		self._add_quick_buttons()
		
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
		
	def _add_quick_buttons(self):
		frame = self.tk_quick_button_frame
		
		button_row = 0
		button_col = 0
		
		def make_quick_button(fun, label):
			nonlocal button_col
			def wrapper_fun():
				if self.selected_object is not None:
					fun(self.selected_object)
				self.refresh_canvas()
			tkinter.Button(frame, text=label, command=wrapper_fun).grid(row=button_row, column=button_col)
			button_col += 1
			
		# Buttons
			
		def fun(obj):
			matr = np.eye(4)
			axis = self.selected_axis
			matr[axis,axis] = -1
			obj.apply_world_transform(matr)
		make_quick_button(fun, 'Flip')
			
		def fun(obj):
			matr = np.eye(4)
			axis_a, axis_b = [x for x in range(3) if x != self.selected_axis]
			matr[axis_a,axis_a] = 0
			matr[axis_a,axis_b] = -1
			matr[axis_b,axis_a] = 1
			matr[axis_b,axis_b] = 0
			obj.apply_world_transform(matr)
		make_quick_button(fun, 'Rot 90')
	
	def _on_user_save(self):
		print('User requested save')
		if self.current_scene_fname is None:
			self._on_user_save_as()
		else:
			self.save_to_file(self.current_scene_fname)
		
	def _on_user_open(self):
		print('User requested open')
		fname = tkinter.filedialog.askopenfilename(filetypes=(('JSON scenes', '*.json'),))
		print('fname = {}'.format(fname))
		self.load_from_file(fname)
		
	def _on_user_save_as(self):
		print('User requested save as')
		fname = tkinter.filedialog.asksaveasfilename(filetypes=(('JSON scenes', '*.json'),))
		print('fname = {}'.format(fname))
		self.save_to_file(fname)
	
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
		
		json_data = io_utils.json_load(fname_transform)
		matr = io_utils.load_matrix_from_json(json_data['image_plane_matrix'])
		
		self.renderer.add_pano_obj(fname_image, matr)
		
		return self.renderer.add_pano_obj(fname_image, matr)
				
	def add_skybox_from_file(self, fname_image):
		return self.renderer.add_skybox(fname_image)
		
	def save_to_file(self, fname):
		json_data = self.save_to_json()
		io_utils.json_save(json_data, fname)
		print('Save scene to: {}'.format(fname))
		
	def save_to_json(self):
		json_data = {}
		
		obj_list = []
		json_data['objs'] = obj_list
		
		for obj in self.renderer.pano_objs:
			obj_json = {}
			obj_json['src'] = obj.source_fname
			obj_json['matr'] = io_utils.save_matrix_to_json(obj.model_matr)
			obj_json['skybox'] = obj.is_skybox
			obj_list.append(obj_json)
			
		return json_data
		
	def load_from_file(self, fname):
		json_data = io_utils.json_load(fname)
		self.load_from_json(json_data)
		self.current_scene_fname = fname
		print('Load scene from: {}'.format(fname))
		
	def load_from_json(self, json_data):
		self.renderer.clear_all()
		
		obj_list = json_data['objs']
		
		for obj_json in obj_list:
			is_skybox = obj_json['skybox']
			source_fname = obj_json['src']
			matr = io_utils.load_matrix_from_json(obj_json['matr'])
			if is_skybox:
				self.renderer.add_skybox(source_fname, matr)
			else:
				self.renderer.add_pano_obj(source_fname, matr)
		self.refresh_canvas()
	
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
	
