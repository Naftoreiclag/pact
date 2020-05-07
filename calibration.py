
from PIL import Image
from PIL import ImageTk
import numpy as np
import io_utils
import os
import render
import tkinter
import draw_utils

class Control_Line:
	
	def __init__(self, start_pos, end_pos, channel):
		self.start_pos = np.array(start_pos)
		self.end_pos = np.array(end_pos)
		self.channel = channel
		
	def save_to_json(self):
		return {
			'start' : [self.start_pos[0], self.start_pos[1]],
			'end' : [self.end_pos[0], self.end_pos[1]],
			'channel' : self.channel,
		}
		
	def load_from_json(data):
		start_pos = np.array(data['start'])
		end_pos = np.array(data['end'])
		channel = int(data['channel'])
		return Control_Line(start_pos, end_pos, channel)

def approximate_intersection(a_vecs, b_vecs):
	if len(a_vecs) < 2:
		return None
	dim = b_vecs[0].shape[0]
	
	quad_B = np.zeros((dim, dim))
	quad_a = np.zeros(dim)
	
	for a_vec, b_vec in zip(a_vecs, b_vecs):
		P = np.outer(b_vec, b_vec) / np.inner(b_vec, b_vec)
		P_I = np.eye(dim) - P
		
		quad_B += P_I
		quad_a += P_I @ a_vec
	
	# quadratic is of form
	# x'Bx - 2a'x + c
	
	_, singular_vals, _ = np.linalg.svd(quad_B)
	small_sing = singular_vals[1]
	small_sing_n = small_sing / len(a_vecs)
	
	# It is easily provable that sum(singular_vals) = trace(quad_B) = N
	# Since tr(P_I) = 1
	
	# Heuristic
	if small_sing_n < 1e-4:
		return None
	
	# solve:
	try:
		return np.linalg.solve(quad_B, quad_a)
	except np.linalg.LinAlgError as e:
		return None
	
def compute_centroid(tri_points):
	
	def simple_orthogonal(direction):
		return np.array((direction[1], -direction[0]))
	
	dir_ab = tri_points[1] - tri_points[0]
	dir_bc = tri_points[2] - tri_points[1]
	dir_ca = tri_points[0] - tri_points[2]
	
	line_starts = tri_points
	line_dirs = [simple_orthogonal(x) for x in [dir_bc, dir_ca, dir_ab]]
	
	intersection = approximate_intersection(line_starts, line_dirs)
	
	midpoint_ab = (tri_points[0] + tri_points[1]) / 2
	midpoint_bc = (tri_points[1] + tri_points[2]) / 2
	midpoint_ca = (tri_points[2] + tri_points[0]) / 2
	
	radius_ab = np.linalg.norm(dir_ab) / 2
	radius_bc = np.linalg.norm(dir_bc) / 2
	radius_ca = np.linalg.norm(dir_ca) / 2
	
	dist_ab_i = np.linalg.norm(midpoint_ab - intersection)
	dist_bc_i = np.linalg.norm(midpoint_bc - intersection)
	dist_ca_i = np.linalg.norm(midpoint_ca - intersection)
	
	height_ab = radius_ab ** 2 - dist_ab_i ** 2
	height_bc = radius_bc ** 2 - dist_bc_i ** 2
	height_ca = radius_ca ** 2 - dist_ca_i ** 2
	
	assert(np.allclose(height_ab, height_bc))
	assert(np.allclose(height_ab, height_ca))
	
	if height_ab < 0:
		dual_dist = None
	else:
		dual_dist = np.sqrt(height_ab)
	
	return intersection, dual_dist
	
class Calibration_Editor:
	
	def __init__(self, tk_master, opengl_context, image_fname, save_data=None):
		
		self.image_fname = image_fname
		self.image = Image.open(image_fname)
		self.image_dimensions = np.array((self.image.width, self.image.height))
		self.tk_master = tk_master
		self.setup_interface()
		
		
		self.renderer = render.Single_Image_Renderer(opengl_context, self.image)
		
		self.control_lines = []
		self.scale_factor = 0.0
		
		self.selected_channel = 0
		self.num_channels = 3
		self.channel_colors = ['red', 'green', 'blue']
		
		self.selected_control_line = None
		self.selected_control_line_point = None
		
		self.intersection_points = []
		
		self.centroid_preview = None
		
		self.look_pos = np.zeros(2,)
		
		
		if save_data is not None:
			self.load_from_json(save_data)
			
	def setup_interface(self):
		
		self.tk_canvas = tkinter.Canvas(self.tk_master, width=800, height=800)
		self.tk_canvas.pack()
		
		self.tk_canvas.bind('<Configure>', self._on_canvas_reconfig)
		
		self.tk_canvas.bind('<Key>', self._on_canvas_key)
		
		self.tk_canvas.bind('<ButtonPress-2>', self._on_canvas_press_m2)
		self.tk_canvas.bind('<B2-Motion>', self._on_canvas_drag_m2)
		self.tk_canvas.bind('<ButtonRelease-2>', self._on_canvas_release_m2)
		self.tk_canvas.bind('<ButtonPress-1>', self._on_canvas_press_m1)
		self.tk_canvas.bind('<B1-Motion>', self._on_canvas_drag_m1)
		self.tk_canvas.bind('<ButtonRelease-1>', self._on_canvas_release_m1)
		self.tk_canvas.bind('<MouseWheel>', self._on_canvas_mousewheel)
		
		fname_leafless, fname_leaf = os.path.splitext(self.image_fname)
		fname_transform = fname_leafless + '.json'
	
		def on_button_save():
			json_data = self.save_to_json()
			io_utils.json_save(json_data, fname_transform)
			print('Save to {}'.format(fname_transform))
		def on_button_load():
			json_data = io_utils.json_load(fname_transform)
			self.load_from_json(json_data)
			print('Load from {}'.format(fname_transform))
			
		button1 = tkinter.Button(self.tk_master, text='save', command=on_button_save)
		button1.pack()
		button2 = tkinter.Button(self.tk_master, text='load', command=on_button_load)
		button2.pack()

	def _on_canvas_mousewheel(self, event):
		
		new_scale = self.scale_factor * np.exp(event.delta / 30)
		new_scale = np.clip(new_scale, 0.01, 10)
		scale_factor_change = new_scale / self.scale_factor
		
		# Find where the mouse is 
		mouse_canvas_pos = np.array((event.x, event.y))
		
		origin_point = self._calc_origin_point()
		dist_to_origin = origin_point - mouse_canvas_pos
		
		self.look_pos += dist_to_origin
		self.look_pos -= dist_to_origin * scale_factor_change
		
		self.scale_factor = new_scale
		self.refresh_canvas()

	def _calc_origin_point(self):
		
		canvas_size = np.array((self.renderer.get_width(), self.renderer.get_height()))
		origin_point = (canvas_size / 2) - self.look_pos
		
		return origin_point

	def refresh_canvas(self):
		
		canvas_size = np.array((self.renderer.get_width(), self.renderer.get_height()))
		origin_point = self._calc_origin_point()
		
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
			
			
			draw_utils.draw_line_segment(self.tk_canvas, start, end, fill='black', width=5)
			draw_utils.draw_disk(self.tk_canvas, start, 6, fill='black')
			draw_utils.draw_disk(self.tk_canvas, end, 6, fill='black')
			
			color = self.channel_colors[con_line.channel]
			
			draw_utils.draw_line_segment(self.tk_canvas, start, end, fill=color, width=3)
			draw_utils.draw_disk(self.tk_canvas, start, 5, fill=color)
			draw_utils.draw_disk(self.tk_canvas, end, 5, fill=color)
			
			draw_utils.draw_line(self.tk_canvas, start, end, fill=color, width=1)
		
		for channel_idx, point in enumerate(self.intersection_points):
			if point is None:
				continue
			draw_at = image_draw_point + (point * self.scale_factor)
			
			draw_utils.draw_disk(self.tk_canvas, draw_at, 9, fill='black')
			
			color = self.channel_colors[channel_idx]
			draw_utils.draw_disk(self.tk_canvas, draw_at, 8, fill=color)
	
		if self.centroid_preview is not None:
			
			draw_at = image_draw_point + (self.centroid_preview * self.scale_factor)
			draw_utils.draw_disk(self.tk_canvas, draw_at, 4, fill='black')
			draw_utils.draw_disk(self.tk_canvas, draw_at, 3, fill='white')
			
	
	def _on_canvas_press_m2(self, event):
		self.tk_canvas.focus_set()
		self.last_m2_mouse_pos = np.array((event.x, event.y))
		
	def _recalc_intersections(self):
		self.intersection_points = solve_vanishing_points(self.control_lines, self.num_channels)
		if not any(x is None for x in self.intersection_points):
			self.centroid_preview, _ = compute_centroid(self.intersection_points)
		else:
			self.centroid_preview = None
		
	def _on_canvas_drag_m2(self, event):
		new_pos = np.array((event.x, event.y))
		
		delta = new_pos - self.last_m2_mouse_pos
		
		self.look_pos -= delta
		
		self.last_m2_mouse_pos = new_pos
		self.refresh_canvas()
		
	def _on_canvas_release_m2(self, event):
		pass
		
	def canvas_to_image_coords(self, canvas_pos):
		
		image_center = self.image_dimensions / 2
		origin_point = self._calc_origin_point()
		
		to_canvas = canvas_pos - origin_point
		to_canvas /= self.scale_factor
		
		return image_center + to_canvas
		
	def image_to_canvas_coords(self, image_pos):
		
		image_center = self.image_dimensions / 2
		origin_point = self._calc_origin_point()
		
		to_image = image_pos - image_center
		to_image *= self.scale_factor
		
		return origin_point + to_image
		
	def get_selection(self, image_pos, select_rad=10):
		
		click_canvas_coords = self.image_to_canvas_coords(image_pos)
		for con_line in self.control_lines[::-1]:
			for point_name, point in [('start', con_line.start_pos), ('end', con_line.end_pos)]:
				point_canvas_coords = self.image_to_canvas_coords(point)
				if np.linalg.norm(click_canvas_coords - point_canvas_coords) < select_rad:
					return point_name, con_line
		return None
				
		
	def _on_canvas_press_m1(self, event):
		
		self.tk_canvas.focus_set()
		click_pos = self.canvas_to_image_coords(np.array((event.x, event.y)))
		
		selection = self.get_selection(click_pos)
		if selection is not None:
			point_name, con_line = selection
			self.selected_control_line = con_line
			self.selected_control_line_point = point_name
		else:
			new_line = Control_Line(click_pos, click_pos, self.selected_channel)
			self.control_lines.append(new_line)
			self.selected_control_line = new_line
			self.selected_control_line_point = 'start'
			self.refresh_canvas()
		
	def _on_canvas_drag_m1(self, event):
		click_pos = self.canvas_to_image_coords(np.array((event.x, event.y)))
		if self.selected_control_line is not None:
			if self.selected_control_line_point == 'start':
				self.selected_control_line.start_pos = click_pos
			elif self.selected_control_line_point == 'end':
				self.selected_control_line.end_pos = click_pos
				
			self._recalc_intersections()
			self.refresh_canvas()
		
	def _on_canvas_release_m1(self, event):
		if self.selected_control_line is not None:
			apparent_start = self.image_to_canvas_coords(self.selected_control_line.start_pos)
			apparent_end = self.image_to_canvas_coords(self.selected_control_line.end_pos)
			length = np.linalg.norm(apparent_start - apparent_end)
			
			if length < 20:
				self.control_lines.remove(self.selected_control_line)
				self.selected_control_line = None
			
			self._recalc_intersections()
			self.refresh_canvas()
	
	def _on_canvas_reconfig(self, event):
		# On bootup, get a good default size
		if self.scale_factor == 0.0:
			self.scale_factor = max(event.width, event.height) / min(self.image.width, self.image.height)
			self.scale_factor = np.clip(self.scale_factor, 0.1, 10)
			
		if self.renderer.get_width() == event.width and self.renderer.get_height() == event.width:
			return
		self.canvas_size = np.array((event.width, event.height))
		self.renderer.resize(event.width, event.height)
		print('Resize canvas to: {}'.format([event.width, event.height]))
		self.refresh_canvas()
		
	def _on_canvas_key(self, event):
		
		binds = {
			'1' : 0,
			'2' : 1,
			'3' : 2,
		}
		
		key = event.char
	
		if key in binds:
			self.selected_channel = binds[key]
			print('Selected channel {}'.format(self.selected_channel))
			
	def save_to_json(self):
		retval = {}
		
		retval['control_lines'] = [x.save_to_json() for x in self.control_lines]
		
		try:
			image_plane_matrix = solve_perspective(self.control_lines, self.image_dimensions[0], self.image_dimensions[1])
			retval['image_plane_matrix'] = io_utils.save_matrix_to_json(image_plane_matrix)
		except RuntimeError as e:
			print('Warn! Unable to solve for perspective: {}'.format(e))
		
		return retval
		
	def load_from_json(self, data):
		
		self.control_lines = load_control_lines_from_json(data)
		self._recalc_intersections()
		self.refresh_canvas()
		
def load_control_lines_from_json(json_data):
	control_lines = [Control_Line.load_from_json(x) for x in json_data['control_lines']]
	return control_lines
		
def solve_vanishing_points(control_lines, num_channels):
	if type(control_lines) == dict:
		control_lines = load_control_lines_from_json(control_lines)
	intersection_points = []
	
	for channel in range(num_channels):
		
		a_vecs = []
		b_vecs = []
		
		for con_line in control_lines:
			if con_line.channel == channel:
				a_vecs.append(con_line.start_pos)
				b_vecs.append(con_line.end_pos - con_line.start_pos)
		
		intersect = approximate_intersection(a_vecs, b_vecs)
		intersection_points.append(intersect)
		
	return intersection_points
		
def solve_perspective(control_lines, image_width, image_height):
	if type(control_lines) == dict:
		control_lines = load_control_lines_from_json(control_lines)
	
	vanishing_points = solve_vanishing_points(control_lines, 3)
	
	# Heuristic: check how parallel the lines are
	
	num_points = sum(1 for x in vanishing_points if x is not None)
	
	if num_points == 0:
		raise RuntimeError('Requires at least 1 vanishing point')
	elif num_points == 1:
		raise RuntimeError('One-point perspective not implemented')
	elif num_points == 2:
		raise RuntimeError('Two-point perspective not implemented')
	elif num_points == 3:
		
		centroid, dist = compute_centroid(vanishing_points)
		
		if dist is None:
			raise RuntimeError('Impossible camera settings')
		
		image_plane_matr = np.eye(4)
		image_plane_matr[0, 3] = centroid[0]
		image_plane_matr[1, 3] = centroid[1]
		image_plane_matr[2, 3] = dist
		image_plane_matr[0, 0] = -image_width
		image_plane_matr[1, 1] = -image_height
		
		to_x_vanish = np.zeros(4,)
		to_y_vanish = np.zeros(4,)
		to_z_vanish = np.zeros(4,)
		
		to_x_vanish[:2] = -(vanishing_points[0] - centroid)
		to_y_vanish[:2] = -(vanishing_points[1] - centroid)
		to_z_vanish[:2] = -(vanishing_points[2] - centroid)
		
		to_x_vanish[2] = dist
		to_y_vanish[2] = dist
		to_z_vanish[2] = dist
		
		to_x_vanish /= np.linalg.norm(to_x_vanish)
		to_y_vanish /= np.linalg.norm(to_y_vanish)
		to_z_vanish /= np.linalg.norm(to_z_vanish)
		
		image_rotation = np.eye(4)
		image_rotation[:, 0] = to_x_vanish
		image_rotation[:, 1] = to_y_vanish
		image_rotation[:, 2] = to_z_vanish
		
		undo_image_rotation = image_rotation.T
		
		downscale_matr = np.eye(4)
		downscale_matr[0, 0] = 1/dist
		downscale_matr[1, 1] = 1/dist
		downscale_matr[2, 2] = 1/dist
		
		heuristic_matr = np.eye(4)
		
		# Small heuristic: if the y vanishing point is below the other two, then flip the image
		if vanishing_points[1][1] > centroid[1]:
			heuristic_matr[1, 1] = -1
			
		# Similarly, if the x vanishing point is to the right of the z vanishing point, then flip the image horizontally
		if vanishing_points[0][0] > vanishing_points[2][0]:
			heuristic_matr[0, 0] = -1
		
		return heuristic_matr @ undo_image_rotation @ downscale_matr @ image_plane_matr
	else:
		assert(False)
