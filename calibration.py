
from PIL import Image
from PIL import ImageTk
import numpy as np
import render

class Control_Line:
	
	def __init__(self, start_pos, end_pos):
		self.start_pos = np.array(start_pos)
		self.end_pos = np.array(end_pos)

def approximate_intersection(a_vecs, b_vecs):
	dim = b_vecs[0].shape[0]
	
	quad_A = np.zeros((dim, dim))
	quad_b = np.zeros(dim)
	
	for a_vec, b_vec in zip(a_vecs, b_vecs):
		P = (b_vec @ b_vec.T) / (b_vec.T @ b_vec)
		P_I = P - np.eye(dim)
		P_I_sq = P_I @ P_I
		
		quad_A += P_I_sq
		quad_b += P_I_sq @ a_vec
	
	# quadratic is of form
	# x'Ax - 2b'x + c
	
	# solve:
	quad_A_inv = np.linalg.inv(quad_A)
	
	ans = quad_A_inv @ quad_b
	
	return ans
	
class Calibration:
	
	def __init__(self, tk_canvas, opengl_context):
		
		self.image = Image.open('ignore/bears.jpg')
		self.image_dimensions = np.array((self.image.width, self.image.height))
		self.tk_canvas = tk_canvas
		
		self.renderer = render.Single_Image_Renderer(opengl_context, self.image)
		
		self.control_lines = []
		self.scale_factor = 0.1
		
		self.selected_control_line = None
		self.selected_control_line_point = None
		
		self.intersection_points = []
		
		self.control_lines.append(Control_Line([0, 0], [4000, 3000]))
		self.look_pos = np.zeros(2,)
		
		self.setup_binds()
		
	def setup_binds(self):
		self.tk_canvas.bind('<Configure>', self._on_canvas_reconfig)
		self.tk_canvas.bind('<ButtonPress-2>', self._on_canvas_press_m2)
		self.tk_canvas.bind('<B2-Motion>', self._on_canvas_drag_m2)
		self.tk_canvas.bind('<ButtonRelease-2>', self._on_canvas_release_m2)
		self.tk_canvas.bind('<ButtonPress-1>', self._on_canvas_press_m1)
		self.tk_canvas.bind('<B1-Motion>', self._on_canvas_drag_m1)
		self.tk_canvas.bind('<ButtonRelease-1>', self._on_canvas_release_m1)
		self.tk_canvas.bind('<MouseWheel>', self._on_canvas_mousewheel)

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
		
		def draw_disk(pos, radius, color):
			bbox_min = pos - radius
			bbox_max = pos + radius
			self.tk_canvas.create_oval(bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1], fill=color, width=0)
		
		def draw_line_segment(start, end, color, width):
			self.tk_canvas.create_line(start[0], start[1], end[0], end[1], fill=color, width=width)
		
		def draw_line(p1, p2, color, width):
			bbox_max = canvas_size
			bbox_min = np.zeros(2)
			
			line_dir = p2 - p1
			line_ori = p1
			
			# Simple hack for removing divide-by-zero
			eps = 1e-20
			line_dir[line_dir == 0] = eps
			
			
			time_hit_a = (bbox_min - line_ori) / line_dir
			time_hit_b = (bbox_max - line_ori) / line_dir
			
			time_hit_max = np.maximum(time_hit_a, time_hit_b)
			time_hit_min = np.minimum(time_hit_a, time_hit_b)
			
			time_enter = np.max(time_hit_min)
			time_exit = np.min(time_hit_max)
			
			pos_enter = line_ori + (time_enter * line_dir)
			pos_exit = line_ori + (time_exit * line_dir)
			
			draw_line_segment(pos_enter, pos_exit, color, width)
		
		for point in self.intersection_points:
			draw_disk(point, 9, 'black')
			draw_disk(point, 8, 'blue')
				
		
		for con_line in self.control_lines:
			
			start = image_draw_point + (con_line.start_pos * self.scale_factor)
			end = image_draw_point + (con_line.end_pos * self.scale_factor)
			
			
			draw_line_segment(start, end, 'black', 5)
			draw_disk(start, 6, 'black')
			draw_disk(end, 6, 'black')
			
			draw_line_segment(start, end, 'red', 3)
			draw_disk(start, 5, 'red')
			draw_disk(end, 5, 'red')
			
			draw_line(start, end, 'red', 1)
	
	def _on_canvas_press_m2(self, event):
		self.last_m2_mouse_pos = np.array((event.x, event.y))
		
	def _recalc_intersections(self):
		self.intersection_points = []
		
		a_vecs = []
		b_vecs = []
		
		for con_line in self.control_lines:
			a_vecs.append(con_line.start_pos)
			b_vecs.append(con_line.end_pos - con_line.start_pos)
		
		intersect = approximate_intersection(a_vecs, b_vecs)
		self.intersection_points.append(intersect)
		
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
		click_pos = self.canvas_to_image_coords(np.array((event.x, event.y)))
		
		selection = self.get_selection(click_pos)
		if selection is not None:
			point_name, con_line = selection
			self.selected_control_line = con_line
			self.selected_control_line_point = point_name
		
		
		else:
			new_line = Control_Line(click_pos, click_pos)
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
			
			self.refresh_canvas()
	
	def _on_canvas_reconfig(self,event):
		self.canvas_size = np.array((event.width, event.height))
		self.renderer.resize(event.width, event.height)
		self.refresh_canvas()
