
import numpy as np

def draw_disk(tk_canvas, pos, radius, **kwargs):
	bbox_min = pos - radius
	bbox_max = pos + radius
	tk_canvas.create_oval(bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1], width=0, **kwargs)

def draw_line_segment(tk_canvas, start, end, **kwargs):
	tk_canvas.create_line(start[0], start[1], end[0], end[1], **kwargs)

def draw_line(tk_canvas, p1, p2, **kwargs):
	bbox_max = np.array((tk_canvas.winfo_width(), tk_canvas.winfo_height()))
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
	
	draw_line_segment(tk_canvas, pos_enter, pos_exit, **kwargs)
