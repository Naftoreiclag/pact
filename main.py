import tkinter
from PIL import ImageTk
import render
import time
import numpy as np

class Asdf():
	def __init__(self, tk_canvas):
		self.anchor_xy = np.zeros((2, ))
		self.tk_canvas = tk_canvas
		self.renderer = render.Renderer()
		self.refresh = None
		
	def refresh_canvas(self):
		image = self.renderer.render()
		
		tk_image = ImageTk.PhotoImage(image)
		self.tk_canvas.usr_image_ref = tk_image
		self.tk_canvas.delete('all')
		self.tk_canvas.create_image(0, 0, image=tk_image, anchor='nw')
	
	def on_canvas_press_m1(self, event):
		self.anchor_xy = np.array((event.x, event.y))
		
	def on_canvas_drag_m1(self, event):
		new_anchor_xy = np.array((event.x, event.y))
		
		diff = new_anchor_xy - self.anchor_xy
		self.renderer.view_params.yaw_rad -= (diff[0] / self.renderer.get_width()) * 2
		self.renderer.view_params.pitch_rad += (diff[1] / self.renderer.get_height()) * 2
		
		self.renderer.view_params.pitch_rad = np.clip(self.renderer.view_params.pitch_rad, -np.pi/2, np.pi/2)
		
		self.anchor_xy = new_anchor_xy
		self.refresh_canvas()
	
	
	def on_canvas_reconfig(self,event):
		width = event.width
		height = event.height
		
		self.renderer.resize(width, height)
		
		self.refresh_canvas()
		
	def on_canvas_release_m1(self, event):
		self.anchor_xy = None
	

def main():
	tk_root = tkinter.Tk()
	tk_root.title('hello world')
	
	tk_canvas = tkinter.Canvas(tk_root, width=800, height=800, bg='black')
	tk_canvas.pack(expand=True, fill='both')
	
	
	asdf = Asdf(tk_canvas)
		
	tk_canvas.bind('<Configure>', asdf.on_canvas_reconfig)
	tk_canvas.bind('<ButtonPress-1>', asdf.on_canvas_press_m1)
	tk_canvas.bind('<B1-Motion>', asdf.on_canvas_drag_m1)
	tk_canvas.bind('<ButtonRelease-1>', asdf.on_canvas_release_m1)

	def clicky_1():
		renderer._debug_scalar *= 0.9
		asdf.refresh_canvas()
	def clicky_2():
		renderer._debug_scalar /= 0.9
		asdf.refresh_canvas()

	def update_canvas():
		start = time.time()
		image = renderer.render()
		print("render:", (time.time() - start)*1000)
		start = time.time()
		tk_image = ImageTk.PhotoImage(image)
		tk_canvas.usr_image_ref = tk_image
		tk_canvas.delete('all')
		tk_canvas.create_image(0, 0, image=tk_image, anchor='nw')
		print("update canvas:", (time.time() - start)*1000)
		tk_root.after(10, update_canvas)
	#tk_root.after(10, update_canvas)

	button1 = tkinter.Button(tk_root, text='clicky1', command=clicky_1)
	button1.pack()
	button2 = tkinter.Button(tk_root, text='clicky2', command=clicky_2)
	button2.pack()
	tk_root.mainloop()

	print('application closed')

if __name__ == '__main__':
	main()

