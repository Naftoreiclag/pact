import tkinter
from PIL import ImageTk
import render
import time
import numpy as np

def main():
	tk_root = tkinter.Tk()
	tk_root.title('hello world')
	
	tk_canvas = tkinter.Canvas(tk_root, width=800, height=800, bg='black')
	tk_canvas.pack(expand=True, fill='both')
	
	renderer = render.Renderer()
	
	def refresh_canvas():
		image = renderer.render()
		
		tk_image = ImageTk.PhotoImage(image)
		tk_canvas.usr_image_ref = tk_image
		tk_canvas.delete('all')
		tk_canvas.create_image(0, 0, image=tk_image, anchor='nw')
	
	def on_canvas_reconfig(event):
		width = event.width
		height = event.height
		
		renderer.resize(width, height)
		
		refresh_canvas()
		
	anchor_xy = np.zeros((2, ))
	def on_canvas_press_m1(event):
		nonlocal anchor_xy
		#print('Press', event.x, event.y)
		anchor_xy = np.array((event.x, event.y))
		
	def on_canvas_drag_m1(event):
		nonlocal anchor_xy
		#print('Drag', event.x, event.y)
		new_anchor_xy = np.array((event.x, event.y))
		
		diff = new_anchor_xy - anchor_xy
		renderer.view_params.yaw_rad -= (diff[0] / renderer.get_width()) * 2
		renderer.view_params.pitch_rad += (diff[1] / renderer.get_height()) * 2
		
		renderer.view_params.pitch_rad = np.clip(renderer.view_params.pitch_rad, -np.pi/2, np.pi/2)
		
		anchor_xy = new_anchor_xy
		refresh_canvas()
		
	def on_canvas_release_m1(event):
		nonlocal anchor_xy
		#print('Release', event.x, event.y)
		anchor_xy = None
		
	tk_canvas.bind('<Configure>', on_canvas_reconfig)
	tk_canvas.bind('<ButtonPress-1>', on_canvas_press_m1)
	tk_canvas.bind('<B1-Motion>', on_canvas_drag_m1)
	tk_canvas.bind('<ButtonRelease-1>', on_canvas_release_m1)

	def clicky_1():
		renderer._debug_scalar *= 0.9
		refresh_canvas()
	def clicky_2():
		renderer._debug_scalar /= 0.9
		refresh_canvas()

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

