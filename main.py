import tkinter
from PIL import ImageTk
import render
import time

def main():
	tk_root = tkinter.Tk()
	tk_root.title('hello world')
	
	tk_canvas = tkinter.Canvas(tk_root, width=800, height=800, bg='black')
	tk_canvas.pack(expand=True, fill='both')
	
	renderer = render.Renderer()
	
	def on_canvas_reconfig(event):
		width = event.width
		height = event.height
		
		renderer.resize(width, height)
		image = renderer.render()
		
		tk_image = ImageTk.PhotoImage(image)
		tk_canvas.usr_image_ref = tk_image
		tk_canvas.delete('all')
		tk_canvas.create_image(0, 0, image=tk_image, anchor='nw')
		
	def on_canvas_press_m1(event):
		print('Press', event.x, event.y)
		
	def on_canvas_drag_m1(event):
		print('Drag', event.x, event.y)
		
	def on_canvas_release_m1(event):
		print('Release', event.x, event.y)
		
	tk_canvas.bind('<Configure>', on_canvas_reconfig)
	tk_canvas.bind('<ButtonPress-1>', on_canvas_press_m1)
	tk_canvas.bind('<B1-Motion>', on_canvas_drag_m1)
	tk_canvas.bind('<ButtonRelease-1>', on_canvas_release_m1)

	def clicky():
		renderer.view_params.yaw_rad += (3.14159/180)*10
		image = renderer.render()

		tk_image = ImageTk.PhotoImage(image)
		tk_canvas.usr_image_ref = tk_image
		tk_canvas.delete('all')
		tk_canvas.create_image(0, 0, image=tk_image, anchor='nw')

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

	button = tkinter.Button(tk_root, text='clicky', command=clicky)
	button.pack()
	tk_root.mainloop()

	print('application closed')

if __name__ == '__main__':
	main()

