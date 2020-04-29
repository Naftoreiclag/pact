import tkinter
from PIL import ImageTk
import render
import time

def main():
	tk_root = tkinter.Tk()
	tk_root.title('hello world')

	# Needs to happen after tk init for some reason
	glfw_win = render.opengl_context_init()
	
	tk_canvas = tkinter.Canvas(tk_root, width=800, height=600, bg='black')
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
		
	tk_canvas.bind('<Configure>', on_canvas_reconfig)

	def clicky():
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
	tk_root.after(10, update_canvas)

	button = tkinter.Button(tk_root, text='clicky', command=clicky)
	button.pack()
	tk_root.mainloop()
	render.opengl_context_cleanup(glfw_win)

	print('application closed')

if __name__ == '__main__':
	main()

