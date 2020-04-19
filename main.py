import tkinter
from PIL import ImageTk
import render

def main():
	tk_root = tkinter.Tk()
	tk_root.title('hello world')

	# Needs to happen after tk init for some reason
	glfw_win = render.opengl_context_init()
	
	tk_canvas = tkinter.Canvas(tk_root, width=500, height=500, bg='black')
	tk_canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
	
	renderer = render.Renderer()

	def clicky():
		image = renderer.render()

		tkimage = ImageTk.PhotoImage(image)
		
		if False:
			label = tkinter.Label(tk_root, image=tkimage)
			label.james_image_reference = tkimage
			label.pack()
		
		tk_canvas.j_image_ref = tkimage
		tk_canvas.delete(tkinter.ALL)
		tk_canvas.create_image(0, 0, image=tkimage)

	button = tkinter.Button(tk_root, text='clicky', command=clicky)
	button.pack()

	tk_root.mainloop()
	render.opengl_context_cleanup(glfw_win)

	print('application closed')

if __name__ == '__main__':
	main()

