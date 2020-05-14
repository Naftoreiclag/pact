import tkinter
import moderngl
import scene_editor
import calibration

def main():
	tk_root = tkinter.Tk()
	tk_root.title('Perspective-Aware Photo Collaging Tool')
	
	tk_root.lift()
	tk_root.focus()
	tk_root.focus_force()
	tk_root.grab_set()
	tk_root.grab_release()
		
	ctx = moderngl.create_standalone_context()
	
	editor = scene_editor.Scene_Editor(tk_root, ctx)
	
	tk_root.mainloop()

	print('application closed')

if __name__ == '__main__':
	main()

