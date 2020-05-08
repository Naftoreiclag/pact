import tkinter
import moderngl
import scene_editor
import calibration

def main():
	tk_root = tkinter.Tk()
	tk_root.title('hello world')
	
	tk_root.lift()
	tk_root.focus()
	tk_root.focus_force()
	tk_root.grab_set()
	tk_root.grab_release()
		
	ctx = moderngl.create_standalone_context()
	
	fname = 'ignore/data/barn.jpg'
	if True:
		editor = scene_editor.Scene_Editor(tk_root, ctx)
		obj2 = editor.add_skybox_from_file('ignore/Bridge2/')
		obj = editor.add_pano_obj_from_file(fname)
		editor.select_object(obj)
	else:
		editor = calibration.Calibration_Editor(tk_root, ctx, fname)
	
	tk_root.mainloop()

	print('application closed')

if __name__ == '__main__':
	main()

