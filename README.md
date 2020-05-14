# Perspective-Aware Collaging Tool

The following libraries are needed:
+ `moderngl`
+ `Pillow`
+ `numpy`
+ `pyrr`

Requires OpenGL 3.

(Tested only on MacOS as of writing. However, since it only uses Python's built-in `tkinter` for GUI, there should be minimal issues running on other platforms.)

## Quickstart Guide:

(TODO: add link for example data for people who download from GitHub.)

Run `python3 main.py`. Go to `File -> Open` and open one of the example scenes (e.g. `scenes/LemonadeCollage.json`).

Interface:
+ Left-hand side: Tool selection.
	+ `AA-Scale`: Axis-aligned scaling.
	+ `AA-Rot`: Axis-aligned rotation.
	+ `Masking`: Brush for removing/adding to an image's alpha channel.
+ Right-hand side: Layer slection.
	+ Checkbox: select layer for transformation.
	+ Arrows: Change layer order.
+ Top Panel: Single-click tools
	+ `Flip`: Flip the image along the selected axis.
	+ `Rot 90`: Rotate the image by 90 degrees along the selected axis.
	
Controls:
+ `1,2,3`: Select an axis (x, y, z respectively) to apply a transformation along.
+ `Left Click/Drag`: Apply the transformation by dragging horizontally.
+ `Right Click/Drag`: Pan camera.

Menubar:
+ `File -> Save` to save the current project. (Warning: some bugs are here, I would use Save-as instead.)
+ `File -> Export` save as a cubemap.
+ `Layer -> Add Photo` add a new object to the scene. Example objects are in `data/...`. Select the image. To add more images, use the calibration tool (below).
+ `Layer -> Add Skybox` add a new skybox to the scene. Example skyboxes are in `data/skybox/...`. Select the folder containing the skybox data.
+ `Layer -> Open Calibration Tool` open the tool for solving for a photo's principal point (optical center). Save by pressing `save`.
+ `Layer -> Duplicate Selected` duplicate the selected layers.
+ `Layer -> Erase Selected` delete the selected layers. Asks for user confirmation.

Have fun!
