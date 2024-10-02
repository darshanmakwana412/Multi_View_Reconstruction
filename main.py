from parameter_esimation import ObjView, Scene

obj_dir = "bottle"
views = ObjView.load_from(obj_dir)
scene = Scene(views, obj_dir)