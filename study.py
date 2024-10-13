sample_path = "train/conditioning_images/020728.png train/images/020728.png"
condation_img_path,normal_img_path = sample_path.split() 
scene_name = sample_path.split('/')[0]
#img_name, img_ext = sample_path.split('/')[1]
print(scene_name)
print(condation_img_path)
#print(img_name,img_ext)