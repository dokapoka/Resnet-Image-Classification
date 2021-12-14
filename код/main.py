from pytube import YouTube

from moviepy.editor import *
import os
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
from torchvision import transforms

import torch
import streamlit as st
import glob

# Здесь скачиваем видео с youtube и сохраняем в папку
# ссылка на видео с youtube
#link = 'https://www.youtube.com/watch?v=OIiER9ZkgN0'
#link = 'https://www.youtube.com/watch?v=Yw75c7w1-Is'
link = st.text_input(label = 'Ссылка на видео')
if st.button('Поехали!'):
	if link == '':
		st.warning('Нет ссылки :(')
		st.stop()
		
	# путь до папки, куда скачиваем видео
	save_path = '/home/alex/Documents/Prog/Project/Video'
	
	#удалим то, что было скачено ранее
	files = glob.glob(save_path + '/*')
	for f in files:
		os.remove(f)

	try:
		yt = YouTube(link) 
	except:
		st.warning('Не работает :(')
		st.stop() 

	# настраиваем формат видео, выбираем худшее качество для экономии памяти, ускорения работы    
	video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[0]
	try:
		video.download(save_path)
		print('Видео загружено в папку', save_path) 
	except:
		print('Ошибка загрузки') 

	# Здесь достаем скачанное видео из папки. В данном случаем у нас 1 видео
	videos = []

	for root, dirs, files in os.walk(save_path):
		for file in files:
			if os.path.splitext(file)[1] == '.mp4':
				# объединяем в полный путь
				filePath = os.path.join(root, file)
				# загружаем видео
				video = VideoFileClip(filePath)
				videos.append(video)
				
	final_clip = concatenate_videoclips(videos)


	def get_class(frame):
		input_image = frame
		input_image = Image.fromarray(np.uint8(input_image)).convert('RGB')
		
		preprocess = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		input_tensor = preprocess(input_image)
		input_batch = input_tensor.unsqueeze(0)
		model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
		model.eval()

		if torch.cuda.is_available():
			input_batch = input_batch.to('cuda')
			model.to('cuda')

		with torch.no_grad():
			output = model(input_batch)
		probabilities = torch.nn.functional.softmax(output[0], dim=0)
		
		with open('/home/alex/Documents/Prog/Project/imagenet_classes.txt', "r") as f:
			categories = [s.strip() for s in f.readlines()]
		top5_prob, top5_catid = torch.topk(probabilities, 5)
		return categories[top5_catid[0]], top5_prob[0].item()


	k = 0
	for i in range(2, 20, 2):
		k += 1
		frame = final_clip.get_frame(i)
		# рисуем кадр
		fig, ax = plt.subplots()
		ax.imshow(frame, interpolation ='nearest')
		st.write('Класс объекта: ', get_class(frame)[0], ', вероятность принадлежности объекта к классу =',int(round(get_class(frame)[1]*100, 0)), '%', 'Кадр №', k)
		st.pyplot(fig)
