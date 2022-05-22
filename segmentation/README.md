- Link data preprocess for segmentation https://drive.google.com/file/d/1is5oG6xMQukPcraT7Jn8R3hO1E97NfGo/view?fbclid=IwAR0XoH5dKVej4p0KqTEamlz15EOznJLv3xbnul-WtsQe1TeLGJHdvnZVrfU 

- To create above data foler, you can follow by:
    - From the organizer's data, collect all photos and masks together in 2 folders image and mask
    - Next, you run code to augmentation data: 
        - <code>python data.py</code>
    - After that, you move all image into folder data, you run code to make noisy background for image trainning
        - <code>python make_noisy.py</code>

    - So you can have my data folder to training

- To training model, you can run:
    - <code>python new_train.py</code>

- To detect model, you can run:
    - <code>python predict.py</code>
    - we provide pretrained model file in res folder to user predict,but also you can use you pretrained
    - After that, you can view result mask in res/data

