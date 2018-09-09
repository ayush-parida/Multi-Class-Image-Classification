Multi class vehicle classifier
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Requirements
1. Python 3.5.x and Pip
	Install for Ubuntu
		Run "sudo apt-get install python-pip python-dev" in terminal.
	
2. TensorFlow
	Install for windows
		Run "pip3 install --upgrade tensorflow" in terminal.
	Install for Ubuntu
		Run "pip3 install tensorflow" in terminal.
    Note: For Windows users, TensorFlow only supports version 3.5.x of Python.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
How to create your own image classifier

1. Install Python 3.5.x and Pip

2. Install TensorFlow

3. Validate Your Installation
    For both users, invoke python from your shell as follows:

	$ python      # Ubuntu
	C:\> python   # Windows
	Enter the following short program inside the python interactive shell:
	
	>>> import tensorflow as tf
	>>> hello = tf.constant('Hello, TensorFlow!')
	>>> sess = tf.Session()
	>>> print(sess.run(hello))
   If the system outputs the following, then Tensorflow has been successfully installed:

	Hello, TensorFlow!

4. Prepare your training data
    Training data is what you want your computer to learn, to recognize the objects in the images.There is a super convenient Google plug-in called “Fatkun Batch     Download Image” may help us downloading amount of images at once.

5. Place the images into different folder for each different type as they are arranged in data folder.

6. Start Retraining

    At first, get the latest sample code from git TensorFlow repository. The sample code will be in "/tensorflow/tensorflow/examples/image_retraining/retrain.py"

	For training type the following command in terminal

		python {$your-working_directory}/retrain.py
		--bottleneck_dir=/{$your-working_directory}/bottlenecks 
		--how_many_training_steps 500
		--model_dir=/{$your-working_directory}/inception
		--output_graph=/{$your-working_directory}/retrained_graph.pb
		--output_labels=/{$your-working_directory}/retrained_labels.txt
		--image_dir /{$your-working_directory}/${your_training_data_path}

   The command I used was:
   retrain.py --bottleneck_dir=bottleneck --how_many_training_steps 2000 --model_dir=model --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir data

7. Test your image classifier

    Go to your work directory and create a file called "label_image.py".
    Type the following lines of code and save it.

import tensorflow as tf, sys
image_path = sys.argv[1]
# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
    in tf.gfile.GFile("D:/tensorflow_work/retrained_labels.txt")]
# Unpersists graph from file
with tf.gfile.FastGFile("D:/tensorflow_work/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
# Feed the image_data as input to the graph and get first prediction
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
     predictions = sess.run(softmax_tensor, 
     {'DecodeJpeg/contents:0': image_data})
     # Sort to show labels of first prediction in order of confidence
     top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
     for node_id in top_k:
         human_string = label_lines[node_id]
         score = predictions[0][node_id]
         print('%s (score = %.5f)' % (human_string, score))

   Run the command in terminal
	"python label_image.py Image_to_test_path/Image_name.extension"

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Notes:

1. There should be atleast 50 images of each type in the dataset for proper training of the model.
2. The model may take 20 minutes to 2 hours depending on the amount of data and configuration of the machine.
3. The above is made and tested on Python 3.5.2 only.