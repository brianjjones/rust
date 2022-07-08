import tensorflow as tf
from neural_compressor.experimental import Graph_Optimization

# default input shape 299x299x3
model = tf.keras.applications.inception_v3.InceptionV3(
    weights='imagenet',
    input_shape=(299, 299, 3)
    # classifier_activation='softmax'
)
directory = "examples/inceptionv3"
model.save(directory, save_format="tf")

graph_opt = Graph_Optimization()
graph_opt.model = directory   # the path to saved_model dir
# output = graph_opt() output.save('examples/inceptionv3/optimized_model')
graph_opt.output = 'examples/inceptionv3/optimized_model'
optimized_model = graph_opt()
model.save('examples/inceptionv3/optimized_model')
# save the model



######################################################
# Check the prediction results for the sample image. #
######################################################
# load sample image
fname = "examples/mobilenetv3/sample_image/macaque.jpg"
buf = tf.io.read_file(fname)
img = tf.image.decode_jpeg(buf)

# clip to the square and resize to (299, 299)
small = tf.image.resize(img[:, 100:-100], (299, 299), antialias=True)

# dump the content to use from Rust later
small = tf.cast(small, tf.uint8)
buf = tf.image.encode_png(small)
tf.io.write_file(directory + "/sample.png", buf)

# check model prediction
predict = model(small[tf.newaxis, :, :, :])
predict = predict.numpy()
decoded = tf.keras.applications.imagenet_utils.decode_predictions(predict, top=1)[0]

print(f"""
argmax={predict.argmax(axis=1)[0]}
""")
print("class_name | class_description | score")
print("-----------+-------------------+------")
print(f"{decoded[0][0]:>10} | {decoded[0][1]:>17} | {decoded[0][2]:0.3f}")
