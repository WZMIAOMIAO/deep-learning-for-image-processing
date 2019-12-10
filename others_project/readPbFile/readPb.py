import tensorflow as tf
import configparser
from distutils.version import StrictVersion
import cv2
import glob
from using_function import draw_box, read_pbtxt, get_inAndout_tensor, convert_type, read_image

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# 读取参数配置文件
conf = configparser.ConfigParser()
conf.read('info.config')
path_to_frozen_graph = conf.get('tensorflow', 'path_to_frozen_graph')
path_to_labels = conf.get('tensorflow', 'path_to_labels')
path_to_images = conf.get('tensorflow', 'path_to_images')
probability_thresh = float(conf.get('tensorflow', 'probability_thresh'))

# 读取pbtxt标签信息
category_index = read_pbtxt(path_to_labels)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    with tf.Session() as sess:
        # Get handles to input and output tensors
        tensor_dict, image_tensor = get_inAndout_tensor()
        test_image_paths = glob.glob(path_to_images)
        for image_path in test_image_paths:
            image_BGR, image_np_expanded = read_image(image_path)

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image_np_expanded})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            convert_type(output_dict)

            draw_box(image_BGR,
                     output_dict['detection_boxes'],
                     output_dict['detection_classes'],
                     output_dict['detection_scores'],
                     category_index,
                     thresh=probability_thresh,
                     line_thickness=5)
            cv2.namedWindow("prediction", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("prediction", image_BGR)
            cv2.waitKey(0)
