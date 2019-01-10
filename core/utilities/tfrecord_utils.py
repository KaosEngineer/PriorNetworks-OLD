import tensorflow as tf

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature_list(values):
    return tf.train.FeatureList(feature=[int64_feature([value]) for value in values])

def bytes_feature_list(values):
    return tf.train.FeatureList(feature=[bytes_feature([value]) for value in values])

def float_feature_list(values):
    return tf.train