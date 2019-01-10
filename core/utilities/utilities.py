import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

try:
    import cPickle as pickle
except:
    import pickle


# __all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


def kl(mu1, mu2, sig1, sig2):
    """
    Mean K-L divergence between two univariate Gaussian distributions
    :param mu1: Mean of first Gaussian
    :param mu2: Mean of second Gaussian
    :param sig1: Standard deviation of first Gaussian
    :param sig2: Standard deviation of second Gaussian
    :return:
    """
    return tf.reduce_mean(0.5 * (((mu1 - mu2) ** 2 + sig1 ** 2) / (sig2 ** 2) + tf.log(sig2) - tf.log(sig1)))


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


activation_dict = {'relu': tf.nn.relu,
                   'tanh': tf.nn.tanh,
                   'sigmoid': tf.nn.sigmoid,
                   'elu': tf.nn.elu,
                   'selu': tf.nn.selu,
                   'softmax': tf.nn.softmax,
                   'softsign': tf.nn.softsign,
                   'softplus': tf.nn.softplus,
                   'exp': tf.exp,
                   'lrelu': lrelu
                   }

initializer_dict = {'xavier': tf.contrib.layers.xavier_initializer  # ,
                    # 'glorot' : tf.contrib.layers.glorot_initializer
                    }

activation_fn_list = ['relu', 'tanh', 'sigmoid', 'elu', 'softmax', 'softsign', 'softplus', 'exp', 'lrelu', 'selu']
initializer_list = ['xavier', 'glorot']


def load_data_local(path, bulats=False, shuffle=False, integer=False, seed=100):
    """
    Helper function used to ALTA dataset into numpy arrays along with targets and metadata (L1, etc...)
    name: Name of dataset along with Acoustic Model and feature generation. Ex: BLXXXgrd01/HC3/F2
    directory: directory where it is located.
    shuffle: If true, pre-shuffle data
    seed: seed for shuffling.
    """

    features_raw = np.loadtxt(path + '/features_L1.dat', dtype=np.float32)
    if bulats:
        targets = np.loadtxt(path + '/targets_bulats.dat', dtype=np.float32)
    else:
        targets = np.loadtxt(path + '/targets.dat', dtype=np.float32)

    """
    with open("Data/sim.csv") as f:
      ncols = len(f.readline().split(','))

  data = np.loadtxt("Data/sim.csv", delimiter=',', skiprows=1, usecols=range(1,ncols+1))
  """

    if shuffle:
        data = np.c_[features_raw.reshape(len(features_raw), -1), targets.reshape(len(targets), -1)]
        np.random.seed(seed)
        np.random.shuffle(data)
        features_raw = data[:, :features_raw.size // len(features_raw)].reshape(features_raw.shape)
        targets = data[:, features_raw.size // len(features_raw):].reshape(targets.shape)

    features = features_raw[:, :-1]
    L1_indicator = np.asarray(features_raw[:, -1], dtype=np.int32)

    if integer:
        return np.around(targets[:, -1:]), features, L1_indicator
    else:
        return targets[:, -1:], features, L1_indicator


def load_data(path, use_aux=False, expert=False, shuffle=False, integer=False, seed=100):
    """
    Helper function used to ALTA dataset into numpy arrays along with targets and metadata (L1, etc...)
    name: Name of dataset along with Acoustic Model and feature generation. Ex: BLXXXgrd01/HC3/F2
    directory: directory where it is located.
    shuffle: If true, pre-shuffle data
    seed: seed for shuffling.
    """

    if expert:
        gfile = 'grades-expert.txt'
    else:
        gfile = 'grades.txt'
    with open(path + '/features.txt') as f:
        ncols = len(f.readline().split())
        ncols = len(f.readline().split())
        features = np.loadtxt(path + '/features.txt', dtype=np.float32, delimiter=' ', skiprows=1,
                              usecols=range(1, ncols))
        print features.shape
    if use_aux:
        with open(path + '/aux.txt') as f:
            ncols = len(f.readline().split())
            aux = np.loadtxt(path + '/aux.txt', dtype=np.float32, skiprows=1, usecols=range(1, ncols))
    with open(path + '/' + gfile) as f:
        ncols = len(f.readline().split())
        targets = np.loadtxt(path + '/' + gfile, dtype=np.float32, skiprows=1, usecols=range(1, ncols))

    if len(targets.shape) < 2:
        targets = targets[:, np.newaxis]
    if shuffle:
        if use_aux:
            data = np.c_[
                features.reshape(len(features), -1), aux.reshape(len(aux), -1), targets.reshape(len(targets), -1)]
            np.random.seed(seed)
            np.random.shuffle(data)
            print features.size // len(features)
            features = data[:, :features.size // len(features)].reshape(features.shape)
            aux = data[:, features.size // len(features):aux.size // len(aux) + features.size // len(features)].reshape(
                aux.shape)
            targets = data[:, aux.size // len(aux) + features.size // len(features):].reshape(targets.shape)
        else:
            data = np.c_[features.reshape(len(features), -1), targets.reshape(len(targets), -1)]
            np.random.seed(seed)
            np.random.shuffle(data)
            features = data[:, :features.size // len(features)].reshape(features.shape)
            targets = data[:, features.size // len(features):].reshape(targets.shape)

    if integer:
        targets = np.around(targets)

    if use_aux:
        return targets[:, -1:], features, np.asarray(aux, dtype=np.int32)
    else:
        return targets[:, -1:], features


def create_dict(index):
    dict = {}
    path = os.path.join(index)
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split()
            dict[line[1]] = int(line[0]) + 1
    return dict


def word_to_id(data, index):
    vocab = create_dict(index)
    return [[vocab[word] if vocab.has_key(word) else 0 for word in line] for line in data]


def load_text(data_path, input_index, strip_start_end=True):
    with open(data_path, 'r') as f:
        data = []
        slens = []
        for line in f.readlines():
            line = line.replace('\n', '').split()
            if strip_start_end:
                line = line[1:-1]  # strip off sentence start and sentence end
            if len(line) == 0:
                pass
            else:
                data.append(line)
                slens.append(len(line))
    data = word_to_id(data, input_index)
    return data, slens


def text_to_array(data_path, input_index, strip_start_end=True):
    data, slens = load_text(data_path, input_index, strip_start_end=strip_start_end)

    slens = np.asarray(slens, dtype=np.int32)
    processed_data = np.zeros((len(slens), np.max(slens)), dtype=np.int32)
    for i, length in zip(xrange(len(data)), slens):
        processed_data[i][0:slens[i]] = data[i]

    return processed_data, slens


def process_data_lm(data, path, input_index, output_index, bptt, spId=False):
    data_path = os.path.join(path, data)
    with open(data_path, 'r') as f:
        data = []
        slens = []
        for line in f.readlines():
            line = line.replace('\n', '').split()
            if spId:
                line = line[1:]
            data.append(line)
            slens.append(len(line) - 1)
    in_data = word_to_id(data, path, input_index)
    out_data = word_to_id(data, path, output_index)

    if bptt == None:
        slens = np.asarray(slens, dtype=np.int32)
        input_processed_data = np.zeros((len(slens), np.max(slens)), dtype=np.int32)
        target_processed_data = np.zeros((len(slens), np.max(slens)), dtype=np.int32)

        for i in xrange(len(in_data)):
            input = in_data[i][:-1]
            output = out_data[i][1:]
            input_processed_data[i][0:slens[i]] = input
            target_processed_data[i][0:slens[i]] = output
        return target_processed_data, input_processed_data, slens
    else:
        sequence_lengths = []
        for s in slens:
            if s <= bptt:
                sequence_lengths.append(s)
            else:
                lines = int(np.floor(s / float(bptt)))
                lens = [bptt] * lines
                if len(lens) > 0: sequence_lengths.extend(lens)
                s = s % bptt
                if s > 0:
                    sequence_lengths.append(s)
        sequence_lengths = np.asarray(sequence_lengths, dtype=np.int32)
        # print np.mean(sequence_lengths), np.std(sequence_lengths),

        # print sequence_lengths.shape[0], len(id_data)
        input_processed_data = np.zeros((len(sequence_lengths), bptt), dtype=np.int32)
        target_processed_data = np.zeros((len(sequence_lengths), bptt), dtype=np.int32)
        row = 0
        for i, length in zip(xrange(len(in_data)), slens):
            input = in_data[i][:-1]
            output = out_data[i][1:]
            lines = int(np.ceil(length / float(bptt)))
            for j in xrange(lines):
                input_processed_data[row + j][0:sequence_lengths[row + j]] = input[j * bptt:(j + 1) * bptt]
                target_processed_data[row + j][0:sequence_lengths[row + j]] = output[j * bptt:(j + 1) * bptt]
            row += lines

        return target_processed_data, input_processed_data, sequence_lengths


def process_data_bucket(data, path, spId, input_index):
    data_path = os.path.join(path, data)
    with open(data_path, 'r') as f:
        data = []
        slens = []
        for line in f.readlines():
            line = line.replace('\n', '').split()
            if spId:
                line = line[1:]
            data.append(line)
            slens.append(len(line))
    data = word_to_id(data, path, input_index)

    slens = np.array(slens, dtype=np.int32)
    print np.mean(slens), np.std(slens), np.max(slens)

    processed_data = []

    for dat in data:
        processed_data.append(np.asarray(dat, dtype=np.int32))

    return processed_data, slens, np.int32(np.max(slens) - 1)


def split_train_valid(data_list, valid_size):
    """ Helper Function
    Args:
      data_list: list of numpy arrays
      valid_size : int
    Returns:
      returns two lists of np arrays for eval and train data
    """
    # !!!THIS MAY NOT BE EFFICIENT AT ALL!!!
    vld_data_list = []
    trn_data_list = []
    for data in data_list:
        try:
            vld_data_list.append(data[:valid_size, :])
            trn_data_list.append(data[valid_size:, :])
        except:
            vld_data_list.append(data[:valid_size])
            trn_data_list.append(data[valid_size:])
    return vld_data_list, trn_data_list


# def check_vocab(vocab_file, out_dir, check_special_token=True, sos=None, eos=None, unk=None):
#     """Check if vocab_file doesn't exist, create from corpus_file."""
#     if tf.gfile.Exists(vocab_file):
#         print("# Vocab file %s exists" % vocab_file)
#         vocab = []
#         with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
#             vocab_size = 0
#             for word in f:
#                 vocab_size += 1
#                 vocab.append(word.strip())
#         if check_special_token:
#             # Verify if the vocab starts with unk, sos, eos
#             # If not, prepend those tokens & generate a new vocab file
#             if not unk: unk = UNK
#             if not sos: sos = SOS
#             if not eos: eos = EOS
#             assert len(vocab) >= 3
#             if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
#                 print("The first 3 vocab words [%s, %s, %s]"
#                       " are not [%s, %s, %s]" %
#                       (vocab[0], vocab[1], vocab[2], unk, sos, eos))
#                 vocab = [unk, sos, eos] + vocab
#                 vocab_size += 3
#                 new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
#                 with codecs.getwriter("utf-8")(
#                         tf.gfile.GFile(new_vocab_file, "wb")) as f:
#                     for word in vocab:
#                         f.write("%s\n" % word)
#                 vocab_file = new_vocab_file
#     else:
#         raise ValueError("vocab_file '%s' does not exist." % vocab_file)
#
#     vocab_size = len(vocab)
#     return vocab_size, vocab_file
#
# # NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
# class BatchedInput(
#     collections.namedtuple("BatchedInput",
#                            ("initializer", "source", "source_sequence_length"))):
#     pass

# def get_infer_iterator(src_dataset,
#                        src_vocab_table,
#                        batch_size,
#                        source_reverse,
#                        eos,
#                        src_max_len=None):
#   src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
#   src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
#
#   if src_max_len:
#     src_dataset = src_dataset.map(lambda src: src[:src_max_len])
#   # Convert the word strings to ids
#   src_dataset = src_dataset.map(
#       lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
#   if source_reverse:
#     src_dataset = src_dataset.map(lambda src: tf.reverse(src, axis=[0]))
#   # Add in the word counts.
#   src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))
#
#   def batching_func(x):
#     return x.padded_batch(
#         batch_size,
#         # The entry is the source line rows;
#         # this has unknown-length vectors.  The last entry is
#         # the source row size; this is a scalar.
#         padded_shapes=(
#             tf.TensorShape([None]),  # src
#             tf.TensorShape([])),  # src_len
#         # Pad the source sequences with eos tokens.
#         # (Though notice we don't generally need to do this since
#         # later on we will be masking out calculations past the true sequence.
#         padding_values=(
#             src_eos_id,  # src
#             0))  # src_len -- unused
#
#   batched_dataset = batching_func(src_dataset)
#   batched_iter = batched_dataset.make_initializable_iterator()
#   (src_ids, src_seq_len) = batched_iter.get_next()
#   return BatchedInput(
#       initializer=batched_iter.initializer,
#       source=src_ids,
#       target_input=None,
#       target_output=None,
#       source_sequence_length=src_seq_len,
#       target_sequence_length=None)
#
# def get_iterator(dataset,
#                  vocab_file,
#                  batch_size,
#                  random_seed,
#                  num_buckets,
#                  sos="<s>",
#                  UNK_ID = 0,
#                  UNK = "<unk>",
#                  eos="</s>",
#                  max_len=None,
#                  num_threads=4,
#                  output_buffer_size=None,
#                  skip_count=None,
#                  num_shards=1,
#                  shard_index=0):
#
#     vocab_table = lookup_ops.index_table_from_file(vocab_file, default_value=UNK_ID)
#
#     if not output_buffer_size:
#         output_buffer_size = batch_size * 1000
#     src_eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)
#
#     dataset = dataset.shard(num_shards, shard_index)
#     if skip_count is not None:
#         dataset = dataset.skip(skip_count)
#
#     dataset = dataset.shuffle(output_buffer_size, random_seed)
#
#     dataset = dataset.map(
#         lambda src: tf.string_split([src]).values,
#         num_threads=num_threads,
#         output_buffer_size=output_buffer_size)
#
#     if max_len:
#         dataset = dataset.map(
#             lambda src: src[:max_len],
#             num_threads=num_threads,
#             output_buffer_size=output_buffer_size)
#
#     # Convert the word strings to ids.  Word strings that are not in the
#     # vocab get the lookup table's default_value integer.
#     dataset = dataset.map(
#         lambda src: tf.cast(vocab_table.lookup(src), tf.int32),
#         num_threads=num_threads, output_buffer_size=output_buffer_size)
#
#     # Add in sequence lengths.
#     dataset = dataset.map(
#         lambda src: src, tf.size(src),
#         num_threads=num_threads,
#         output_buffer_size=output_buffer_size)
#
#     # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
#     def batching_func(x):
#         return x.padded_batch(
#             batch_size,
#             # The first three entries are the source and target line rows;
#             # these have unknown-length vectors.  The last two entries are
#             # the source and target row sizes; these are scalars.
#             padded_shapes=(
#                 tf.TensorShape([None]),  # src
#                 tf.TensorShape([])),  # src_len
#             # Pad the source and target sequences with eos tokens.
#             # (Though notice we don't generally need to do this since
#             # later on we will be masking out calculations past the true sequence.
#             padding_values=(
#                 src_eos_id,  # src
#                 0))  # src_len -- unused
#
#     if num_buckets > 1:
#
#         def key_func(unused_1, src_len):
#             # Calculate bucket_width by maximum source sequence length.
#             # Pairs with length [0, bucket_width) go to bucket 0, length
#             # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
#             # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
#             if max_len:
#                 bucket_width = (max_len + num_buckets - 1) // num_buckets
#             else:
#                 bucket_width = 10
#
#             # Bucket sentence pairs by the length of their source sentence and target
#             # sentence.
#             bucket_id = tf.maximum(src_len // bucket_width)
#             return tf.to_int64(tf.minimum(num_buckets, bucket_id))
#
#         def reduce_func(unused_key, windowed_data):
#             return batching_func(windowed_data)
#
#         batched_dataset = dataset.group_by_window(key_func=key_func,
#                                                   reduce_func=reduce_func,
#                                                   window_size=batch_size)
#
#     else:
#         batched_dataset = batching_func(dataset)
#     batched_iter = batched_dataset.make_initializable_iterator()
#     (src_ids, seq_len) = (batched_iter.get_next())
#     return BatchedInput(
#         initializer=batched_iter.initializer,
#         source=src_ids,
#         source_sequence_length=seq_len)

def create_train_op(total_loss,
                    learning_rate,
                    optimizer,
                    optimizer_params,
                    n_examples,
                    batch_size,
                    learning_rate_decay=None,
                    staircase=False,
                    global_step=0,
                    update_ops=None,
                    variables_to_train=None,
                    clip_gradient_norm=0,
                    summarize_gradients=False,
                    gate_gradients=tf.train.Optimizer.GATE_OP,
                    aggregation_method=None,
                    colocate_gradients_with_ops=False,
                    gradient_multipliers=None,
                    check_numerics=True):
    """Creates an `Operation` that evaluates the gradients and returns the loss.
       This functions wrap the tf.contrib.slim create_train_op to add a decaying learning
       rate
    Args:
      total_loss: A `Tensor` representing the total loss.
      learning_rate: A 'Tensor' or float representing base learning rate.
      optimizer: A tf.Optimizer to use for computing the gradients.
      optimzer_params: a dictionary of parameter values to pass to optimizer
      n_examples: A float representing number of examples per epoch.
      batch_size: A float representing batch size.
      learning_rate decay: A 'tensor' or float representing the decay factor for exponentially decay lr.
      staircase: A bool representing whther to staircase or not exponential decay
      global_step: A `Tensor` representing the global step variable. If left as
        `_USE_GLOBAL_STEP`, then slim.variables.global_step() is used.
      update_ops: An optional list of updates to execute. If `update_ops` is
        `None`, then the update ops are set to the contents of the
        `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
        it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`,
        a warning will be displayed.
      variables_to_train: an optional list of variables to train. If None, it will
        default to all tf.trainable_variables().
      clip_gradient_norm: If greater than 0 then the gradients would be clipped
        by it.
      summarize_gradients: Whether or not add summaries for each gradient.
      gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: Whether or not to try colocating the gradients
        with the ops that generated them.
      gradient_multipliers: A dictionary of either `Variables` or `Variable` op
        names to the coefficient by which the associated gradient should be
        scaled.
      check_numerics: Whether or not we apply check_numerics.
    Returns:
      A `Tensor` that when evaluated, computes the gradients and returns the total
        loss value.
    """

    if learning_rate_decay:
        # Variables that affect learning rate.
        decay_steps = n_examples / batch_size
        # Decay the learning rate exponentially based on the number of steps.
        learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=decay_steps,
                                                   decay_rate=learning_rate_decay,
                                                   staircase=staircase)
    return slim.learning.create_train_op(total_loss=total_loss,
                                         optimizer=optimizer(learning_rate, **optimizer_params),
                                         global_step=global_step,
                                         update_ops=update_ops,
                                         variables_to_train=variables_to_train,
                                         clip_gradient_norm=clip_gradient_norm,
                                         summarize_gradients=summarize_gradients,
                                         gate_gradients=gate_gradients,
                                         aggregation_method=aggregation_method,
                                         colocate_gradients_with_ops=colocate_gradients_with_ops,
                                         gradient_multipliers=gradient_multipliers,
                                         check_numerics=check_numerics)