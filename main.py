from data_utils import *
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=307)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pre', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--encode_dim', type=int, default=2)
parser.add_argument('--file_path', type=str, default='PEMS04')

args = parser.parse_args()

blocks = [[1, 32, 64], [64, 128, 256]]

data_radio = 0.6, 0.2, 0.2

df_train, df_val, df_test, df_mean, df_std = data_gen(args.file_path, data_radio, args.n_route, args.n_his, args.n_pre, day_slot=288)

inputs_placeholder = tf.placeholder(tf.float32, [None, args.n_his + args.n_pre, args.n_route, 1], name='data_input')
# print()
model = GSTGCN(inputs=inputs_placeholder, blocks=blocks, args=args, len_train=len(df_train), std=df_std, mean=df_mean)

model.train(df_train, df_val)
model.test(df_test, './output/')


