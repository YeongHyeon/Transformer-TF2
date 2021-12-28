import argparse, time, os, operator

import tensorflow as tf
import source.utils as utils
import source.connector as con
import source.tf_process as tfp
import source.datamanager as dman

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    dataset = dman.DataSet()

    agent = con.connect(nn=FLAGS.nn).Agent(\
        dim_s=dataset.dim_h, dim_f=dataset.dim_w, \
        dim_model=FLAGS.d_model, dim_ff=FLAGS.d_ff, depth=FLAGS.depth, num_head=FLAGS.num_head, learning_rate=FLAGS.lr, \
        path_ckpt='Checkpoint')

    time_tr = time.time()
    tfp.training(agent=agent, dataset=dataset, batch_size=FLAGS.batch, epochs=FLAGS.epochs)
    time_te = time.time()
    best_dict, num_model = tfp.test(agent=agent, dataset=dataset)
    time_fin = time.time()

    print("Time (TR): %.5f [sec]" %(tr_time))
    print("Time (TE): %.5f (%.5f [sec/sample])" %(te_time, te_time/num_model/dataset.num_te))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0", help='')

    parser.add_argument('--nn', type=int, default=0, help='')

    parser.add_argument('--d_model', type=int, default=18, help='')
    parser.add_argument('--d_ff', type=int, default=128, help='')
    parser.add_argument('--depth', type=int, default=3, help='')
    parser.add_argument('--num_head', type=int, default=2, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')

    parser.add_argument('--batch', type=int, default=32, help='')
    parser.add_argument('--epochs', type=int, default=10, help='')

    FLAGS, unparsed = parser.parse_known_args()

    main()
