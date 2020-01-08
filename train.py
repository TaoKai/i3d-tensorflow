from i3d import buildMyNet
from data_prepare import MyData
import tensorflow as tf
import numpy as np

def train(epoch=10):
    sess = tf.Session()
    dataflow = MyData('./data', batch_size=8)
    tr_num = dataflow.get_train_num()
    te_num = dataflow.get_test_num()
    pred, inputs, drop_out, loss, labels, opt = buildMyNet(sess)
    best_cost = 0.0
    cost = 0
    saver = tf.train.Saver()
    save_path = './models/best_model.ckpt'
    for i in range(epoch):
        for j in range(tr_num):
            batch_data = dataflow.next_train()
            if batch_data[1].shape[0]<8:
                continue
            cur_cost, _ = sess.run([loss, opt], {inputs:batch_data[0], labels:batch_data[1], drop_out:0.5})
            cost += cur_cost
            print('epoch', i, 'step', j, 'current cost:', cost/(j+1))
        cost = cost/tr_num
        total = 0
        total_pass = 0
        for k in range(te_num):
            batch_data = dataflow.next_test()
            if batch_data[1].shape[0]<8:
                continue
            probs = sess.run(pred, {inputs:batch_data[0], drop_out:1.0})
            pred_labels = np.argmax(probs, axis=1)
            acc_vec = (pred_labels==batch_data[1]).astype(np.int32)
            total += batch_data[1].shape[0]
            total_pass += np.sum(acc_vec)
            print('Evaluating the test set ...', total_pass, '/', total)
        acc = total_pass*1.0/total
        print('Evaluating accuracy is', acc)
        f = open('log.txt', 'a')
        f.write('acc:'+str(acc)+'\n')
        f.close()
        if acc > best_cost:
             best_cost = acc
             saver.save(sess, save_path)
        cost = 0
    print('training finished ... best accuracy', best_cost)


if __name__ == "__main__":
    train(epoch=100)