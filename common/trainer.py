# coding: utf-8
import sys
sys.path.append('..')
import time
import matplotlib.pyplot as plt
import numpy as np
from common.utils import clip_grads, get_batch


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # 뒤섞기
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 기울기 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 공유된 가중치를 하나로 모음
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 손실 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('Iteration (x' + str(self.eval_interval) + ')')
        #plt.xlabel('반복 (x' + str(self.eval_interval) + ')')
        plt.ylabel('Loss')
        #plt.ylabel('손실')
        plt.show()


class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.lost_list = None
        self.eval_interval = None
        self.current_epoch = 0
        self.lost_list = []
        self.acc_list = []

    def fit(self, corpus, label, time_size, hidden_size=64, max_epoch=10, batch_size=16, max_grad=None, eval_interval=20):

        self.time_idx = 0
        # 에폭 동안 측정된 loss, acc를 저장할 리스트
        # 몇 번 반복마다 loss를 출력할지
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0.0  # loss 계산을 위한 누적 손실
        total_acc = 0.0  # loss 계산을 위한 누적 손실
        loss_count = 0    # 손실을 몇번 쌓았는지 카운트
        data_size = len(corpus)
        # batch_size: 한 번에 학습할 아동의 인터뷰 수
        # time_size: RNN이 한 번에 펼쳐서 보는 time step 길이, 한 아동의 인터뷰의 길이 중 최대값으로 함
        # 한 epoch 동안 몇번의 미니배치를 뽑을 수 있는지 계산
        max_iters = max(1, data_size // batch_size)
        # print('max_iters:', max_iters)
        start_time = time.time()

        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_label = get_batch(corpus, label, time_size, batch_size)
                # 기울기를 구해 매개변수 갱신
                loss, acc = model.forward(batch_x, batch_label)
                # print('forward done')
                model.backward()
                # print('backward done')
                params, grads = remove_duplicate(model.params, model.grads)  # 공유된 가중치를 하나로 모음
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                # before = [p.copy() for p in params] 
                # 파라미터 업데이트
                optimizer.update(params, grads)
                # for i in range(len(before)):
                #     diff = np.max(np.abs(params[i] - before[i]))
                #     print(f"param[{i}] max diff:", diff)
                total_loss += loss
                total_acc += acc
                loss_count += 1
                
                # loss 평가
                if (eval_interval is not None) and ((iters+1) % eval_interval) == 0:
                    # ppl = np.exp(total_loss / loss_count)
                    avg_loss = total_loss / loss_count
                    avg_acc = total_acc / loss_count
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | loss %.4f | 정확도 %.4f' 
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss, avg_acc))
                    self.lost_list.append(float(avg_loss))
                    self.acc_list.append(float(avg_acc))
                    total_loss, total_acc, loss_count = 0.0, 0.0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.lost_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.lost_list, label='train')
        plt.xlabel('Iteration (x' + str(self.eval_interval) + ')')
        #plt.xlabel('반복 (x' + str(self.eval_interval) + ')')
        plt.ylabel('perplexity')
        #plt.ylabel('퍼플렉서티')
        plt.show()

class PoolingRnnlmTrainer:
    def __init__(self, model, optimizer, mode):
        self.model = model
        self.optimizer = optimizer
        self.mode = mode
        self.time_idx = None
        self.lost_list = None
        self.eval_interval = None
        self.current_epoch = 0
        self.lost_list = []
        self.acc_list = []

    def fit(self, corpus, label, time_size, hidden_size=64, max_epoch=10, batch_size=16, max_grad=None, eval_interval=20):

        self.time_idx = 0
        # 에폭 동안 측정된 loss, acc를 저장할 리스트
        # 몇 번 반복마다 loss를 출력할지
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0.0  # loss 계산을 위한 누적 손실
        total_acc = 0.0  # loss 계산을 위한 누적 손실
        loss_count = 0    # 손실을 몇번 쌓았는지 카운트
        data_size = len(corpus)
        # batch_size: 한 번에 학습할 아동의 인터뷰 수
        # time_size: RNN이 한 번에 펼쳐서 보는 time step 길이, 한 아동의 인터뷰의 길이 중 최대값으로 함
        # 한 epoch 동안 몇번의 미니배치를 뽑을 수 있는지 계산
        max_iters = max(1, data_size // batch_size)
        # print('max_iters:', max_iters)
        start_time = time.time()

        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_label = get_batch(corpus, label, time_size, batch_size)
                # 기울기를 구해 매개변수 갱신
                loss, acc = model.forward(batch_x, batch_label, self.mode)
                # print('forward done')
                model.backward()
                # print('backward done')
                params, grads = remove_duplicate(model.params, model.grads)  # 공유된 가중치를 하나로 모음
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                # before = [p.copy() for p in params] 
                # 파라미터 업데이트
                optimizer.update(params, grads)
                # for i in range(len(before)):
                #     diff = np.max(np.abs(params[i] - before[i]))
                #     print(f"param[{i}] max diff:", diff)
                total_loss += loss
                total_acc += acc
                loss_count += 1
                
                # loss 평가
                if (eval_interval is not None) and ((iters+1) % eval_interval) == 0:
                    # ppl = np.exp(total_loss / loss_count)
                    avg_loss = total_loss / loss_count
                    avg_acc = total_acc / loss_count
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | loss %.4f | 정확도 %.4f' 
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss, avg_acc))
                    self.lost_list.append(float(avg_loss))
                    self.acc_list.append(float(avg_acc))
                    total_loss, total_acc, loss_count = 0.0, 0.0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.lost_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.lost_list, label='train')
        plt.xlabel('Iteration (x' + str(self.eval_interval) + ')')
        #plt.xlabel('반복 (x' + str(self.eval_interval) + ')')
        plt.ylabel('perplexity')
        #plt.ylabel('퍼플렉서티')
        plt.show()


def remove_duplicate(params, grads):
    '''
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다.
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads