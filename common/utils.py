import re
from pathlib import Path
from typing import List
from dataclasses import dataclass
import numpy as np
import sys

def count_utterance_by_speaker(cha_file: str) -> dict:
    """
    실제 발화한 화자만 반환 (0 발화 제외)
    """
    content = Path(cha_file).read_text(errors='ignore')
    
    # 실제 *SPEAKER 발화만
    speaker_utts = re.findall(r'\*([A-Z][A-Za-z ]+?):\s*(.*?)(?=\n\*[A-Z][A-Za-z ]+?:|\n%|\n@|\Z)', 
                             content, re.DOTALL | re.I)
    
    active_speakers = {}
    for speaker, text in speaker_utts:
        # 클리닝 후 길이 체크
        text = re.sub(r'\d+_\d+|\[\w+\]|\b\d+\b|\bxxx\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) > 1:
            if speaker not in active_speakers:
                active_speakers[speaker] = 0
            active_speakers[speaker] += 1
    
    # 발화수 순 정렬
    sorted_speakers = dict(sorted(active_speakers.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_speakers

# # 테스트
# speakers = count_utterance_by_speaker("ENNI/SLI/A/413.cha")
# # print(f"👥 화자 {list(speakers.keys())}")
# print("📊 발화 분포:", speakers)


@dataclass
class Utterance:
    order: int
    speaker: str
    text: str
    clean_text: str

def clean(text: str) -> str:
    """타임스탬프 + 모든 특수기호 제거"""
    
    # 1. 타임스탬프 123_456
    text = re.sub(r'\d+_\d+', '', text)
    
    # 2. 나머지 특수기호
    text = re.sub(r'\[[^\]]*\]|\b(?:xxx|www|0)\b|[/]|[/=]|&=', '', text)
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_utterances(cha_file: str, speakers: List[str]) -> List[Utterance]:
    """
    지정 화자들의 발화만 순서대로 추출
    
    Args:
        cha_file: '413.cha'
        speakers: ['CHI', 'MOT'] 또는 ['CHI', 'EXA']
    
    Returns:
        List[Utterance]: 순서 유지된 발화 리스트
    """
    content = Path(cha_file).read_text(errors='ignore')
    
    # 지정 화자 패턴 (대소문자 무시)
    speaker_pattern = '|'.join([re.escape(s) for s in speakers])
    pattern = rf'\*({speaker_pattern}):\s*(.*?)(?=\n\*[A-Z][A-Za-z ]+?:|\n%|\n@|\Z)'
    
    matches = list(re.finditer(pattern, content, re.DOTALL | re.I))
    
    utterances = []
    for i, match in enumerate(matches, 1):
        speaker = match.group(1).strip()
        raw_text = match.group(2).strip()
        
        # 클리닝
        clean_text = clean(raw_text)
        
        if len(clean_text) > 1:  # 의미있는 발화만
            utterances.append(Utterance(
                order=i,
                speaker=speaker,
                text=raw_text,
                clean_text=clean_text
            ))
    
    return utterances


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

def get_batch(corpus, label, time_size, batch_size, is_random=True, count=0):
    # 아동의 수에서 batch_size만큼 수를 뽑음
    if is_random:
        nums = np.random.choice(len(corpus), size=batch_size, replace=False)
    else:
        nums = np.arange(count*batch_size, (count+1)*batch_size)
    xs_batch = corpus[nums]

    label_batch = label[nums]
    for i, j in enumerate(xs_batch):
        xs_batch[i] = j[:-2]    # 마지막은 label
        

    def pad_to(word_list, size):
        length = len(word_list)
        if length >= size:
            return word_list[:size]
        else:
            return word_list + [-1] * (size - length)
    
    xs_batch = np.array([pad_to(list(x), time_size) for x in xs_batch], dtype=int)

    return xs_batch, label_batch



def eval_perplexity(model, corpus, label, batch_size=10, time_size=35):
    print('퍼플렉서티 평가 중 ...')
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush()

    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl


def eval_loss(model, corpus, label, batch_size, time_size):
    print('loss, 정확도 평가 중 ...')
    corpus_size = len(corpus)
    total_loss, total_acc = 0.0, 0.0
    max_iters = max(1, corpus_size // batch_size)

    for iters in range(max_iters):
        xs_dev, label_dev = get_batch(corpus, label, time_size, batch_size, is_random=False, count=iters) 
        try:
            loss, acc = model.forward(xs_dev, label_dev, train_flg=False)
        except TypeError:
            loss = model.forward(xs_dev, label_dev)
        total_loss += loss
        total_acc += acc


    print('')
    avg_loss = total_loss / max_iters
    avg_acc = total_acc / max_iters
    return avg_loss, avg_acc