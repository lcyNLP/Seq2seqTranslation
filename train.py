# 从io工具包导入open方法
from io import open
# 用于字符规范化
import unicodedata
import os
# 用于正则表达式
import re
# 用于随机生成数据
import random
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch中预定义的优化方法工具包
from torch import optim
# 导入时间和数学工具包
import time
import math

# 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# 保存的后缀
SETING = "03"
TRAINTYPR = "SGD"

# 是否进行训练模型
TRAIN_RNN = True
TRAIN_ATTN = True

# 是否加载之前模型迭代训练
LOADMODEL_RNN = False
LOADMODEL_ATTN = False

data_path = './data/eng-fra.txt'
log_dir_rnn = "./model/rnn" + SETING + ".pt"
log_dir_attn = "./model/attn" + SETING + ".pt"

# 设置迭代步数
n_iters = 300000
# 设置日志打印间隔
print_every = 10000
# 设置画图间隔
plot_every = 1000
# 设置隐层大小为256 ，也是词嵌入维度
hidden_size = 256
# learning_rate
learning_rate_rnn = 0.1
learning_rate_attn = 0.001

# 设置teacher_focing比率为0.5
teacher_forcing_ratio = 1

# 预测的个数
n_predict = 10

# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1

# 设置组成句子中单词或者标点的最多个数
MAX_LENGTH = 8
# 过滤出符合我们要求的语言对
# 选择带有指定前缀的语言特征数据作为训练数据
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


class Lang:
    def __init__(self, name):
        """初始化函数中参数name代表传入某种语言的名字"""
        # 将name传入类中
        self.name = name
        # 初始化词汇对应自然数值的字典
        self.word2index = {}
        # 初始化自然数值对应词汇的字典, 其中0，1对应的SOS和EOS已经在里面了
        self.index2word = {0: "SOS", 1: "EOS"}
        # 初始化词汇对应的自然数索引，这里从2开始，因为0，1已经被开始和结束标志占用了
        self.n_words = 2

    def addSentence(self, sentence):
        """添加句子函数, 即将句子转化为对应的数值序列, 输入参数sentence是一条句子"""
        # 根据一般国家的语言特性(我们这里研究的语言都是以空格分个单词)
        # 对句子进行分割，得到对应的词汇列表
        for word in sentence.split(' '):
            # 然后调用addWord进行处理
            self.addWord(word)

    def addWord(self, word):
        """添加词汇函数, 即将词汇转化为对应的数值, 输入参数word是一个单词"""
        # 首先判断word是否已经在self.word2index字典的key中
        if word not in self.word2index:
            # 如果不在, 则将这个词加入其中, 并为它对应一个数值，即self.n_words
            self.word2index[word] = self.n_words
            # 同时也将它的反转形式加入到self.index2word中
            self.index2word[self.n_words] = word
            # self.n_words一旦被占用之后，逐次加1, 变成新的self.n_words
            self.n_words += 1


# 将unicode转为Ascii, 我们可以认为是去掉一些语言中的重音标记：Ślusàrski
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """字符串规范化函数, 参数s代表传入的字符串"""
    # 使字符变为小写并去除两侧空白符, z再使用unicodeToAscii去掉重音标记
    s = unicodeToAscii(s.lower().strip())
    # 在.!?前加一个空格
    s = re.sub(r"([.!?])", r" \1", s)
    # 使用正则表达式将字符串中不是大小写字母和正常标点的都替换成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def timeSince(since):
    "获得每次打印的训练耗时, since是训练开始时间"
    # 获得当前时间
    now = time.time()
    # 获得时间差，就是训练耗时
    s = now - since
    # 将秒转化为分钟, 并取整
    m = math.floor(s / 60)
    # 计算剩下不够凑成1分钟的秒数
    s -= m * 60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)


def readLangs(lang1, lang2):
    """读取语言函数, 参数lang1是源语言的名字, 参数lang2是目标语言的名字
       返回对应的class Lang对象, 以及语言对列表"""
    # 从文件中读取语言对并以/n划分存到列表lines中
    lines = open(data_path, encoding='utf-8'). \
        read().strip().split('\n')
    # 对lines列表中的句子进行标准化处理，并以\t进行再次划分, 形成子列表, 也就是语言对
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # 然后分别将语言名字传入Lang类中, 获得对应的语言对象, 返回结果
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def filterPair(p):
    """语言对过滤函数, 参数p代表输入的语言对, 如['she is afraid.', 'elle malade.']"""
    # p[0]代表英语句子，对它进行划分，它的长度应小于最大长度MAX_LENGTH并且要以指定的前缀开头
    # p[1]代表法文句子, 对它进行划分，它的长度应小于最大长度MAX_LENGTH
    return len(p[0].split(' ')) < MAX_LENGTH and \
           p[0].startswith(eng_prefixes) and \
           len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    """对多个语言对列表进行过滤, 参数pairs代表语言对组成的列表, 简称语言对列表"""
    # 函数中直接遍历列表中的每个语言对并调用filterPair即可
    return [pair for pair in pairs if filterPair(pair)]


# 整合数据预处理的函数
def prepareData(lang1, lang2):
    '''

    :param lang1:代表语言的名字，英文
    :param lange2: 代表目标语言的名字，法文
    :return:
    '''
    # 第一步通过readLangs()函数得到两个类对象,并得到字符串类型的语言对的列表
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    # 第二部对字符串类型的列表进行过滤操作
    pairs = filterPairs(pairs)
    # 对过滤后的语言进行遍历操作
    for pair in pairs:
        # 并使用input_lang和output_lang的addSentence方法对其进行数值映射
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        # 返回数值映射后的对象, 和过滤后语言对
    return input_lang, output_lang, pairs


# 声明全局变量获取与处理之后的数据
input_lang, output_lang, pairs = prepareData("eng", "fra")


# 讲语言转化为模型输入需要的张量
def tensorsFromSentence(lang, sentence):
    '''
    :param lang: 代表传入的Lang的实例化对象
    :param sentence: 代表传入的语句
    :return:
    '''
    indexes = [lang.word2index[word] for word in sentence.split(" ")]
    # 注意要在列表的最后添加一个句子结束符号
    indexes.append(EOS_token)
    # 将其封装成torch.tensor类型，并改变形状为 n+2,并改变它的形状为nx1, 以方便后续计算
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    '''
    :param pair: 代表一个语言对(英文，法文)
    :return:
    '''
    # 依次调用具体的处理函数,分别处理源语言和目标语言
    input_tensor = tensorsFromSentence(input_lang, pair[0])
    target_tensor = tensorsFromSentence(output_lang, pair[1])
    # 最后返回它们组成的元组
    return (input_tensor, target_tensor)


# 取pairs的第一条
pair = pairs[0]
pair_tensor = tensorsFromPair(pair)


# 构建编码器类
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        '''
        :param input_size: 代表编码器输入尺寸,就是英文词表大小
        :param hidden_size: 代表gru的隐层神经元数,同事也是词嵌入的维度
        '''
        super(EncoderRNN, self).__init__()
        # 将参数hidden_size传入类中
        self.hidden_size = hidden_size
        # 实例化nn中预定义的Embedding层, 它的参数分别是input_size, hidden_size
        # 这里的词嵌入维度即hidden_size
        # nn.Embedding的演示在该代码下方
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 然后实例化nn中预定义的GRU层, 它的参数是hidden_size
        # nn.GRU的演示在该代码下方
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input1, hidden):
        '''
        :param input1:代表源语言中的输入张量
        :param hidden: 代表初始化隐藏层张量
        :return:
        '''
        # 注意：经过EMbedding处理之后，张量是一个二维张量,但是GRU要求的输入是三维张量
        # 所以要对结果进行扩充维度view(),同时让任意词嵌入映射后的尺寸是[1, embedding]
        output = self.embedding(input1).view(1, 1, -1)
        # 然后将embedding层的输出和传入的初始hidden作为gru的输入传入其中,
        # 获得最终gru的输出output和对应的隐层张量hidden， 并返回结果
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        '''将隐藏层张量初始化1*1*self.hidden_size'''
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 构建解码器类
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        '''
        :param hidden_size:代表隐藏层神经元个数,同时也是解码器的输入尺寸
        :param output_size: 代表解码器的输出尺寸,指定的尺寸也是目标语言的单词总数
        '''
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 实例化Embedding对象，输入参数分别是目标语言的单词总数，和词嵌入维度
        self.emmbedding = nn.Embedding(output_size, hidden_size)
        # 实例化GRU
        self.gru = nn.GRU(hidden_size, hidden_size)
        # 实例化线性层的对象,对GRU的输出做线性变换，得到希望的输出尺寸output_size
        self.out = nn.Linear(hidden_size, output_size)
        # 最后进入softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input1, hidden):
        '''
        :param input1: 代表目标语言的输入张量
        :param hidden: 代表初始化GRU隐藏张量
        :return:
        '''
        # 经历了Embedding层处理后，要将张量形状改变为三维张量
        output = self.emmbedding(input1).view(1, 1, -1)
        # 使用relu函数进行处理，似的embedding矩阵更稀疏，防止过拟合
        output = F.relu(output)
        # 将张量传入GRU解码器中
        output, hidden = self.gru(output, hidden)
        # 经历了GRU处理后的张量是三维张量，但是全连接层需要二维张量，利用output[0]来降维
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        """初始化隐层张量函数"""
        # 将隐层张量初始化成为1x1xself.hidden_size大小的0张量
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 构建局域GRU和Attention的解码器类
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        """初始化函数中的参数有4个, hidden_size代表解码器中GRU的输入尺寸，也是它的隐层节点数
           output_size代表整个解码器的输出尺寸, 也是我们希望得到的指定尺寸即目标语言的词表大小
           dropout_p代表我们使用dropout层时的置零比率，默认0.1, max_length代表句子的最大长度"""
        super(AttnDecoderRNN, self).__init__()
        # 将以下参数传入类中
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 实例化一个Embedding层, 输入参数是self.output_size和self.hidden_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 根据attention的QKV理论，attention的输入参数为三个Q，K，V，
        # 第一步，使用Q与K进行attention权值计算得到权重矩阵, 再与V做矩阵乘法, 得到V的注意力表示结果.
        # 这里常见的计算方式有三种:
        # 1，将Q，K进行纵轴拼接, 做一次线性变化, 再使用softmax处理获得结果最后与V做张量乘法
        # 2，将Q，K进行纵轴拼接, 做一次线性变化后再使用tanh函数激活, 然后再进行内部求和, 最后使用softmax处理获得结果再与V做张量乘法
        # 3，将Q与K的转置做点积运算, 然后除以一个缩放系数, 再使用softmax处理获得结果最后与V做张量乘法

        # 说明：当注意力权重矩阵和V都是三维张量且第一维代表为batch条数时, 则做bmm运算.

        # 第二步, 根据第一步采用的计算方法, 如果是拼接方法，则需要将Q与第二步的计算结果再进行拼接,
        # 如果是转置点积, 一般是自注意力, Q与V相同, 则不需要进行与Q的拼接.因此第二步的计算方式与第一步采用的全值计算方法有关.
        # 第三步，最后为了使整个attention结构按照指定尺寸输出, 使用线性层作用在第二步的结果上做一个线性变换. 得到最终对Q的注意力表示.

        # 我们这里使用的是第一步中的第一种计算方式, 因此需要一个线性变换的矩阵, 实例化nn.Linear
        # 因为它的输入是Q，K的拼接, 所以输入的第一个参数是self.hidden_size * 2，第二个参数是self.max_length
        # 这里的Q是解码器的Embedding层的输出, K是解码器GRU的隐层输出，因为首次隐层还没有任何输出，会使用编码器的隐层输出
        # 而这里的V是编码器层的输出
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # 接着我们实例化另外一个线性层, 它是attention理论中的第四步的线性层，用于规范输出尺寸
        # 这里它的输入来自第三步的结果, 因为第三步的结果是将Q与第二步的结果进行拼接, 因此输入维度是self.hidden_size * 2
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # 接着实例化一个nn.Dropout层，并传入self.dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        # 之后实例化nn.GRU, 它的输入和隐层尺寸都是self.hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # 最后实例化gru后面的线性层，也就是我们的解码器输出层.
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input1, hidden, encoder_outputs):
        '''
        :param input1: 输入张量
        :param hidden: 初始的隐层张量
        :param encoder_outputs: 解码器的输出张量
        :return:
        '''

        # 根据结构计算图, 输入张量进行Embedding层并扩展维度
        embedded = self.embedding(input1).view(1, 1, -1)
        # 使用dropout进行随机丢弃，防止过拟合
        embedded = self.dropout(embedded)

        # 进行attention的权重计算, 哦我们呢使用第一种计算方式：
        # 将Q，K进行纵轴拼接, 做一次线性变化, 最后使用softmax处理获得结果
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        # 然后进行第一步的后半部分, 将得到的权重矩阵与V做矩阵乘法计算, 当二者都是三维张量且第一维代表为batch条数时, 则做bmm运算
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # 之后进行第二步, 通过取[0]是用来降维, 根据第一步采用的计算方法, 需要将Q与第一步的计算结果再进行拼接
        output = torch.cat((embedded[0], attn_applied[0]), 1)

        # 最后是第三步, 使用线性层作用在第三步的结果上做一个线性变换并扩展维度，得到输出
        output = self.attn_combine(output).unsqueeze(0)

        # attention结构的结果使用relu激活
        output = F.relu(output)

        # 将激活后的结果作为gru的输入和hidden一起传入其中
        output, hidden = self.gru(output, hidden)

        # 最后将结果降维并使用softmax处理得到最终的结果
        output = F.log_softmax(self.out(output[0]), dim=1)
        # 返回解码器结果，最后的隐层张量以及注意力权重张量
        return output, hidden, attn_weights

    def initHidden(self):
        """初始化隐层张量函数"""
        # 将隐层张量初始化成为1x1xself.hidden_size大小的0张量
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train_attn(input_tensor, target_tensor, encoder, decoder,
               encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    '''
    :param input_tensor:代表原语言的输入张量
    :param target_tensor:代表目标语言的输入张量
    :param encoder:编码器的实例化对象
    :param decoder:解码器的实例化对象
    :param encoder_optimizer:编码器的优化器
    :param decoder_optimizer:解码器的优化器
    :param criterion:损失函数
    :param max_length:代表句子的最大长度
    :return:
    '''
    # 初始化编码器隐藏张量
    encoder_hidden = encoder.initHidden()

    # 训练前将编码器和解码器的优化器梯度归零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 根据源文本和目标文本张量获取对应的长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # 初始化编码器输出张量，形状是max_lengthxencoder.hidden_size的0张量
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # 设置损失初始值
    loss = 0

    # 遍历输入张量
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # 每一轮次的输出encoder_output是三维张量,使用[0,0]进行降维到以为列表,赋值给输出
        encoder_outputs[ei] = encoder_output[0, 0]

    # 初始化解码器的第一个输入字符
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # 初始化解码器隐藏层张量，复制给最后一次编码器的隐藏张量
    decoder_hidden = encoder_hidden

    # 判断时候使用teacher_focing
    use_teacher_focing = True if random.random() < teacher_forcing_ratio else False

    # 如果使用teacher_focing
    if use_teacher_focing:
        # 遍历目标张量
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 使用损失函数计算损失值，冰进行累加
            loss += criterion(decoder_output, target_tensor[di])
            # 因为使用了teacher_focing, 所以将下一步的解码器输入设定为“正确答案”
            decoder_input = target_tensor[di]

    # 如果不使用teacher_focing
    else:
        # 遍历目标张量
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 预测值变为输出张量中概率最大的那一个
            topv, topi = decoder_output.topk(1)
            # 使用损失函数计算损失值, 并进行累加
            loss += criterion(decoder_output, target_tensor[di])
            # 如果某一步的解码器结果是句子的终止符号,则解码直接结果，跳出循环
            if topi.squeeze().item() == EOS_token:
                break
            # 下一步解码器的输入要设定为当前步最大概率值的哪一个
            decoder_input = topi.squeeze().detach()
    # 应用反向传播计算梯度
    loss.backward()
    # 利用编码器和解码器优化器进行参数更新
    encoder_optimizer.step()
    decoder_optimizer.step()

    # 最后返回平均损失
    return loss.item() / target_length


def train_rnn(input_tensor, target_tensor, encoder, decoder,
              encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    '''
    :param input_tensor:代表原语言的输入张量
    :param target_tensor:代表目标语言的输入张量
    :param encoder:编码器的实例化对象
    :param decoder:解码器的实例化对象
    :param encoder_optimizer:编码器的优化器
    :param decoder_optimizer:解码器的优化器
    :param criterion:损失函数
    :param max_length:代表句子的最大长度
    :return:
    '''
    # 初始化编码器隐藏张量
    encoder_hidden = encoder.initHidden()

    # 训练前将编码器和解码器的优化器梯度归零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 根据源文本和目标文本张量获取对应的长度
    target_length = target_tensor.size(0)

    # 设置损失初始值
    loss = 0

    # 初始化解码器的第一个输入字符
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # 初始化解码器隐藏层张量，复制给最后一次编码器的隐藏张量
    decoder_hidden = encoder_hidden
    # 判断时候使用teacher_focing
    use_teacher_focing = True if random.random() < teacher_forcing_ratio else False

    # 如果使用teacher_focing
    if use_teacher_focing:
        # 遍历目标张量
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            # 使用损失函数计算损失值，冰进行累加
            loss += criterion(decoder_output, target_tensor[di])
            # 因为使用了teacher_focing, 所以将下一步的解码器输入设定为“正确答案”
            decoder_input = target_tensor[di]

    # 如果不使用teacher_focing
    else:
        # 遍历目标张量
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            # 预测值变为输出张量中概率最大的那一个
            topv, topi = decoder_output.topk(1)
            # 使用损失函数计算损失值, 并进行累加
            loss += criterion(decoder_output, target_tensor[di])
            # 如果某一步的解码器结果是句子的终止符号,则解码直接结果，跳出循环
            if topi.squeeze().item() == EOS_token:
                break
            # 下一步解码器的输入要设定为当前步最大概率值的哪一个
            decoder_input = topi.squeeze().detach()
    # 应用反向传播计算梯度
    loss.backward()
    # 利用编码器和解码器优化器进行参数更新
    encoder_optimizer.step()
    decoder_optimizer.step()

    # 最后返回平均损失
    return loss.item() / target_length


# 导入plt以便绘制损失曲线
import matplotlib.pyplot as plt


def trainIters(encoder, decoder, n_iters, type_train, start_epoch, print_every=1000, plot_every=100,
               learning_rate=0.01):
    '''
    训练迭代函数
    :param encoder:代表编码器的实例化对象
    :param decoder:代表解码器的实例化对象
    :param n_iters:训练的总迭代步数
    :param print_every:每隔多少轮次进行一次训练日志的打印
    :param plot_every:每隔多少轮次进行一次损失值的添加，为了后续绘制损失曲线
    :param learning_rate:学利率
    :return:
    '''
    # 获取训练开始的时间
    start = time.time()
    # 初始化存放平均损失的值的列表
    plot_losses = []
    # 每隔打印时间间隔的总损失值
    print_loss_total = 0
    # 每个绘制曲线损失值的列表
    plot_loss_total = 0

    # 定义编码器和解码器的优化器
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.9)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.9)

    # 定义损失函数
    criterion = nn.NLLLoss()

    # 按照设定的总迭代次数进行迭代训练
    for iter in range(1, n_iters + 1):
        # 每次从语言对的列表中随机抽取一条样本作为本轮迭代的训练数据
        training_pair = tensorsFromPair(random.choice(pairs))
        # 一次讲选取出来的语句对作为输入张量，和输出张量
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        if type_train == "attn":
            # 通过train函数获得模型运行的损失
            loss = train_attn(input_tensor, target_tensor, encoder,
                              decoder, encoder_optimizer, decoder_optimizer, criterion)
        elif type_train == "rnn":
            # 通过train函数获得模型运行的损失
            loss = train_rnn(input_tensor, target_tensor, encoder,
                             decoder, encoder_optimizer, decoder_optimizer, criterion)
        # 将损失进行累和
        print_loss_total += loss
        plot_loss_total += loss

        # 当迭代步达到日志打印间隔时
        if iter % print_every == 0:
            # 通过总损失除以间隔得到平均损失
            print_loss_avg = print_loss_total / print_every
            # 将总损失归0
            print_loss_total = 0
            # 打印日志，日志内容分别是：训练耗时，当前迭代步，当前进度百分比，当前平均损失
            print('%s %s (%d %d%%) %.4f' % (type_train, timeSince(start),
                                            iter + start_epoch,
                                            (iter + start_epoch) / (n_iters + start_epoch) * 100,
                                            print_loss_avg))

        # 当迭代步达到损失绘制间隔时
        if iter % plot_every == 0:
            # 通过总损失除以间隔得到平均损失
            plot_loss_avg = plot_loss_total / plot_every
            # 将平均损失装进plot_losses列表
            plot_losses.append(plot_loss_avg)
            # 总损失归0
            plot_loss_total = 0

        # 绘制损失曲线
    plt.figure()
    plt.plot(plot_losses)
    # 保存到指定路径
    plt.savefig("./plt/s2s_loss_" + type_train + "_" + SETING + ".png")


# 评估函数
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    '''
    评估函数
    :param encoder: 代表编码器对象
    :param decoder: 代表解码器对象
    :param sentence: 待评估的源语句
    :param max_length: 句子的最大长度
    :return:
    '''
    # 注意：整个评估过程梯度不进行改变
    with torch.no_grad():
        # 对输入语句进行张量表示
        input_tensor = tensorsFromSentence(input_lang, sentence)
        # 获得输入的句子长度
        input_length = input_tensor.size(0)
        # 初始化编码器的隐藏层张量
        encoder_hidden = encoder.initHidden()

        # 初始化输出张量， 矩阵的形状为max_length * hidden_size
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # 遍历输入张量
        for ei in range(input_length):
            # 根据索引从input_tensor取出对应的单词的张量表示，和初始化隐层张量一同传入encoder对象中
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            # 将每次获得的输出encoder_output(三维张量), 使用[0, 0]降两维变成向量依次存入到encoder_outputs
            # 这样encoder_outputs每一行存的都是对应的句子中每个单词通过编码器的输出结果
            encoder_outputs[ei] += encoder_output[0, 0]

            # 初始化解码器的第一个输入，即起始符
            decoder_input = torch.tensor([[SOS_token]], device=device)
            # 初始化解码器的隐层张量即编码器的隐层输出
            decoder_hidden = encoder_hidden

            # 初始化预测的词汇列表
            decoded_words = []
            # 初始化attention张量
            decoder_attentions = torch.zeros(max_length, max_length)
            # 开始循环解码
            for di in range(max_length):
                # 将decoder_input, decoder_hidden, encoder_outputs传入解码器对象
                # 获得decoder_output, decoder_hidden, decoder_attention
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

                # 取所有的attention结果存入初始化的attention张量中
                decoder_attentions[di] = decoder_attention.data
                # 从解码器输出中获得概率最高的值及其索引对象
                topv, topi = decoder_output.data.topk(1)
                # 从索引对象中取出它的值与结束标志值作对比
                if topi.item() == EOS_token:
                    # 如果是结束标志值，则将结束标志装进decoded_words列表，代表翻译结束
                    decoded_words.append('<EOS>')
                    # 循环退出
                    break

                else:
                    # 否则，根据索引找到它在输出语言的index2word字典中对应的单词装进decoded_words
                    decoded_words.append(output_lang.index2word[topi.item()])

                # 最后将本次预测的索引降维并分离赋值给decoder_input，以便下次进行预测
                decoder_input = topi.squeeze().detach()
            # 返回结果decoded_words， 以及完整注意力张量, 把没有用到的部分切掉
            return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=6):
    """随机测试函数, 输入参数encoder, decoder代表编码器和解码器对象，n代表测试数"""
    # 对测试数进行循环
    for i in range(n):
        # 从pairs随机选择语言对
        pair = random.choice(pairs)
        # > 代表输入
        print('>', pair[0])
        # = 代表正确的输出
        print('=', pair[1])
        # 调用evaluate进行预测
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        # 将结果连成句子
        output_sentence = ' '.join(output_words)
        # < 代表模型的输出
        print('<', output_sentence)
        print('')


if __name__ == '__main__':

    # 通过input_lang.n_words获取输入词汇总数，与hidden_size一同传入EncoderRNN类中
    # 得到编码器对象encoder1
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    # 实例化RNN解码器对象
    rnn_decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)
    # 通过output_lang.n_words获取目标词汇总数，与hidden_size和dropout_p一同传入AttnDecoderRNN类中
    # 得到解码器对象attn_decoder1
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    if os.path.exists(log_dir_attn) and LOADMODEL_ATTN and TRAIN_ATTN:
        checkpoint = torch.load(log_dir_attn, map_location=device)
        encoder1.load_state_dict(checkpoint['rnn_encoder1'])
        attn_decoder1.load_state_dict(checkpoint['attn_decoder1'])
        encoder1.train()
        attn_decoder1.train()
        start_epoch = checkpoint['epoch']

        print("-------------------Attn_decoder_"+TRAINTYPR+"---------------------")
        print('attn加载 epoch {} 成功！'.format(start_epoch))
        trainIters(encoder1, attn_decoder1, n_iters, type_train="attn",
                   start_epoch=start_epoch, print_every=print_every,
                   plot_every=plot_every, learning_rate=learning_rate_attn)
        # 模型保存
        torch.save({
            "epoch": n_iters + start_epoch,
            "rnn_encoder1": encoder1.state_dict(),
            "attn_decoder1": attn_decoder1.state_dict()
        }, "./model/attn" + SETING + ".pt")
    elif TRAIN_ATTN:
        start_epoch = 0
        print('无保存attn模型，将从头开始训练！')
        print("-------------------Attn_decoder_"+TRAINTYPR+"---------------------")
        trainIters(encoder1, attn_decoder1, n_iters, type_train="attn",
                   start_epoch=start_epoch, print_every=print_every,
                   plot_every=plot_every, learning_rate=learning_rate_attn)
        # 模型保存
        torch.save({
            "epoch": n_iters + start_epoch,
            "rnn_encoder1": encoder1.state_dict(),
            "attn_decoder1": attn_decoder1.state_dict()
        }, "./model/attn" + SETING + ".pt")


    # 如果有保存的模型，则加载模型，并在其基础上继续训练RNN
    if os.path.exists(log_dir_rnn) and LOADMODEL_RNN and TRAIN_RNN:
        checkpoint = torch.load(log_dir_rnn, map_location=device)
        encoder1.load_state_dict(checkpoint['rnn_encoder1'])
        rnn_decoder1.load_state_dict(checkpoint['rnn_decoder1'])
        encoder1.train()
        rnn_decoder1.train()

        start_epoch = checkpoint['epoch']
        print("-------------------RNN_decoder_" + TRAINTYPR + "----------------------")
        print('rnn加载 epoch {} 成功！'.format(start_epoch))
        # 调用trainIters进行模型训练，将编码器对象encoder1，码器对象attn_decoder1，迭代步数，日志打印间隔传入其中
        trainIters(encoder1, rnn_decoder1, n_iters, type_train="rnn",
                   start_epoch=start_epoch, print_every=print_every,
                   plot_every=plot_every, learning_rate=learning_rate_rnn)
        # 模型保存
        torch.save({
            "epoch": n_iters + start_epoch,
            "rnn_encoder1": encoder1.state_dict(),
            "rnn_decoder1": rnn_decoder1.state_dict()
        }, "./model/rnn" + SETING + ".pt")

    elif TRAIN_RNN:
        start_epoch = 0
        print('无保存rnn模型，将从头开始训练！')
        print("-------------------RNN_decoder_" + TRAINTYPR + "----------------------")
        trainIters(encoder1, rnn_decoder1, n_iters, type_train="rnn",
                   start_epoch=start_epoch, print_every=print_every,
                   plot_every=plot_every, learning_rate=learning_rate_rnn)
        # 模型保存
        torch.save({
            "epoch": n_iters + start_epoch,
            "rnn_encoder1": encoder1.state_dict(),
            "rnn_decoder1": rnn_decoder1.state_dict()
        }, "./model/rnn" + SETING + ".pt")

    print("----------------预测的结果-----------------")
    # 调用evaluateRandomly进行模型测试，将编码器对象encoder1，码器对象attn_decoder1传入其中
    evaluateRandomly(encoder1, attn_decoder1, n=n_predict)

    print("----------------Attention张量制图-----------------")
    # Attention张量制图:
    sentence = "we re both teachers ."
    # 调用评估函数
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, sentence)

    print(output_words)

    # 将attention张量转化成numpy, 使用matshow绘制
    plt.matshow(attentions.numpy())
    # 保存图像
    plt.savefig("./plt/s2s_rnn_" + SETING + ".png")

    plt.matshow(attentions.numpy())
    # 保存图像
    plt.savefig("./plt/s2s_attn_" + SETING + ".png")



