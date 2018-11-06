#encoding: utf-8

'''
 Back-Propagation Neural Networks
      Baseado no trabalho de
 Neil Schemenauer <nas@arctrix.com>
'''

# Bibliotecas utilizadas
import math
import random
import string

# Usado para inicialização de números aleatórios (não utilizado)
#random.seed(0)

# Gera um número randômico calculado entre a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Executa uma matriz (pode-se usar a biblioteca NumPy para otimizar a velocidade)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# Função logística utilizada (ou tangente hiperbólica comentada) e sua correpondente derivativa 
def sigmoid(x):
    #return math.tanh(x)
    return 1.0/(1.0+math.exp(-x))

def dsigmoid(y):
    #return 1.0 - y**2
    return y*(1.0-y)

class NN:
    def __init__(self, ni, nh, no):
        # Número de nós na entrada, camada oculta e saída 
        self.ni = ni + 1
        self.nh = nh
        self.no = no 

        # Nós após ativação
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # Geração dos pesos
        self.wi =  makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        self.wo_bias = makeMatrix(1, 1)

        # Pesos atribuídos na entrada
        self.wi[0][0] = 0.4  # Nó #1
        self.wi[0][1] = 0.8
        self.wi[1][0] = 0.5  # Nó #2
        self.wi[1][1] = 0.8
        self.wi[2][0] = -0.6 # Bias
        self.wi[2][1] = -0.2
        print('Pesos de entrada: {}'.format(self.wi))

        self.wo[0][0] = -0.4
        self.wo[1][0] = 0.9
        self.wo_bias[0] = -0.3
        print('Pesos de saída: {}, {}'.format(self.wo,self.wo_bias))

        # Guarda valores dos últimos pesos para uso do termo momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)
        self.co_bias = 0.0

    def update(self, inputs):
        if len(inputs) != self.ni-1: 
            raise ValueError('Valores incorretos de entrada')

        # Para ativação dos nós de entrada
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]
        #raw_input('Entrada: {}'.format(self.ai))

        # Para ativação dos nós na camada oculta
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)
            self.ah_bias = sigmoid(self.wo_bias[0])
        #raw_input('Ativação da camada oculta: {}'.format(self.ah))

        # Para ativação dos nós na camada de saída
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            sum = sum + (self.wo_bias[0]) # Bias
            self.ao[k] = sigmoid(sum)
        #raw_input('Ativação da camada de saída: {}'.format(self.ao))
        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('Valores incorretos de alvo')

        # Cálculo dos erros na camada de saída
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error
       
        # Cálculo dos erros na camada intermediária
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # Pesos de saída são atualizados
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                change_bias = output_deltas[k]*self.ah_bias
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                self.wo_bias[0] = self.wo_bias[0] + N*change_bias + M*self.co_bias
                self.co_bias = change_bias
        #raw_input('Pesos de saída: {}, {}'.format(self.wo, self.wo_bias))

        # Pesos de entrada são atualizados
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change
        #raw_input('Pesos de entrada: {}'.format(self.wi))

        #self.weights()

        # Erro é calculado
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Pesos de entrada:')
        for i in range(self.ni):
            raw_input(self.wi[i])
        print()
        print('Pesos de saída:')
        for j in range(self.nh):
            raw_input(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.):
        # N: taxa de aprendizado
        # M: termo momentum (desligado)
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)

    # Imprime os erros a cada 100 iterações
            if i % 100 == 0:
                print('Erro %-.5f' % error)


'''
Rotina executada
'''
def demo():
    # Treinamento da rede para função XOR (ou-exclusivo)
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]
    # Cria uma rede com dois nós de entrada, dois nós intermediários e um de saída
    n = NN(2, 2, 1)

    # Treinamento com os dados fornecidos acima
    n.train(pat)

    # Teste disso
    n.test(pat)



if __name__ == '__main__':
    demo()

