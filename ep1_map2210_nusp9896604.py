"""

==============================================================
EP1 - MAP2210
1o. Semestre -2020 

Author : Pedro Blöss Braga

==============================================================


### OBS 1: o arquivo requirements.txt contém os nomes das bibliotecas utilizadas e suas versões, para que possa instalar por pip install -r requirements.txt 
### OBS 2: Pode alterar o diretório (path) em "plt.savefig()" para guardar os arquivos no diretório que queira. 

## Parte1
- Eliminação Gaussiana sem pivotamento
- Eliminação Gaussiana com pivotamento parcial

Utilizar matriz de Hilbert H_{ij} = 1/ (i+j - 1)
ordens crescentes n = 2,4,8,16, ...

Solucionar o sistema Hx = b, com cada b_{i} = \sum_i H_{ij}

i) Resolver sistema com os algoritmos, calcular norma2 da diferença entre solução obtida e solução exata
ii) comparar a norma erro com o determinante det(H)
ii) O que se pode concluir?


## Parte 2
i) implementar Decomposição de Cholesky e substituições para resolver o sistema M x = b, 
com M uma matriz pseudo-aleatória e novamente b o vetor formado pela soma das componentes de cada linha de M,
e calcular norma2
ii) resolver com Cholesky, eliminação gaussiana sem pivotamento e linalgsolve para comparar
iii) Comparar precisão e tempo e o determinante
iv) o que se pode concluir?

"""

## bibliotecas utilizadas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import time
import os

import scipy.linalg as LA
import math
from random import randint
import statistics as stat

##################################
# Parte 1
##################################



#########################################
##### Funções auxiliares
#########################################

def gera_matriz_Hilbert(n):
    """ 
    Gerador de Matriz de Hilbert := H_ij = 1/((i+1)+(j+1)-1) 
    params: 
        n : Tamanho da matriz quadrada
    """
    H=[]
    for j in range(n):
        H.append([])
        for i in range(n):
            H[j].append(1/((i+1)+(j+1)-1))
    return(np.array(H))

##################################################################################

def gera_b(M):
    b = []
    for i in range(len(M)):
        b.append(sum(M[i]))
    return np.array(b)

##################################################################################

def norma2(x, x_real):
    norm2=0
    for i in range(len(x)):
        norm2+=(x_real[i] - x[i])**2
    return np.sqrt(norm2)

##################################################################################

def laplace(M, verbose=False):
    """
    det(M) = \sum_{k=1}^{n} a_{ik}C_{ik} onde C_{ij}=(-1)^{i+j}M_{ij} é o cofator
    """
    n = np.shape(M)[0]
    if (n==1):
        det = M
    elif (n==2):
        det = M[0,0]*M[1,1] - M[0,1]*M[1,0]

    else:
        det =0
    for i in range(n):
        novaMatriz = M[1:, :] # matriz que não contém a linha e a coluna dos elementos escolhidos
        novaMatriz = np.delete(novaMatriz, i, axis=1)
        det = det+math.pow(-1,1+i+1)*M[0,i]*(laplace(novaMatriz))
    
    return det
  
##################################################################################

def LU(x, verbose=False):
    dim = np.shape(x)
    det = 1 # elemento neutro da multiplicação

    lower = np.identity(dim[0])
    for j in range(0, dim[1]):
        for i in range(j, dim[0]): # eliminação de Gauss
            if (i == j and x[i,j]==0 and i!= dim[1]-1): #permutação de linhas para aque pivot !=0
                p=-2 # contador
                k=i

                while p!=k and k <dim[1]-1: # trocar linha pivot por outra com elemento nao nula em coluna pivot
                    if x[k,j] !=0:
                        p=k
                    else:
                        k+=1
                if p ==-2: #caso nãão tenha (coluan inteira nula), i.e. matriz singular
                    raise ValueError('Det=0 (Matriz singular)')
                
                else:
                    x[i,],x[p,]=x[p,].copy(),x[i,].copy() # permutação de linhas
                    det=(-1)*det # trocar sinal do determinante (pela propriedade)
            elif (i!=j): # com o pivô não-nulo:
                xt0=x[j,j] 
                xt1=x[i,j]
                ft=xt1/xt0
                x[i,:]=x[i,:]-x[j,:]*ft # operação-linha número 3
                lower[i,j]=ft # o fator que multiplica a outra linha é atribuída à matriz L - não se troca o sinal pois a operação já é de substração
            det=det*x[j,j] # após zerar todos os elementos abaixo do elemento pivô na respectiva coluna, agrupar o elemento da diagonal principal ao determinante - ao final de todas as colunas, o número resultante será o produtório de todos os elementos dessa diagonal, e portanto o determinante da matriz original (por construção, a matriz identidade de L será sempre 1)
    if verbose == True:
        print("U=",x, '\n', ' ', '\n',
        "L=",lower,'\n',' ', '\n',
        "LU = ",np.dot(lower,x), "= X",'\n',' ', '\n',
        "Determinante = ",det
        )
    return det 

##################################################################################




#########################################
##### Funções de solução
#########################################

##################################################################################
def soluciona_sem_pivotamento(M, b, verbose=False):
    """ 
    M : matriz nxn formada pelos coeficientes do sistema de equações
    b : vetor nx1 de valores associados a cada equação
    """
    print(20*'-', '\n', 'Sem pivotamento', '\n', 20*'-')
    # tamanho das linhas e colunas da matriz quadrada M
    n = len(M)
    if b.size != n:
        raise ValueError(f"Inválido: b deve ser n x 1 e M deve ser n x n. Recebido: b {b.size}x1 e M {n}x{n}")

    for linha_pivot in range(n-1):
        for linha in range(linha_pivot+1, n):
            # coeficiente m = razao entre primeiros valores das linhas escolhidas
            m = M[linha][linha_pivot]/M[linha_pivot][linha_pivot]

            # rodando as colunas
            for col in range(linha_pivot+1, n):
                M[linha][col] = M[linha][col] - m*M[linha_pivot][col]
            # coluna de solução da equação
            b[linha] = b[linha] - m*b[linha_pivot]

    if verbose == True:
        print(f'M : {M} \n b: {b}')

    x = np.zeros(n)
    k=n-1
    x[k] = b[k]/M[k, k]
    while k >= 0:
        x[k] = (b[k] - np.dot(M[k, k+1:], x[k+1:]))/M[k,k]
        k-=1
    if verbose == True:
        for i in range(n):
            print('\n {} \n x{} = {} \n {}'.format(10*'=', i+1, x[i], 10*'='))
        print(' \n {} \n x = {} \n {}'.format(20*'=', x, 20*'='))
    return x


    ##################################################################################


def soluciona_com_pivotamento_parcial(M, b, verbose=False):
    print(20*'-', '\n', 'Pivotamento parcial', '\n', 20*'-')
    # quantidade de linhas e colunas da matriz quadrada M 
    n = len(M)

    if b.size != n:
        raise ValueError(f"Inválido: b deve ser n x 1 e M deve ser n x n. Recebido: b {b.size}x1 e M {n}x{n}")

    # percorreremos a k-ésima linha de pivot.
    for k in range(n-1):
        # escolha do maior pivot
        maximo = abs(M[k:, k]).argmax() + k

        if M[maximo, k] ==0:
            raise ValueError('Impossível solucionar, pois a Matriz é singular.')
        
        # troca de linhas
        if maximo != k:
            # troca k-ésima linha pela linha de índice = "maximo"
            M[[k, maximo]] = M[[maximo ,k]]
            b[[k, maximo]] = b[[maximo, k]]

        for lin in range(k+1, n):
            # multiplicador da matriz para atualização L_i = L_i + m_ij* L_j
            m = M[lin][k]/M[k][k]

            # a unica nesta coluna ja que o resto é zero
            M[lin][k] = m

            for col in range(k+1, n):
                M[lin][col] = M[lin][col] - m * M[k][col]
            
            #coluna de solução
            b[lin] -= m*b[k]

    if verbose == True:
        print(f' M: {M} \n b: {b} \n')

    # hora de substituir e achar a solução x
    x = np.zeros(n)
    k = n-1
    x[k] = b[k]/M[k, k]
    
    while k >=0:
        x[k] = (b[k] - np.dot(M[k, k+1:], x[k+1:]))/M[k,k]
        k -=1
    
    if verbose == True:
        for i in range(n):
            print('\n {} \n x{} = {} \n {}'.format(10*'=', i+1, x[i], 10*'='))
        print(' \n {} \n x = {} \n {}'.format(20*'=', x, 20*'='))
    return x
    

##################################################################################

def soluciona_linalg(M, b, verbose=False):
    print(20*'-', '\n', 'linalg solve', '\n', 20*'-')
    n = len(M)
    x = LA.solve(M, b)
    if verbose == True:
        for i in range(n):
            print('\n {} \n x{} = {} \n {}'.format(10*'=', i+1, x[i], 10*'='))
        print(' \n {} \n x = {} \n {}'.format(20*'=', x, 20*'='))
    return x

##################################################################################





#########################################
##### Funções de processamento
#########################################

##################################################################################

def multiplos_testes1(K=100):
    tempo_sempivot, tempo_pivotparcial, tempo_linalg = [],[],[]
    norma2_sempivot, norma2_pivotparcial=[],[]
    determinantes_LU, determinantes_Laplace =[],[]
    
    n=2 # tamanho inicial da matriz
    ns=[]
    c=0
    while n<=K:

        print(f' Solucionando matriz de ordem {n}')

        ################################
        # solução Eliminação Gaussiana sem pivotamento
        ################################

        # gera matriz M (de Hilbert) e vetor b
        M = gera_matriz_Hilbert(n)
        b = gera_b(M)
        # Eliminação Gaussiana sem pivotamento 
        t1 = time.time()
        try:
            x_sempivot = soluciona_sem_pivotamento(M, b)
        except Exception as e:
            print(f' \n PARTE1 - Gauss_SEMPIVOT, erro: \n {e} \n ')
            x_sempivot=0
        dt1 = time.time() - t1

        ################################
        # solução Eliminação Gaussiana com pivotamento parcial
        ################################

        # gera matriz M (de Hilbert) e vetor b
        M = gera_matriz_Hilbert(n)
        b = gera_b(M)
        # Eliminação Gaussiana com pivotamento parcial
        t2 = time.time()
        try:
            x_pivotparcial = soluciona_com_pivotamento_parcial(M, b)
        except Exception as e:
            print(f' \n PARTE1 - Gauss_PIVOTPARCIAL, erro: \n {e} \n ')
            x_pivotparcial=0
        dt2 = time.time() - t2

        ################################
        # solução exata -> linalg
        ################################

        # gera matriz M (de Hilbert) e vetor b
        M = gera_matriz_Hilbert(n)
        b = gera_b(M)
        # Solução com scipy.linalg:
        t3 = time.time()
        try:
            x_linalg = soluciona_linalg(M, b)
        except Exception as e:
            print(f' \n PARTE1 - linalg, erro: \n {e} \n ')
            x_linalg=0
        dt3 = time.time() - t3

        # guarda tempos em listas
        tempo_sempivot.append(dt1)
        tempo_pivotparcial.append(dt2)
        tempo_linalg.append(dt3)

        # guarda normas erro e determinantes em listas
        norma2_sempivot.append(norma2(x=x_sempivot,x_real=x_linalg))
        norma2_pivotparcial.append(norma2(x=x_pivotparcial, x_real=x_linalg))
        determinantes_LU.append(LU(gera_matriz_Hilbert(n)))
        #determinantes_Laplace.append(laplace(gera_matriz_Hilbert(n)))

        # guardando os tamanhos de matrizes utilizados
        ns.append(str(n))

        # dobrando o tamanho da matriz
        n*=2

        # contador de iteração 
        print(20*'=', '\n', c*100/(math.floor(math.log2(K))), '%', '\n', 20*'=', '\n', ' ')
        c+=1
    
    return tempo_sempivot, tempo_pivotparcial, tempo_linalg, norma2_sempivot, norma2_pivotparcial, determinantes_LU, determinantes_Laplace, ns

################################################


def parte1(graf_dir, K=100, plota = True):
    ################################
    # Soluções
    ################################

    tempo_sempivot, tempo_pivotparcial, tempo_linalg, norma2_sempivot, norma2_pivotparcial, determinantes_LU, determinantes_Laplace, ns = multiplos_testes1(K=K)
    grade = 20*'='
    print(f'{grade} \n Solucionou! \n {grade}')
    
    if plota == True:
        ################################
        # Gráficos de tempos
        ################################

        EPOCHS = list(range(len(determinantes_LU)))

        # gráfico tempos separados
        plt.figure(figsize=(10,6))

        plt.subplot(3,1,1)
        plt.scatter(EPOCHS, tempo_sempivot, color='red')
        plt.plot(EPOCHS, tempo_sempivot, label='Tempo sem pivotamento')
        plt.legend(loc='best', fontsize=15)
        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Tempo (s)')

        plt.subplot(3,1,2)
        plt.scatter(EPOCHS, tempo_pivotparcial, color='red')
        plt.plot(EPOCHS, tempo_pivotparcial, label='Tempo pivotamento parcial')
        plt.legend(loc='best', fontsize=15)
        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Tempo (s)')

        plt.subplot(3,1,3)
        plt.scatter(EPOCHS, tempo_linalg, color='red')
        plt.plot(EPOCHS, tempo_linalg, label='Tempo linalg')
        plt.legend(loc='best', fontsize=15)
        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Tempo (s)')

        plt.tight_layout()
        plt.savefig(graf_dir+'/pt1_tempos_sub.png')
        #plt.show()


        ## gráfico tempos juntos
        plt.figure(figsize=(10,6))

        plt.scatter(EPOCHS, tempo_sempivot, color='red')
        plt.plot(EPOCHS, tempo_sempivot, color='blue', label='Tempo sem pivotamento')
        plt.legend(loc='best', fontsize=15)

        plt.scatter(EPOCHS, tempo_pivotparcial, color='red')
        plt.plot(EPOCHS, tempo_pivotparcial, color='green', label='Tempo pivotamento parcial')
        plt.legend(loc='best', fontsize=15)

        plt.scatter(EPOCHS, tempo_linalg, color='red')
        plt.plot(EPOCHS, tempo_linalg, color='orange', label='Tempo linalg')
        plt.legend(loc='best', fontsize=15)

        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Tempo (s)')
        plt.savefig(graf_dir+'/pt1_tempos.png')
        #plt.show()


        ################################
        ## Gráficos de erro
        ################################


        # gráfico normas2 e det separados
        plt.figure(figsize=(10,6))

        plt.subplot(3,1,1)
        plt.scatter(EPOCHS, norma2_sempivot, color='red')
        plt.plot(EPOCHS, norma2_sempivot, label='norma2 sem pivotamento')
        plt.legend(loc='best', fontsize=15)
        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Norma2')

        plt.subplot(3,1,2)
        plt.scatter(EPOCHS, norma2_pivotparcial, color='red')
        plt.plot(EPOCHS, norma2_pivotparcial, label='norma2 pivotamento parcial')
        plt.legend(loc='best', fontsize=15)
        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Norma2')

        plt.subplot(3,1,3)
        plt.scatter(EPOCHS, determinantes_LU, color='red')
        plt.plot(EPOCHS, determinantes_LU, label='determinante (LU)')
        plt.legend(loc='best', fontsize=15)
        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Determinante')

        plt.tight_layout()
        plt.savefig(graf_dir+'/pt1_normas_sub.png')
        #plt.show()


        ## gráfico normas2 e det juntos
        plt.figure(figsize=(10,6))

        plt.scatter(EPOCHS, norma2_sempivot, color='red')
        plt.plot(EPOCHS, norma2_sempivot, color='blue', label='Norma2 sem pivotamento')
        plt.legend(loc='best', fontsize=15)

        plt.scatter(EPOCHS, norma2_pivotparcial, color='red')
        plt.plot(EPOCHS, norma2_pivotparcial, color='green', label='Norma2 pivotamento parcial')
        plt.legend(loc='best', fontsize=15)

        plt.scatter(EPOCHS, determinantes_LU, color='red')
        plt.plot(EPOCHS, determinantes_LU, color='orange', label='determinante (LU)')
        plt.legend(loc='best', fontsize=15)

        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.savefig(graf_dir+'/pt1_normas.png')
        #plt.show()
        
        plt.figure(figsize=(10,6))
        vals = [media(norma2_sempivot), media(norma2_pivotparcial)] #, media(determinantes_LU)]
        labels = ['Gauss \n Sem_Pivot', 'Gauss \n Pivot_Parcial'] #, 'determinante \n (LU)']
        plt.title('Média dos erros')
        plt.bar(labels, vals)
        plt.xticks(labels)
        plt.savefig(graf_dir+'/pt1_medias.png')
        #plt.show()


    return tempo_sempivot, tempo_pivotparcial, tempo_linalg, norma2_sempivot, norma2_pivotparcial, determinantes_LU, ns


########################################################


##################################
# Parte 2
##################################



#########################################
##### Funções auxiliares
#########################################

def checa_singularidade(M):
    """
    Uma matriz é singular se:
    - det(M) = 0
    - Ax = 0 para algum vetor x não nulo
    - Ax= b nãão possui solução ou possui infinitas soluções
    """
    return LU(M) == 0 #or laplace(M) == 0

########################################################

def gera_matriz_pseudo_aleatoria2(n):
    return np.random.rand(n,n)

########################################################

def gera_matriz_pseudo_aleatoria(n):
    M = []
    for i in range(n):
        M.append([0]*n)
        for j in range(n):
            M[i][j] = randint(0, n+i+j)
    return np.array(M)

########################################################

def retira_nulos(M):
    for k in range(len(M)):
        for j in range(len(M[k])):
            if M[k][j] == 0.0:
                M[k][j] = 1

########################################################

def retira_nans_lista(l):
    return [x for x in l if str(x) != 'nan']

########################################################

def media(l):
    soma = 0
    for i in range(len(l)):
        if l[i] >=0. or l[i] <0.: # ignorando nans
            soma+=l[i]

    return soma / len(l)

########################################################
           
def gera_matriz_simetrica_posdef(n):
    """
    Retorna a matriz resultante do produto de uma matriz pseudo-aleatória pela sua transposta.
    Esta deve ser simétrica positiva definida (se a original não for singular)
    """
    M = gera_matriz_pseudo_aleatoria2(n)
    try:
        if checa_singularidade(M): # se for singular, tenta achar outra
            gera_matriz_simetrica_posdef(n)
        
        retira_nulos(M)
        
        return np.dot(M, M.T)
    except Exception as e:
        print(f' \n Erro em gegera_matriz_simetrica_posdef(): \n {e} \n ')
        gera_matriz_simetrica_posdef(n)

########################################################


#########################################
##### Funções de solução
#########################################

##################################################################################


def Cholesky(M):
    """
    Decomposição de Cholesky de M, sendo M uma matriz simétrica ,positiva definida.
    A função retorna a variante triangular inferior L (Lower)
    """
    print(20*'-', '\n', 'Cholesky', '\n', 20*'-')
    n = len(M)

    # matriz cheia de zeros
    L  = [[0.] * n for i in range(n)]

    # performando a decomposição
    for i in range(n):
        for k in range(i+1):
            soma = sum(L[i][j] * L[k][j] for j in range(k))

            if (i==k): #elementos da diagonal
                #l_{kk} = \sqrt{ m_{kk} - \sum_{j=1}^{k-1} l_{kj}^{2}}
                L[i][k] = np.sqrt(M[i][i] - soma)
            else:
                L[i][k] = (1. / L[k][k] * (M[i][k] - soma))

    return L

##################################################################################

def substitui_x(M, b, verbose=False):
    # hora de substituir e achar a solução x
    n = len(M)
    M = np.array(M)
    x = np.zeros(n)
    k = n-1

    x[k] = b[k]/M[k, k]
    
    while k >=0:
        x[k] = (b[k] - np.dot(M[k, k+1:], x[k+1:]))/M[k,k]
        k -=1
    
    if verbose == True:
        for i in range(n):
            print('\n {} \n x{} = {} \n {}'.format(10*'=', i+1, x[i], 10*'='))
        print(' \n {} \n x = {} \n {}'.format(20*'=', x, 20*'='))
    return x

##################################################################################


#########################################
##### Funções de processamento
#########################################

##################################################################################


def multiplos_testes2(K=100):
    tempo_cholesky, tempo_Gauss, tempo_linalg = [],[],[]
    norma2_cholesky, norma2_Gauss=[],[]
    determinantes_LU=[]

    n=2 # tamanho inicial de matriz
    ns=[]
    c=0

    while n<=K:

        print(f' Solucionando matriz de ordem {n}')

        ################################
        # Cholesky
        ################################

        # gera matriz M (simetrica positiva definida) e vetor b
        M = gera_matriz_simetrica_posdef(n)
        b = gera_b(M)
        # Decomposição de Cholesky 
        t1 = time.time()
        try:
            L = Cholesky(M)
            # substituição reversa
            x_cholesky = substitui_x(L, b = gera_b(M))
        except Exception as e:
            print(f' \n PARTE2 - Cholesky, erro: \n {e} \n ')
            x_cholesky=0
        dt1 = time.time() - t1

        ################################
        # Eliminação Gaussiana Sem pivotamento
        ################################

        # gera matriz M (simetrica positiva definida) e vetor b
        M = gera_matriz_simetrica_posdef(n)
        b = gera_b(M)
        # Eliminação Gaussiana sem pivotamento 
        t2 = time.time()
        try:
            x_sempivot = soluciona_sem_pivotamento(M, b)
        except Exception as e:
            print(f' \n PARTE2 - Gauss_SEMPIVOT, erro: \n {e} \n ')
            x_sempivot = 0
        dt2 = time.time() - t2

        ################################
        # solução exata -> linalg
        ################################

        # gera matriz M (simetrica positiva definida) e vetor b
        M = gera_matriz_simetrica_posdef(n)
        b = gera_b(M)
        # Solução com scipy.linalg:
        t3 = time.time()
        try:
            x_linalg = soluciona_linalg(M, b)
        except Exception as e:
            print(f' \n PARTE 2 - linalg, erro: \n {e} \n ')
            x_linalg=0
        dt3 = time.time() - t3
                   

        # guarda tempos em listas
        tempo_cholesky.append(dt1)
        tempo_Gauss.append(dt2)
        tempo_linalg.append(dt3)


        # guarda normas e determinantes em listas
        norma2_cholesky.append(norma2(x=x_cholesky, x_real=x_linalg))
        norma2_Gauss.append(norma2(x=x_sempivot, x_real=x_linalg))
        determinantes_LU.append(LU(gera_matriz_Hilbert(n)))
        #determinantes_Laplace.append(laplace(gera_matriz_Hilbert(n)))
                
        # guardando os tamanhos de matrizes utilizados
        ns.append(str(n))

        # dobrando o tamanho da matriz
        n*=2

        # contador de iteração 
        print(20*'=', '\n', c*100/(math.floor(math.log2(K))), '%', '\n', 20*'=', '\n', ' ')
        c+=1


    # poderia adicionar determinantes_Laplace mas por custo computacional deixei para calcular apenas com LU()
    return tempo_cholesky, tempo_Gauss, tempo_linalg, norma2_cholesky, norma2_Gauss, determinantes_LU, ns

##################################################################################

def parte2(graf_dir, K=100, plota=True):
    ################################
    # Soluções
    ################################

    tempo_cholesky, tempo_Gauss, tempo_linalg, norma2_cholesky, norma2_Gauss, determinantes_LU, ns = multiplos_testes2(K=K)
    grade = 20*'='
    print(f'{grade} \n Solucionou! \n {grade}')
    
    if plota == True:
        ################################
        # Gráficos de tempos
        ################################

        EPOCHS = list(range(len(determinantes_LU)))

        # gráfico tempos separados
        plt.figure(figsize=(10,6))

        plt.subplot(3,1,1)
        plt.scatter(EPOCHS, tempo_cholesky, color='red')
        plt.plot(EPOCHS, tempo_cholesky, label='Tempo Cholesky')
        plt.legend(loc='best', fontsize=15)
        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Tempo (s)')

        plt.subplot(3,1,2)
        plt.scatter(EPOCHS, tempo_Gauss, color='red')
        plt.plot(EPOCHS, tempo_Gauss, label='Tempo Gauss_sempivotamento')
        plt.legend(loc='best', fontsize=15)
        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Tempo (s)')

        plt.subplot(3,1,3)
        plt.scatter(EPOCHS, tempo_linalg, color='red')
        plt.plot(EPOCHS, tempo_linalg, label='Tempo linalg')
        plt.legend(loc='best', fontsize=15)
        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Tempo (s)')

        plt.tight_layout()
        plt.savefig(graf_dir+'/pt2_tempos_sub.png')
        #plt.show()
        
    
        
        ## gráfico tempos juntos
        plt.figure(figsize=(10,6))

        plt.scatter(EPOCHS, tempo_cholesky, color='red')
        plt.plot(EPOCHS, tempo_cholesky, color='blue', label='Tempo Cholesky')
        plt.legend(loc='best', fontsize=15)

        plt.scatter(EPOCHS, tempo_Gauss, color='red')
        plt.plot(EPOCHS, tempo_Gauss, color='green', label='Tempo Gauss_sempivotamento')
        plt.legend(loc='best', fontsize=15)

        plt.scatter(EPOCHS, tempo_linalg, color='red')
        plt.plot(EPOCHS, tempo_linalg, color='orange', label='Tempo linalg')
        plt.legend(loc='best', fontsize=15)

        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Tempo (s)')
        plt.savefig(graf_dir+'/pt2_tempos.png')
        #plt.show()
        

        ################################
        ## Gráficos de erro
        ################################

        # gráfico normas2 e det separados
        plt.figure(figsize=(10,6))

        plt.subplot(3,1,1)
        plt.scatter(EPOCHS, norma2_cholesky, color='red')
        plt.plot(EPOCHS, norma2_cholesky, label='norma2 Cholesky')
        plt.legend(loc='best', fontsize=15)
        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Norma2')

        plt.subplot(3,1,2)
        plt.scatter(EPOCHS, norma2_Gauss, color='red')
        plt.plot(EPOCHS, norma2_Gauss, label='norma2 Gauss_sempivotamento')
        plt.legend(loc='best', fontsize=15)
        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Norma2')

        plt.subplot(3,1,3)
        plt.scatter(EPOCHS, determinantes_LU, color='red')
        plt.plot(EPOCHS, determinantes_LU, label='determinante (LU)')
        plt.legend(loc='best', fontsize=15)
        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.ylabel('Determinante')

        plt.tight_layout()
        plt.savefig(graf_dir+'/pt2_normas_sub.png')
        #plt.show()


        ## gráfico normas2 e det juntos
        plt.figure(figsize=(10,6))

        plt.scatter(EPOCHS, norma2_cholesky, color='red')
        plt.plot(EPOCHS, norma2_cholesky, color='blue', label='Norma2 Cholesky')
        plt.legend(loc='best', fontsize=15)

        plt.scatter(EPOCHS, norma2_Gauss, color='red')
        plt.plot(EPOCHS, norma2_Gauss, color='green', label='norma2 Gauss_sempivotamento')
        plt.legend(loc='best', fontsize=15)
        
        plt.scatter(EPOCHS, determinantes_LU, color='red')
        plt.plot(EPOCHS, determinantes_LU, color='orange', label='determinante (LU)')
        plt.legend(loc='best', fontsize=15)

        plt.xticks(EPOCHS, ns)
        plt.xlabel('Tamanho da Matriz')
        plt.savefig(graf_dir+'/pt2_normas.png')
        #plt.show()

        vals = [media(norma2_cholesky), media(norma2_Gauss)] #, media(determinantes_LU)]
        labels= ['Cholesky', 'Gauss \n SemPivot'] #, 'determinante \n (LU)']

        plt.figure(figsize=(10,6))
        plt.title('Média dos erros - Parte 2')
        plt.bar(labels, vals)
        plt.xticks(labels)
        plt.savefig(graf_dir+'/pt2_medias.png')
        #plt.show()

    return tempo_cholesky, tempo_Gauss, tempo_linalg, norma2_cholesky, norma2_Gauss, determinantes_LU, ns

##################################################################################

def gera_data_atual():
    now = datetime.now()
    data = now.strftime("%d%m%Y")
    hora = now.strftime("%H%M%S")
    d1 = str(data+'_'+hora)
    return d1

##########################################################################################################################################################################



#########################################
##### Função Principal
#########################################



def main(K, plota=True, path='/home/pedro/Desktop/MAP2210/'):

    print(__doc__)

    dt_atual = gera_data_atual()
    nome_arq = f'saidas_{dt_atual}'
    final_dir = os.path.join(path, nome_arq)
    os.mkdir(final_dir)

    graf_dir = os.path.join(final_dir, 'graficos')
    os.mkdir(graf_dir)



    grade = 20*'='
    print(f' \n {grade} \n Iniciando PARTE 1 \n {grade}')
    tempo_sempivot, tempo_pivotparcial, tempo_linalg1, norma2_sempivot, norma2_pivotparcial, determinantes_LU1, ns = parte1(K=K, plota=plota, graf_dir=graf_dir)
    print(f' \n {grade} \n Finalizou PARTE 1 \n {grade}')

    print(f' \n {grade} \n Iniciando PARTE 2 \n {grade}')
    tempo_cholesky, tempo_Gauss, tempo_linalg2, norma2_cholesky, norma2_Gauss, determinantes_LU2, ns = parte2(K=K, plota=plota, graf_dir=graf_dir)
    print(f' \n {grade} \n Finalizou PARTE 2 \n {grade}')

    # dataframe com os valores
    tabela1, tabela2 = {},{}


    d1 = {
        'Tamanho da Matrix': ns,
        'Tempos GaussSemPivot':tempo_sempivot,
        'Tempos GaussPivotParcial':tempo_pivotparcial,
        'Tempos LinalgSolve': tempo_linalg1,

        'Norma2 GaussSemPivot':norma2_sempivot,
        'Norma2 GaussPivotParcial':norma2_pivotparcial,
        'Determinante (LU)': determinantes_LU1
    }
    col1 = [
        'Tamanho da Matrix',
        'Tempos GaussSemPivot', 'Tempos GaussPivotParcial', 'Tempos LinalgSolve',
        'Norma2 GaussSemPivot', 'Norma2 GaussPivotParcial','Determinante (LU)'
    ]
    col2 = [
        'Tamanho da Matrix',
        'Tempos Cholesky', 'Tempos GaussSemPivot', 'Tempos LinalgSolve',
        'Norma2 GaussSemPivot', 'Norma2 GaussSemPivot', 'Determinante (LU)'
    ]
    d2 = {
        'Tamanho da Matrix': ns,
        'Tempos Cholesky':tempo_cholesky,
        'Tempos GaussSemPivot':tempo_Gauss,
        'Tempos LinalgSolve': tempo_linalg2,

        'Norma2 GaussSemPivot':norma2_cholesky,
        'Norma2 GaussSemPivot':norma2_Gauss,
        'Determinante (LU)': determinantes_LU2
    }

    tabela1 = pd.DataFrame(d1, columns = col1, 
        index=list(range(len(ns))))

    tabela2 = pd.DataFrame(d2, columns = col2, 
        index=list(range(len(ns))))

    print(' ', '\n', 20*'=', '\n', ' Parte 1', '\n', ' ')
    print(tabela1)
    print(' ', '\n', 20*'=', '\n', ' ')

    print(' ', '\n', 20*'=', '\n', ' Parte 2', '\n', ' ')
    print(tabela2)
    print(' ', '\n', 20*'=', '\n', ' ')


    # dataframes das médias (media mais geral)
    df1, df2 = {},{}


    data1 = {
        'Média Tempos': [media(tempo_sempivot), media(tempo_pivotparcial), media(tempo_linalg1)],
        'Média Erros' :[media(norma2_sempivot), media(norma2_pivotparcial), media(determinantes_LU1)]
    }
    data2 = {
        'Média Tempos': [media(tempo_cholesky), media(tempo_Gauss), media(tempo_linalg2)],
        'Média Erros' : [media(norma2_cholesky), media(norma2_Gauss), media(determinantes_LU2)]
    }
    df1 = pd.DataFrame(data1, columns = ['Média Tempos', 'Média Erros'], 
        index=['Gauss_SemPivot', 'Gauss_PivotParcial', 'Linalg'])
    df2 = pd.DataFrame(data2, columns = ['Média Tempos', 'Média Erros'], 
        index=['Cholesky', 'Gauss_SemPivot', 'Linalg'])

    tabela1.to_csv(final_dir+'/tabela1.csv')
    tabela2.to_csv(final_dir+'/tabela2.csv')
    df1.to_csv(final_dir+'/df1.csv')
    df2.to_csv(final_dir+'/df2.csv')


    print(' ', '\n', 20*'=', '\n', ' Parte 1', '\n', ' ')
    print(df1)

    print(' ', '\n', 20*'=', '\n', ' Parte 2', '\n', ' ')
    print(df2)


    print(' \n (PARTE 1) Com relação à comparação com os determinantes: ',
    '\n A média dos determinantes (LU) é {:.2E} vezes a média das normas2 da Eliminação Gaussiana Sem Pivotamento. \n '.format(media(determinantes_LU1) / media(norma2_sempivot)),
    '\n A média dos determinantes (LU) é {:.2E} vezes a média das normas2 da Eliminação Gaussiana Com Pivotamento Parcial. \n '.format(media(determinantes_LU1) / media(norma2_pivotparcial)),
    )

    print(' \n (PARTE 2) Com relação à comparação com os determinantes: ',
    '\n A média dos determinantes (LU) é {:.2E} vezes a média das normas2 de Cholesky. \n '.format(media(determinantes_LU2) / media(norma2_cholesky)),
    '\n A média dos determinantes (LU) é {:.2E} vezes a média das normas2 da Eliminação Gaussiana Sem Pivotamento. \n '.format(media(determinantes_LU2) / media(norma2_Gauss)),
    )



if __name__ == "__main__":
    ti = time.time()
    K = int( input('\n Qual valor de K utilizar? \n (K é o limite em que paro de gerar matrizes múltiplas de 2 : \n n = 2, 4, 8, 16, ..., K) \n Insira número INTEIRO \n'))
    main(K, plota=True)
    dt = time.time() - ti
    print(f'Tempo total de Processamento: {dt} s (contando com tempo de gráficos abertos).')
