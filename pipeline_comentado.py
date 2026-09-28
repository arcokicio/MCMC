# =============================================================================
# pipeline_comentado.py
# O post inteiro, comentado linha a linha: SINTAXE (Python) e MATEMATICA.
#
# Rode:   py pipeline_comentado.py
# Depois: mude B para 0.3 e rode de novo. Compare as duas saidas.
#
# Legenda dos comentarios:
#   [PY]  = o que a sintaxe Python faz
#   [MAT] = que conta matematica esta acontecendo
#   [POR QUE] = qual o papel disso na historia do post
# =============================================================================

import numpy as np
# [PY] "import X as Y" carrega a biblioteca X e da a ela o apelido Y.
#      Daqui pra frente, tudo que comeca com "np." vem do NumPy.
# [PY] NumPy e a biblioteca de tabelas de numeros (arrays). Um array e uma
#      grade de numeros: 1 linha = vetor; varias linhas = matriz.


# -----------------------------------------------------------------------------
# 0. PARAMETROS DO MUNDO
# -----------------------------------------------------------------------------

N, M = 64, 200
# [PY] Atribuicao multipla: N recebe 64, M recebe 200, na mesma linha.
# [MAT] N = numero de dimensoes. Cada vetor e uma lista de 64 numeros.
#       M = numero de conceitos. Cada conceito sera uma seta nesse espaco.
# [POR QUE] M > N  =>  as setas NAO cabem todas perpendiculares => superposicao.

DANO, SOTAQUE, TOPICO = 0, 1, 2
# [PY] So nomes para os numeros 0, 1, 2. Servem de indice (endereco de linha).
#      Escrever D[DANO] e mais legivel que D[0].
# [POR QUE] Os 3 primeiros conceitos tem papel na historia; os outros 197
#           sao "enchimento" (o resto do que um texto carrega).

B = 0.0   # MUDE: 0.0 = juiz justo | 0.3 = juiz com vies
# [POR QUE] B e o vies que VOCE planta. E a verdade-terreno: voce sabe
#           a resposta certa porque escolheu este numero.

rng = np.random.default_rng(0)
# [PY] Cria um gerador de numeros aleatorios. O 0 e a semente: com a mesma
#      semente, a mesma sequencia de numeros sai sempre (reproduzivel).
#      "rng" = random number generator.


# -----------------------------------------------------------------------------
# 1. CONCEITOS: 200 setas de comprimento 1 em 64 dimensoes
# -----------------------------------------------------------------------------

D = rng.normal(size=(M, N))
# [PY] rng.normal sorteia numeros da curva normal (media 0, desvio 1).
#      size=(M, N) pede uma tabela de M linhas por N colunas: (200, 64).
# [MAT] Cada linha D[i] e um vetor de 64 numeros = uma seta com direcao
#       aleatoria. Sortear de uma normal em todas as coordenadas garante
#       que nenhuma direcao e privilegiada (a seta aponta "para qualquer lado").

D /= np.linalg.norm(D, axis=1, keepdims=True)
# [PY] "a /= b" e o mesmo que "a = a / b".
#      np.linalg.norm(..., axis=1) calcula o comprimento de CADA LINHA.
#        axis=0 seria "descendo pelas colunas"; axis=1 e "andando pela linha".
#      keepdims=True devolve o resultado com forma (200, 1) em vez de (200,),
#        para que a divisao encaixe: cada linha e dividida pelo SEU comprimento.
#        (Isso se chama "broadcasting": o NumPy estica o (200,1) para (200,64).)
# [MAT] Comprimento de um vetor v = raiz(v1^2 + v2^2 + ... + v64^2).
#       Dividir o vetor pelo proprio comprimento => comprimento vira 1.
#       Com comprimento 1, o produto escalar de duas setas = cosseno do angulo.


# -----------------------------------------------------------------------------
# Funcao que fabrica "prompts" (vetores de ativacao)
# -----------------------------------------------------------------------------

def prompts(n, dano, sotaque, topico):
    # [PY] "def nome(argumentos):" cria uma funcao. Tudo indentado abaixo
    #      pertence a ela. Ela so roda quando alguem a chama: prompts(...).
    """Cada prompt = soma das setas dos conceitos que ele tem + 8 de enchimento."""
    # [PY] Texto entre aspas triplas logo apos o def = documentacao da funcao.

    z = np.zeros((n, M))
    # [PY] Tabela de zeros com n linhas (prompts) e M colunas (conceitos).
    # [MAT] z[i, j] = "quanto o prompt i contem do conceito j". Comeca tudo 0.

    z[:, DANO], z[:, SOTAQUE], z[:, TOPICO] = dano, sotaque, topico
    # [PY] z[:, 0] = "todas as linhas (:), coluna 0". Ou seja, a coluna inteira.
    #      Preenche as colunas 0, 1, 2 com os valores recebidos.
    #      Os valores podem ser um numero so (vale para todos os prompts)
    #      ou um array de n numeros (um por prompt).

    for i in range(n):
        # [PY] Laco: repete o bloco abaixo para i = 0, 1, 2, ..., n-1.
        cols = rng.choice(np.arange(3, M), 8, replace=False)
        # [PY] np.arange(3, M) = [3, 4, 5, ..., 199] (os conceitos de enchimento).
        #      rng.choice(lista, 8, replace=False) sorteia 8 deles SEM repetir.
        z[i, cols] = rng.random(8)
        # [PY] Na linha i, nessas 8 colunas, coloca 8 numeros entre 0 e 1.
        # [MAT] Cada prompt ativa 8 conceitos aleatorios com intensidades
        #       aleatorias. Isso e esparsidade: poucos ligados, a maioria zero.

    return z @ D + rng.normal(0, 0.1, (n, N))
    # [PY] "@" e multiplicacao de matrizes. (n, 200) @ (200, 64) = (n, 64).
    #      rng.normal(0, 0.1, (n, N)) = ruido pequeno (desvio 0.1), mesma forma.
    #      "return" devolve o resultado para quem chamou a funcao.
    # [MAT] Para cada prompt i:
    #         vetor_i = z[i,0]*D[0] + z[i,1]*D[1] + ... + z[i,199]*D[199] + ruido
    #       Ou seja: o prompt e a SOMA das setas dos conceitos que ele contem,
    #       cada uma multiplicada pela intensidade. Isso e a superposicao:
    #       200 conceitos misturados dentro de 64 numeros.


# -----------------------------------------------------------------------------
# 2. A REGRA SECRETA DO JUIZ
# -----------------------------------------------------------------------------

n = 4000
dano, sot, top = (rng.random((3, n)) < 0.5).astype(float)
# [PY] rng.random((3, n)) = tabela 3 x 4000 de numeros entre 0 e 1.
#      "< 0.5" transforma cada numero em True/False (metade True, em media).
#      .astype(float) converte True->1.0 e False->0.0.
#      Desempacotamento: as 3 linhas vao para dano, sot, top.
# [MAT] Cada prompt recebe, ao acaso e independente, sim/nao para
#       "e nocivo?", "tem sotaque periferico?", "e do topico X?".

X = prompts(n, dano, sot, top)
# [PY] Chama a funcao. X tem forma (4000, 64): 4000 prompts, 64 numeros cada.

recusa = 1.0*dano + B*sot + 0.55*top + rng.normal(0, 0.3, n)
# [PY] Operacoes elemento a elemento: cada uma das 4000 posicoes e
#      calculada separadamente. O resultado e um array de 4000 numeros.
# [MAT] recusa_i = 1.0*dano_i + B*sotaque_i + 0.55*topico_i + acaso_i
#       E uma combinacao linear: cada fator tem um peso.
# [POR QUE] ESTE E O CORACAO DA VERDADE-TERRENO.
#           Com B = 0, o sotaque tem peso zero: nao muda a recusa. Juiz justo.
#           Com B > 0, o sotaque aumenta a recusa. Juiz com vies.
#           Note que TOPICO tambem pesa (0.55): e o confundidor que vai
#           sujar a direcao e precisar ser retirado no passo 4.


# -----------------------------------------------------------------------------
# 3. DIRECAO DE RECUSA = media(recusou muito) - media(recusou pouco)
# -----------------------------------------------------------------------------

def direcao(X, grupo):
    v = X[grupo].mean(0) - X[~grupo].mean(0)
    # [PY] "grupo" e um array de True/False com 4000 posicoes.
    #      X[grupo] seleciona so as linhas onde grupo e True ("mascara booleana").
    #      ~grupo inverte: True vira False e vice-versa (o grupo oposto).
    #      .mean(0) tira a media descendo pelas linhas (axis 0):
    #        de (k, 64) sai um vetor de 64 numeros, o "prompt medio" do grupo.
    # [MAT] v = media_A - media_B  (diferenca de medias, Arditi et al.)
    #       A parte comum aos dois grupos se cancela na subtracao;
    #       sobra o que os diferencia.
    return v / np.linalg.norm(v)
    # [MAT] Normaliza: comprimento 1. So interessa a DIRECAO, nao o tamanho.

r = direcao(X, recusa > np.median(recusa))
# [PY] np.median = valor do meio. "recusa > mediana" = True para a metade
#      que mais recusou, False para a outra metade.
# [MAT] r e a seta que aponta de "recusou pouco" para "recusou muito".
# [POR QUE] Como dano e topico influenciam a recusa, r carrega pedacos
#           das setas de DANO e TOPICO misturados. Ela esta "suja".


# -----------------------------------------------------------------------------
# 4. RESIDUALIZAR: tirar da seta a sombra sobre DANO e TOPICO
# -----------------------------------------------------------------------------

def residualiza(v):
    Q, _ = np.linalg.qr(D[[DANO, TOPICO]].T)
    # [PY] D[[0, 2]] pega as linhas 0 e 2 de D: forma (2, 64).
    #      .T transpoe: (64, 2), cada seta vira uma coluna.
    #      np.linalg.qr decompoe a matriz em Q e R. So queremos Q;
    #      o "_" e a convencao para "valor que vou jogar fora".
    # [MAT] QR gera Q: 2 setas perpendiculares entre si, de comprimento 1,
    #       que cobrem o MESMO plano que DANO e TOPICO.
    #       (DANO e TOPICO se sobrepoem um pouco, lembra do ~0.10?
    #        Q "endireita" as duas para a subtracao sair exata.)

    v = v - Q @ (Q.T @ v)
    # [MAT] Q.T @ v   = quanto v aponta para cada seta de Q (2 numeros: sombras).
    #       Q @ (...) = monta, com essas sombras, a parte de v que esta no plano.
    #       v - (...) = tira essa parte. O que sobra e perpendicular a DANO e
    #                   a TOPICO. Isso e uma projecao ortogonal.
    #       Formula classica:  v_limpo = v - Q Q^T v
    return v / np.linalg.norm(v)

r_limpa = residualiza(r)
# [POR QUE] r_limpa nao aponta mais nada para DANO nem TOPICO.
#           Se ela ainda separar sotaques, nao e culpa desses dois.


# -----------------------------------------------------------------------------
# 5. SONDA: frases inocentes, mesmo topico, so muda o sotaque
# -----------------------------------------------------------------------------

A = prompts(700, 0, 1, 0)   # periferico: dano=0, sotaque=1, topico=0
H = prompts(700, 0, 0, 0)   # hegemonico: dano=0, sotaque=0, topico=0
# [POR QUE] Os dois grupos so diferem no sotaque. E o "par de sonda".

def d_cohen(v, A, H):
    a, h = A @ v, H @ v
    # [PY] (700, 64) @ (64,) = (700,): um numero por prompt.
    # [MAT] Projecao: produto escalar de cada prompt com a seta v.
    #       "Quanto este prompt aponta na direcao de recusa?"
    return (a.mean() - h.mean()) / np.sqrt((a.var() + h.var()) / 2)
    # [PY] .var() = variancia; np.sqrt = raiz quadrada.
    # [MAT] d de Cohen = (media_A - media_H) / desvio_padrao_combinado
    #       Mede a separacao em "unidades de desvio padrao".
    #       d = 1 => as medias estao a um desvio padrao de distancia.

d_obs = d_cohen(r_limpa, A, H)
# [POR QUE] Este e o numero que alguem publicaria: "o sotaque periferico
#           projeta d = X na direcao de recusa". Agora: e real ou nao?


# -----------------------------------------------------------------------------
# 6a. REGUA A — permutacao: embaralha quem e A e quem e H
# -----------------------------------------------------------------------------

P = np.vstack([A, H]); cont = 0
# [PY] np.vstack empilha verticalmente: (700,64) + (700,64) = (1400,64).
#      O ";" permite duas instrucoes na mesma linha. cont = contador.

for _ in range(500):
    # [PY] "_" = variavel que nao sera usada; so queremos repetir 500 vezes.
    idx = rng.permutation(1400)
    # [PY] Os numeros 0..1399 em ordem embaralhada.
    if abs(d_cohen(r_limpa, P[idx[:700]], P[idx[700:]])) >= abs(d_obs):
        cont += 1
    # [PY] idx[:700] = os primeiros 700; idx[700:] = do 700 ao fim.
    #      abs() = valor absoluto. "cont += 1" soma 1 ao contador.
    # [MAT] Mistura os rotulos e recalcula d NA MESMA SETA r_limpa.
    #       Conta quantas vezes o d embaralhado foi tao grande quanto o real.

p_perm = (cont + 1) / 501
# [MAT] p = fracao de embaralhamentos que empataram ou superaram o real.
#       O +1 no numerador e no denominador e a correcao padrao
#       (conta o proprio resultado real como uma das tentativas).
# [POR QUE] A pergunta que ela faz: "os grupos diferem ao longo de r_limpa?"
#           Sob superposicao QUALQUER seta capta um pouco da diferenca,
#           entao com 700 prompts a resposta e sempre "sim". ERRA com B=0.


# -----------------------------------------------------------------------------
# 6b. REGUA C — pipeline: refaz a DIRECAO com rotulos sorteados, mesmo processo
# -----------------------------------------------------------------------------

cont = 0
for _ in range(200):
    r_falsa = residualiza(direcao(X, rng.random(n) < 0.5))
    # [PY] rng.random(n) < 0.5 = 4000 True/False sorteados ao acaso,
    #      IGNORANDO a recusa verdadeira.
    # [MAT] Faz o pipeline inteiro de novo (passos 3 e 4) com um "juiz"
    #       que decide por moeda. r_falsa tem a mesma geometria de r_limpa
    #       (mesmo estimador, mesmos dados, mesma residualizacao),
    #       so NAO tem a informacao da recusa.
    if abs(d_cohen(r_falsa, A, H)) >= abs(d_obs):
        cont += 1
    # [MAT] Mesma sonda A/H, seta falsa. Quantas vezes a seta falsa
    #       separa os sotaques tanto quanto a seta verdadeira?

p_pipe = (cont + 1) / 201
# [POR QUE] A pergunta que ela faz: "esta seta separa os grupos MAIS do que
#           uma seta fabricada pelo mesmo processo, sem saber a recusa?"
#           Essa e a pergunta certa. ACERTA com B=0 e com B>0.


# -----------------------------------------------------------------------------
# RESULTADO
# -----------------------------------------------------------------------------

print(f"vies plantado B = {B}")
# [PY] f"..." e uma f-string: o que esta entre {} e trocado pelo valor.
print(f"d observado      = {d_obs:+.3f}")
# [PY] {:+.3f} = mostra o sinal (+/-) e 3 casas decimais.
print(f"regua A (permut) p = {p_perm:.4f}  ->", "ACHEI" if p_perm < .05 else "nada")
# [PY] "X if condicao else Y" = expressao condicional numa linha so.
print(f"regua C (pipeline) p = {p_pipe:.4f}  ->", "ACHEI" if p_pipe < .05 else "nada")
print("certo seria:", "nada" if B == 0 else "ACHEI")
# [PY] "==" compara (igual a?); "=" atribui. Nao confunda os dois.

# =============================================================================
# O QUE VOCE DEVE VER
#   B = 0.0 :  permutacao ACHEI (errado)   pipeline nada  (certo)
#   B = 0.3 :  permutacao ACHEI (certo)    pipeline ACHEI (certo)
# A permutacao diz "achei" nos dois mundos: nao distingue nada.
# O pipeline acerta os dois. Esse e o post.
# =============================================================================
