# pipeline_minimo.py — o post inteiro em ~60 linhas. So NumPy.
# Rode:  py pipeline_minimo.py      Depois mude B (linha 8) para 0.3 e rode de novo.
import numpy as np

N, M = 64, 200          # sala: 64 numeros por vetor, 200 conceitos
DANO, SOTAQUE, TOPICO = 0, 1, 2
B = 0.0                 # MUDE: 0.0 = juiz justo | 0.3 = juiz com vies
rng = np.random.default_rng(0)

# 1. CONCEITOS: 200 setas de comprimento 1
D = rng.normal(size=(M, N)); D /= np.linalg.norm(D, axis=1, keepdims=True)

def prompts(n, dano, sotaque, topico):
    """Cada prompt = soma das setas dos conceitos que ele tem + 8 de enchimento."""
    z = np.zeros((n, M))
    z[:, DANO], z[:, SOTAQUE], z[:, TOPICO] = dano, sotaque, topico
    for i in range(n):
        z[i, rng.choice(np.arange(3, M), 8, replace=False)] = rng.random(8)
    return z @ D + rng.normal(0, 0.1, (n, N))

# 2. REGRA SECRETA do juiz (voce sabe a resposta porque escreveu)
n = 4000
dano, sot, top = (rng.random((3, n)) < 0.5).astype(float)
X = prompts(n, dano, sot, top)
recusa = 1.0*dano + B*sot + 0.55*top + rng.normal(0, 0.3, n)

# 3. DIRECAO DE RECUSA = media(recusou muito) - media(recusou pouco)
def direcao(X, grupo):
    v = X[grupo].mean(0) - X[~grupo].mean(0); return v / np.linalg.norm(v)
r = direcao(X, recusa > np.median(recusa))

# 4. RESIDUALIZAR: tirar da seta a sombra sobre DANO e TOPICO
def residualiza(v):
    Q, _ = np.linalg.qr(D[[DANO, TOPICO]].T)
    v = v - Q @ (Q.T @ v); return v / np.linalg.norm(v)
r_limpa = residualiza(r)

# 5. SONDA: frases inocentes, mesmo topico, so muda o sotaque
A = prompts(700, 0, 1, 0)   # periferico
H = prompts(700, 0, 0, 0)   # hegemonico
def d_cohen(v, A, H):
    a, h = A @ v, H @ v
    return (a.mean() - h.mean()) / np.sqrt((a.var() + h.var()) / 2)
d_obs = d_cohen(r_limpa, A, H)

# 6a. REGUA A — permutacao: embaralha quem e A e quem e H
P = np.vstack([A, H]); cont = 0
for _ in range(500):
    idx = rng.permutation(1400)
    if abs(d_cohen(r_limpa, P[idx[:700]], P[idx[700:]])) >= abs(d_obs): cont += 1
p_perm = (cont + 1) / 501

# 6b. REGUA C — pipeline: refaz a DIRECAO com rotulos sorteados, mesmo processo
cont = 0
for _ in range(200):
    r_falsa = residualiza(direcao(X, rng.random(n) < 0.5))
    if abs(d_cohen(r_falsa, A, H)) >= abs(d_obs): cont += 1
p_pipe = (cont + 1) / 201

print(f"vies plantado B = {B}")
print(f"d observado      = {d_obs:+.3f}")
print(f"regua A (permut) p = {p_perm:.4f}  ->", "ACHEI" if p_perm < .05 else "nada")
print(f"regua C (pipeline) p = {p_pipe:.4f}  ->", "ACHEI" if p_pipe < .05 else "nada")
print("certo seria:", "nada" if B == 0 else "ACHEI")
