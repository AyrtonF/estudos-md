# Estudo Didático: Testes de Hipóteses Estatísticas

---

## 1. Introdução aos Testes de Hipóteses

Testes de hipóteses são ferramentas estatísticas que nos ajudam a tomar decisões baseadas em dados.  
Você propõe uma hipótese (chamada de hipótese nula, H0) e, com base nos dados, decide se há evidência suficiente para rejeitá-la em favor da hipótese alternativa (H1).

### Exemplo prático
Um hospital quer saber se um novo remédio reduz o tempo de recuperação.

- **Hipótese Nula (H0):** O remédio não tem efeito (tempo médio = 20 dias)
- **Hipótese Alternativa (H1):** O remédio tem efeito (tempo médio ≠ 20 dias)

---

## 2. Teste t para Uma Amostra

### Quando usar?
Quando você quer comparar a média de uma amostra com um valor conhecido/histórico.

### Fórmula da Estatística t

```
t = (x̄ - μ0) / (s / sqrt(n))
```

- x̄ = Média da amostra
- μ0 = Média esperada sob H0
- s = Desvio padrão da amostra
- n = Tamanho da amostra

#### Passo a Passo

1. Calcule a média (x̄) e o desvio padrão (s) dos dados.
2. Subtraia a média esperada (μ0) da média da amostra.
3. Divida o desvio padrão pela raiz quadrada do tamanho da amostra (s / sqrt(n)).
4. Divida o resultado do passo 2 pelo do passo 3.

#### Exemplo Numérico

- **Amostra:** 30 pacientes  
- x̄ = 17 dias  
- s = 2.5  
- n = 30  
- **H0:** μ0 = 20 dias

```
t = (17 - 20) / (2.5 / sqrt(30))
t = (-3) / (2.5 / 5.477)
t = (-3) / 0.456
t ≈ -6.58
```

**Interpretação:**  
O valor t indica que a média observada está 6.58 desvios padrões abaixo da média nula.

**Valor-p:**  
É a probabilidade de obter um resultado tão extremo quanto o observado, assumindo que H0 é verdadeira.  
Se p < 0.05: rejeitamos H0.

#### Código Python

```python
from scipy import stats
dados = [16, 18, 17, 15, ...]  # Valores da amostra
t_stat, p_valor = stats.ttest_1samp(dados, popmean=20)
print(f"t = {t_stat:.2f}, p = {p_valor:.4f}")
```

**Conclusão:**  
Se p < 0.05, rejeitamos H0. Ou seja, o remédio reduz o tempo de recuperação.

---

## 3. Teste t para Duas Amostras Independentes

### Quando usar?
Quando você quer comparar as médias de dois grupos distintos (ex: controle vs. tratado).

### Fórmula

```
t = (x̄1 - x̄2) / sqrt( (s1^2 / n1) + (s2^2 / n2) )
```

- x̄1, x̄2: médias dos grupos 1 e 2
- s1, s2: desvios padrões dos grupos
- n1, n2: tamanhos dos grupos

#### Passo a Passo

1. Calcule a média e o desvio padrão de cada grupo.
2. Subtraia as médias (x̄1 - x̄2).
3. Calcule (s1^2 / n1) e (s2^2 / n2), some e tire a raiz quadrada.
4. Divida a diferença de médias pelo resultado do passo 3.

#### Exemplo

- Grupo A: x̄1 = 20, s1 = 2.5, n1 = 30
- Grupo B: x̄2 = 18, s2 = 2.3, n2 = 30

```
t = (20 - 18) / sqrt( (2.5^2 / 30) + (2.3^2 / 30) )
t = 2 / sqrt( 6.25/30 + 5.29/30 )
t = 2 / sqrt(0.2083 + 0.1763)
t = 2 / sqrt(0.3846)
t = 2 / 0.620
t ≈ 3.23
```

**Interpretação:**  
Se t é alto (positivo ou negativo), indica diferença nas médias.

**Valor-p:**  
Se p < 0.05: diferença estatisticamente significativa.

#### Código Python

```python
from scipy import stats
grupo_A = [19, 21, 20, ...]
grupo_B = [17, 18, 19, ...]
t_stat, p_valor = stats.ttest_ind(grupo_A, grupo_B)
print(f"t = {t_stat:.2f}, p = {p_valor:.4f}")
```

**Conclusão:**  
Se p < 0.05, o novo tratamento é mais eficaz.

---

## 4. Teste Qui-Quadrado de Independência

### Quando usar?
Para verificar se existe associação entre duas variáveis categóricas (ex: plano de saúde vs. comparecimento).

### Fórmula

```
χ² = Σ ( (O - E)² / E )
```

- O: valores observados
- E: valores esperados sob independência

#### Passo a Passo

1. Monte a tabela de contingência (linhas = categoria 1, colunas = categoria 2).
2. Calcule o esperado para cada célula:
   - E = (Total linha * Total coluna) / Total geral
3. Para cada célula, calcule (O - E)² / E.
4. Some os valores para obter χ².

#### Exemplo

|           | Compareceu | Não Compareceu | Total |
|-----------|------------|----------------|-------|
| Plano A   | 40         | 10             | 50    |
| Plano B   | 25         | 15             | 40    |
| Total     | 65         | 25             | 90    |

**Cálculo para Plano A, Compareceu:**
```
E = (50 * 65) / 90 ≈ 36.11
```

**Cálculo para Plano A, Não Compareceu:**
```
E = (50 * 25) / 90 ≈ 13.89
```

**Agora, para cada célula:**
```
χ² = ((40-36.11)^2 / 36.11) + ((10-13.89)^2 / 13.89) + ... ≈ 5.12
```

**Valor-p:**  
Se p < 0.05, as variáveis são associadas.

#### Código Python

```python
import pandas as pd
from scipy.stats import chi2_contingency

tabela = pd.DataFrame({
    'Compareceu': [40, 25],
    'Não Compareceu': [10, 15]
})
chi2, p, dof, esperado = chi2_contingency(tabela)
print(f"χ² = {chi2:.2f}, p = {p:.4f}")
```

**Conclusão:**  
Se p < 0.05, o tipo de plano afeta o comparecimento.

---

## 5. Teste de Proporções

### Quando usar?
Para comparar uma proporção observada com uma esperada.

### Fórmula

```
z = (p̂ - p0) / sqrt( p0*(1-p0)/n )
```

- p̂: proporção observada (ex: 85/100 = 0.85)
- p0: proporção esperada (ex: 0.70)
- n: tamanho da amostra

#### Passo a Passo

1. Calcule a proporção observada (p̂).
2. Subtraia a proporção esperada (p0).
3. Calcule o denominador: p0*(1-p0)/n, tire a raiz quadrada.
4. Divida o resultado do passo 2 pelo do passo 3.

#### Exemplo

- Sucesso: 85 de 100 (p̂ = 0.85)
- Esperado: p0 = 0.70
- n = 100

```
z = (0.85 - 0.70) / sqrt( 0.7 * 0.3 / 100 )
z = 0.15 / sqrt( 0.0021 )
z = 0.15 / 0.0458
z ≈ 3.27
```

**Valor-p:**  
Se p < 0.05, a proporção observada é significativamente maior que a esperada.

#### Código Python

```python
from statsmodels.stats.proportion import proportions_ztest

stat, pval = proportions_ztest(count=85, nobs=100, value=0.7, alternative='larger')
print(f"z = {stat:.2f}, p = {pval:.4f}")
```

**Conclusão:**  
Se p < 0.05, a proporção de sucesso é maior que a esperada.

---

## 6. Resumo dos Testes

| Teste                 | Quando Usar?                | Fórmula Chave                                    | Decisão (p < 0.05)           |
|-----------------------|-----------------------------|--------------------------------------------------|------------------------------|
| t uma amostra         | Média amostral vs. valor    | t = (x̄ - μ0) / (s / sqrt(n))                   | Rejeita H0                   |
| t duas amostras       | Médias de 2 grupos          | t = (x̄1 - x̄2) / sqrt( (s1^2/n1) + (s2^2/n2) ) | Diferença significativa       |
| Qui-quadrado          | Assoc. entre categorias     | χ² = Σ ( (O-E)² / E )                           | Variáveis associadas          |
| Proporções            | Proporção observada vs. esp.| z = (p̂ - p0) / sqrt( p0*(1-p0)/n )             | Proporção diferente do esperado |

---

## 7. Dicas Finais

- Sempre escreva as hipóteses (H0 e H1) antes de calcular!
- Mostre cada passo do cálculo.
- Valor-p < 0.05 = resultado significativo (rejeite H0!).
- Cite que os testes partem do pressuposto de amostras aleatórias e independentes.
- Use exemplos numéricos para mostrar seu raciocínio na prova.
