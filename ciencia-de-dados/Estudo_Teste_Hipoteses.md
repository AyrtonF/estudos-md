# Estudo Didático: Testes de Hipóteses Estatísticas

---

## 1. Introdução aos Testes de Hipóteses

Testes de hipóteses são ferramentas estatísticas que nos ajudam a tomar decisões baseadas em dados.  
Você propõe uma hipótese (chamada de hipótese nula, H₀) e, com base nos dados, decide se há evidência suficiente para rejeitá-la em favor da hipótese alternativa (H₁).

### Exemplo prático
Um hospital quer saber se um novo remédio reduz o tempo de recuperação.

- **Hipótese Nula (H₀):** O remédio não tem efeito (tempo médio = 20 dias)
- **Hipótese Alternativa (H₁):** O remédio tem efeito (tempo médio ≠ 20 dias)

---

## 2. Teste t para Uma Amostra

### Quando usar?
Quando você quer comparar a média de uma amostra com um valor conhecido/histórico.

### Fórmula da Estatística t

\[
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
\]

- \(\bar{x}\) = Média da amostra
- \(\mu_0\) = Média esperada sob H₀
- \(s\) = Desvio padrão da amostra
- \(n\) = Tamanho da amostra

#### Passo a Passo

1. Calcule a média (\(\bar{x}\)) e o desvio padrão (\(s\)) dos dados.
2. Subtraia a média esperada (\(\mu_0\)) da média da amostra.
3. Divida o desvio padrão pelo raiz quadrada do tamanho da amostra (\(s / \sqrt{n}\)).
4. Divida o resultado do passo 2 pelo do passo 3.

#### Exemplo Numérico

- **Amostra:** 30 pacientes  
- \(\bar{x}\) = 17 dias  
- \(s\) = 2.5  
- \(n\) = 30  
- **H₀:** \(\mu_0 = 20\) dias

\[
t = \frac{17 - 20}{2.5 / \sqrt{30}} = \frac{-3}{2.5 / 5.477} = \frac{-3}{0.456} \approx -6.58
\]

**Interpretação:**  
O valor t indica que a média observada está 6.58 desvios padrões abaixo da média nula.

**Valor-p:**  
É a probabilidade de obter um resultado tão extremo quanto o observado, assumindo que H₀ é verdadeira.  
Se \(p < 0.05\): rejeitamos H₀.

#### Código Python

```python
from scipy import stats
dados = [16, 18, 17, 15, ...]  # Valores da amostra
t_stat, p_valor = stats.ttest_1samp(dados, popmean=20)
print(f"t = {t_stat:.2f}, p = {p_valor:.4f}")
```

**Conclusão:**  
Se \(p < 0.05\), rejeitamos H₀. Ou seja, o remédio reduz o tempo de recuperação.

---

## 3. Teste t para Duas Amostras Independentes

### Quando usar?
Quando você quer comparar as médias de dois grupos distintos (ex: controle vs. tratado).

### Fórmula

\[
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
\]

- \(\bar{x}_1, \bar{x}_2\): médias dos grupos 1 e 2
- \(s_1, s_2\): desvios padrões dos grupos
- \(n_1, n_2\): tamanhos dos grupos

#### Passo a Passo

1. Calcule a média e o desvio padrão de cada grupo.
2. Subtraia as médias (\(\bar{x}_1 - \bar{x}_2\)).
3. Calcule \(\frac{s_1^2}{n_1}\) e \(\frac{s_2^2}{n_2}\), some e tire a raiz quadrada.
4. Divida a diferença de médias pelo resultado do passo 3.

#### Exemplo

- Grupo A: \(\bar{x}_1 = 20\), \(s_1 = 2.5\), \(n_1 = 30\)
- Grupo B: \(\bar{x}_2 = 18\), \(s_2 = 2.3\), \(n_2 = 30\)

\[
t = \frac{20 - 18}{\sqrt{\frac{2.5^2}{30} + \frac{2.3^2}{30}}}
= \frac{2}{\sqrt{\frac{6.25}{30} + \frac{5.29}{30}}}
= \frac{2}{\sqrt{0.2083 + 0.1763}}
= \frac{2}{\sqrt{0.3846}}
= \frac{2}{0.620}
\approx 3.23
\]

**Interpretação:**  
Se \(t\) é alto (positivo ou negativo), indica diferença nas médias.

**Valor-p:**  
Se \(p < 0.05\): diferença estatisticamente significativa.

#### Código Python

```python
from scipy import stats
grupo_A = [19, 21, 20, ...]
grupo_B = [17, 18, 19, ...]
t_stat, p_valor = stats.ttest_ind(grupo_A, grupo_B)
print(f"t = {t_stat:.2f}, p = {p_valor:.4f}")
```

**Conclusão:**  
Se \(p < 0.05\), o novo tratamento é mais eficaz.

---

## 4. Teste Qui-Quadrado de Independência

### Quando usar?
Para verificar se existe associação entre duas variáveis categóricas (ex: plano de saúde vs. comparecimento).

### Fórmula

\[
\chi^2 = \sum \frac{(O - E)^2}{E}
\]

- \(O\): valores observados
- \(E\): valores esperados sob independência

#### Passo a Passo

1. Monte a tabela de contingência (linhas = categoria 1, colunas = categoria 2).
2. Calcule o esperado para cada célula:
   - \(E = \frac{\text{Total linha} \times \text{Total coluna}}{\text{Total geral}}\)
3. Para cada célula, calcule \((O - E)^2 / E\).
4. Some os valores para obter \(\chi^2\).

#### Exemplo

|           | Compareceu | Não Compareceu | Total |
|-----------|------------|----------------|-------|
| Plano A   | 40         | 10             | 50    |
| Plano B   | 25         | 15             | 40    |
| Total     | 65         | 25             | 90    |

**Cálculo para Plano A, Compareceu:**
\[
E = \frac{50 \times 65}{90} \approx 36.11
\]

**Cálculo para Plano A, Não Compareceu:**
\[
E = \frac{50 \times 25}{90} \approx 13.89
\]

**Agora, para cada célula:**
\[
\chi^2 = \sum \frac{(O - E)^2}{E}
\]
\[
\chi^2 = \frac{(40-36.11)^2}{36.11} + \frac{(10-13.89)^2}{13.89} + ... \approx 5.12
\]

**Valor-p:**  
Se \(p < 0.05\), as variáveis são associadas.

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
Se \(p < 0.05\), o tipo de plano afeta o comparecimento.

---

## 5. Teste de Proporções

### Quando usar?
Para comparar uma proporção observada com uma esperada.

### Fórmula

\[
z = \frac{\hat{p} - p_0}{\sqrt{ \frac{p_0(1-p_0)}{n} } }
\]

- \(\hat{p}\): proporção observada (ex: 85/100 = 0.85)
- \(p_0\): proporção esperada (ex: 0.70)
- \(n\): tamanho da amostra

#### Passo a Passo

1. Calcule a proporção observada (\(\hat{p}\)).
2. Subtraia a proporção esperada (\(p_0\)).
3. Calcule o denominador: \(p_0(1-p_0)/n\), tire a raiz quadrada.
4. Divida o resultado do passo 2 pelo do passo 3.

#### Exemplo

- Sucesso: 85 de 100 (\(\hat{p} = 0.85\))
- Esperado: \(p_0 = 0.70\)
- \(n = 100\)

\[
z = \frac{0.85 - 0.70}{ \sqrt{ \frac{0.7 \times 0.3}{100} } }
= \frac{0.15}{ \sqrt{ 0.0021 } }
= \frac{0.15}{0.0458}
\approx 3.27
\]

**Valor-p:**  
Se \(p < 0.05\), a proporção observada é significativamente maior que a esperada.

#### Código Python

```python
from statsmodels.stats.proportion import proportions_ztest

stat, pval = proportions_ztest(count=85, nobs=100, value=0.7, alternative='larger')
print(f"z = {stat:.2f}, p = {pval:.4f}")
```

**Conclusão:**  
Se \(p < 0.05\), a proporção de sucesso é maior que a esperada.

---

## 6. Resumo dos Testes

| Teste                 | Quando Usar?                | Fórmula Chave                | Decisão (p < 0.05)           |
|-----------------------|-----------------------------|------------------------------|------------------------------|
| t uma amostra         | Média amostral vs. valor    | \( t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}} \) | Rejeita H₀                   |
| t duas amostras       | Médias de 2 grupos          | \( t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}} \) | Diferença significativa       |
| Qui-quadrado          | Assoc. entre categorias     | \( \chi^2 = \sum \frac{(O-E)^2}{E} \) | Variáveis associadas          |
| Proporções            | Proporção observada vs. esp.| \( z = \frac{\hat{p} - p_0}{\sqrt{ \frac{p_0(1-p_0)}{n} } } \) | Proporção diferente do esperado |

---

## 7. Dicas Finais

- Sempre escreva as hipóteses (H₀ e H₁) antes de calcular!
- Mostre cada passo do cálculo.
- Valor-p < 0.05 = resultado significativo (rejeite H₀!).
- Cite que os testes partem do pressuposto de amostras aleatórias e independentes.
- Use exemplos numéricos para mostrar seu raciocínio na prova.